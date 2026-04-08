import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm.auto import tqdm

from model.policy_head import PolicyHead
from utils.libero_utils import build_libero_env
from utils.libero_utils.viz import make_grid_video_from_numpy, render_done_to_boundary, video_pad_time


def parse_args():
    parser = argparse.ArgumentParser(description="LIBERO policy evaluation script.")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--num_env_rollouts", type=int, default=10)
    parser.add_argument("--vec_env_num", type=int, default=10, help="Number of parallel environments.")
    parser.add_argument("--track_pred_nfe", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--rollout_horizon", type=int, default=600)
    parser.add_argument("--mix_precision", action="store_true")
    parser.add_argument(
        "--amplify_libero_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "LIBERO" / "libero" / "libero"),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "LIBERO" / "datasets"),
    )
    parser.add_argument(
        "--task_emb_cache_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "LIBERO" / "task_embedding_caches" / "task_emb_bert.npy"),
    )
    return parser.parse_args()


def merge_results(results: list[dict]):
    merged = {}
    for result in results:
        merged.update(result)

    returns = np.array([v for k, v in merged.items() if k.startswith("rollout/return_env")], dtype=np.float32)
    horizons = np.array([v for k, v in merged.items() if k.startswith("rollout/horizon_env")], dtype=np.float32)
    successes = np.array([v for k, v in merged.items() if k.startswith("rollout/success_env")], dtype=np.float32)

    merged["rollout/return_env_avg"] = float(np.mean(returns))
    merged["rollout/horizon_env_avg"] = float(np.mean(horizons))
    merged["rollout/success_env_avg"] = float(np.mean(successes))
    return merged


@torch.no_grad()
def rollout_env(
    env_idx,
    env,
    policy,
    num_env_rollouts,
    task_emb,
    horizon,
    progress_every=25,
):
    policy.eval()
    policy_ref = policy.module if hasattr(policy, "module") else policy

    if isinstance(task_emb, torch.Tensor):
        task_emb = task_emb.detach().cpu().numpy()
    task_emb = np.asarray(task_emb)

    obs_key_mapping = {
        "gripper_states": "robot0_gripper_qpos",
        "joint_states": "robot0_joint_pos",
    }

    all_rewards = []
    all_success = []
    all_horizons = []
    videos = []

    for rollout_i in tqdm(range(num_env_rollouts), desc=f"env{env_idx} rollouts", leave=False):
        print(
            f"[progress] env={env_idx} rollout={rollout_i + 1}/{num_env_rollouts} reset",
            flush=True,
        )
        obs = env.reset()
        policy_ref.reset()

        n_envs = obs["image"].shape[0]
        success = np.zeros(n_envs, dtype=bool)
        reward = np.zeros(n_envs, dtype=np.float32)
        episode_frames = []

        for step_i in range(horizon):
            rgb = obs["image"]

            step_task_emb = obs.get("task_emb", task_emb)
            if isinstance(step_task_emb, torch.Tensor):
                step_task_emb = step_task_emb.detach().cpu().numpy()
            step_task_emb = np.asarray(step_task_emb)
            if step_task_emb.ndim == 1:
                step_task_emb = np.repeat(step_task_emb[None, :], rgb.shape[0], axis=0)

            extra_states = {k: obs[obs_key_mapping[k]] for k in policy_ref.extra_state_keys}
            action, _ = policy_ref.act(rgb, step_task_emb, extra_states)
            obs, r, done, info = env.step(action)

            reward += np.asarray(r, dtype=np.float32)
            success = np.logical_or(success, np.asarray(info.get("success", done), dtype=bool))
            done_all = bool(np.all(success))

            if step_i == 0 or ((step_i + 1) % progress_every == 0) or done_all:
                print(
                    f"[progress] env={env_idx} rollout={rollout_i + 1}/{num_env_rollouts} "
                    f"step={step_i + 1}/{horizon} done={done_all} success_count={int(success.sum())}/{len(success)}",
                    flush=True,
                )

            video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
            b, _, c, h, _ = video_img.shape
            frame = np.concatenate(
                [video_img[:, 0], np.ones((b, c, h, 2), dtype=np.uint8) * 255, video_img[:, 1]],
                axis=-1,
            )
            frame = render_done_to_boundary(frame, success.tolist())
            episode_frames.append(frame)

            if done_all:
                break

        episode_videos = np.stack(episode_frames, axis=1)
        videos.extend(list(episode_videos))
        all_rewards.extend(reward.tolist())
        all_success.extend(success.astype(np.float32).tolist())
        all_horizons.append(step_i + 1)

        print(
            f"Finished rollout env={env_idx} rollout={rollout_i + 1}/{num_env_rollouts} "
            f"success={success.tolist()} reward={reward.tolist()} steps={step_i + 1}",
            flush=True,
        )

    videos = video_pad_time(videos)
    return {
        f"rollout/return_env{env_idx}": float(np.mean(all_rewards)),
        f"rollout/horizon_env{env_idx}": float(np.mean(all_horizons)),
        f"rollout/success_env{env_idx}": float(np.mean(all_success)),
        f"rollout/vis_env{env_idx}": videos,
    }


def evaluate(accelerator: Accelerator, args, video_save_dir: str):
    model = PolicyHead(track_pred_nfe=args.track_pred_nfe, track_pred_cfg=args.cfg_scale)
    model.load_state_dict(load_file(args.ckpt_path, device="cpu"))
    model = accelerator.prepare(model)

    os.makedirs(video_save_dir, exist_ok=True)

    suite_dir = os.path.join(args.amplify_libero_path, "init_files", args.suite)
    total_envs = len(sorted(os.listdir(suite_dir)))

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    envs_per_rank = math.ceil(total_envs / world_size)
    env_idx_start = envs_per_rank * rank
    env_idx_end = min(envs_per_rank * (rank + 1), total_envs)

    print(f"Rank {rank}/{world_size}: evaluating envs [{env_idx_start}, {env_idx_end})", flush=True)

    per_rank_results = []
    per_rank_success = []
    rollouts_per_env = args.num_env_rollouts // args.vec_env_num

    for env_idx in range(env_idx_start, env_idx_end):
        env_cfg = OmegaConf.create(
            {
                "task_suite": args.suite,
                "task_no": env_idx,
                "img_size": args.img_size,
                "action_dim": 7,
                "n_envs": args.vec_env_num,
                "dataset_path": args.dataset_path,
                "libero_path": args.amplify_libero_path,
                "task_emb_cache_path": args.task_emb_cache_path,
            }
        )
        env, _, task_emb = build_libero_env(**env_cfg)

        result = rollout_env(
            env_idx=env_idx,
            env=env,
            policy=model,
            num_env_rollouts=rollouts_per_env,
            task_emb=task_emb,
            horizon=args.rollout_horizon,
        )

        video = result.pop(f"rollout/vis_env{env_idx}")
        for vid_idx in range(video.shape[0]):
            video_path = os.path.join(video_save_dir, f"env_{env_idx}_rollout_{vid_idx}.mp4")
            make_grid_video_from_numpy(
                [rearrange(video[vid_idx], "t c h w -> t h w c")],
                ncol=1,
                output_name=video_path,
            )

        per_rank_success.append(result[f"rollout/success_env{env_idx}"])
        print(
            f"Rank {rank}: env {env_idx} success={result[f'rollout/success_env{env_idx}']:.4f} "
            f"(running mean={sum(per_rank_success) / len(per_rank_success):.4f})",
            flush=True,
        )
        per_rank_results.append(result)
        del env

    return merge_results(per_rank_results)


def main():
    args = parse_args()
    print(f"Starting evaluation for checkpoint: {args.ckpt_path}", flush=True)

    eval_result_dir = os.path.join(args.save_path, "eval_results")
    video_save_dir = os.path.join(eval_result_dir, f"video_{args.suite}", "ours")
    os.makedirs(eval_result_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16" if args.mix_precision else "no", cpu=False)
    local_results = evaluate(accelerator=accelerator, args=args, video_save_dir=video_save_dir)

    accelerator.wait_for_everyone()
    gathered_results = accelerator.gather_object(local_results) if accelerator.num_processes > 1 else [local_results]

    if accelerator.is_main_process:
        merged = merge_results(gathered_results)
        success_rates = {k: v for k, v in merged.items() if k.startswith("rollout/success_env")}
        print(f"Success rates: {success_rates}", flush=True)
        print(f"Mean success rate: {merged['rollout/success_env_avg']:.4f}", flush=True)


if __name__ == "__main__":
    main()
