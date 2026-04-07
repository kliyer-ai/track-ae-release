import math

import einops
import torch
from jaxtyping import Float
from torch import nn

from model.dino import MinDino
from model.rope import centers, make_axial_pos_2d
from model.transformer import FeedForwardBlock, InputMLP, Level, RMSNorm, TransformerLayer
from model.vae import TrackVAE


class CondTokenConcatMerge(nn.Module):
    def __init__(self, in_features: int, out_features: int, cond_features: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.cond_proj = nn.Linear(cond_features, out_features, bias=False)

    def forward(self, x, pos, extra_tokens, extra_pos, **kwargs):
        return torch.cat((self.proj(x), self.cond_proj(extra_tokens)), dim=1), torch.cat((pos, extra_pos), dim=1)


class CondTokenConcatSplit(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_extra_tokens: int):
        super().__init__()
        self.num_extra_tokens = num_extra_tokens
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, **kwargs):
        return self.proj(x[:, : -self.num_extra_tokens])


class MappingNetwork(nn.Module):
    def __init__(self, depth, width, d_ff):
        super().__init__()
        self.in_norm = RMSNorm(width)
        self.blocks = nn.ModuleList([FeedForwardBlock(width, d_ff) for _ in range(depth)])
        self.out_norm = RMSNorm(width)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        features = 2 * math.pi * x @ self.weight.T
        return torch.cat((features.cos(), features.sin()), dim=-1)


def _broadcast_channel_stat(stat: torch.Tensor | None, latents: torch.Tensor) -> torch.Tensor | None:
    if stat is None:
        return None
    stat = stat.to(device=latents.device, dtype=torch.float64)
    if stat.ndim == 0:
        return stat.view(1, 1, 1)
    if stat.ndim == 1:
        return stat.view(1, 1, -1)
    raise ValueError(f"Expected scalar or per-channel stat, got shape {tuple(stat.shape)}")


class TrackFM(nn.Module):
    """
    Clean generator implementation for the released `traj_gen_v2` model.

    This module intentionally supports only the configuration used in
    `configs/experiment/traj_gen_v2.yaml`:
    - 1 x 16 x 16 latent grid with 16 latent channels
    - 24-layer 1024-dim transformer backbone
    - DINOv2-B/14 start-frame tokens
    - 8 endpoint conditions with Poisson condition dropping during training
    """

    def __init__(
        self,
        vae: TrackVAE,
        depth: int = 24,
        width: int = 1024,
        d_cross: int = 768,
        d_cond_norm: int = 256,
        vae_shift: float = -0.175,
        vae_scale: float = 11.6,
        n_cond: int = 8,
        poisson_rate: float = 2.5,
    ):
        super().__init__()

        self.vae = vae

        self.depth = depth
        self.width = width
        self.d_cross = d_cross
        self.d_cond_norm = d_cond_norm

        self.grid_size = (1, 16, 16)
        self.latent_dim = 16
        self.n_cond = n_cond
        self.poisson_rate = float(poisson_rate)
        self.cfg_scale = float(1)

        self.img_embedder = MinDino("dinov2_vitb14_reg", out="features", requires_grad=False)
        self.track_cond_emb = InputMLP(in_features=3, dim=d_cross, random_fourier=True, theta=10.0 * math.pi)
        self.time_emb = FourierFeatures(1, 256)
        self.time_in_proj = nn.Linear(256, 256, bias=False)
        self.mapping = MappingNetwork(depth=2, width=256, d_ff=768)
        self.in_proj = CondTokenConcatMerge(in_features=self.latent_dim, out_features=1024, cond_features=d_cross)

        self.backbone = Level(
            [
                TransformerLayer(
                    d_model=width,
                    d_cross=d_cross,
                    d_cond_norm=d_cond_norm,
                )
                for _ in range(depth)
            ]
        )

        self.out_proj = CondTokenConcatSplit(
            in_features=1024,
            out_features=self.latent_dim,
            num_extra_tokens=self.grid_size[1] * self.grid_size[2],
        )

        self.learnable_emb_dino = nn.Parameter(torch.randn(d_cross) * 0.02)

        self.vae_shift = torch.tensor(vae_shift)
        self.vae_scale = torch.tensor(vae_scale)

    def get_pos(self, x: Float[torch.Tensor, "b l c"]) -> Float[torch.Tensor, "b l 3"]:
        batch_size = x.shape[0]
        num_latent_tokens = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        assert x.shape[1] == num_latent_tokens, f"Expected {num_latent_tokens} latent tokens, got {x.shape[1]}"

        time_pos = einops.repeat(
            centers(-1, 1, self.grid_size[0], device=x.device),
            "t -> (t h w) 1",
            h=self.grid_size[1],
            w=self.grid_size[2],
        )
        spatial_pos = einops.repeat(
            make_axial_pos_2d(self.grid_size[1], self.grid_size[2], device=x.device),
            "(h w) c -> (t h w) c",
            t=self.grid_size[0],
            h=self.grid_size[1],
            w=self.grid_size[2],
        )
        pos = torch.cat((time_pos, spatial_pos), dim=-1)
        return pos.unsqueeze(0).expand(batch_size, -1, -1)

    def _time_condition(self, t: Float[torch.Tensor, "b"]) -> Float[torch.Tensor, "b c"]:
        time_features = self.time_emb(t[:, None])
        time_features = self.time_in_proj(time_features)
        return self.mapping(time_features)

    def get_frame_embed(self, start_frame, time_idx: int):
        frame_embed = self.img_embedder(start_frame)
        batch_size, height, width, channels = frame_embed.shape
        frame_embed = einops.rearrange(frame_embed, "b h w c -> b (h w) c")
        time_pos = centers(-1, 1, self.grid_size[0], device=frame_embed.device)[time_idx].expand(height * width, 1)
        pos = torch.cat(
            (time_pos, make_axial_pos_2d(height, width, device=frame_embed.device)),
            dim=-1,
        )
        return frame_embed, pos.unsqueeze(0).expand(batch_size, -1, -1)

    def get_start_frame_embed(self, start_frame):
        frame_embed, frame_pos = self.get_frame_embed(start_frame, time_idx=0)
        frame_embed = frame_embed + einops.rearrange(self.learnable_emb_dino, "c -> 1 1 c")
        return frame_embed, frame_pos

    def get_static_conditioning(self, start_frame, track_conds):
        batch_size = start_frame.shape[0]
        start_embed, start_pos = self.get_start_frame_embed(start_frame)

        start_cond = track_conds[..., :2]
        end_cond = track_conds[..., 2:]
        track_cond_emb = self.track_cond_emb(end_cond)
        num_cond = track_cond_emb.shape[1]

        pos_cross = torch.cat(
            (
                centers(-1, 1, self.grid_size[0], device=start_cond.device)[0].expand(batch_size, num_cond, 1),
                start_cond,
            ),
            dim=-1,
        )

        if self.training:
            n_drop = torch.poisson(torch.full((batch_size,), self.poisson_rate, device=start_cond.device))
            n_drop = n_drop.clamp(0, num_cond).long()
            keep_mask = torch.arange(num_cond, device=start_cond.device)[None] < (num_cond - n_drop[:, None])
            track_cond_emb = track_cond_emb * keep_mask[..., None]
            pos_cross = pos_cross * keep_mask[..., None]

        return {
            "extra_tokens": start_embed,
            "extra_pos": start_pos,
            "x_cross": track_cond_emb,
            "pos_cross": pos_cross,
        }

    def get_static_unconditioning(self, start_frame):
        batch_size = start_frame.shape[0]
        start_embed, start_pos = self.get_start_frame_embed(start_frame)
        return {
            "extra_tokens": start_embed,
            "extra_pos": start_pos,
            "x_cross": torch.zeros(
                batch_size,
                self.n_cond,
                self.d_cross,
                dtype=start_frame.dtype,
                device=start_frame.device,
            ),
            "pos_cross": torch.zeros(
                (batch_size, self.n_cond, 3),
                dtype=start_frame.dtype,
                device=start_frame.device,
            ),
        }

    def get_track_cond(self, tracks: Float[torch.Tensor, "b n t 2"]):
        batch_size, num_tracks, num_points, _ = tracks.shape
        assert self.n_cond <= num_tracks, "traj_gen_v2 samples endpoint conditions without replacement"

        traj_idx = torch.stack(
            [torch.randperm(num_tracks, device=tracks.device)[: self.n_cond] for _ in range(batch_size)]
        )
        batch_idx = einops.repeat(
            torch.arange(batch_size, device=tracks.device),
            "b -> b n_cond",
            n_cond=self.n_cond,
        )
        end_t = torch.full((batch_size, self.n_cond), num_points - 1, device=tracks.device)

        start_points = tracks[batch_idx, traj_idx, 0]
        end_points = tracks[batch_idx, traj_idx, end_t]
        end_time = end_t.unsqueeze(-1).float() / (num_points - 1)
        return torch.cat((start_points, end_points, end_time), dim=-1), traj_idx

    def normalize_latents(self, latents: Float[torch.Tensor, "b l c"]) -> Float[torch.Tensor, "b l c"]:
        vae_shift = _broadcast_channel_stat(self.vae_shift, latents)
        vae_scale = _broadcast_channel_stat(self.vae_scale, latents)

        if vae_shift is not None:
            latents = latents - vae_shift
        if vae_scale is not None:
            latents = latents * (math.sqrt(0.5) / vae_scale.sqrt())
        return latents

    def denormalize_latents(self, latents: Float[torch.Tensor, "b l c"]) -> Float[torch.Tensor, "b l c"]:
        vae_shift = _broadcast_channel_stat(self.vae_shift, latents)
        vae_scale = _broadcast_channel_stat(self.vae_scale, latents)

        if vae_scale is not None:
            latents = latents * (vae_scale.sqrt() / math.sqrt(0.5))
        if vae_shift is not None:
            latents = latents + vae_shift
        return latents

    def _predict_velocity(self, x_t, t, static_cond):
        cond_norm = self._time_condition(t.to(x_t.dtype))
        pos = self.get_pos(x_t)
        x_t, pos = self.in_proj(x_t, pos, **static_cond)
        x_t = self.backbone(x_t, pos=pos, cond_norm=cond_norm, **static_cond)
        return self.out_proj(x_t)

    def _rf_loss(self, latents, start_frame, track_conds):
        batch_size = latents.shape[0]
        t = torch.sigmoid(torch.randn((batch_size,), device=latents.device, dtype=latents.dtype))
        t_expanded = t.view(batch_size, 1, 1)
        noise = torch.randn_like(latents)
        x_t = (1 - t_expanded) * latents + t_expanded * noise

        static_cond = self.get_static_conditioning(start_frame=start_frame, track_conds=track_conds)
        velocity = self._predict_velocity(x_t, t, static_cond)
        return ((noise - latents - velocity) ** 2).mean(dim=(1, 2))

    def forward(
        self,
        tracks_enc_yx: Float[torch.Tensor, "b n t 2"],
        start_frame: Float[torch.Tensor, "b h w c"],
        **kwargs,
    ):
        with torch.no_grad():
            encoded = self.vae.encode(tracks_enc_yx, start_frame=start_frame)
            latents = self.normalize_latents(encoded.mean)
            track_conds, _ = self.get_track_cond(tracks_enc_yx)

        rf_loss = self._rf_loss(latents, start_frame=start_frame, track_conds=track_conds)
        return dict(rf_loss=rf_loss), dict()

    @torch.no_grad()
    def sample(
        self,
        z: Float[torch.Tensor, "b l c"],
        points_per_traj: int,
        query_pos: Float[torch.Tensor, "b n 2"],
        track_conds: Float[torch.Tensor, "b n_cond 5"],
        start_frame: Float[torch.Tensor, "b h w c"],
        sample_steps: int = 50,
        decode_latent: bool = True,
    ):
        batch_size = z.shape[0]
        dt = torch.full((batch_size, 1, 1), 1.0 / sample_steps, device=z.device, dtype=z.dtype)

        static_cond = self.get_static_conditioning(start_frame=start_frame, track_conds=track_conds)
        static_uncond = self.get_static_unconditioning(start_frame=start_frame) if self.cfg_scale > 1.0 else None

        latents = z
        for step in range(sample_steps, 0, -1):
            t = torch.full((batch_size,), step / sample_steps, device=z.device, dtype=z.dtype)
            cond_velocity = self._predict_velocity(latents, t, static_cond)

            if static_uncond is None:
                latents = latents - dt * cond_velocity
                continue

            uncond_velocity = self._predict_velocity(latents, t, static_uncond)
            guided_velocity = uncond_velocity + self.cfg_scale * (cond_velocity - uncond_velocity)
            latents = latents - dt * guided_velocity

        latents = self.denormalize_latents(latents)
        if not decode_latent:
            return latents

        return self.vae.decode(
            latents=latents,
            query_pos=query_pos,
            points_per_track=points_per_traj,
            start_frame=start_frame,
        )
