# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich
# Adapted from https://github.com/stefan-baumann/flow-poke-transformer/blob/main/scripts/demo/app.py by Nick Stracke

import glob
import math
import os
import tempfile
from dataclasses import dataclass, field
from functools import partial

import einops
import fire
import gradio as gr
import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import UInt8
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image

from model.gen import TrackFM
from model.rope import make_axial_pos_2d
from model.vae import TrackVAE

VAL_SHAPE = [256, 16]


def visualize_trajectories(
    start_frame: UInt8[np.ndarray, "H W C"] | None,
    trajectories: UInt8[np.ndarray, "N T 2"] | list[UInt8[np.ndarray, "T 2"]],
    elev=10,
    azim=75,
    show_colorbar=False,
    show_points=False,
    show_lines=True,
    proj_type="ortho",
    shadows=True,
    keep_aspect=True,
    center_crop: bool = False,
):
    """
    Visualize particle trajectories in 3D with time as the third dimension.

    Parameters:
    -----------
    start_frame : numpy.ndarray
        The initial frame image (2D or 3D array)
    trajectories : list of numpy.ndarray
        List of trajectories, where each trajectory is an array of shape (N, 2) or (N, 3)
        containing (x, y) or (x, y, t) coordinates over time
    """
    H, W, _ = start_frame.shape
    min_size = min(H, W)

    if center_crop:  # only center crop the visualization
        start_h = (H - min_size) // 2
        start_w = (W - min_size) // 2
        start_frame = start_frame[start_h : start_h + min_size, start_w : start_w + min_size, :]
        H, W, _ = start_frame.shape
        # adjust trajectory pixel coords accordingly
        trajectories = (trajectories - np.array([start_w, start_h], dtype=trajectories.dtype)) * np.array(
            [min_size / W, min_size / H], dtype=trajectories.dtype
        )

    fig = plt.figure(figsize=(5, 5))
    # computed_zorder=False to allow manual z-ordering
    # this is because .scatter() is buggy with zorder otherwise
    ax: Axes3D = fig.add_subplot(111, projection="3d", computed_zorder=False, proj_type=proj_type)

    if keep_aspect:
        ax.set_box_aspect((W / H, 1, 1))  # this is the only way I found to adjust the aspect ratio
    else:
        ax.set_box_aspect((1, 1, 1))  # this is needed bc otherwise images are somehow slightly non-square

    # not sure if these two lines are needed, but they do no harm
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    # Display the start frame as a textured surface at y=0
    if start_frame is not None:
        # Use actual pixel coordinates for full resolution
        x_img = np.arange(W)
        z_img = np.arange(H)
        X_img, Z_img = np.meshgrid(x_img, z_img)
        Y_img = np.zeros_like(X_img)

        # Handle grayscale or color images
        if len(start_frame.shape) == 2:
            colors = plt.cm.gray(start_frame / 255.0)
        else:
            # Normalize if needed
            if start_frame.max() > 1.0:
                colors = start_frame / 255.0
            else:
                colors = start_frame

        ax.plot_surface(
            X_img,
            Y_img,
            Z_img,
            facecolors=colors,
            shade=False,
            alpha=0.7,
            antialiased=True,
            rcount=H,
            ccount=W,
            zorder=-1,  # ensure the image is at the back
        )

    # Create a colormap for time progression (blue to red)
    cmap = cm.get_cmap("jet")

    # Find the maximum time across all trajectories for normalization
    max_time = 0
    for traj in trajectories:
        if traj.shape[1] >= 3:
            max_time = max(max_time, traj[:, 2].max())
        else:
            max_time = max(max_time, len(traj))

    # Plot shadows
    if shadows:
        for traj in trajectories:
            if len(traj) < 2:
                continue

            # Extract coordinates
            if traj.shape[1] >= 3:
                x, z, t = traj[:, 0], traj[:, 1], traj[:, 2]
            else:
                x, z = traj[:, 0], traj[:, 1]
                t = np.arange(len(traj))

            # Fake blurry variant
            for lw in np.exp(np.linspace(math.log(1), math.log(16), 50)):
                ax.plot(
                    x,
                    t,
                    np.full(z.shape, H),
                    color="black",
                    linewidth=lw,
                    alpha=0.005,
                    zorder=-10,
                    marker="",
                )

    # Plot each trajectory
    for traj in trajectories:
        if len(traj) < 2:
            continue

        # Extract coordinates
        if traj.shape[1] >= 3:
            x, z, t = traj[:, 0], traj[:, 1], traj[:, 2]
        else:
            x, z = traj[:, 0], traj[:, 1]
            t = np.arange(len(traj))

        # Normalize time for coloring
        t_norm = t / max_time if max_time > 0 else t

        if show_lines:
            # Plot the trajectory line segments with color gradient
            for i in range(len(x) - 1):
                color = cmap(t_norm[i])
                current_t = t[i]
                ax.plot(
                    [x[i], x[i + 1]],
                    [t[i], t[i + 1]],
                    [z[i], z[i + 1]],
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    zorder=current_t + 50,
                    marker="",
                )
        if show_points:
            # Add dots at each point
            colors_scatter = [cmap(tn) for tn in t_norm]
            for i in range(len(x)):
                current_t = t[i]
                ax.scatter(
                    [x[i]],
                    [t[i]],
                    [z[i]],
                    c=[colors_scatter[i]],
                    s=30,
                    alpha=1.0,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=current_t + 50 + 1e-6,  # ensure points are on top of lines
                )

    # Set labels and title
    ax.set_ylabel("  Time    ", labelpad=-12)

    # Time axis
    ax.set_yticks([], minor=True)
    ax.set_yticks([16, 32, 48], minor=False)
    ax.set_yticklabels([])
    x_ticks = [(W / 5) * i for i in range(1, 5)]
    ax.set_xticks(x_ticks, minor=False)
    ax.set_zticks([50, 100, 150, 200], minor=False)
    # Image axes
    ax.set_xticks([], minor=True)
    ax.set_zticks([], minor=True)
    ax.tick_params(axis="x", which="major", length=0, width=0, zorder=-100)
    ax.tick_params(axis="z", which="major", length=0, width=0, zorder=-100)
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.set_facecolor("#F1F1F1")
    ax.zaxis.pane.set_facecolor("#F1F1F1")

    ax.plot([0, W - 1], [0, 0], [0, 0], color="black", linewidth=1, alpha=0.66, zorder=10, clip_on=False)
    ax.plot([0, 0], [0, 0], [0, H], color="black", linewidth=1, alpha=0.66, zorder=10, clip_on=False)
    ax.plot([0, W - 1], [0, 0], [H, H], color="black", linewidth=1, alpha=0.66, zorder=10, clip_on=False)
    ax.plot([W - 1, W - 1], [0, 0], [0, H], color="black", linewidth=1, alpha=0.66, zorder=10, clip_on=False)
    ax.plot([0, W], [max_time, max_time], [0, 0], color="black", linewidth=1, alpha=0.66, zorder=1000, clip_on=False)
    ax.plot([0, 0], [max_time, max_time], [0, H], color="black", linewidth=1, alpha=0.66, zorder=1000, clip_on=False)
    ax.plot([W, W], [max_time, max_time], [0, H], color="black", linewidth=1, alpha=0.66, zorder=1000, clip_on=False)
    ax.plot([W - 1, W], [0, max_time], [0, 0], color="black", linewidth=1, alpha=0.66, zorder=10, clip_on=False)
    ax.plot(
        [0, 0],
        [0, max_time],
        [0, 0],
        color="black",
        linewidth=1,
        alpha=0.66,
        linestyle=(0, (5, 5)),
        zorder=1000,
        clip_on=False,
    )

    # Set the viewing angle for better visualization
    ax.view_init(elev=elev, azim=azim)

    # Add a colorbar
    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_time))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("Time (frames)", fontsize=12)

    # Set axis limits
    if start_frame is not None:
        # the origin (0,0) is at the top-left corner of the image
        ax.set_xlim(W, 0)
        ax.set_zlim(H, 0)
    ax.set_ylim(0, max_time)

    return fig, ax


def draw_trajectories_on_frame(
    frame: torch.Tensor,  # [H, W, C] float in [0,1]
    tracks: torch.Tensor,  # [N, T, 2] pixel coords (x, y)
    start_cond: torch.Tensor | None = None,  # [N_cond, 3] (x, y, t)
    end_cond: torch.Tensor | None = None,  # [N_cond, 3] (x, y, t)
    traj_idx: torch.Tensor | None = None,  # [N_cond]
    line_width: float = 0.005,  # relative to image diagonal
    marker_size: float = 0.01,  # relative to image diagonal
    cond_size: float = 0.015,  # relative to image diagonal
    return_pil_image: bool = False,
    draw_pokes: bool = False,
    draw_trajectories: bool = True,
    poke_scale: float = 0.05,  # relative to image diagonal
    outline_width: float = 0.001,  # white edge width, relative to diagonal
    dpi: int = 100,
):
    """
    Draw trajectories with gradient lines and optional arrows using matplotlib.
    All sizes are relative to image diagonal for consistent appearance across resolutions.
    """
    assert frame.ndim == 3 and frame.shape[-1] == 3, "frame must be [H,W,3] RGB in [0,1]"
    H, W, _ = frame.shape

    # Scale all sizes relative to image diagonal
    # scale_factor = np.sqrt(H**2 + W**2)
    scale_factor = min(H, W) * 1.5
    line_px = math.ceil(line_width * scale_factor)
    marker_px = math.ceil(marker_size * scale_factor)
    cond_size_px = math.ceil(cond_size * scale_factor)
    poke_scale_px = math.ceil(poke_scale * scale_factor)
    outline_px = math.ceil(outline_width * scale_factor)
    # Convert to numpy
    frame_np = frame.float().clamp(0, 1).cpu().numpy()
    tracks_np = tracks.float().cpu().numpy()  # [N, T, 2]
    N, T, _ = tracks_np.shape

    # Create figure
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(frame_np, extent=(0, W, H, 0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    cmap = cm.get_cmap("jet")

    # Draw each trajectory
    for n in range(N):
        pts = tracks_np[n]  # [T, 2]
        valid = ~np.isnan(pts).any(axis=1)
        pts_valid = pts[valid]

        if not valid.any() or len(pts_valid) < 2:
            continue

        start_marker_color = cmap(0.0)
        end_marker_color = cmap(1.0)
        start_end_dist = np.linalg.norm(pts_valid[-1] - pts_valid[0])
        if draw_pokes:  # less than 0.5% of movement
            start_marker_color = "black"
            end_marker_color = "black"

            if start_end_dist > scale_factor * 0.005:
                start = pts_valid[0]
                end = pts_valid[-1]
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="simple,head_length=0.8,head_width=0.6",
                    facecolor="black",
                    edgecolor="white",
                    linewidth=outline_px,
                    mutation_scale=poke_scale_px,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=10,
                )
                ax.add_patch(arrow)

        start_pos = pts_valid[0]
        start_circle = Circle(
            start_pos,
            marker_px,
            facecolor=start_marker_color,
            edgecolor="white",
            linewidth=outline_px,
            zorder=4,
        )
        ax.add_patch(start_circle)

        end_pos = pts_valid[-1]
        end_circle = Circle(
            end_pos,
            marker_px,
            facecolor=end_marker_color,
            edgecolor="white",
            linewidth=outline_px,
            zorder=4,
        )
        ax.add_patch(end_circle)

        if not draw_trajectories:
            continue

        # Draw gradient trajectory lines
        for t in range(len(pts_valid) - 1):
            x1, y1 = pts_valid[t]
            x2, y2 = pts_valid[t + 1]
            ratio = t / max(len(pts_valid) - 1, 1)
            color = cmap(ratio)

            # White halo
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="white",
                linewidth=line_px + 2,
                alpha=0.6,
                solid_capstyle="round",
                zorder=2,
            )
            # Colored line
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=line_px, solid_capstyle="round", zorder=3)

    # Draw condition markers if provided
    if start_cond is not None and end_cond is not None and traj_idx is not None:
        sc = start_cond.float().cpu().numpy()
        ec = end_cond.float().cpu().numpy()

        for i in range(len(sc)):
            sx, sy = sc[i, :2]
            ex, ey = ec[i, :2]

            # Start condition (cyan square)
            start_rect = Rectangle(
                (sx - cond_size_px / 2, sy - cond_size_px / 2),
                cond_size_px,
                cond_size_px,
                facecolor="none",
                edgecolor="cyan",
                linewidth=2,
                zorder=6,
            )
            ax.add_patch(start_rect)

            # End condition (magenta square)
            end_rect = Rectangle(
                (ex - cond_size_px / 2, ey - cond_size_px / 2),
                cond_size_px,
                cond_size_px,
                facecolor="none",
                edgecolor="magenta",
                linewidth=2,
                zorder=6,
            )
            ax.add_patch(end_rect)

    # Convert to image
    fig.canvas.draw()
    img_array = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if return_pil_image:
        from PIL import Image

        return Image.fromarray(img_array)

    return img_array


@dataclass
class ModelState:
    model: TrackFM
    device: torch.device


@dataclass
class SampleState:
    image: UInt8[np.ndarray, "h w c"]
    all_tracks: list[list[tuple[float, float]]] = field(default_factory=lambda: [[]])

    def add_point_ui(self, point: tuple[float, float]):
        self.all_tracks[-1].append(point)


def render_input(inputs: SampleState) -> UInt8[np.ndarray, "h w c"]:
    H, W, _ = inputs.image.shape
    dpi = 100
    px_scale = math.ceil(min(H, W) * 0.015)

    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(inputs.image, origin="upper", extent=(0, W, 0, H), aspect="equal")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for track in inputs.all_tracks:
        if len(track) == 0:
            continue
        track_px = [(x * W, H - y * H) for (x, y) in track]  # we need to vertically flip bc of ax.imshow coord system

        if len(track) == 1:
            ax.scatter(track_px[0][0], track_px[0][1], color="C1", marker="x", s=150, linewidths=3)
        elif len(track_px) == 2 and np.linalg.norm(np.array(track_px[0]) - np.array(track_px[1]), ord=2) <= 0.01:
            # If there is ~no movement, draw a circle to prevent an invisible arrow
            arrow = Circle(xy=track_px[0], radius=px_scale, facecolor="black", edgecolor="white")
            ax.add_patch(arrow)
        else:
            for start, end in zip(track_px[:-1], track_px[1:]):
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="simple,head_length=0.8,head_width=0.6",
                    facecolor="black",
                    edgecolor="white",
                    linewidth=1,
                    mutation_scale=20,
                    shrinkA=0,
                    shrinkB=0,
                )
                ax.add_patch(arrow)

    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img


def draw_3d(start_frame, trajectories, **kwargs) -> UInt8[np.ndarray, "h w c"]:
    fig, ax = visualize_trajectories(start_frame, trajectories, elev=10, show_points=False, **kwargs)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.margins(0)

    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img


def get_3d_plot(start_frame, trajectories, save_path=None, **kwargs):
    fig, ax = visualize_trajectories(start_frame, trajectories, elev=10, show_points=False, **kwargs)

    # Remove margins and fill the whole figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.margins(0)

    # Draw the canvas before reading pixel data
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]  # Get RGB image for viz

    # Optionally save as vector PDF
    if save_path is not None:
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0, transparent=True)

    plt.close(fig)
    return img, save_path


@torch.no_grad()
def predict(
    inputs: SampleState,
    model_state: ModelState,
    cfg_scale: float = 1.0,
    n_vis_tracks: int = 80,
    n_decoder_tracks: int = 80,
    decode_grid: bool = False,
    seed: int = 42,
    use_3d_viz: bool = False,
    azim: float = 75.0,
    use_ortho: bool = False,
    keep_aspect: bool = False,
    center_crop_viz: bool = False,
) -> tuple[UInt8[np.ndarray, "h w c"], None]:
    assert inputs.image is not None, "Image input is required for prediction."

    batch_size = 4

    # print(inputs, flush=True)
    device = model_state.device
    dtype = torch.bfloat16
    model = model_state.model
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    flatteneted_pokes = []
    for track in inputs.all_tracks:
        n_end = len(track) - 1

        if n_end < 1:
            print("Skipping track with less than 2 points:", track, flush=True)
            continue

        start = [c * 2 - 1 for c in track[0]]
        for i, end in enumerate(track[1:]):
            end = [c * 2 - 1 for c in end]
            flatteneted_pokes.append([start[1], start[0], end[1], end[0], (i + 1) / n_end])  # end_t in [0, 1]

    if len(flatteneted_pokes) > 0:
        track_conds = torch.tensor(flatteneted_pokes, device=device, dtype=dtype)  # (n_pokes, 5)
    else:
        track_conds = torch.zeros((0, 5), device=device, dtype=dtype)  # (n_pokes, 5)

    query_points = []
    for track in inputs.all_tracks:
        if len(track) > 0:
            start = track[0]
            query_points.append([start[1], start[0]])  # [0, 1] range
    query_points = torch.tensor(query_points, device=device, dtype=dtype)  # (n_query, 2)
    query_points = torch.cat(
        [
            query_points,
            torch.rand(n_decoder_tracks - query_points.shape[0], 2, device=device, dtype=dtype, generator=g),
        ],
        dim=0,
    )

    with torch.autocast(device_type="cuda", dtype=dtype):
        model.cfg_scale = cfg_scale
        pred_tracks = model.sample(
            z=torch.randn(batch_size, *VAL_SHAPE, device=device, dtype=dtype, generator=g),
            points_per_traj=64,
            query_pos=query_points.expand(batch_size, -1, -1) * 2 - 1,
            start_frame=torch.from_numpy(inputs.image).to(device).expand(batch_size, -1, -1, -1).float() / 127.5 - 1.0,
            track_conds=track_conds.expand(batch_size, -1, -1),  # (batch_size, n_pokes, 5)
        )

    H, W, C = inputs.image.shape

    pred_tracks = (pred_tracks.flip(-1) + 1) / 2 * torch.tensor([W, H], device=device)

    out_frames = []
    # Save individual frames as PDF files for Gradio File outputs.
    pdf_paths = []

    for pred_track in pred_tracks:
        # tracks are in [-1, 1] range
        # convert the tracks to pixel coordinates
        if use_3d_viz:
            # out_frame = draw_3d(
            #     inputs.image,
            #     pred_track[:n_vis_tracks].int().cpu().numpy(),
            #     azim=azim,
            #     proj_type="ortho" if use_ortho else "persp",
            #     keep_aspect=keep_aspect,
            #     center_crop=center_crop_viz,
            # )
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.close()

            out_frame, pdf_path = get_3d_plot(
                inputs.image,
                pred_track[:n_vis_tracks].int().cpu().numpy(),
                azim=azim,
                proj_type="ortho" if use_ortho else "persp",
                keep_aspect=keep_aspect,
                center_crop=center_crop_viz,
                save_path=tmp.name,
            )
            pdf_paths.append(pdf_path)

        else:
            out_frame = draw_trajectories_on_frame(
                torch.from_numpy(inputs.image) / 255.0,
                pred_track[:n_vis_tracks].cpu(),
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.close()
            pdf_paths.append(tmp.name)
            pil = Image.fromarray(out_frame)
            pil.save(tmp.name, "PDF", resolution=300)

        out_frames.append(out_frame)

    out_video = None
    if decode_grid:
        H = W = int(math.sqrt(n_decoder_tracks))

        batch_size = 1
        grid = make_axial_pos_2d(H, W, device=device, dtype=dtype)

        with torch.autocast(device_type="cuda", dtype=dtype):
            pred_tracks = model.sample(
                z=torch.randn(batch_size, *VAL_SHAPE, device=device, dtype=dtype),
                points_per_traj=64,
                query_pos=grid.expand(batch_size, -1, -1),
                start_frame=torch.from_numpy(inputs.image).to(device).expand(batch_size, -1, -1, -1).float() / 127.5
                - 1.0,
                track_conds=track_conds.expand(batch_size, -1, -1),  # (batch_size, n_pokes, 5)
            )
        grid = einops.rearrange(pred_tracks, "b (h w) t c -> b t c h w", h=H, w=W)[0]
        flow = torch.diff(grid, dim=0)
        flow_video = flow_to_image(flow.float())
        flow_video = resize(flow_video, size=[256, 256])

        tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        imageio.mimwrite(tmpfile.name, flow_video.movedim(1, -1).cpu().numpy(), format="mp4", fps=15)

        out_video = tmpfile.name
    return [np.concatenate(out_frames, axis=1), *pdf_paths, out_video]
    # if use_3d_viz:
    #     return [np.concatenate(out_frames, axis=1), *pdf_paths, out_video]
    # else:
    #     return [np.concatenate(out_frames, axis=1), *out_frames, out_video]


def load_video(
    selected_filename: str | None,
    inputs: SampleState,
    model_state: ModelState,
    center_crop: bool,
    start_idx: int,
    video_dir: str,
):
    """Load first frame of a selected video and return (frame, SampleState)."""
    if selected_filename is None:
        return None, inputs
    full_path = os.path.join(video_dir, selected_filename)
    try:
        reader = imageio.get_reader(full_path)
        frame = reader.get_data(start_idx)
    except Exception as e:
        print(f"Failed to read video {full_path}: {e}", flush=True)
        return None, inputs

    # Normalize frame to RGB uint8
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]

    # center-crop & resize to target size (reuse same logic as preprocess_image)
    H, W, C = frame.shape
    target_size = 256
    min_size = min(H, W)

    if center_crop:
        start_h = (H - min_size) // 2
        start_w = (W - min_size) // 2
        frame = frame[start_h : start_h + min_size, start_w : start_w + min_size, :]
        frame = np.array(Image.fromarray(frame).resize((target_size, target_size), Image.BICUBIC))
    else:  # keep aspect ratio, resize smaller side to target_size
        scale = target_size / min_size
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        frame = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.BICUBIC))

    inputs = SampleState(image=frame)
    return frame, inputs


def demo(
    device: str = "cuda",
    compile: bool = False,  # Faster inference, at the cost of compilation time whenever a prediction config is first encountered
    warmup_compiled_paths: bool = False,
    # Gradio settings
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 55555,
    video_pexel_dir="/export/scratch/ra49veb/cvpr-2026/track-ae/wan_samples_2",
    video_davis_dir="/export/group/datasets/DAVIS/videos",
    # Profiler
    profile: bool = False,
):
    with gr.Blocks() as demo:
        gr.Markdown("## Motion Spaces Demo")

        vae = TrackVAE()
        model = TrackFM(vae=vae)

        sd = torch.load(
            "/export/scratch/ra49veb/checkpoints/track-project/327613-n16-endt-unlock-repro-actually-rebalanced/checkpoints/step-650000/model.pt"
        )

        sd = {
            k.replace(".mid_level.", ".")
            .replace("unet", "backbone")
            .replace("ae", "vae")
            .replace("backbone.mid_merge", "in_proj")
            .replace("extra_proj", "cond_proj")
            .replace("backbone.mid_split", "out_proj"): v
            for k, v in sd.items()
            if "dummy_query_pos" not in k
        }

        model.load_state_dict(sd)

        model.eval()
        model.to(device)
        model.requires_grad_(False)

        # model.use_compile = compile

        gr_model = gr.State(ModelState(model=model, device=torch.device(device)))

        with gr.Row():
            # Input & poke/query interaction handling
            with gr.Column():
                inputs = gr.State(None)
                image_input = gr.Image(
                    label="Input Image", interactive=True, type="numpy", image_mode="RGB", format="png"
                )

                center_crop = gr.Checkbox(label="center crop", value=False)

                # slider to pick start frame for video loaders
                start_idx = gr.Slider(label="Start frame index", minimum=0, maximum=100, step=1, value=0)

                # Directory with Pexel videos to choose from
                pexel_video_paths = sorted(
                    sum(
                        (
                            glob.glob(os.path.join(video_pexel_dir, f"original*.{ext}"))
                            for ext in ["mp4", "avi", "mov", "mkv"]
                        ),
                        [],
                    )
                )
                pexel_video_choices = [os.path.basename(p) for p in pexel_video_paths]
                with gr.Row():  # dropdown and load button side-by-side
                    video_pexel_dropdown = gr.Dropdown(
                        choices=pexel_video_choices,
                        label="Select Pexels Video",
                        value=pexel_video_choices[0] if pexel_video_choices else None,
                    )
                    load_pexel_video_button = gr.Button("Load video")

                # register module-level load_video, binding demo's video_dir via partial
                load_pexel_video_button.click(
                    partial(load_video, video_dir=video_pexel_dir),
                    inputs=[video_pexel_dropdown, inputs, gr_model, center_crop, start_idx],
                    outputs=[image_input, inputs],
                )

                # Directory with DAVIS videos to choose from
                davis_video_paths = sorted(
                    sum(
                        (glob.glob(os.path.join(video_davis_dir, f"*.{ext}")) for ext in ["mp4", "avi", "mov", "mkv"]),
                        [],
                    )
                )
                davis_video_choices = [os.path.basename(p) for p in davis_video_paths]
                with gr.Row():  # dropdown and load button side-by-side
                    video_davis_dropdown = gr.Dropdown(
                        choices=davis_video_choices,
                        label="Select DAVIS Video",
                        value=davis_video_choices[0] if davis_video_choices else None,
                    )
                    load_davis_video_button = gr.Button("Load video")

                # register module-level load_video, binding demo's video_dir via partial
                load_davis_video_button.click(
                    partial(load_video, video_dir=video_davis_dir),
                    inputs=[video_davis_dropdown, inputs, gr_model, center_crop, start_idx],
                    outputs=[image_input, inputs],
                )

                new_track_button = gr.Button("Start New Track")
                new_track_button.click(
                    lambda inputs: inputs.all_tracks.append([]) or inputs,
                    inputs=[inputs],
                    outputs=[inputs],
                )
                reset_tracks_button = gr.Button("Reset Tracks")
                reset_tracks_button.click(
                    lambda inputs: setattr(inputs, "all_tracks", [[]]) or inputs,
                    inputs=[inputs],
                    outputs=[inputs],
                )

                @torch.no_grad()
                def preprocess_image(
                    image_input: UInt8[np.ndarray, "h w c"] | None,
                    inputs: SampleState,
                    model: ModelState,
                    center_crop: bool = False,
                ):
                    if image_input is None:
                        return None, None
                    H, W, C = image_input.shape
                    assert C == 3, f"Expected 3 channels in the image, got {C} channels."
                    target_size: int = 256
                    min_size = min(H, W)

                    if center_crop:
                        start_h = (H - min_size) // 2
                        start_w = (W - min_size) // 2
                        frame = image_input[start_h : start_h + min_size, start_w : start_w + min_size, :]
                        frame = np.array(Image.fromarray(frame).resize((target_size, target_size), Image.BICUBIC))
                    else:  # keep aspect ratio, resize smaller side to target_size
                        scale = target_size / min_size
                        new_h, new_w = int(round(H * scale)), int(round(W * scale))
                        frame = np.array(Image.fromarray(image_input).resize((new_w, new_h), Image.BICUBIC))

                    inputs = SampleState(
                        image=frame,
                    )
                    return frame, inputs

                image_input.upload(
                    preprocess_image, inputs=[image_input, inputs, gr_model, center_crop], outputs=[image_input, inputs]
                )

                def input_click(image_input, inputs: SampleState, evt: gr.SelectData):
                    if image_input is None:
                        return image_input, inputs
                    # (x, y) indexing
                    inputs.add_point_ui((evt.index[0] / image_input.shape[1], evt.index[1] / image_input.shape[0]))
                    return render_input(inputs), inputs

                image_input.select(input_click, inputs=[image_input, inputs], outputs=[image_input, inputs])

            # Output visualization
            with gr.Column():
                image_output = gr.Image(label="Prediction", type="numpy", image_mode="RGB", format="png")
                with gr.Row():
                    # image_outputs_sep = [
                    #     gr.Image(label=f"Prediction {i+1}", type="numpy", image_mode="RGB", format="png", scale=1)
                    #     for i in range(4)
                    # ]
                    image_outputs_sep = [
                        gr.File(label=f"Prediction {i + 1} (PDF)", file_types=[".pdf"]) for i in range(4)
                    ]

                video_output = gr.Video(label="Prediction Video", format="mp4")

                predict_button = gr.Button("Predict", variant="primary")

                with gr.Accordion("Advanced Settings"):
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=7, step=0.1, value=1)
                    n_decoder_tracks = gr.Number(label="Number of Decoder Tracks", value=80, precision=0, minimum=1)
                    n_vis_tracks = gr.Number(label="Number of Visualization Tracks", value=80, precision=0, minimum=1)
                    decode_grid = gr.Checkbox(label="Use Decoding Grid", value=False)
                    use_3d_viz = gr.Checkbox(label="Use 3D Visualization (Slow!!)", value=False)
                    keep_aspect_ratio = gr.Checkbox(label="Keep aspect ratio", value=False)
                    center_crop_viz = gr.Checkbox(label="Center crop visualization", value=True)
                    azim = gr.Slider(label="3D Viz Azimuth", minimum=0, maximum=90, step=1, value=75)
                    use_ortho = gr.Checkbox(label="Use Orthographic Projection for 3D Viz", value=False)
                    seed = gr.Slider(label="Random Seed", minimum=0, maximum=2**32 - 1, step=1, value=42)

                predict_button.click(
                    predict,
                    inputs=[
                        inputs,
                        gr_model,
                        cfg_scale,
                        n_vis_tracks,
                        n_decoder_tracks,
                        decode_grid,
                        seed,
                        use_3d_viz,
                        azim,
                        use_ortho,
                        keep_aspect_ratio,
                        center_crop_viz,
                    ],
                    outputs=[
                        image_output,
                        *image_outputs_sep,
                        video_output,
                    ],  # outputs=[image_output, image_outputs_sep, video_output],
                    show_progress=True,
                )

    print(f"demo launched at {server_name}:{server_port}", flush=True)
    demo.launch(share=share, server_name=server_name, server_port=server_port, debug=True)


if __name__ == "__main__":
    # Allow TF32, make model go brrrrrrrrrr
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable benchmarking for matmuls
    torch.backends.cudnn.benchmark = True
    # Increase the compilation cache size, so that we don't stop compiling when hitting many different paths
    torch._dynamo.config.cache_size_limit = max(2**12, torch._dynamo.config.cache_size_limit)

    # This seems to be crucial for compilation with cudagraphs to be stable (as of torch ~2.8) with gradio's multithreading
    # If you do inference using a normal script, leave these enabled, as it'll likely give you slight speedups
    # Also, if you run out of memory, consider enabling these again; as it might help with that, although you might pay
    # with the aforementioned stability issues
    torch._inductor.config.triton.cudagraph_trees = False

    fire.Fire(demo)
