# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Nick Stracke et al., CompVis @ LMU Munich
import os

from safetensors.torch import load_file
from torch.hub import download_url_to_file, get_dir

dependencies = ["torch", "safetensors"]

_MODEL_URLS = {
    "zipmo_planner_dense": "https://huggingface.co/CompVis/ZipMo/resolve/main/zipmo_planner_dense.safetensors",
    "zipmo_planner_sparse": "https://huggingface.co/CompVis/ZipMo/resolve/main/zipmo_planner_sparse.safetensors",
    "zipmo_planner_libero": "https://huggingface.co/CompVis/ZipMo/resolve/main/zipmo_planner_libero.safetensors",
    "zipmo_vae": "https://huggingface.co/CompVis/ZipMo/resolve/main/zipmo_vae.safetensors",
    "zipmo_policy_head": "https://huggingface.co/CompVis/ZipMo/resolve/main/zipmo_policy_head.safetensors",
}


def _download_safetensors(url: str, filename: str) -> str:
    hub_dir = get_dir()
    checkpoints_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    cached_file = os.path.join(checkpoints_dir, filename)
    if not os.path.exists(cached_file):
        download_url_to_file(url, cached_file, progress=True)

    return cached_file


def zipmo_planner_dense(pretrained: bool = False, **kwargs):
    from zipmo.planner import ZipMoPlanner_Dense
    from zipmo.vae import ZipMoVAE

    vae = ZipMoVAE()
    model = ZipMoPlanner_Dense(vae=vae, **kwargs)

    if pretrained:
        path = _download_safetensors(
            _MODEL_URLS["zipmo_planner_dense"],
            "zipmo_planner_dense.safetensors",
        )
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_planner_sparse(pretrained: bool = False, **kwargs):
    from zipmo.planner import ZipMoPlanner_Sparse
    from zipmo.vae import ZipMoVAE

    vae = ZipMoVAE()
    model = ZipMoPlanner_Sparse(vae=vae, **kwargs)

    if pretrained:
        path = _download_safetensors(
            _MODEL_URLS["zipmo_planner_sparse"],
            "zipmo_planner_sparse.safetensors",
        )
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_planner_libero(pretrained: bool = False, **kwargs):
    from zipmo.planner import ZipMoPlanner_Libero
    from zipmo.vae import ZipMoVAE

    vae = ZipMoVAE()
    model = ZipMoPlanner_Libero(vae=vae, **kwargs)

    if pretrained:
        path = _download_safetensors(
            _MODEL_URLS["zipmo_planner_libero"],
            "zipmo_planner_libero.safetensors",
        )
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_policy_head(pretrained: bool = False, **kwargs):
    from zipmo.policy_head import PolicyHead

    model = PolicyHead(**kwargs)

    if pretrained:
        path = _download_safetensors(
            _MODEL_URLS["zipmo_policy_head"],
            "zipmo_policy_head.safetensors",
        )
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_vae(pretrained: bool = False, **kwargs):
    from zipmo.vae import ZipMoVAE

    model = ZipMoVAE()

    if pretrained:
        path = _download_safetensors(
            _MODEL_URLS["zipmo_vae"],
            "zipmo_vae.safetensors",
        )
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model
