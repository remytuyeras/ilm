"""Shared checkpoint loading for the controlled nanoGPT character baseline."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def get_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_character_data(data_dir: str) -> Dict[str, Any]:
    with (Path(data_dir) / "meta.pkl").open("rb") as handle:
        return pickle.load(handle)


def load_nanogpt_checkpoint(
        checkpoint_path: str,
        nanogpt_dir: str,
        device: torch.device,
        ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    source_dir = str(Path(nanogpt_dir).resolve())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from model import GPT, GPTConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    compiled_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(compiled_prefix):
            state_dict[key[len(compiled_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint
