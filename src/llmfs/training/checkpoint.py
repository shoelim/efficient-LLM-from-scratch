"""Trusted local checkpoint snapshots including optimizer, scaler and RNG state."""

import os
from dataclasses import asdict
from pathlib import Path

import torch

from llmfs.config import GPTConfig, TrainConfig
from llmfs.utils.randomness import capture_rng


def save_training_checkpoint(path: str | Path, *, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer, step: int,
                             model_config: GPTConfig, train_config: TrainConfig,
                             batch_generator: torch.Generator, data_fingerprint: str,
                             scaler=None, distributed_rng_states=None, world_size: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "step": step,  # number of optimizer steps already completed
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng(batch_generator),
        "data_fingerprint": data_fingerprint,
        "world_size": world_size,
        "distributed_rng_states": distributed_rng_states,
    }
    # Rename atomically to avoid leaving a truncated last.pt after an interruption.
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, *, map_location="cpu") -> dict:
    """ONLY load trusted locally created .pt files; PyTorch pickle can execute code."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1 or "model" not in payload:
        raise ValueError("Unsupported checkpoint format")
    return payload
