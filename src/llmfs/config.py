"""Explicit configuration with validation (no dynamic magic, no hidden defaults)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    norm: str = "layernorm"  # layernorm | rmsnorm
    position: str = "learned"  # learned | rope
    mlp: str = "gelu"  # gelu | swiglu
    n_kv_head: int | None = None  # None means full multi-head attention
    attention_impl: str = "sdpa"  # sdpa fast default | manual reference
    rope_base: float = 10000.0
    fused_qkv: bool = False  # learning-compatible separate Q/K/V; fast configs enable fusion

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("Sizes, layer count, and number of heads must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        kv = self.n_head if self.n_kv_head is None else self.n_kv_head
        if kv < 1 or self.n_head % kv:
            raise ValueError("n_kv_head must divide n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.norm not in ("layernorm", "rmsnorm"):
            raise ValueError("Unsupported norm")
        if self.position not in ("learned", "rope"):
            raise ValueError("Unsupported position encoding")
        if self.position == "rope" and (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.mlp not in ("gelu", "swiglu"):
            raise ValueError("Unsupported MLP type")
        if self.attention_impl not in ("manual", "sdpa"):
            raise ValueError("attention_impl must be manual or sdpa")
        if self.rope_base <= 1:
            raise ValueError("rope_base must be > 1")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 1
    steps: int = 200
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 10
    log_interval: int = 10
    save_interval: int = 50
    seed: int = 1337
    device: str = "auto"  # auto | cpu | cuda | mps
    precision: str = "fp32"  # fp32 | bf16 | fp16 (CUDA only)
    optimizer: str = "auto"  # auto | torch | fused | scratch
    compile: bool = False  # CUDA torch.compile is opt-in
    compile_mode: str = "default"  # default | reduce-overhead | max-autotune
    batch_device: str = "cpu"  # cpu | device (preload entire token corpus into accelerator)
    tf32: bool = True
    activation_checkpointing: bool = False  # trade compute for lower activation memory

    def __post_init__(self) -> None:
        positive = (self.batch_size, self.grad_accum_steps, self.steps, self.eval_interval,
                    self.eval_batches, self.log_interval, self.save_interval)
        if any(v <= 0 for v in positive):
            raise ValueError("Batch, steps, evaluation and save intervals must be positive")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError("warmup_steps must be between 0 and steps")
        if not 0 <= self.min_lr <= self.learning_rate:
            raise ValueError("Need 0 <= min_lr <= learning_rate")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer hyperparameters")
        if self.optimizer not in ("auto", "torch", "fused", "scratch"):
            raise ValueError("Unsupported optimizer")
        if self.compile_mode not in ("default", "reduce-overhead", "max-autotune"):
            raise ValueError("Unsupported compile_mode")
        if self.batch_device not in ("cpu", "device"):
            raise ValueError("batch_device must be cpu or device")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32, bf16 or fp16")


def load_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)
    if not isinstance(obj, dict) or not isinstance(obj.get("model"), dict):
        raise ValueError("Config YAML needs a 'model' mapping")
    train = obj.get("train", {})
    if not isinstance(train, dict):
        raise ValueError("'train' must be a mapping")
    return obj["model"], train


def config_as_dict(config: GPTConfig | TrainConfig) -> dict[str, Any]:
    return asdict(config)
