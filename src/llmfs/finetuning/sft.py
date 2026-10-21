"""Simple supervised fine-tuning with response-only cross-entropy.

For instructional transparency, examples that exceed block_size are rejected
rather than quietly truncated (which can invalidate response supervision).
"""

import json
import random
import shutil
from pathlib import Path

import torch

from llmfs.config import GPTConfig
from llmfs.finetuning.lora import adapter_state_dict, apply_lora
from llmfs.model.gpt import GPT
from llmfs.tokenization.io import Tokenizer, load_tokenizer
from llmfs.training.checkpoint import load_checkpoint
from llmfs.training.optim import create_optimizer
from llmfs.utils.randomness import seed_everything, select_device


def encode_example(prompt: str, response: str, tokenizer: Tokenizer,
                   block_size: int) -> tuple[list[int], list[int]]:
    if not response:
        raise ValueError("Response cannot be empty")
    prompt_ids = tokenizer.encode(prompt) or [tokenizer.eos_id]
    response_ids = tokenizer.encode(response)
    whole = prompt_ids + response_ids + [tokenizer.eos_id]
    if len(whole) < 2 or len(whole) - 1 > block_size:
        raise ValueError("SFT example too long for block_size; split examples explicitly")
    x = whole[:-1]
    y = whole[1:]
    # Targets in prompt positions are masked; first response token is supervised.
    y[:len(prompt_ids) - 1] = [-100] * (len(prompt_ids) - 1)
    return x, y


def collate_sft(examples: list[tuple[list[int], list[int]]], *,
                eos_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty SFT batch")
    length = max(len(x) for x, _ in examples)
    x_batch = torch.full((len(examples), length), eos_id, dtype=torch.long)
    y_batch = torch.full((len(examples), length), -100, dtype=torch.long)
    for row, (x, y) in enumerate(examples):
        x_batch[row, :len(x)] = torch.tensor(x, dtype=torch.long)
        y_batch[row, :len(y)] = torch.tensor(y, dtype=torch.long)
    return x_batch.to(device), y_batch.to(device)


def load_sft_examples(jsonl_path: str | Path, tokenizer: Tokenizer,
                      block_size: int) -> list[tuple[list[int], list[int]]]:
    examples = []
    seen_pairs = set()  # Keep exact duplicates on only one side of the holdout.
    for line_number, line in enumerate(Path(jsonl_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("prompt"), str) or not isinstance(record.get("response"), str):
            raise ValueError(f"Line {line_number}: requires prompt and response strings")
        identity = (record["prompt"], record["response"])
        if identity in seen_pairs:
            continue
        seen_pairs.add(identity)
        examples.append(encode_example(*identity, tokenizer, block_size))
    if len(examples) < 2:
        raise ValueError("Need at least 2 SFT examples for document-level holdout")
    return examples


def run_sft(checkpoint: str | Path, data_path: str | Path, output_dir: str | Path, *,
            steps: int = 50, batch_size: int = 2, lr: float = 1e-4,
            use_lora: bool = False, lora_rank: int = 4, lora_alpha: float = 8.0,
            seed: int = 42, device_name: str = "auto",
            precision: str = "fp32", activation_checkpointing: bool = False) -> Path:
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("steps, batch_size and learning rate must be positive")
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Unsupported precision")
    seed_everything(seed)
    checkpoint = Path(checkpoint)
    payload = load_checkpoint(checkpoint)
    tokenizer_file = checkpoint.parent / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_file)
    cfg = GPTConfig(**payload["model_config"])
    model = GPT(cfg)
    model.load_state_dict(payload["model"])
    device = select_device(device_name)
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed-precision SFT requires CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Device does not support bf16")
    model.to(device)
    model.gradient_checkpointing = activation_checkpointing
    examples = load_sft_examples(data_path, tokenizer, cfg.block_size)
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * 0.2))
    train, val = examples[n_val:], examples[:n_val]
    if use_lora:
        targets = ("qkv_proj",) if cfg.fused_qkv else ("q_proj", "v_proj")
        replaced = apply_lora(model, rank=lora_rank, alpha=lora_alpha, targets=targets)
        print(f"LoRA adapted: {', '.join(replaced)}")
    optimizer = create_optimizer(model, lr, weight_decay=0.0,
                                 implementation="auto", device_type=device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    model.train()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_file, output_dir / "tokenizer.json")
    for step in range(1, steps + 1):
        batch = [train[rng.randrange(len(train))] for _ in range(batch_size)]
        x, y = collate_sft(batch, eos_id=tokenizer.eos_id, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=precision != "fp32"):
            loss = model(x, y).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step == steps or step % 10 == 0:
            print(f"SFT step {step}/{steps}: train response-only NLL={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for example in val:
            x, y = collate_sft([example], eos_id=tokenizer.eos_id, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=precision != "fp32"):
                val_loss += model(x, y).loss.item()
        val_loss /= len(val)
    print(f"SFT validation response-only NLL={val_loss:.4f}")
    if use_lora:
        path = output_dir / "adapter.pt"
        torch.save({"type": "lora_adapter", "source_checkpoint": str(checkpoint),
                    "targets": targets, "rank": lora_rank,
                    "alpha": lora_alpha, "adapter": adapter_state_dict(model),
                    "val_loss": val_loss}, path)
    else:
        path = output_dir / "sft.pt"
        torch.save({"format_version": 1, "model": model.state_dict(),
                    "model_config": payload["model_config"],
                    "type": "sft_full", "source_checkpoint": str(checkpoint),
                    "val_loss": val_loss}, path)
    return path
