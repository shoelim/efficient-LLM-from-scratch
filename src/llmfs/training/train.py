"""Single-device or PyTorch DDP pretraining with optimized PyTorch kernels.

The default optimized path uses SDPA, an AdamW implementation selected by
hardware, vectorized document-packed token windows, fp32 master parameters,
AMP (on request), accumulation, clipping, and exact resumable checkpoint state.
Under torchrun, ranks independently sample batches and average gradients via DDP.
"""

import json
import os
from contextlib import nullcontext
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from llmfs.config import GPTConfig, TrainConfig, load_config
from llmfs.data.fast_batcher import TokenBatcher
from llmfs.data.prepare import dataset_fingerprint, load_tokens
from llmfs.evaluation.metrics import evaluate_loss
from llmfs.model.gpt import GPT
from llmfs.tokenization.io import load_tokenizer
from llmfs.training.checkpoint import load_checkpoint, save_training_checkpoint
from llmfs.training.optim import create_optimizer
from llmfs.training.schedule import cosine_lr
from llmfs.utils.randomness import capture_rng, restore_rng, seed_everything, select_device


def setup_training(data_dir: str | Path, config_path: str | Path) -> tuple[GPTConfig, TrainConfig]:
    model_dict, train_dict = load_config(config_path)
    tokenizer = load_tokenizer(Path(data_dir) / "tokenizer.json")
    model_dict = dict(model_dict)
    if "vocab_size" in model_dict and model_dict["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("Config vocabulary and tokenizer disagree")
    model_dict["vocab_size"] = tokenizer.vocab_size
    return GPTConfig(**model_dict), TrainConfig(**train_dict)


def run_training(data_dir: str | Path, config_path: str | Path,
                 output_dir: str | Path, *, resume: str | Path | None = None,
                 stop_after_step: int | None = None) -> Path:
    """Run on one device; under torchrun WORLD_SIZE>1 use standard DDP.

    Distributed runs sample independent random training windows per rank. The
    effective per-update batch is batch_size * grad_accum_steps * WORLD_SIZE.
    Ranks synchronize gradients only at the last accumulation microstep.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return _run_training(data_dir, config_path, output_dir, resume=resume,
                             rank=0, world=1, local_rank=0, stop_after_step=stop_after_step)
    _, tcfg = setup_training(data_dir, config_path)
    requested = select_device(tcfg.device)
    local_rank = int(os.environ["LOCAL_RANK"])
    if requested.type == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl" if requested.type == "cuda" else "gloo")
    try:
        return _run_training(data_dir, config_path, output_dir, resume=resume,
                             rank=dist.get_rank(), world=world, local_rank=local_rank,
                             stop_after_step=stop_after_step)
    finally:
        dist.destroy_process_group()


def _run_training(data_dir: str | Path, config_path: str | Path,
                  output_dir: str | Path, *, resume: str | Path | None,
                  rank: int, world: int, local_rank: int,
                  stop_after_step: int | None) -> Path:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    mcfg, tcfg = setup_training(data_dir, config_path)
    train_tokens, val_tokens = load_tokens(data_dir, "train"), load_tokens(data_dir, "val")
    if min(len(train_tokens), len(val_tokens)) <= mcfg.block_size:
        raise ValueError("Train AND val token arrays must be longer than block_size")
    if world > 1:
        # Hash very large corpora only once instead of reading the same files
        # N times in parallel on distributed startup.
        fingerprints = [dataset_fingerprint(data_dir) if rank == 0 else None]
        dist.broadcast_object_list(fingerprints, src=0)
        fingerprint = fingerprints[0]
    else:
        fingerprint = dataset_fingerprint(data_dir)
    # A new run must not silently append metrics to an earlier run or replace
    # its checkpoint. In DDP, broadcast rank 0's decision so every worker
    # raises before entering the next collective barrier.
    run_conflict = None
    if rank == 0 and resume is None and output_dir.exists() and any(output_dir.iterdir()):
        run_conflict = (f"Output directory {output_dir} is not empty. "
                        "Choose a new --output path, or use --resume to continue this run.")
    if world > 1:
        conflicts = [run_conflict]
        dist.broadcast_object_list(conflicts, src=0)
        run_conflict = conflicts[0]
    if run_conflict is not None:
        raise FileExistsError(run_conflict)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    target_tokenizer = output_dir / "tokenizer.json"
    source_tokenizer = data_dir / "tokenizer.json"
    if rank == 0:
        if target_tokenizer.exists() and target_tokenizer.read_bytes() != source_tokenizer.read_bytes():
            raise ValueError("Output directory already belongs to a different tokenizer")
        shutil.copy2(source_tokenizer, target_tokenizer)
    if world > 1:
        dist.barrier()

    device = select_device(tcfg.device)
    if device.type == "cuda" and world > 1:
        device = torch.device("cuda", local_rank)
    if tcfg.compile and world > 1:
        raise ValueError("Do not combine torch.compile and DDP in this verified path")
    if tcfg.precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision currently requires CUDA; use fp32 on CPU/MPS")
    if tcfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bf16")
    if tcfg.compile and device.type != "cuda":
        raise ValueError("CUDA required for the supported torch.compile training path")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tcfg.tf32
        torch.backends.cudnn.allow_tf32 = tcfg.tf32
    seed_everything(tcfg.seed + rank)
    batch_rng = torch.Generator(device="cpu").manual_seed(tcfg.seed + 1 + rank)
    model = GPT(mcfg).to(device)
    model.gradient_checkpointing = tcfg.activation_checkpointing
    optimizer = create_optimizer(model, tcfg.learning_rate, tcfg.weight_decay,
                                 implementation=tcfg.optimizer, device_type=device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.precision == "fp16")
    start_step = 0
    if resume is not None:
        payload = load_checkpoint(resume)
        if payload["model_config"] != asdict(mcfg) or payload["train_config"] != asdict(tcfg):
            raise ValueError("Resume requires identical model and training configuration")
        if payload["data_fingerprint"] != fingerprint:
            raise ValueError("Resume data/tokenizer differs from checkpoint")
        if payload.get("world_size", 1) != world:
            raise ValueError("Resume requires identical distributed world size")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        start_step = payload["step"]
        saved_rng = (payload["distributed_rng_states"][rank] if world > 1
                     else payload["rng"])
        restore_rng(saved_rng, batch_rng)
    elif rank == 0:
        (output_dir / "run_config.json").write_text(
            json.dumps({"model": asdict(mcfg), "train": asdict(tcfg),
                        "data_fingerprint": fingerprint, "parameters": model.count_parameters(),
                        "world_size": world},
                       indent=2), encoding="utf-8")
    if start_step >= tcfg.steps:
        raise ValueError("Checkpoint has completed all configured steps")
    finish_step = tcfg.steps if stop_after_step is None else stop_after_step
    if not start_step < finish_step <= tcfg.steps:
        raise ValueError("stop_after_step must be > current step and <= configured steps")

    batcher = TokenBatcher(train_tokens, block_size=mcfg.block_size,
                           batch_size=tcfg.batch_size, device=device, storage=tcfg.batch_device)
    # Compile after loading raw model parameters, before training; checkpoints
    # always serialize the unwrapped model (avoids _orig_mod state prefixes).
    forward_model = (torch.compile(model, mode=tcfg.compile_mode, dynamic=False)
                     if tcfg.compile else model)
    if world > 1:
        forward_model = DDP(forward_model,
                            device_ids=[local_rank] if device.type == "cuda" else None,
                            broadcast_buffers=False)
    amp_dtype = torch.float16 if tcfg.precision == "fp16" else torch.bfloat16
    log_path = output_dir / "metrics.jsonl"
    beginning = time.perf_counter()
    interval_begin = beginning
    last_logged_step = start_step
    if rank == 0:
        print(f"Model: {model.count_parameters():,} parameters | device: {device} "
              f"| world: {world} | optimizer: {tcfg.optimizer} | SDPA: {mcfg.attention_impl} "
              f"| compile: {tcfg.compile} | resuming at step {start_step}", flush=True)
    for step in range(start_step, finish_step):
        lr = cosine_lr(step, tcfg.steps, tcfg.warmup_steps,
                       tcfg.learning_rate, tcfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss_sum = torch.zeros((), device=device)
        for micro in range(tcfg.grad_accum_steps):
            sync_context = (forward_model.no_sync() if world > 1 and micro + 1 < tcfg.grad_accum_steps
                            else nullcontext())
            with sync_context:
                x, y = batcher.next(batch_rng)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=tcfg.precision != "fp32"):
                    out = forward_model(x, y)
                    loss = out.loss / tcfg.grad_accum_steps
                loss_sum += loss.detach()
                scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        completed = step + 1
        wants_eval = completed % tcfg.eval_interval == 0 or completed == finish_step
        wants_log = completed % tcfg.log_interval == 0 or wants_eval
        if wants_log and world > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            loss_sum.div_(world)
        if wants_log and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            interval_steps = completed - last_logged_step
            last_logged_step = completed
            tps = interval_steps * tcfg.batch_size * tcfg.grad_accum_steps * mcfg.block_size * world / max(
                now - interval_begin, 1e-9)
            interval_begin = now
            row = {"step": completed, "train_loss": float(loss_sum.item()), "lr": lr,
                   "grad_norm": float(grad_norm.item()),
                   "tokens_seen": completed * tcfg.batch_size * tcfg.grad_accum_steps * mcfg.block_size * world,
                   "train_tokens_per_second": round(tps, 2),
                   "elapsed_seconds": round(now - beginning, 3)}
            if wants_eval:
                val = evaluate_loss(model, val_tokens, block_size=mcfg.block_size,
                                    batch_size=tcfg.batch_size, batches=tcfg.eval_batches,
                                    device=device, precision=tcfg.precision)
                row["val_loss"] = val
                row["val_perplexity"] = math.exp(val)
                interval_begin = time.perf_counter()  # exclude eval time from next tps
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if wants_eval and world > 1:
            dist.barrier()
        if completed % tcfg.save_interval == 0 or completed == finish_step:
            if world > 1:
                all_rng: list[dict | None] = [None] * world
                dist.all_gather_object(all_rng, capture_rng(batch_rng))
            else:
                all_rng = None
            if rank == 0:
                save_training_checkpoint(output_dir / "last.pt", model=model,
                                         optimizer=optimizer, step=completed,
                                         model_config=mcfg, train_config=tcfg,
                                         batch_generator=batch_rng,
                                         data_fingerprint=fingerprint, scaler=scaler,
                                         distributed_rng_states=all_rng, world_size=world)
            if world > 1:
                dist.barrier()
    return output_dir / "last.pt"
