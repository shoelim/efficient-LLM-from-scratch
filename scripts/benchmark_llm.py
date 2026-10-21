"""Matched, hardware-local throughput comparisons (no invented GPU measurements).

Run on your own GPU (recommended) or CPU:
 python scripts/benchmark_llm.py --device cpu --threads 1 --repeats 3
 python scripts/benchmark_llm.py --device cuda --precision bf16 --repeats 20

Report includes model, PyTorch, hardware, tokens/sec, decode tok/sec and timing.
Compilation time is not included; the production training CLI exposes --compile
through YAML. No claim of universal speedup is inferred from one device.
"""

import argparse
import json
import platform
import time

import torch

from llmfs.config import GPTConfig
from llmfs.inference.sample import generate
from llmfs.model.gpt import GPT
from llmfs.training.optim import create_optimizer


def benchmark(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    if args.precision != "fp32" and args.device != "cuda":
        raise ValueError("Mixed-precision timings require CUDA")
    if args.seq < 1 or args.new_tokens < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("Invalid benchmark dimensions")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    cfg = GPTConfig(vocab_size=args.vocab, block_size=args.seq + args.new_tokens + 4,
                    n_layer=args.layers, n_head=args.heads, n_kv_head=args.kv_heads,
                    n_embd=args.width, attention_impl="sdpa", fused_qkv=True,
                    norm="rmsnorm", position="rope", mlp="swiglu", dropout=0, bias=False)
    model = GPT(cfg).to(device)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Device cannot use bf16")
    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
    y = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
    optimizer = create_optimizer(model, 3e-4, 0.1, implementation="auto", device_type=device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        for _ in range(args.repeats):
            fn()
        sync()
        seconds = (time.perf_counter() - begin) / args.repeats
        memory_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                     if device.type == "cuda" else None)
        return round(seconds, 6), round(memory_mb, 2) if memory_mb is not None else None

    model.train()
    def train_one():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=args.precision != "fp32"):
            loss = model(x, y).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    train_sec, train_mb = measure(train_one)

    model.eval()
    generation = {}
    results = []
    for label, options in [("uncached", {"use_cache": False}),
                           ("dynamic_cache", {"use_cache": True, "cache_mode": "dynamic"}),
                           ("static_cache", {"use_cache": True, "cache_mode": "static"})]:
        def decode():
            return generate(model, x, max_new_tokens=args.new_tokens,
                            temperature=0, precision=args.precision, **options)
        with torch.inference_mode():
            result = decode()
        results.append(result)
        seconds, peak_mb = measure(decode)
        generation[label] = {"seconds": seconds,
                             "decode_tokens_per_second": round(args.batch * args.new_tokens / seconds, 2),
                             "peak_allocated_gpu_mb": peak_mb}
    if not torch.equal(results[0], results[1]) or not torch.equal(results[0], results[2]):
        raise AssertionError("Cached decode differs from uncached decode")
    output = {
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__,
                        "device": str(device), "processor": platform.processor(),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "threads": args.threads, "precision": args.precision},
        "model": {"parameters": model.count_parameters(), "layers": args.layers,
                  "heads": args.heads, "kv_heads": args.kv_heads, "width": args.width,
                  "vocab": args.vocab, "batch": args.batch, "train_seq": args.seq,
                  "new_tokens": args.new_tokens, "warmup": args.warmup,
                  "repeats": args.repeats},
        "training": {"seconds_per_step": train_sec,
                     "tokens_per_second": round(args.batch * args.seq / train_sec, 2),
                     "peak_allocated_gpu_mb": train_mb},
        "inference": generation, "greedy_output_parity": True,
        "caveats": "Local microbenchmark: no compilation/distribution, data IO, load, or tokenization measured."
    }
    print(json.dumps(output, indent=2))
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--seq', type=int, default=64)
    parser.add_argument('--new-tokens', type=int, default=12)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--kv-heads', type=int, default=2)
    parser.add_argument('--vocab', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', type=str, default=None)
    benchmark(parser.parse_args())
