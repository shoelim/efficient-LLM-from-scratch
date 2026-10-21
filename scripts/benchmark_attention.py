"""Compare manual attention vs SDPA performance while checking outputs.

Example:
    python scripts/benchmark_attention.py --device cpu --seq 64 --repeats 20
This is a microbenchmark, not a validated production throughput claim.
"""

import argparse
import json
import time

import torch

from llmfs.config import GPTConfig
from llmfs.model.attention import CausalSelfAttention


def benchmark(args):
    torch.manual_seed(7)
    device = torch.device(args.device)
    cfg = dict(vocab_size=32, block_size=args.seq, n_layer=1, n_embd=args.embd,
               n_head=args.heads, dropout=0)
    manual = CausalSelfAttention(GPTConfig(**cfg, attention_impl="manual")).to(device).eval()
    sdpa = CausalSelfAttention(GPTConfig(**cfg, attention_impl="sdpa")).to(device).eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(args.batch, args.seq, args.embd, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def run(module):
        with torch.inference_mode():
            for _ in range(3):
                module(x)
            sync()
            started = time.perf_counter()
            for _ in range(args.repeats):
                module(x)
            sync()
        return (time.perf_counter() - started) / args.repeats

    with torch.inference_mode():
        error = float((manual(x)[0] - sdpa(x)[0]).abs().max())
    result = {"device": str(device), "batch": args.batch, "seq": args.seq,
              "embd": args.embd, "heads": args.heads, "repeats": args.repeats,
              "maximum_output_difference": error,
              "manual_seconds_per_forward": run(manual),
              "sdpa_seconds_per_forward": run(sdpa)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--embd", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    benchmark(parser.parse_args())
