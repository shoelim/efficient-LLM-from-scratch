"""CLI that exposes every stage separately. Run `llmfs --help`."""

import argparse
import os
import json
from pathlib import Path

import torch

from llmfs.config import GPTConfig
from llmfs.data.prepare import dataset_fingerprint, load_tokens, prepare_data
from llmfs.evaluation.baselines import BigramLM
from llmfs.evaluation.metrics import evaluate_loss, perplexity
from llmfs.finetuning.lora import apply_lora, load_adapter_state_dict
from llmfs.finetuning.sft import run_sft
from llmfs.inference.sample import generate
from llmfs.inference.export import export_inference, load_export
from llmfs.model.gpt import GPT
from llmfs.tokenization.bpe import ByteBPETokenizer
from llmfs.tokenization.io import load_tokenizer
from llmfs.training.checkpoint import load_checkpoint
from llmfs.training.train import run_training
from llmfs.utils.randomness import select_device


def load_inference_model(checkpoint: str | Path, device: torch.device,
                         adapter: str | Path | None = None) -> GPT:
    if Path(checkpoint).is_dir():
        model = load_export(checkpoint, device)
    else:
        payload = load_checkpoint(checkpoint)
        model = GPT(GPTConfig(**payload["model_config"]))
        model.load_state_dict(payload["model"])
    if adapter is not None:
        a = torch.load(adapter, map_location="cpu", weights_only=False)
        if a.get("type") != "lora_adapter":
            raise ValueError("Not a LoRA adapter checkpoint")
        apply_lora(model, rank=a["rank"], alpha=a["alpha"], targets=tuple(a["targets"]))
        load_adapter_state_dict(model, a["adapter"])
    model.to(device)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfs", description="From-scratch LLM: efficient training, inference, evaluation and SFT")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Split raw documents, train tokenizer, save token IDs")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    prep.add_argument("--vocab-size", type=int, default=512)
    prep.add_argument("--val-fraction", type=float, default=0.2)
    prep.add_argument("--seed", type=int, default=1337)
    train = sub.add_parser("train", help="Train all weights from random initialization")
    train.add_argument("--data", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", default=None, help="Trusted local last.pt only")
    train.add_argument("--stop-after-step", type=int, default=None,
                       help="End early WITHOUT changing the configured LR schedule (useful for resuming)")
    base = sub.add_parser("baseline", help="Count-based bigram baseline on the token corpus")
    base.add_argument("--data", required=True)
    base.add_argument("--alpha", type=float, default=1.0)
    ev = sub.add_parser("evaluate", help="Held-out validation loss and perplexity")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--batches", type=int, default=20)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--seed", type=int, default=2026)
    ev.add_argument("--device", default="auto")
    gen = sub.add_parser("generate", help="Autoregressive text sampling")
    gen.add_argument("--checkpoint", required=True, help="Trusted last.pt or exported model directory")
    gen.add_argument("--adapter", default=None)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=80)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--top-p", type=float, default=None)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--no-cache", action="store_true")
    gen.add_argument("--cache-mode", choices=["static", "dynamic"], default="static")
    gen.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    gen.add_argument("--device", default="auto")
    ex = sub.add_parser("export", help="Save inference-only safetensors with tokenizer and config")
    ex.add_argument("--checkpoint", required=True, help="Trusted local training checkpoint")
    ex.add_argument("--output", required=True)
    sft = sub.add_parser("sft", help="Response-only supervised fine-tuning or LoRA")
    sft.add_argument("--checkpoint", required=True)
    sft.add_argument("--data", required=True, help="Prompt/response JSONL")
    sft.add_argument("--output", required=True)
    sft.add_argument("--steps", type=int, default=50)
    sft.add_argument("--batch-size", type=int, default=2)
    sft.add_argument("--lr", type=float, default=1e-4)
    sft.add_argument("--lora", action="store_true")
    sft.add_argument("--rank", type=int, default=4)
    sft.add_argument("--alpha", type=float, default=8)
    sft.add_argument("--device", default="auto")
    sft.add_argument("--seed", type=int, default=42)
    sft.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    sft.add_argument("--activation-checkpointing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_data(args.input, args.output, tokenizer_type=args.tokenizer,
                              vocab_size=args.vocab_size, val_fraction=args.val_fraction,
                              seed=args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        checkpoint = run_training(args.data, args.config, args.output, resume=args.resume,
                                  stop_after_step=args.stop_after_step)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Checkpoint: {checkpoint}")
    elif args.command == "baseline":
        tokenizer = load_tokenizer(Path(args.data) / "tokenizer.json")
        train = load_tokens(args.data, "train")
        val = load_tokens(args.data, "val")
        baseline = BigramLM(tokenizer.vocab_size, alpha=args.alpha).fit(train.tolist())
        val_nll = baseline.nll(val.tolist())
        print(json.dumps({"bigram_heldout_nll": val_nll,
                          "bigram_perplexity": perplexity(val_nll), "alpha": args.alpha}))
    elif args.command == "evaluate":
        device = select_device(args.device)
        model = load_inference_model(args.checkpoint, device)
        ckpt = load_checkpoint(args.checkpoint)
        if ckpt.get("data_fingerprint") is not None and (
            dataset_fingerprint(args.data) != ckpt["data_fingerprint"]
        ):
            raise ValueError("Evaluation data differs from checkpoint training data")
        val = load_tokens(args.data, "val")
        nll = evaluate_loss(model, val, block_size=model.config.block_size,
                            batch_size=args.batch_size, batches=args.batches,
                            device=device, seed=args.seed)
        print(json.dumps({"heldout_window_nll": nll, "perplexity": perplexity(nll),
                          "batches": args.batches, "batch_size": args.batch_size}))
    elif args.command == "export":
        print(f"Inference export: {export_inference(args.checkpoint, args.output)}")
    elif args.command == "generate":
        device = select_device(args.device)
        model = load_inference_model(args.checkpoint, device, args.adapter)
        tokenizer_root = (Path(args.checkpoint) if Path(args.checkpoint).is_dir()
                          else Path(args.checkpoint).parent)
        tokenizer = load_tokenizer(tokenizer_root / "tokenizer.json")
        prompt = tokenizer.encode(args.prompt)
        if not prompt:
            prompt = [tokenizer.eos_id]
        if device.type == "mps":
            # MPS generators are not consistently supported in all PyTorch versions.
            torch.manual_seed(args.seed)
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(args.seed)
        ids = generate(model, torch.tensor([prompt], device=device),
                       max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p, eos_id=tokenizer.eos_id,
                       use_cache=not args.no_cache, cache_mode=args.cache_mode,
                       precision=args.precision, generator=generator)
        if isinstance(tokenizer, ByteBPETokenizer):
            print(tokenizer.decode(ids[0].tolist(), errors="replace"))
        else:
            print(tokenizer.decode(ids[0].tolist()))
    elif args.command == "sft":
        path = run_sft(args.checkpoint, args.data, args.output, steps=args.steps,
                       batch_size=args.batch_size, lr=args.lr,
                       use_lora=args.lora, lora_rank=args.rank, lora_alpha=args.alpha,
                       seed=args.seed, device_name=args.device, precision=args.precision,
                       activation_checkpointing=args.activation_checkpointing)
        print(f"Fine-tuning artifact: {path}")


if __name__ == "__main__":
    main()
