"""Export inference-only artifacts (no optimizer, RNG, or pickle execution).

Tied LM head / token embedding shares underlying storage; safetensors requires
saving it once. Reconstruct the alias from the verified model configuration.
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from llmfs.config import GPTConfig
from llmfs.model.gpt import GPT
from llmfs.training.checkpoint import load_checkpoint


def export_inference(checkpoint: str | Path, output_dir: str | Path) -> Path:
    """Export from a *trusted local* .pt training checkpoint to safetensors."""
    checkpoint = Path(checkpoint)
    payload = load_checkpoint(checkpoint)
    cfg = GPTConfig(**payload["model_config"])
    model = GPT(cfg)
    model.load_state_dict(payload["model"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint.parent / "tokenizer.json", out / "tokenizer.json")
    state = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in model.state_dict().items()
             if not (name == "lm_head.weight" and cfg.tie_weights)}
    save_file(state, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps({"format_version": 1,
                                                  "model_config": asdict(cfg)}, indent=2),
                                    encoding="utf-8")
    return out


def load_export(path: str | Path, device: torch.device) -> GPT:
    path = Path(path)
    obj = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if obj.get("format_version") != 1:
        raise ValueError("Unsupported export format")
    cfg = GPTConfig(**obj["model_config"])
    model = GPT(cfg)
    state = load_file(path / "model.safetensors", device="cpu")
    result = model.load_state_dict(state, strict=False)
    allowed_missing = ["lm_head.weight"] if cfg.tie_weights else []
    if sorted(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise ValueError(f"Invalid export weights: {result}")
    return model.to(device).eval()
