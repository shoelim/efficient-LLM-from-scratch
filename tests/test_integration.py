import json
from pathlib import Path

import torch
import pytest

from llmfs.cli import load_inference_model, main
from llmfs.config import GPTConfig
from llmfs.data.prepare import prepare_data
from llmfs.finetuning.sft import run_sft
from llmfs.model.gpt import GPT
from llmfs.training.checkpoint import load_checkpoint
from llmfs.training.optim import create_optimizer
from llmfs.training.train import run_training
from llmfs.utils.randomness import capture_rng, restore_rng, seed_everything


def test_full_prepare_train_evaluate_generate_pipeline(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    run = tmp_path / "run"
    prepare_data(root / "examples/toy_corpus.txt", data, tokenizer_type="char")
    ckpt = run_training(data, root / "configs/smoke.yaml", run)
    assert ckpt.exists()
    assert (run / "tokenizer.json").exists()
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 4
    model = load_inference_model(ckpt, torch.device("cpu"))
    assert model.config.vocab_size > 26
    main(["evaluate", "--checkpoint", str(ckpt), "--data", str(data),
          "--batches", "1", "--batch-size", "2", "--device", "cpu"])
    main(["generate", "--checkpoint", str(ckpt), "--prompt", "once",
          "--max-new-tokens", "3", "--temperature", "0", "--device", "cpu"])
    assert "perplexity" in capsys.readouterr().out


def test_new_run_refuses_populated_directory(tmp_path):
    """A second fresh training run must not append to metrics or overwrite weights."""
    root = Path(__file__).resolve().parents[1]
    data, run = tmp_path / "data", tmp_path / "run"
    prepare_data(root / "examples/toy_corpus.txt", data, tokenizer_type="char")
    checkpoint = run_training(data, root / "configs/smoke.yaml", run)
    before = (run / "metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="--resume"):
        run_training(data, root / "configs/smoke.yaml", run)
    assert checkpoint.exists()
    assert (run / "metrics.jsonl").read_bytes() == before


def test_sft_and_lora_artifact_can_be_loaded(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data, run = tmp_path / "data", tmp_path / "run"
    prepare_data(root / "examples/toy_corpus.txt", data, tokenizer_type="char")
    ckpt = run_training(data, root / "configs/smoke.yaml", run)
    small_sft = tmp_path / "short_sft.jsonl"
    small_sft.write_text("".join(json.dumps({"prompt": p, "response": r}) + "\n"
                                  for p, r in [("cat", " fox."), ("bird", " sky."),
                                               ("small", " big."), ("once", " time.")]))
    adapter_path = run_sft(ckpt, small_sft, tmp_path / "adapter_run",
                           steps=2, batch_size=2, use_lora=True, device_name="cpu")
    assert adapter_path.exists()
    model = load_inference_model(ckpt, torch.device("cpu"), adapter_path)
    assert model(torch.tensor([[0, 1]])).logits.isfinite().all()
    sft_path = run_sft(ckpt, small_sft, tmp_path / "full_run",
                       steps=1, batch_size=2, device_name="cpu")
    assert load_inference_model(sft_path, torch.device("cpu")).count_parameters() > 0


def test_rng_save_restore_includes_batch_generator(tmp_path):
    seed_everything(9)
    config = GPTConfig(vocab_size=9, block_size=8, n_layer=1, n_head=1, n_embd=8)
    model = GPT(config)
    optimizer = create_optimizer(model, 0.001, 0.01)
    batch_rng = torch.Generator().manual_seed(21)
    state = capture_rng(batch_rng)
    numbers_expected = torch.rand(3)
    indices_expected = torch.randint(0, 100, (5,), generator=batch_rng)
    restore_rng(state, batch_rng)
    assert torch.equal(torch.rand(3), numbers_expected)
    assert torch.equal(torch.randint(0, 100, (5,), generator=batch_rng), indices_expected)


def test_exact_cpu_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    """Real 4-step run resumed after step 2, including stochastic dropout."""
    import shutil

    import yaml

    import llmfs.training.train as train_module

    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    prepare_data(root / "examples/toy_corpus.txt", data, tokenizer_type="char")
    settings = yaml.safe_load((root / "configs/smoke.yaml").read_text())
    settings["model"]["dropout"] = 0.2
    config = tmp_path / "resume_config.yaml"
    config.write_text(yaml.safe_dump(settings))
    midpoint = tmp_path / "midpoint.pt"
    original = train_module.save_training_checkpoint

    def capture_midpoint(path, **kwargs):
        original(path, **kwargs)
        if kwargs["step"] == 2:
            shutil.copy2(path, midpoint)

    monkeypatch.setattr(train_module, "save_training_checkpoint", capture_midpoint)
    uninterrupted = run_training(data, config, tmp_path / "full")
    monkeypatch.setattr(train_module, "save_training_checkpoint", original)
    resumed = run_training(data, config, tmp_path / "resumed", resume=midpoint)
    a, b = load_checkpoint(uninterrupted), load_checkpoint(resumed)
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    # Verify optimizer state, not just model weights.
    opt_a = a["optimizer"]["state"]
    opt_b = b["optimizer"]["state"]
    for key in opt_a:
        for attribute in opt_a[key]:
            va, vb = opt_a[key][attribute], opt_b[key][attribute]
            if torch.is_tensor(va):
                assert torch.equal(va, vb)
            else:
                assert va == vb


def test_byte_bpe_cli_training_path(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "bpe"
    prepare_data(root / "examples/toy_corpus.txt", data, tokenizer_type="bpe", vocab_size=280)
    ckpt = run_training(data, root / "configs/smoke.yaml", tmp_path / "run_bpe")
    main(["generate", "--checkpoint", str(ckpt), "--prompt", "dragon 🐉",
          "--max-new-tokens", "3", "--temperature", "0", "--device", "cpu"])
