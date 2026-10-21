#!/usr/bin/env python3
"""Download a bounded TinyStories subset, preserving the official data splits.

Data is saved as one JSON object per story: {"text": "..."}. This format is
accepted directly by `llmfs prepare`. The dataset is NOT distributed in this
repository. The optional Hugging Face datasets package is needed to download.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Callable

DATASET_ID = "roneneldan/TinyStories"
DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
DATASET_LICENSE = "cdla-sharing-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_to_jsonl(dataset, path: Path, count: int, seen: set[str]) -> int:
    saved = 0
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("Unexpected TinyStories row: expected a 'text' string")
            story = row["text"].strip()
            if not story:
                continue
            key = hashlib.sha256(story.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            output.write(json.dumps({"text": story}, ensure_ascii=False) + "\n")
            saved += 1
            if saved == count:
                break
    if saved != count:
        raise ValueError(f"Requested {count} unique stories, but only {saved} were available")
    return saved


def download_tinystories(
    output_dir: str | Path,
    *,
    train_stories: int = 20_000,
    validation_stories: int = 1_000,
    seed: int = 1337,
    shuffle_buffer: int = 10_000,
    revision: str | None = None,
    loader: Callable | None = None,
) -> dict:
    """Fetch subsets without loading the full dataset into memory.

    `loader` is injectable for offline tests. Passing `revision` pins a specific
    Hugging Face commit. Without it, source content may change over time.
    """
    if min(train_stories, validation_stories, shuffle_buffer) < 1:
        raise ValueError("Counts and shuffle buffer must be positive")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {out}")
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
        loader = load_dataset
    out.parent.mkdir(parents=True, exist_ok=True)
    # All the partial files remain outside the destination until BOTH splits
    # finish and pass validation. Failures don't leave misleading prepared data.
    with tempfile.TemporaryDirectory(prefix="tinystories-", dir=str(out.parent)) as temp:
        stage = Path(temp)
        seen: set[str] = set()
        common = {"split": "train", "streaming": True}
        if revision is not None:
            common["revision"] = revision
        train = loader(DATASET_ID, **common)
        train = train.shuffle(seed=seed, buffer_size=shuffle_buffer)
        n_train = _stream_to_jsonl(train, stage / "train.jsonl", train_stories, seen)
        common["split"] = "validation"
        validation = loader(DATASET_ID, **common)
        validation = validation.shuffle(seed=seed + 1, buffer_size=shuffle_buffer)
        n_validation = _stream_to_jsonl(
            validation, stage / "validation.jsonl", validation_stories, seen
        )
        manifest = {
            "dataset": DATASET_ID,
            "url": DATASET_URL,
            "license": DATASET_LICENSE,
            "source_revision": revision,
            "official_splits": {"train": "train", "validation": "validation"},
            "train_stories": n_train,
            "validation_stories": n_validation,
            "shuffle_seed": seed,
            "shuffle_buffer": shuffle_buffer,
            "train_sha256": _sha256(stage / "train.jsonl"),
            "validation_sha256": _sha256(stage / "validation.jsonl"),
            "notes": ("Bounded, buffered-shuffle sample of the original dataset. "
                      "Do not compare directly to full-corpus benchmark results."),
        }
        (stage / "source.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        # This directory is ignored by Git; still refuse silent replacement.
        out.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl", "source.json"):
            (stage / name).replace(out / name)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/tinystories/raw")
    parser.add_argument("--train-stories", type=int, default=20_000)
    parser.add_argument("--validation-stories", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--revision", default=None,
                        help="Optional Hugging Face Git commit SHA to pin the source")
    args = parser.parse_args(argv)
    manifest = download_tinystories(
        args.output, train_stories=args.train_stories,
        validation_stories=args.validation_stories,
        seed=args.seed, shuffle_buffer=args.shuffle_buffer,
        revision=args.revision,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Saved {args.output}/train.jsonl and {args.output}/validation.jsonl")


if __name__ == "__main__":
    main()
