"""Document-first splitting, scratch tokenization, and memory-mapped token arrays."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from llmfs.tokenization.bpe import ByteBPETokenizer
from llmfs.tokenization.char import CharTokenizer
from llmfs.tokenization.io import save_tokenizer


def read_documents(path: str | Path) -> list[str]:
    """A .txt file contains documents separated by blank lines, or pass a folder of .txt.

    Each nonempty paragraph / file is a document. No network downloads are performed.
    """
    path = Path(path)
    if path.is_dir():
        documents = [p.read_text(encoding="utf-8") for p in sorted(path.glob("*.txt"))]
    elif path.is_file():
        import re

        documents = re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"No such input text: {path}")
    # Duplicate documents can otherwise end up on both sides of the split.
    return list(dict.fromkeys(d.strip() for d in documents if d.strip()))


def split_documents(documents: list[str], val_fraction: float = 0.2,
                    seed: int = 1337) -> tuple[list[str], list[str]]:
    if not 0 < val_fraction < 1 or len(documents) < 2:
        raise ValueError("Need >= 2 documents and 0 < val_fraction < 1")
    unique = list(dict.fromkeys(documents))
    if len(unique) < 2:
        raise ValueError("Need >= 2 distinct documents")
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, min(len(unique) - 1, round(len(unique) * val_fraction)))
    return unique[n_val:], unique[:n_val]


def _hash(path: Path) -> str:
    # Stream large .npy corpora instead of reading the whole file into memory.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_documents(documents: list[str], tokenizer: CharTokenizer | ByteBPETokenizer
                     ) -> np.ndarray:
    ids: list[int] = []
    for doc in documents:
        ids.extend(tokenizer.encode(doc))
        ids.append(tokenizer.eos_id)
    return np.asarray(ids, dtype=np.uint32)


def prepare_data(input_path: str | Path, output_dir: str | Path, *,
                 tokenizer_type: str = "char", vocab_size: int = 512,
                 val_fraction: float = 0.2, seed: int = 1337) -> dict:
    docs = read_documents(input_path)
    training, validation = split_documents(docs, val_fraction, seed)
    if tokenizer_type == "char":
        tokenizer = CharTokenizer.train(training)
    elif tokenizer_type == "bpe":
        tokenizer = ByteBPETokenizer.train(training, vocab_size=vocab_size)
    else:
        raise ValueError("tokenizer_type must be 'char' or 'bpe'")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, out / "tokenizer.json")
    np.save(out / "train.npy", encode_documents(training, tokenizer))
    np.save(out / "val.npy", encode_documents(validation, tokenizer))
    # Validation character vocab may include chars absent from train: fail explicitly,
    # not silently leak validation vocabulary into the tokenizer.
    manifest = {
        "input": str(input_path), "tokenizer": tokenizer_type,
        "vocab_size": tokenizer.vocab_size, "eos_id": tokenizer.eos_id,
        "n_train_documents": len(training), "n_val_documents": len(validation),
        "train_tokens": int(np.load(out / "train.npy", mmap_mode="r").size),
        "val_tokens": int(np.load(out / "val.npy", mmap_mode="r").size),
        "train_utf8_bytes": sum(len(d.encode("utf-8")) for d in training),
        "val_utf8_bytes": sum(len(d.encode("utf-8")) for d in validation),
        "seed": seed, "val_fraction": val_fraction,
        "train_document_sha256": [hashlib.sha256(d.encode()).hexdigest() for d in training],
        "val_document_sha256": [hashlib.sha256(d.encode()).hexdigest() for d in validation],
    }
    for name in ("tokenizer.json", "train.npy", "val.npy"):
        manifest[f"{name}_sha256"] = _hash(out / name)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_tokens(data_dir: str | Path, split: str) -> np.ndarray:
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    path = Path(data_dir) / f"{split}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint32 or array.ndim != 1:
        raise ValueError("Expected 1-D uint32 token array")
    return array


def dataset_fingerprint(data_dir: str | Path) -> str:
    """Identifier that detects changes to tokenizer and either token sequence."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("tokenizer.json", "train.npy", "val.npy"):
        if _hash(Path(data_dir) / filename) != manifest[f"{filename}_sha256"]:
            raise ValueError(f"Dataset file changed after preparation: {filename}")
    text = "".join(manifest[f"{name}_sha256"] for name in
                   ("tokenizer.json", "train.npy", "val.npy"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()
