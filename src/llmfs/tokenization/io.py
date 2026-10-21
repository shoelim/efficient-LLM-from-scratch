"""Stable JSON tokenizer persistence (no pickle for user-supplied vocabularies)."""

import json
from pathlib import Path
from typing import TypeAlias

from llmfs.tokenization.bpe import ByteBPETokenizer
from llmfs.tokenization.char import CharTokenizer

Tokenizer: TypeAlias = CharTokenizer | ByteBPETokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokenizer.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") == "char":
        return CharTokenizer.from_dict(data)
    if data.get("type") == "byte_bpe":
        return ByteBPETokenizer.from_dict(data)
    raise ValueError(f"Unknown tokenizer type: {data.get('type')!r}")
