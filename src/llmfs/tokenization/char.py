"""Transparent character-level tokenizer, with a dedicated end-of-document id."""

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    chars: str

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("Duplicate characters in vocabulary")
        self._encoder = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def train(cls, documents: list[str]) -> "CharTokenizer":
        if not documents:
            raise ValueError("Need at least one training document")
        return cls("".join(sorted(set("".join(documents)))))

    @property
    def eos_id(self) -> int:
        return len(self.chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        try:
            return [self._encoder[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"Character {exc.args[0]!r} absent from training vocabulary") from exc

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        parts = []
        for idx in ids:
            if idx == self.eos_id:
                if not skip_special:
                    parts.append("<eos>")
            elif 0 <= idx < len(self.chars):
                parts.append(self.chars[idx])
            else:
                raise ValueError(f"Invalid character token id: {idx}")
        return "".join(parts)

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(str(data["chars"]))
