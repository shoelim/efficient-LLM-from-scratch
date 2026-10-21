"""Educational byte-level BPE: deterministic but deliberately not speed-optimized.

Vocabulary: bytes 0..255, EOS 256, merged byte sequences 257 onwards.
Counts never span document boundaries. Encoding uses a heap of adjacent-pair merge ranks and a linked list, while
training deliberately recomputes counts after each merge. Large-corpus tokenizer
training remains a separate systems exercise.
"""

import base64
import heapq
from collections import Counter
from dataclasses import dataclass, field


def _merge_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        for next_id, (a, b) in enumerate(self.merges, start=257):
            if a not in self.vocab or b not in self.vocab:
                raise ValueError(f"Merge references an undefined token: {(a, b)}")
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]

    @property
    def eos_id(self) -> int:
        return 256

    @property
    def vocab_size(self) -> int:
        return 257 + len(self.merges)

    @classmethod
    def train(cls, documents: list[str], *, vocab_size: int = 512,
              min_pair_count: int = 2) -> "ByteBPETokenizer":
        if not documents or vocab_size < 257 or min_pair_count < 1:
            raise ValueError("Need documents, vocab_size >= 257, min_pair_count >= 1")
        sequences = [list(doc.encode("utf-8")) for doc in documents]
        merges: list[tuple[int, int]] = []
        while len(merges) + 257 < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            # Prefer higher frequency; tie-break by the smallest pair of token IDs.
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_pair_count:
                break
            new_id = 257 + len(merges)
            sequences = [_merge_pair(seq, pair, new_id) for seq in sequences]
            merges.append(pair)
        return cls(merges)

    def encode_reference(self, text: str) -> list[int]:
        """Clear O(number of merges × number of tokens) reference encoder."""
        sequence = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            sequence = _merge_pair(sequence, pair, 257 + i)
        return sequence

    def encode(self, text: str) -> list[int]:
        """Heap-based, linked-list BPE encoder; equivalent to rank-order merges.

        Each candidate adjacent pair has a merge rank. Extract the lowest rank,
        verify it is still adjacent, merge in-place and update two neighbors.
        Candidate ties use the original leftmost position (same as reference).
        This avoids scanning the *whole* text once for every learned merge.
        """
        vals = list(text.encode("utf-8"))
        n = len(vals)
        if n < 2 or not self.merges:
            return vals
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1
        alive = [True] * n
        heap: list[tuple[int, int]] = []
        ranks = self.merge_ranks

        def push(i: int) -> None:
            if i < 0 or not alive[i] or nxt[i] < 0:
                return
            rank = ranks.get((vals[i], vals[nxt[i]]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        for i in range(n - 1):
            push(i)
        while heap:
            rank, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j < 0 or not alive[j] or ranks.get((vals[i], vals[j])) != rank:
                continue
            vals[i] = 257 + rank
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] >= 0:
                prev[nxt[j]] = i
            push(prev[i])
            push(i)
        return [val for i, val in enumerate(vals) if alive[i]]

    def decode(self, ids: list[int], *, skip_special: bool = True,
               errors: str = "strict") -> str:
        pieces = []
        for idx in ids:
            if idx == self.eos_id:
                if not skip_special:
                    pieces.append(b"<eos>")
            else:
                if idx not in self.vocab:
                    raise ValueError(f"Invalid BPE token id: {idx}")
                pieces.append(self.vocab[idx])
        return b"".join(pieces).decode("utf-8", errors=errors)

    def to_dict(self) -> dict:
        return {"type": "byte_bpe", "merges": [list(pair) for pair in self.merges],
                "vocab_base64": [base64.b64encode(self.vocab[i]).decode("ascii")
                                 for i in sorted(self.vocab)]}

    @classmethod
    def from_dict(cls, data: dict) -> "ByteBPETokenizer":
        model = cls([tuple(map(int, pair)) for pair in data["merges"]])
        # Verify serialized merge definitions, not just blindly trusting the file.
        if "vocab_base64" in data:
            entries = [base64.b64decode(b) for b in data["vocab_base64"]]
            if entries != [model.vocab[i] for i in sorted(model.vocab)]:
                raise ValueError("Tokenizer vocabulary and merges disagree")
        return model
