"""Count-based unsmoothed and smoothed language models for week 1."""

import math
from collections import Counter


class BigramLM:
    """Add-alpha conditional bigram model for discrete token IDs.

    This is a *separate* baseline: it has no trainable neural components.
    """

    def __init__(self, vocab_size: int, alpha: float = 1.0) -> None:
        if vocab_size <= 0 or alpha <= 0:
            raise ValueError("vocab_size and alpha must be positive")
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.counts: Counter[tuple[int, int]] = Counter()
        self.previous: Counter[int] = Counter()

    def fit(self, tokens: list[int]) -> "BigramLM":
        for a, b in zip(tokens, tokens[1:]):
            if not 0 <= a < self.vocab_size or not 0 <= b < self.vocab_size:
                raise ValueError("Out-of-vocabulary token")
            self.counts[a, b] += 1
            self.previous[a] += 1
        return self

    def probability(self, previous: int, next_token: int) -> float:
        if not 0 <= previous < self.vocab_size or not 0 <= next_token < self.vocab_size:
            raise ValueError("Out-of-vocabulary token")
        return ((self.counts[previous, next_token] + self.alpha) /
                (self.previous[previous] + self.alpha * self.vocab_size))

    def nll(self, tokens: list[int]) -> float:
        if len(tokens) < 2:
            raise ValueError("Need at least two tokens")
        return -sum(math.log(self.probability(a, b)) for a, b in zip(tokens, tokens[1:])) / (
            len(tokens) - 1)
