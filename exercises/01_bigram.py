"""Week 1: derive add-alpha conditional bigram probabilities before coding."""


def conditional_bigram_probability(previous: int, nxt: int, tokens: list[int],
                                    vocab_size: int, alpha: float = 1.0) -> float:
    """Count adjacent (previous,next) pairs and apply Laplace smoothing."""
    raise NotImplementedError("Try formula in notes/01_autoregressive_lm.md")
