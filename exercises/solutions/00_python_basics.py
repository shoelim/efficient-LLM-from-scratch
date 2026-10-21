"""Week 0 reference solution; deliberately plain Python."""


def count_tokens(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def make_lm_examples(token_ids: list[int], context_length: int
                     ) -> list[tuple[list[int], int]]:
    if context_length < 1:
        raise ValueError("context_length must be positive")
    return [(token_ids[i - context_length:i], token_ids[i])
            for i in range(context_length, len(token_ids))]
