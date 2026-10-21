"""Week 0. Replace `NotImplementedError` and write your own unit tests."""


def count_tokens(tokens: list[str]) -> dict[str, int]:
    """Return counts for each string, e.g. ['a','b','a'] -> {'a':2,'b':1}."""
    raise NotImplementedError("Week 0: implement without Counter first")


def make_lm_examples(token_ids: list[int], context_length: int
                     ) -> list[tuple[list[int], int]]:
    """For each i>=context_length, return (previous context_length tokens, token i).

    Example: ids=[1,2,3,4], context_length=2 => [([1,2],3), ([2,3],4)].
    Handle empty sequences and invalid context lengths intentionally.
    """
    raise NotImplementedError("Week 0: use slicing and a for loop")
