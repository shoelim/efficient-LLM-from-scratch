"""Hand-checked smoothed conditional probabilities."""

import math

import pytest

from llmfs.evaluation.baselines import BigramLM


def test_smoothed_bigram_hand_calculation():
    lm = BigramLM(2, alpha=1).fit([0, 1, 0, 0])
    assert lm.probability(0, 1) == 0.5
    assert math.isclose(lm.probability(1, 0), 2 / 3)
    expected = - (math.log(.5) + math.log(2 / 3) + math.log(.5)) / 3
    assert math.isclose(lm.nll([0, 1, 0, 0]), expected)
    with pytest.raises(ValueError):
        lm.nll([1])
