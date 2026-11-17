"""Analytic checks for the continuous signed-mixture weight."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'rmc'))
from forward_mixture import difference_weights


def test_mixture_recovers_sub_bin_integral():
    # CE = 2x, low = 1 on [0, 1]. Their total masses are equal, but
    # their signed integral over [0, 1/2] is -1/4, not zero.
    rng = np.random.default_rng(184)
    n = 500_000
    choose_ce = rng.random(n) < 0.3
    u = rng.random(n)
    x = np.where(choose_ce, np.sqrt(u), u)
    w = difference_weights(2 * x, np.ones(n), 1, 1, 0.3)
    score = w * (x < 0.5)
    assert abs(score.mean() + 0.25) < 5 * score.std(ddof=1) / np.sqrt(n)
    assert abs(w.mean()) < 5 * w.std(ddof=1) / np.sqrt(n)


def test_support_and_validation():
    assert difference_weights(1, 0, 1, 1, 0.5) == 2
    assert difference_weights(0, 1, 1, 1, 0.5) == -2
    with pytest.raises(ValueError, match='zero proposal'):
        difference_weights(0, 0, 1, 1, 0.5)
    with pytest.raises(ValueError, match='probability'):
        difference_weights(1, 1, 1, 1, 1)
