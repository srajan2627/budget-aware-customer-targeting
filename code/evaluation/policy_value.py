"""Exploratory IPW point estimates on randomized development validation data."""
import numpy as np


def policy_scores(observed_actions, outcomes, recommended_actions, propensities):
    """Return paired per-customer policy and no-contact IPW scores.

    V(pi) = mean(1[A=pi(X)] * Y / e(A)). The incremental score subtracts
    the same customer's no-contact score, so uncontacted rows cancel exactly.
    These are evaluation scores, not observed individual treatment effects.
    """
    a = np.asarray(observed_actions)
    y = np.asarray(outcomes)
    pi = np.asarray(recommended_actions)
    e = np.asarray(propensities, dtype=float)
    if a.ndim != 1 or y.shape != a.shape or pi.shape != a.shape or not len(a):
        raise ValueError('Expected equally sized nonempty action and outcome vectors.')
    if not np.isin(a, [0, 1, 2]).all() or not np.isin(pi, [0, 1, 2]).all():
        raise ValueError('Action codes must be 0, 1, or 2.')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Conversion outcomes must be binary.')
    if e.shape != (3,) or not np.isfinite(e).all() or (e <= 0).any() or not np.isclose(e.sum(), 1):
        raise ValueError('Provide three positive assignment probabilities summing to one.')
    a = a.astype(int)
    value = (a == pi) * y / e[a]
    control = (a == 0) * y / e[0]
    return value, value - control
