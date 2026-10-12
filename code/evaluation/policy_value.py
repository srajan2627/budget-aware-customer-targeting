"""IPW scores for randomized three-action policy evaluation."""
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


def paired_bootstrap_totals(scores, *, repeats=2000, seed=2026):
    """Bootstrap fixed-policy totals with identical row draws for every column.

    The estimand is n * mean(score). Policies and rankings stay frozen: this
    measures evaluation-sample uncertainty, not training or reranking variation.
    Do not stratify on conversion: preserving purchase counts would suppress
    precisely the uncertainty we need to measure. Batch draws bound memory.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2 or not len(s) or not np.isfinite(s).all():
        raise ValueError('Require a finite, nonempty score matrix.')
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError('Require at least two bootstrap replicates.')
    rng = np.random.default_rng(seed)
    totals = np.empty((repeats, s.shape[1]))
    for start in range(0, repeats, 32):
        size = min(32, repeats - start)
        counts = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=size)
        totals[start:start + size] = counts @ s
    return totals


def interval(samples):
    """Pointwise percentile 95% interval; no multiplicity adjustment."""
    return tuple(float(x) for x in np.quantile(samples, [.025, .975]))
