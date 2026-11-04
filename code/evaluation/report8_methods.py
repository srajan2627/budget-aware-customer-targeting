"""Report 8 diagnostics and known-truth reference calculations.

The original evaluator and historical exports remain unchanged. The conservative
event bounds below apply to homogeneous iid selected populations in the stated
simulation, not to general learned-policy comparisons on Hillstrom.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta


def nested_training_samples(frame, fractions, seed):
    """Paired m-out-of-n bootstraps; all sizes share prefixes within each arm."""
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError('Training fractions must be in (0, 1].')
    rng = np.random.default_rng(seed)
    parts = {f: [] for f in fractions}
    for _, arm in frame.groupby('treatment', sort=True):
        draws = rng.integers(0, len(arm), len(arm))
        for fraction in fractions:
            size = int(np.floor(len(arm) * fraction))
            if size < 1:
                raise ValueError('Each training arm must receive at least one draw.')
            parts[fraction].append(arm.iloc[draws[:size]])
    return {f: pd.concat(rows) for f, rows in parts.items()}


def chance_overlap(population, left, right):
    """Expected intersection / smaller list for independent uniform subsets."""
    if population < 1 or not 0 <= left <= population or not 0 <= right <= population:
        raise ValueError('Invalid population or selection sizes.')
    if min(left, right) == 0:
        return np.nan
    return max(left, right) / population


def adjusted_overlap(observed, expected):
    """Zero at chance and one at identical lists; undefined if chance is one."""
    return (observed - expected) / (1 - expected) if expected < 1 else np.nan


def support_diagnostic(scores, selected, warning_threshold=10):
    scores = np.asarray(scores)
    if selected < 0 or selected > len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid scores or selected count.')
    positive, negative = int((scores > 0).sum()), int((scores < 0).sum())
    if selected == 0:
        if np.any(scores):
            raise ValueError('An empty contact policy must have zero incremental scores.')
        status = 'structural_zero_no_selected_customers'
    elif positive + negative == 0:
        status = 'zero_events_interval_uninformative'
    elif positive + negative < warning_threshold:
        status = 'few_contributing_events'
    else:
        status = 'nonzero_support_not_a_coverage_guarantee'
    return dict(positive_events=positive, negative_events=negative,
                contributing_events=positive + negative, support_status=status,
                degenerate_nonempty=selected > 0 and positive + negative == 0)


def count_bootstrap_intervals(counts, population, repeats, rng, alpha=.05):
    """Exact distributional compression of full-cohort {-3,0,3} row resampling.

    Counts columns are positive and negative scores. Remaining N-counts.sum()
    rows have zero score, including unselected customers. Each simulated trial
    uses its own empirical probabilities; allocation is not conditioned on Y.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 2 or repeats < 2 or population < 1:
        raise ValueError('Expected trial-by-two counts, positive N and repeats >=2.')
    if (counts < 0).any() or (counts != np.floor(counts)).any() or (counts.sum(axis=1) > population).any():
        raise ValueError('Invalid event counts.')
    if not 0 < alpha < 1:
        raise ValueError('Invalid alpha.')
    lower, upper = np.empty(len(counts)), np.empty(len(counts))
    for i, (positive, negative) in enumerate(counts):
        probabilities = [positive / population, negative / population,
                         1 - (positive + negative) / population]
        draws = rng.multinomial(population, probabilities, size=repeats)
        lower[i], upper[i] = np.quantile(3. * (draws[:, 0] - draws[:, 1]),
                                       [alpha / 2, 1 - alpha / 2])
    return lower, upper


def exact_event_bounds(counts, selected, alpha=.05):
    """Conservative simultaneous marginal CP bounds, ONLY for the stated DGM.

    Each marginal interval has error alpha/2 (tail alpha/4). The union bound
    does not require independence of the positive and negative count margins.
    """
    counts = np.asarray(counts)
    if (counts.ndim != 2 or counts.shape[1] != 2 or selected < 1
            or (counts < 0).any() or (counts != np.floor(counts)).any()
            or (counts.sum(axis=1) > selected).any() or not 0 < alpha < 1):
        raise ValueError('Require integer event counts within positive selected size.')
    low = np.zeros(counts.shape, dtype=float)
    high = np.ones(counts.shape, dtype=float)
    mask = counts > 0
    low[mask] = beta.ppf(alpha / 4, counts[mask], selected - counts[mask] + 1)
    mask = counts < selected
    high[mask] = beta.ppf(1 - alpha / 4, counts[mask] + 1, selected - counts[mask])
    return (np.maximum(-selected, 3 * selected * (low[:, 0] - high[:, 1])),
            np.minimum(selected, 3 * selected * (high[:, 0] - low[:, 1])))


def probability_summary(successes, trials):
    """Monte Carlo proportion, standard error and Wilson interval."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError('Invalid simulation count.')
    p = successes / trials
    z = 1.959963984540054
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials ** 2)) / denominator
    # Include p explicitly to remove roundoff-sized negative plotting errors
    # at observed proportions zero and one.
    return dict(rate=p, mc_se=float(np.sqrt(p * (1 - p) / trials)),
                mc_lower=min(p, max(0., float(center - half))),
                mc_upper=max(p, min(1., float(center + half))))
