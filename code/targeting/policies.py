"""Policies assign one action per customer under a combined email budget."""
import numpy as np


def assign_policy(probabilities, strategy, budget, *, seed=42):
    """Return action codes: 0=no email, 1=men's email, 2=women's email.

    Input rows must be in a consistent order across competing methods. Seeded
    random tie-breaking avoids favoring source-row order, especially for trees.
    Budgets are upper bounds; uplift contacts only positive-uplift customers.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite n-by-3 purchase probabilities.')
    if ((p < 0) | (p > 1)).any() or not 0 <= budget <= 1:
        raise ValueError('Probabilities and budget must lie in [0, 1].')
    n = len(p)
    actions = np.zeros(n, dtype=int)
    limit = int(np.floor(n * budget))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    if strategy == 'no_contact':
        return actions
    if strategy in ('random_mens', 'random_womens'):
        actions[order[:limit]] = 1 if strategy == 'random_mens' else 2
        return actions
    if strategy not in ('response', 'uplift'):
        raise ValueError(f'Unknown strategy: {strategy}')
    # Randomize exact email ties with the same seed for both ranking rules.
    best_email = np.where(p[:, 1] > p[:, 2], 1, 2)
    tied = p[:, 1] == p[:, 2]
    best_email[tied] = rng.integers(1, 3, size=tied.sum())
    score = p[np.arange(n), best_email]
    if strategy == 'uplift':
        score = score - p[:, 0]
        order = order[score[order] > 0]
    ranked = order[np.argsort(-score[order], kind='stable')]
    selected = ranked[:limit]
    actions[selected] = best_email[selected]
    return actions
