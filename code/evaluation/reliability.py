"""Separate training-sample variation from fixed-test sampling uncertainty."""
from itertools import combinations

import numpy as np
import pandas as pd


def training_bootstrap(frame, seed):
    """Resample within randomized arms, NOT within purchase outcome strata."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, arm in frame.groupby('treatment', sort=True):
        parts.append(arm.iloc[rng.integers(0, len(arm), len(arm))])
    return pd.concat(parts)


def agreement(left, right):
    """Compare capped policies; report full and contacted-only agreement."""
    a, b = np.asarray(left), np.asarray(right)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('Require aligned nonempty policy vectors.')
    if not np.isin(a, [0, 1, 2]).all() or not np.isin(b, [0, 1, 2]).all():
        raise ValueError('Invalid action code.')
    sa, sb = a != 0, b != 0
    common, union = sa & sb, sa | sb
    minimum = min(sa.sum(), sb.sum())
    return dict(
        contacts_left=int(sa.sum()), contacts_right=int(sb.sum()),
        common_contacts=int(common.sum()),
        selection_overlap=float(common.sum() / minimum) if minimum else np.nan,
        selection_jaccard=float(common.sum() / union.sum()) if union.any() else np.nan,
        treatment_agreement_shared=float((a[common] == b[common]).mean()) if common.any() else np.nan,
        action_agreement_all=float((a == b).mean()),
    )


def pairwise_agreement(policies):
    return [dict(refit_left=i, refit_right=j, **agreement(policies[i], policies[j]))
            for i, j in combinations(range(len(policies)), 2)]
