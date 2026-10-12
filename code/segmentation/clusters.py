"""Development-only cluster choice and interpretable customer profiles."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline

from preprocessing import FEATURE_COLUMNS, build_preprocessor


def cluster_pipeline(k, seed=42):
    return Pipeline([
        ('preprocess', build_preprocessor(segmentation=True)),
        ('cluster', KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])


def choose_clusters(development, *, stability_repeats=5, seed=2026):
    """Choose among 2–5 using attributes only; never use test or response.

    Highest sampled silhouette among candidates with >=5% minimum size and
    >=0.75 mean bootstrap ARI on the same development customers. Explicitly
    mark a fallback if no candidate meets both gates.
    """
    rng = np.random.default_rng(seed)
    rows, fitted = [], {}
    for k in (2, 3, 4, 5):
        model = cluster_pipeline(k).fit(development[FEATURE_COLUMNS])
        labels = model.predict(development[FEATURE_COLUMNS])
        x = model.named_steps['preprocess'].transform(development[FEATURE_COLUMNS])
        silhouette = silhouette_score(x, labels, sample_size=min(2000, len(x)), random_state=seed)
        ari = []
        for repeat in range(stability_repeats):
            sample = development.iloc[rng.integers(0, len(development), len(development))]
            alternative = cluster_pipeline(k, seed + repeat).fit(sample[FEATURE_COLUMNS])
            ari.append(adjusted_rand_score(labels, alternative.predict(development[FEATURE_COLUMNS])))
        sizes = np.bincount(labels, minlength=k)
        rows.append(dict(k=k, silhouette=float(silhouette), min_cluster_fraction=float(sizes.min()/len(labels)),
                         bootstrap_ari_mean=float(np.mean(ari)), bootstrap_ari_min=float(np.min(ari))))
        fitted[k] = model
        print(f'Cluster k={k}: silhouette={silhouette:.3f}, mean ARI={np.mean(ari):.3f}', flush=True)
    comparison = pd.DataFrame(rows)
    comparison['eligible'] = (comparison.min_cluster_fraction >= .05) & (comparison.bootstrap_ari_mean >= .75)
    eligible = comparison.loc[comparison.eligible]
    candidates = eligible if len(eligible) else comparison
    k = int(candidates.sort_values(['silhouette', 'k'], ascending=[False, True]).iloc[0].k)
    comparison['selected'] = comparison.k.eq(k)
    comparison['selection_fallback'] = not len(eligible)
    return fitted[k], comparison


def profiles(frame, labels, cohort):
    rows = []
    for cluster in sorted(np.unique(labels)):
        group = frame.iloc[np.flatnonzero(labels == cluster)]
        rows.append(dict(cohort=cohort, cluster=int(cluster), customers=len(group),
                         fraction=len(group)/len(frame), recency_median=float(group.recency.median()),
                         history_median=float(group.history.median()), mens_past_purchase=float(group.mens.mean()),
                         womens_past_purchase=float(group.womens.mean()), newbie_fraction=float(group.newbie.mean()),
                         most_common_channel=group.channel.mode().iloc[0],
                         most_common_zip_class=group.zip_code.mode().iloc[0]))
    return rows


def wilson(successes, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return center - half, center + half


def cluster_responses(frame, labels):
    """Arm counts, Wilson rates, and conservative difference interval bounds.

    Difference bounds subtract opposite Wilson endpoints. They are descriptive
    uncertainty ranges, not calibrated 95% CIs for differences; the runner adds
    paired IPW cluster-effect bootstrap intervals for inference.
    """
    rows = []
    for cluster in sorted(np.unique(labels)):
        group = frame.iloc[np.flatnonzero(labels == cluster)]
        for action, arm in group.groupby('treatment', sort=True):
            lower, upper = wilson(int(arm.conversion.sum()), len(arm))
            rows.append(dict(cluster=int(cluster), action=action, customers=len(arm), purchases=int(arm.conversion.sum()),
                             conversion_rate=float(arm.conversion.mean()), rate_ci_lower=lower, rate_ci_upper=upper))
    return pd.DataFrame(rows)
