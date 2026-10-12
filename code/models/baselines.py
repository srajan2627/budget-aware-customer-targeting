"""Interpretable arm-specific response models and three-action T-learners."""
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from preprocessing import FEATURE_COLUMNS, build_preprocessor

ACTIONS = ('No E-Mail', 'Mens E-Mail', 'Womens E-Mail')
FAMILIES = ('logistic', 'tree')


def build_model(family, *, include_cluster=False):
    """Fresh preprocessing is fitted separately within each training arm."""
    if family == 'logistic':
        estimator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)
    elif family == 'tree':
        estimator = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=42)
    else:
        raise ValueError(f'Unknown model family: {family}')
    # No class reweighting or resampling: we need purchase probabilities.
    preprocessor = build_preprocessor()
    if include_cluster:
        preprocessor.transformers.append(
            ('cluster', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['cluster'])
        )
    return Pipeline([('preprocess', preprocessor), ('model', estimator)])


class ArmResponseModels:
    """Estimate P(purchase | X, action) for each randomized action.

    The two email predictions supply a response-targeting baseline. Subtracting
    the no-email prediction gives the two T-learner uplift estimates. Sharing
    these response models isolates the difference between the ranking rules.
    """
    def __init__(self, family='logistic', *, include_cluster=False):
        self.family = family
        self.include_cluster = include_cluster

    @property
    def features(self):
        # Earlier Step 3 joblib models predate this optional ablation feature.
        return FEATURE_COLUMNS + (['cluster'] if getattr(self, 'include_cluster', False) else [])

    def fit(self, development_df):
        if set(development_df['treatment'].unique()) != set(ACTIONS):
            raise ValueError('Training data must contain exactly the three Hillstrom actions.')
        self.models_ = {}
        self.prevalence_ = {}
        for action in ACTIONS:
            arm = development_df.loc[development_df['treatment'].eq(action)]
            if set(arm['conversion'].unique()) != {0, 1}:
                raise ValueError(f'{action} must contain binary outcomes and both classes.')
            self.models_[action] = build_model(self.family, include_cluster=self.include_cluster).fit(
                arm[self.features], arm['conversion']
            )
            self.prevalence_[action] = float(arm['conversion'].mean())
        return self

    def predict_probabilities(self, features):
        """Columns follow ACTIONS: control, men's email, women's email."""
        return np.column_stack([
            self.models_[action].predict_proba(features[self.features])[:, 1]
            for action in ACTIONS
        ])

    def predict_uplift(self, features):
        p = self.predict_probabilities(features)
        return p[:, 1:] - p[:, [0]]
