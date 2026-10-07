"""Interpretable arm-specific response models and three-action T-learners."""
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from preprocessing import FEATURE_COLUMNS, build_preprocessor

ACTIONS = ('No E-Mail', 'Mens E-Mail', 'Womens E-Mail')
FAMILIES = ('logistic', 'tree')


def build_model(family):
    """Fresh preprocessing is fitted separately within each training arm."""
    if family == 'logistic':
        estimator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)
    elif family == 'tree':
        estimator = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=42)
    else:
        raise ValueError(f'Unknown model family: {family}')
    # No class reweighting or resampling: we need purchase probabilities.
    return Pipeline([('preprocess', build_preprocessor()), ('model', estimator)])


class ArmResponseModels:
    """Estimate P(purchase | X, action) for each randomized action.

    The two email predictions supply a response-targeting baseline. Subtracting
    the no-email prediction gives the two T-learner uplift estimates. Sharing
    these response models isolates the difference between the ranking rules.
    """
    def __init__(self, family='logistic'):
        self.family = family

    def fit(self, development_df):
        if set(development_df['treatment'].unique()) != set(ACTIONS):
            raise ValueError('Training data must contain exactly the three Hillstrom actions.')
        self.models_ = {}
        self.prevalence_ = {}
        for action in ACTIONS:
            arm = development_df.loc[development_df['treatment'].eq(action)]
            if set(arm['conversion'].unique()) != {0, 1}:
                raise ValueError(f'{action} must contain binary outcomes and both classes.')
            self.models_[action] = build_model(self.family).fit(
                arm[FEATURE_COLUMNS], arm['conversion']
            )
            self.prevalence_[action] = float(arm['conversion'].mean())
        return self

    def predict_probabilities(self, features):
        """Columns follow ACTIONS: control, men's email, women's email."""
        return np.column_stack([
            self.models_[action].predict_proba(features[FEATURE_COLUMNS])[:, 1]
            for action in ACTIONS
        ])

    def predict_uplift(self, features):
        p = self.predict_probabilities(features)
        return p[:, 1:] - p[:, [0]]
