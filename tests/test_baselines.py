"""Check targeting semantics, causal weighting, and arm-specific fitting."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in ('models', 'targeting', 'evaluation'):
    sys.path.insert(0, str(ROOT / 'code' / directory))
from baselines import ACTIONS, ArmResponseModels
from policies import assign_policy
from policy_value import policy_scores
from run_baselines import load_shared_folds


class PolicyTests(unittest.TestCase):
    def test_response_and_uplift_rank_different_customers(self):
        # A buys often without email; B has the larger incremental response.
        p = np.array([[.20, .21, .19], [.02, .08, .07], [.10, .09, .08], [.10, .10, .10]])
        np.testing.assert_array_equal(assign_policy(p, 'response', .25), [1, 0, 0, 0])
        np.testing.assert_array_equal(assign_policy(p, 'uplift', .25), [0, 1, 0, 0])
        self.assertEqual(np.count_nonzero(assign_policy(p, 'uplift', 1)), 2)
        np.testing.assert_array_equal(assign_policy(p, 'no_contact', 1), np.zeros(4))

    def test_combined_budget_and_seeded_random_selection(self):
        p = np.tile([.01, .02, .03], (101, 1))
        for strategy in ('random_mens', 'random_womens', 'response', 'uplift'):
            actions = assign_policy(p, strategy, .10)
            self.assertEqual(np.count_nonzero(actions), 10)
            np.testing.assert_array_equal(actions, assign_policy(p, strategy, .10))
            self.assertEqual(np.count_nonzero(assign_policy(p, strategy, 0)), 0)
        a = assign_policy(p, 'random_mens', .10)
        b = assign_policy(p, 'random_womens', .10)
        np.testing.assert_array_equal(a != 0, b != 0)
        self.assertTrue((a[a != 0] == 1).all())
        self.assertTrue((b[b != 0] == 2).all())

    def test_ipw_known_balanced_trial(self):
        # Repeated identical covariates under all three actions: policy is men's
        # for everyone; men's converts and control never converts.
        a = np.tile([0, 1, 2], 10)
        y = (a == 1).astype(int)
        value, difference = policy_scores(a, y, np.ones(len(a)), [1/3]*3)
        self.assertAlmostEqual(value.mean(), 1.)
        self.assertAlmostEqual(difference.mean(), 1.)
        _, no_contact_difference = policy_scores(a, y, np.zeros(len(a)), [1/3]*3)
        np.testing.assert_array_equal(no_contact_difference, np.zeros(len(a)))
        _, negative = policy_scores(a, (a == 0).astype(int), np.ones(len(a)), [1/3]*3)
        self.assertAlmostEqual(negative.mean(), -1.)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            assign_policy([[0, np.nan, 1]], 'uplift', .1)
        with self.assertRaises(ValueError):
            policy_scores([0], [1], [1], [.2, .2, .2])


class ModelTests(unittest.TestCase):
    def frame(self):
        n = 120
        return pd.DataFrame(dict(
            recency=np.tile(np.arange(1, 11), 12), history=np.arange(n) + 10.,
            mens=np.arange(n) % 2, womens=(np.arange(n) + 1) % 2,
            newbie=np.arange(n) % 2, zip_code=['Urban'] * n, channel=['Web'] * n,
            treatment=np.repeat(ACTIONS, 40), conversion=np.tile([0, 0, 0, 1], 30),
        ))

    def test_arm_fit_and_uplift_subtraction(self):
        df = self.frame()
        for family in ('logistic', 'tree'):
            model = ArmResponseModels(family).fit(df)
            p = model.predict_probabilities(df)
            self.assertEqual(p.shape, (120, 3))
            self.assertTrue(((p >= 0) & (p <= 1)).all())
            np.testing.assert_allclose(model.predict_uplift(df), p[:, 1:] - p[:, [0]])
            for action in ACTIONS:
                pre = model.models_[action].named_steps['preprocess']
                median = pre.named_transformers_['numeric'].named_steps['impute'].statistics_[1]
                self.assertEqual(median, df.loc[df.treatment.eq(action), 'history'].median())
            altered = df.copy()
            altered['conversion'] = 1 - altered['conversion']
            altered['treatment'] = 'unobserved'
            np.testing.assert_allclose(p, model.predict_probabilities(altered))

    def test_saved_folds_follow_source_rows_after_reordering(self):
        df = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'folds.csv'
            original = load_shared_folds(df, path)
            shuffled = df.sample(frac=1, random_state=4)
            restored = load_shared_folds(shuffled, path)
            for (_, a), (_, b) in zip(original, restored):
                self.assertEqual(set(df.index[a]), set(shuffled.index[b]))


if __name__ == '__main__':
    unittest.main()
