"""Independent checks for the reliability and simulation extensions."""
from itertools import combinations
from math import factorial
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/evaluation'))
from policy_value import paired_bootstrap_totals
from reliability import training_bootstrap
from report8_methods import (adjusted_overlap, chance_overlap, count_bootstrap_intervals,
                             exact_event_bounds, nested_training_samples,
                             probability_summary, support_diagnostic)


class Report8Tests(unittest.TestCase):
    def test_chance_reference_matches_exhaustive_unequal_subsets(self):
        population = set(range(5))
        overlaps = [len(set(a) & set(b)) / 2 for a in combinations(population, 2)
                    for b in combinations(population, 3)]
        self.assertAlmostEqual(np.mean(overlaps), chance_overlap(5, 2, 3))
        self.assertEqual(adjusted_overlap(.05, .05), 0)
        self.assertEqual(adjusted_overlap(1., .05), 1.)
        self.assertTrue(np.isnan(chance_overlap(5, 0, 3)))

    def test_nested_samples_are_paired_and_full_size_matches_original(self):
        frame = pd.DataFrame({'treatment': np.repeat(['a', 'b', 'c'], 20),
                              'conversion': np.tile([0, 1], 30)}, index=np.arange(60) + 100)
        unchanged = frame.copy(deep=True)
        samples = nested_training_samples(frame, [.25, .5, 1.], 2026)
        pd.testing.assert_frame_equal(samples[1.], training_bootstrap(frame, 2026))
        for action in ('a', 'b', 'c'):
            full = samples[1.].loc[lambda x: x.treatment.eq(action)]
            half = samples[.5].loc[lambda x: x.treatment.eq(action)]
            pd.testing.assert_frame_equal(full.iloc[:10], half)
        pd.testing.assert_frame_equal(frame, unchanged)
        self.assertTrue(set(samples[1.].index) <= set(frame.index))

    def test_empty_policy_and_nonempty_zero_events_have_different_meanings(self):
        empty = support_diagnostic(np.zeros(100), 0)
        sparse = support_diagnostic(np.zeros(100), 20)
        self.assertFalse(empty['degenerate_nonempty'])
        self.assertTrue(sparse['degenerate_nonempty'])
        self.assertIn('uninformative', sparse['support_status'])
        lower, upper = exact_event_bounds([[0, 0]], 20)
        self.assertLess(lower[0], 0)
        self.assertGreater(upper[0], 0)

    def test_exact_marginal_bounds_match_scipy_and_small_case_coverage(self):
        # One positive count and zero negatives: derive joint bound independently
        # using SciPy's scalar exact binomial intervals at 97.5% each.
        plus = binomtest(1, 8).proportion_ci(.975, method='exact')
        minus = binomtest(0, 8).proportion_ci(.975, method='exact')
        lo, hi = exact_event_bounds([[1, 0]], 8)
        self.assertAlmostEqual(lo[0], max(-8, 24 * (plus.low - minus.high)))
        self.assertAlmostEqual(hi[0], min(8, 24 * (plus.high - minus.low)))
        # Exhaust all possible categorical counts instead of trusting a
        # favorable Monte Carlo seed. Coverage must be conservative under DGM.
        k, qplus, qminus = 8, .12, .03
        truth = 3 * k * (qplus - qminus)
        coverage = 0.
        for p in range(k + 1):
            for m in range(k - p + 1):
                z = k - p - m
                probability = (factorial(k) / (factorial(p) * factorial(m) * factorial(z))
                               * qplus ** p * qminus ** m * (1 - qplus - qminus) ** z)
                low, high = exact_event_bounds([[p, m]], k)
                coverage += probability * bool(low[0] <= truth <= high[0])
        self.assertGreaterEqual(coverage, .95 - 1e-12)

    def test_count_bootstrap_matches_existing_bootstrap_distribution(self):
        # Independent Monte Carlo streams should estimate the same percentile
        # endpoints of the existing row-bootstrap distribution.
        scores = np.r_[np.full(7, 3.), np.full(3, -3.), np.zeros(90)]
        row = paired_bootstrap_totals(scores, repeats=30000, seed=73)[:, 0]
        low, high = count_bootstrap_intervals([[7, 3]], 100, 30000, np.random.default_rng(98))
        np.testing.assert_allclose([low[0], high[0]], np.quantile(row, [.025, .975]), atol=3.)
        zero_low, zero_high = count_bootstrap_intervals([[0, 0]], 100, 100, np.random.default_rng(1))
        self.assertEqual(zero_low[0], 0)
        self.assertEqual(zero_high[0], 0)

    def test_count_model_has_unbiased_ipw_expectation(self):
        # Enumerate all outcomes for two selected customers.
        k, qplus, qminus = 2, .2, .1
        mean = 0.
        for p in range(k + 1):
            for m in range(k - p + 1):
                z = k - p - m
                probability = factorial(k) / (factorial(p) * factorial(m) * factorial(z))
                probability *= qplus ** p * qminus ** m * (1 - qplus - qminus) ** z
                mean += probability * 3 * (p - m)
        self.assertAlmostEqual(mean, k * 3 * (qplus - qminus))

    def test_monte_carlo_uncertainty_not_zero_at_extreme_observed_rate(self):
        self.assertEqual(probability_summary(0, 1000)['mc_lower'], 0)
        self.assertEqual(probability_summary(1000, 1000)['mc_upper'], 1)
        self.assertGreater(probability_summary(0, 1000)['mc_upper'], 0)
        self.assertLess(probability_summary(1000, 1000)['mc_lower'], 1)
        self.assertAlmostEqual(probability_summary(500, 1000)['rate'], .5)

    def test_bad_counts_and_fractions_fail(self):
        with self.assertRaises(ValueError):
            exact_event_bounds([[2, 2]], 3)
        with self.assertRaises(ValueError):
            count_bootstrap_intervals([[1.5, 0]], 10, 20, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            nested_training_samples(pd.DataFrame(), [1.1], 1)


if __name__ == '__main__':
    unittest.main()
