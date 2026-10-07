"""Check leakage boundaries and shared cross-validation membership."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/data"))
from development_cv import make_development_folds
from preprocessing import FEATURE_COLUMNS, build_preprocessor


def sample_frame():
    return pd.DataFrame({
        "recency": [1., 3., np.nan, 5.],
        "history": [10., 20., 30., 40.],
        "mens": [0., 1., np.nan, 0.],
        "womens": [1., 0., 1., 0.],
        "newbie": [0., 0., 1., 1.],
        "zip_code": ["Urban", "Suburban", np.nan, "Urban"],
        "channel": ["Web", "Phone", "Web", "Phone"],
        "conversion": [0, 0, 1, 0],
        "visit": [0, 1, 1, 0],
        "spend": [0., 0., 100., 0.],
        "treatment": ["Mens E-Mail"] * 4,
    })


class PreprocessingTests(unittest.TestCase):
    def test_validation_values_cannot_change_learned_statistics(self):
        train = sample_frame()
        held_out = train.iloc[[0]].copy()
        held_out["history"] = 1_000_000.
        held_out["zip_code"] = "Unseen location"
        held_out["channel"] = np.nan
        for segmentation in [False, True]:
            with self.subTest(segmentation=segmentation):
                transformer = build_preprocessor(segmentation=segmentation)
                before = transformer.fit_transform(train)
                transformed = transformer.transform(held_out)
                self.assertTrue(np.isfinite(transformed).all())
                self.assertEqual(before.shape[1], transformed.shape[1])
                np.testing.assert_array_equal(before, transformer.transform(train))
                encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
                self.assertNotIn("Unseen location", encoder.categories_[0])
        numeric = build_preprocessor().fit(train).named_transformers_["numeric"]
        np.testing.assert_allclose(numeric.named_steps["impute"].statistics_, [3., 25.])
        np.testing.assert_allclose(numeric.named_steps["scale"].mean_, [3., 25.])

    def test_outcomes_and_assignment_never_enter_features(self):
        train = sample_frame()
        for segmentation in [False, True]:
            transformer = build_preprocessor(segmentation=segmentation).fit(train)
            modified = train.copy()
            modified[["conversion", "visit", "spend"]] = 999
            modified["treatment"] = "Other"
            np.testing.assert_array_equal(transformer.transform(train), transformer.transform(modified))
            np.testing.assert_array_equal(
                transformer.transform(train), transformer.transform(train[FEATURE_COLUMNS])
            )
            self.assertFalse(hasattr(clone(transformer), "transformers_"))

    def test_shared_folds_are_stratified_complete_and_reproducible(self):
        df = pd.concat([sample_frame()] * 30, ignore_index=True)
        df["treatment"] = np.repeat(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], 40)
        folds = make_development_folds(df)
        repeated = make_development_folds(df)
        validation_rows = []
        for (train, validation), (train2, validation2) in zip(folds, repeated):
            self.assertEqual(len(set(train) & set(validation)), 0)
            self.assertEqual(len(train) + len(validation), len(df))
            self.assertEqual(df.iloc[validation].groupby(["treatment", "conversion"]).ngroups, 6)
            np.testing.assert_array_equal(train, train2)
            np.testing.assert_array_equal(validation, validation2)
            validation_rows.extend(validation)
        self.assertEqual(sorted(validation_rows), list(range(len(df))))
        # Exercise the intended sklearn integration on synthetic data only.
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        result = cross_validate(
            pipeline, df[FEATURE_COLUMNS], df["conversion"], cv=folds,
            return_estimator=True, error_score="raise",
        )
        for fitted, (train, _) in zip(result["estimator"], folds):
            means = fitted.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"].mean_
            expected = df.iloc[train][["recency", "history"]]
            np.testing.assert_allclose(means, expected.fillna(expected.median()).mean())

    def test_too_few_purchases_for_folds_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stratum"):
            make_development_folds(sample_frame())


if __name__ == "__main__":
    unittest.main()
