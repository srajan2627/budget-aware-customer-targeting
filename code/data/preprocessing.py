"""Unfitted preprocessing pipelines for Hillstrom customer attributes.

Create a fresh preprocessor inside each model pipeline so cross-validation
learns imputation, scaling, and categories only from each training fold.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["recency", "history"]
BINARY_FEATURES = ["mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor(*, segmentation=False):
    """Return a new, unfitted transformer; discard all non-feature columns.

    Supervised models retain raw history and unscaled binary attributes.
    Segmentation uses log1p(history) and scales binary attributes, matching
    the original EDA's feature choices without fitting on held-out data.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    binary_steps = [
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True))
    ]
    if segmentation:
        binary_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    if segmentation:
        history = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        numeric_transformers = [
            ("numeric", numeric, ["recency"]),
            ("history_log", history, ["history"]),
        ]
    else:
        numeric_transformers = [("numeric", numeric, NUMERIC_FEATURES)]
    return ColumnTransformer(
        numeric_transformers + [
            ("binary", Pipeline(binary_steps), BINARY_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
