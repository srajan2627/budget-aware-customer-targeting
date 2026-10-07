"""Create shared treatment-by-conversion folds without loading final test data."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_development_folds(development_df, n_splits=5):
    """Return positional train/validation indices suitable for sklearn cv.

    Reuse this list for all competing methods on the same ordered dataframe.
    Each model must fit its own preprocessing within its training fold.
    """
    if not development_df.index.is_unique:
        raise ValueError("Development source row indices must be unique.")
    columns = development_df[["treatment", "conversion"]]
    if columns.isna().any().any():
        raise ValueError("Treatment and conversion must be present for stratification.")
    strata = columns["treatment"].astype(str) + "_" + columns["conversion"].astype(str)
    if strata.value_counts().min() < n_splits:
        raise ValueError("Each treatment/conversion stratum needs at least n_splits rows.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(cv.split(development_df, strata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-path", type=Path,
        default=PROJECT_ROOT / "datasets/processed/development.csv",
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=PROJECT_ROOT / "datasets/processed/development_folds.csv",
    )
    args = parser.parse_args()
    development_df = pd.read_csv(args.development_path, index_col="source_row")
    folds = make_development_folds(development_df)
    assignments = pd.Series(index=development_df.index, dtype="int64", name="validation_fold")
    print(f"Development customers: {len(development_df):,}")
    for fold, (train, validation) in enumerate(folds, start=1):
        assignments.iloc[validation] = fold
        print(f"Fold {fold}: {len(train):,} training / {len(validation):,} validation customers")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.astype("int64").to_csv(args.output_path, index_label="source_row")
    print(f"Saved shared fold assignments to {args.output_path}")
    print("Final test data was not loaded.")


if __name__ == "__main__":
    main()
