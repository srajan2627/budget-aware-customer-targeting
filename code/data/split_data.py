"""Create the fixed 80% development / 20% final-test Hillstrom split."""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_development_test(df):
    """Stratify by treatment × conversion, retaining source row indices.

    Use development data for training and model selection. Reserve test data
    for final evaluation. Fit future preprocessing within training folds.
    """
    if not df.index.is_unique:
        raise ValueError("Unique source row indices are required to track split membership.")
    if df[["treatment", "conversion"]].isna().any().any():
        raise ValueError("Treatment and conversion must not contain missing values.")
    strata = df["treatment"].astype(str) + "_" + df["conversion"].astype(str)
    development_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=strata
    )
    return development_df.copy(), test_df.copy()


def summarize_split(df):
    """Return customer and purchase counts for each treatment."""
    return df.groupby("treatment").agg(
        customers=("conversion", "size"), purchases=("conversion", "sum")
    )


def main():
    from load_hillstrom import PROJECT_ROOT, load_hillstrom

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "datasets" / "processed",
        help="Directory for development.csv and test.csv (existing files are replaced).",
    )
    args = parser.parse_args()
    development_df, test_df = split_development_test(load_hillstrom())
    print(f"Development customers: {len(development_df):,}")
    print(f"Test customers:        {len(test_df):,}")
    print("\nDevelopment set:")
    print(summarize_split(development_df).to_string())
    print("\nTest set:")
    print(summarize_split(test_df).to_string())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_df.to_csv(args.output_dir / "development.csv", index_label="source_row")
    test_df.to_csv(args.output_dir / "test.csv", index_label="source_row")
    print(f"\nSaved splits to {args.output_dir}")


if __name__ == "__main__":
    main()
