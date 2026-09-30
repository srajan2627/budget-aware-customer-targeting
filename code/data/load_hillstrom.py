"""Load Hillstrom with only deterministic, row-level cleaning."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklift.datasets import fetch_hillstrom


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_hillstrom(data_home=None):
    """Match the EDA dataframe immediately before Section 23.

    Keep duplicate-looking customers. No imputation, scaling, or encoding
    is fitted here. The first call downloads the dataset into data_home.
    """
    if data_home is None:
        data_home = PROJECT_ROOT / "datasets" / "raw"
    hillstrom = fetch_hillstrom(target_col="all", data_home=str(data_home))
    df = pd.concat(
        [
            hillstrom.data.reset_index(drop=True),
            hillstrom.treatment.rename("treatment").reset_index(drop=True),
            hillstrom.target.reset_index(drop=True),
        ],
        axis=1,
    )
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].str.strip()
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    df["history_log1p"] = np.log1p(df["history"])
    return df
