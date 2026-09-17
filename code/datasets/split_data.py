"""
Splits the cleaned data into training and testing sets and saves them
to data/processed/. Parameters are read from params.yaml so they can
be tracked and changed by DVC without touching the code.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from clean_data import INTERIM_DATA_PATH

PROCESSED_DIR = Path("data/processed")
TRAIN_OUT_PATH = PROCESSED_DIR / "train.csv"
TEST_OUT_PATH = PROCESSED_DIR / "test.csv"
PARAMS_PATH = Path("params.yaml")


def load_split_params(path: Path = PARAMS_PATH) -> dict:
    """Load split parameters from params.yaml, with sane defaults."""
    defaults = {"test_size": 0.2, "random_state": 42}

    if not path.exists():
        return defaults

    with open(path, "r") as f:
        params = yaml.safe_load(f) or {}

    split_params = params.get("split", {})
    return {**defaults, **split_params}


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    print(
        f"[split_data] Split into train={train_df.shape[0]} rows, "
        f"test={test_df.shape[0]} rows (test_size={test_size})"
    )
    return train_df, test_df


if __name__ == "__main__":
    if not INTERIM_DATA_PATH.exists():
        raise FileNotFoundError(
            f"'{INTERIM_DATA_PATH}' not found. Run clean_data.py first."
        )

    cleaned_df = pd.read_csv(INTERIM_DATA_PATH, keep_default_na=False, na_values=[""])
    params = load_split_params()

    train_df, test_df = split_data(
        cleaned_df,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUT_PATH, index=False)
    test_df.to_csv(TEST_OUT_PATH, index=False)
    print(f"[split_data] Saved '{TRAIN_OUT_PATH}' and '{TEST_OUT_PATH}'")
