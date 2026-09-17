"""
Reads the raw training data from data/raw/train.csv.
"""

from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/train.csv")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from a CSV file.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Raw data as a DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{path}'. "
            "Make sure train.csv is placed in data/raw/."
        )

    df = pd.read_csv(path)
    print(f"[load_data] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(data.head())
