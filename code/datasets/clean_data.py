"""
Handles missing values and outliers in the dataset.

Missing-value strategy (this dataset encodes "feature absent" as NaN
for many columns - these are NOT random missing values and must not
be imputed with mean/median, or the distribution gets corrupted):

1. Categorical columns where NaN means "this feature does not exist
   for this house" -> filled with the string "None".
2. Numeric columns where NaN means "this feature does not exist,
   so its size/count is 0" -> filled with 0.
3. Genuinely missing values (unknown, not "absent") -> imputed with
   median (numeric) or mode (categorical).

Outlier strategy: IQR-based removal, applied to GrLivArea, which is
the well-known source of a couple of extreme outliers in this dataset
(very large living area sold at an unexpectedly low price).
"""

from pathlib import Path

import pandas as pd

from load_data import load_raw_data

INTERIM_DATA_PATH = Path("data/interim/cleaned.csv")

# Categorical columns where NaN means "feature does not exist"
NONE_FILL_CATEGORICAL = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]

# Numeric columns where NaN means "feature does not exist, so 0"
ZERO_FILL_NUMERIC = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs that encode 'feature absent' rather than 'unknown'."""
    df = df.copy()

    fill_map = {}
    for col in NONE_FILL_CATEGORICAL:
        if col in df.columns:
            fill_map[col] = "NoFeature"
    for col in ZERO_FILL_NUMERIC:
        if col in df.columns:
            fill_map[col] = 0

    if fill_map:
        df = df.fillna(value=fill_map)

    return df


def impute_remaining_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute genuinely unknown missing values.

    LotFrontage is imputed with the median LotFrontage of the same
    Neighborhood (a standard, more accurate trick for this dataset).
    All other remaining numeric columns use the column median, and
    all other remaining categorical columns use the column mode.
    """
    df = df.copy()

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        neighborhood_median = df.groupby("Neighborhood")["LotFrontage"].transform(
            "median"
        )
        overall_median = df["LotFrontage"].median()
        df["LotFrontage"] = df["LotFrontage"].fillna(neighborhood_median)
        df["LotFrontage"] = df["LotFrontage"].fillna(overall_median)

    fill_map = {}

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().any():
            fill_map[col] = df[col].median()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)
            fill_map[col] = mode_value.iloc[0] if not mode_value.empty else "NoFeature"

    if fill_map:
        df = df.fillna(value=fill_map)

    return df


def remove_outliers_iqr(
    df: pd.DataFrame, column: str, k: float = 1.5
) -> pd.DataFrame:
    """Remove rows where `column` falls outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    before = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    removed = before - len(df)
    print(f"[clean_data] Removed {removed} outlier rows based on '{column}'")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline: missing values, then outliers."""
    df = fill_structural_missing(df)
    df = impute_remaining_missing(df)

    remaining_na = df.isnull().sum().sum()
    print(f"[clean_data] Remaining NaNs after imputation: {remaining_na}")
    assert remaining_na == 0, "There are still missing values after cleaning!"

    if "GrLivArea" in df.columns:
        df = remove_outliers_iqr(df, "GrLivArea")

    return df


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    INTERIM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(INTERIM_DATA_PATH, index=False)
    print(
        f"[clean_data] Saved cleaned data to '{INTERIM_DATA_PATH}' "
        f"({cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns)"
    )