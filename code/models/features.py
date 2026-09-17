"""
Transforms the cleaned train/test data (data/processed/*.csv) into
model-ready features:
- Numeric columns are used as-is (Linear Regression handles scaling
  internally via preprocessing below, missing values are already
  handled in Stage 1).
- Categorical columns are one-hot encoded.

The transform is fit ONLY on the training data and then applied to
the test data, to avoid data leakage. The fitted encoder is returned
so it can later be reused at inference time in the API.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"


def split_features_target(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    df = df.copy()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features and
    one-hot encodes categorical features.

    Unknown categories seen at inference time (a value that did not
    appear in training) are safely ignored rather than raising an
    error, which matters once this preprocessor is reused by the API.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )
    return preprocessor


def build_feature_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit a preprocessing pipeline on the training features only."""
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    return preprocessor


if __name__ == "__main__":
    train_df = pd.read_csv(Path("data/processed/train.csv"))
    X_train, y_train = split_features_target(train_df)

    preprocessor = build_feature_pipeline(X_train)
    X_train_transformed = preprocessor.transform(X_train)

    print(f"[features] Raw feature columns: {X_train.shape[1]}")
    print(f"[features] Transformed feature dimensions: {X_train_transformed.shape[1]}")
