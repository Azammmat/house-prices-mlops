"""
Loads train/test data from data/processed/, builds features, trains a
Linear Regression model, evaluates it on the test set, logs
parameters/metrics/model to a local MLflow tracking store backed by
SQLite (mlflow.db, no separate server required), and saves the fitted
pipeline (preprocessing + model bundled together) to
models/model.joblib so the deployment stage can load a single
artifact.
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from features import TARGET_COLUMN, build_feature_pipeline, split_features_target

TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
DEFAULTS_PATH = MODEL_DIR / "feature_defaults.json"
METRICS_PATH = Path("metrics.json")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "house-prices-regression"

# The ~10-15 fields the app/API exposes to the user. Every other
# feature the model needs is filled in from feature_defaults.json
# (median for numeric columns, mode for categorical columns),
# computed from the training data below.
KEY_INPUT_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "Neighborhood",
    "BedroomAbvGr",
    "LotArea",
    "KitchenQual",
    "HouseStyle",
    "CentralAir",
]


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test CSVs produced by Stage 1."""
    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed train/test data not found. Run Stage 1 (dvc repro) first."
        )

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    return train_df, test_df


def compute_feature_defaults(X_train: pd.DataFrame) -> dict:
    """Compute a default value for every feature column, used to fill
    in whatever fields the user does NOT provide through the API/app.

    Numeric columns default to the median, categorical columns default
    to the mode. Values are cast to plain Python types so they can be
    serialized to JSON.
    """
    defaults = {}
    numeric_cols = X_train.select_dtypes(include="number").columns
    categorical_cols = X_train.select_dtypes(exclude="number").columns

    for col in numeric_cols:
        defaults[col] = float(X_train[col].median())

    for col in categorical_cols:
        mode_value = X_train[col].mode(dropna=True)
        defaults[col] = mode_value.iloc[0] if not mode_value.empty else "NoFeature"

    return defaults


def evaluate(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_evaluate() -> dict:
    """Full training routine: load data, build features, train, evaluate,
    log to MLflow, and save the packaged model. Returns the test metrics.
    """
    train_df, test_df = load_processed_data()

    X_train, y_train = split_features_target(train_df, TARGET_COLUMN)
    X_test, y_test = split_features_target(test_df, TARGET_COLUMN)

    preprocessor = build_feature_pipeline(X_train)

    model = LinearRegression()
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("n_train_rows", X_train.shape[0])
        mlflow.log_param("n_test_rows", X_test.shape[0])
        mlflow.log_param("n_raw_features", X_train.shape[1])

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print("[train] Test metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[train] Saved packaged model (preprocessing + model) to '{MODEL_PATH}'")

    defaults = compute_feature_defaults(X_train)
    missing_key_features = [f for f in KEY_INPUT_FEATURES if f not in defaults]
    if missing_key_features:
        raise ValueError(
            f"KEY_INPUT_FEATURES contains columns not found in the training "
            f"data: {missing_key_features}. Check column names."
        )

    with open(DEFAULTS_PATH, "w") as f:
        json.dump(defaults, f, indent=2)
    print(
        f"[train] Saved feature defaults for {len(defaults)} columns "
        f"to '{DEFAULTS_PATH}' ({len(KEY_INPUT_FEATURES)} of them are "
        f"user-editable key inputs)"
    )

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Saved metrics to '{METRICS_PATH}'")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
