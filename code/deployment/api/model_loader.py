"""
Loads the packaged model (preprocessing + Linear Regression, saved by
code/models/train.py as models/model.joblib) and the per-column
defaults (models/feature_defaults.json), and exposes a function that
turns a partial (12-field) user input into the full feature row the
model expects.
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd

# Directory containing model.joblib and feature_defaults.json.
# - In Docker: MODEL_DIR is set explicitly (see Dockerfile: /app/models),
#   so the code below never needs to guess the path.
# - Locally (no MODEL_DIR set): falls back to <repo_root>/models, found
#   by walking up from this file until a "models" sibling exists, or
#   the repo root marker (dvc.yaml) is found. This works regardless of
#   how deeply nested this file is on either system.
def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "dvc.yaml").exists():
            return candidate
    # Fallback: assume repo root is 3 levels up (code/deployment/api/<file>)
    return start.parents[min(3, len(start.parents) - 1)]


_env_model_dir = os.environ.get("MODEL_DIR")
if _env_model_dir:
    MODEL_DIR = Path(_env_model_dir)
else:
    _repo_root = _find_repo_root(Path(__file__).resolve().parent)
    MODEL_DIR = _repo_root / "models"

MODEL_PATH = MODEL_DIR / "model.joblib"
DEFAULTS_PATH = MODEL_DIR / "feature_defaults.json"


class ModelBundle:
    """Holds the trained pipeline and the defaults needed to fill in
    fields the user did not provide.
    """

    def __init__(self, model_path: Path = MODEL_PATH, defaults_path: Path = DEFAULTS_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Run the training pipeline (Stage 1 + Stage 2) first."
            )
        if not defaults_path.exists():
            raise FileNotFoundError(
                f"Feature defaults file not found at '{defaults_path}'. "
                "Run the training pipeline (Stage 1 + Stage 2) first."
            )

        self.pipeline = joblib.load(model_path)
        with open(defaults_path, "r") as f:
            self.defaults = json.load(f)

    def build_full_row(self, user_input: dict) -> pd.DataFrame:
        """Merge user-provided fields with defaults for everything else,
        returning a single-row DataFrame in the shape the model expects.
        """
        full_row = dict(self.defaults)
        full_row.update(user_input)
        return pd.DataFrame([full_row])

    def predict(self, user_input: dict) -> float:
        """Predict SalePrice for a single house given partial input."""
        row = self.build_full_row(user_input)
        prediction = self.pipeline.predict(row)[0]
        return float(prediction)
