# House Prices MLOps Pipeline

An automated MLOps pipeline for predicting house sale prices (Ames Housing
dataset, Kaggle). The pipeline covers three stages — **data engineering**,
**model engineering**, and **deployment** — connected into a single
automated run that repeats every 5 minutes.

- **Data engineering**: DVC pipeline that cleans and splits the raw data.
- **Model engineering**: feature engineering + Linear Regression, tracked
  with MLflow.
- **Deployment**: a FastAPI model API and a Streamlit web app, each running
  in its own Docker container.

## Repository structure

```
├── code
│   ├── datasets/            # Stage 1: data loading, cleaning, splitting
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   └── split_data.py
│   ├── models/               # Stage 2: features, training, MLflow logging
│   │   ├── features.py
│   │   └── train.py
│   └── deployment/           # Stage 3: API + app, Dockerized
│       ├── docker-compose.yml
│       ├── api/
│       │   ├── main.py
│       │   ├── model_loader.py
│       │   ├── schemas.py
│       │   ├── Dockerfile
│       │   └── requirements.txt
│       └── app/
│           ├── app.py
│           ├── Dockerfile
│           └── requirements.txt
├── data
│   ├── raw/                  # train.csv, test.csv (Kaggle House Prices)
│   ├── interim/               # cleaned.csv (generated)
│   └── processed/             # train.csv, test.csv (generated)
├── models/                    # model.joblib, feature_defaults.json (generated)
├── dvc.yaml                   # Stage 1 + Stage 2 pipeline definition
├── params.yaml                 # pipeline parameters (split ratio, etc.)
├── metrics.json                 # latest test metrics (generated, tracked by DVC)
├── mlflow.db                     # local MLflow tracking store (generated)
├── run_pipeline.py                # orchestrates the full pipeline on a schedule
└── requirements.txt
```

## Dataset

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
(Ames Housing), 1460 rows, 81 columns. Place `train.csv` (and optionally
`test.csv`) from Kaggle into `data/raw/` before running the pipeline.

## Prerequisites

- Python 3.11+
- Docker Desktop (with Docker Compose)
- Git

## Setup

1. Clone the repository:
   ```powershell
   git clone https://github.com/Azammmat/house-prices-mlops.git
   cd house-prices-mlops
   ```

2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Place the Kaggle dataset in `data/raw/train.csv` (and `data/raw/test.csv`
   if you have it — it is not used by the pipeline itself, since it has no
   `SalePrice` column to evaluate against).

## Running the pipeline

### Option A: Run everything once, manually

```powershell
dvc repro
docker compose -f code/deployment/docker-compose.yml up --build -d
```

This runs Stage 1 (clean + split), Stage 2 (feature engineering + training +
MLflow logging), and starts Stage 3 (API + app containers).

### Option B: Fully automated, repeating every 5 minutes (required mode)

```powershell
python run_pipeline.py
```

This repeats the full pipeline (`dvc repro` → rebuild/restart Docker
containers) every 5 minutes, so the deployed model always reflects the
latest data and code. Stop it with `Ctrl+C`.

To run a single pass without looping (useful for testing):
```powershell
python run_pipeline.py --once
```

If a full run takes longer than 5 minutes on your machine, increase
`RUN_INTERVAL_SECONDS` at the top of `run_pipeline.py`.

## Accessing the results

Once the containers are running:

- **Web app (Streamlit)**: http://localhost:8501
  Enter the house's key characteristics and click **Predict price**.
- **Model API (FastAPI)**: http://localhost:8000/docs
  Interactive Swagger UI — try `POST /predict` directly, or `GET /health`
  for a liveness check.

To stop the containers:
```powershell
docker compose -f code/deployment/docker-compose.yml down
```

## Inspecting metrics and experiments

- **DVC metrics** (latest test set MAE / RMSE / R²):
  ```powershell
  dvc metrics show
  ```
- **MLflow UI** (full experiment history):
  ```powershell
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```
  Open http://localhost:5000.

## Pipeline details

### Stage 1: Data engineering (DVC)

- Loads `data/raw/train.csv`.
- Fills structurally missing values (e.g. `PoolQC = NaN` means "no pool")
  with a sentinel category, and imputes genuinely missing values
  (e.g. `LotFrontage`) with median/mode.
- Removes outliers in `GrLivArea` using the IQR method.
- Splits into train/test (80/20 by default, see `params.yaml`) and saves
  to `data/processed/`.

### Stage 2: Model engineering (MLflow)

- Builds a preprocessing pipeline (scaling for numeric features, one-hot
  encoding for categorical features).
- Trains a Linear Regression model.
- Evaluates on the test set (MAE, RMSE, R²), logs parameters/metrics/model
  to MLflow, and saves the packaged pipeline to `models/model.joblib`.
- Also saves `models/feature_defaults.json`: median/mode values for every
  feature the model was trained on, used to fill in fields the user does
  not provide through the app/API.

### Stage 3: Deployment (Docker)

- **API** (`code/deployment/api`): FastAPI service exposing `POST /predict`.
  Accepts 12 key house features (overall quality, living area, garage
  size, etc.) and fills in the rest from `feature_defaults.json`.
- **App** (`code/deployment/app`): Streamlit UI with input fields for the
  same 12 features, a predict button, and a result display.
- Both run in separate containers, connected over the Docker Compose
  network. The trained model is mounted as a read-only volume into the
  API container (not baked into the image), so retraining doesn't require
  rebuilding the image — `run_pipeline.py` restarts the API container
  after each retraining run to pick up the latest model.
