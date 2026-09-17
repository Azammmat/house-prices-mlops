# Deployment: Model API container.
#
# The model artifacts (models/model.joblib, models/feature_defaults.json)
# are NOT baked into this image. They are mounted at runtime via a Docker
# volume (see docker-compose.yml), because the training pipeline retrains
# the model every 5 minutes and we don't want to rebuild the image each time.

FROM python:3.11-slim

WORKDIR /app

# Install only what the API needs (kept separate from the full
# repo-level requirements.txt to keep the image small).
COPY code/deployment/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# API source code
COPY code/deployment/api/ /app/

# The model directory is mounted here at runtime (see docker-compose.yml).
# model_loader.py reads MODEL_DIR to find model.joblib / feature_defaults.json.
ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
