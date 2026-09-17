"""
Full pipeline orchestrator.

Runs the complete MLOps pipeline end to end:
  1) Data engineering + model engineering (via `dvc repro`)
  2) Deployment: (re)builds and (re)starts the API + app containers via
     Docker Compose, so the API container always picks up the freshly
     retrained model (mounted as a volume, see docker-compose.yml).

The whole thing repeats automatically every RUN_INTERVAL_SECONDS. If a
run takes longer than the interval, the next run starts immediately
after the previous one finishes instead of overlapping with it.

Usage:
    python run_pipeline.py            # runs forever, every 5 minutes
    python run_pipeline.py --once     # runs a single pass and exits
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = REPO_ROOT / "code" / "deployment" / "docker-compose.yml"

RUN_INTERVAL_SECONDS = 5 * 60  # 5 minutes, per assignment requirements


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_command(command: list[str], cwd: Path = REPO_ROOT) -> None:
    """Run a subprocess, streaming its output, and raise on failure."""
    log(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def run_data_and_model_stages() -> None:
    """Stage 1 + Stage 2: reproduce the DVC pipeline (clean -> split -> train)."""
    log("=== Running data engineering + model engineering (dvc repro) ===")
    run_command(["dvc", "repro"])


def run_deployment_stage() -> None:
    """Stage 3: (re)build and (re)start the API + app containers.

    `docker compose up --build -d` rebuilds only what changed and
    restarts the API container so it picks up the latest model file
    (mounted as a volume from ./models).

    The API container is then explicitly restarted. This is needed
    because the model is mounted as a volume (not baked into the
    image), so a plain `up --build` will NOT reload it if the
    Dockerfile/code didn't change - restarting forces the API process
    to re-read model.joblib / feature_defaults.json from disk.
    """
    log("=== Running deployment stage (docker compose up --build) ===")
    run_command(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "up",
            "--build",
            "-d",
        ]
    )

    log("=== Restarting API container to pick up the latest trained model ===")
    run_command(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "restart",
            "api",
        ]
    )


def run_full_pipeline_once() -> None:
    """Run all three stages once, end to end."""
    start = time.monotonic()
    try:
        run_data_and_model_stages()
        run_deployment_stage()
        elapsed = time.monotonic() - start
        log(f"Pipeline run completed successfully in {elapsed:.1f}s")
    except Exception as exc:
        elapsed = time.monotonic() - start
        log(f"Pipeline run FAILED after {elapsed:.1f}s: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MLOps pipeline.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pass and exit, instead of looping every 5 minutes.",
    )
    args = parser.parse_args()

    if args.once:
        run_full_pipeline_once()
        return

    log(
        f"Starting scheduled pipeline runner "
        f"(every {RUN_INTERVAL_SECONDS // 60} minutes). Press Ctrl+C to stop."
    )
    while True:
        run_full_pipeline_once()
        log(f"Sleeping for {RUN_INTERVAL_SECONDS // 60} minutes...")
        time.sleep(RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped by user.")
        sys.exit(0)
