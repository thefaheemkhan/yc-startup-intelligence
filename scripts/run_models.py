"""CLI: run forecasting backtests + status model; writes reports/ and experiments/runs.jsonl"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402
from src.run_models import run_models  # noqa: E402

if __name__ == "__main__":
    pipeline.setup_logging()
    run_models()
