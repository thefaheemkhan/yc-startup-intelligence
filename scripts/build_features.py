"""CLI: build derived features (incl. AI classification)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.setup_logging()
    pipeline.run_features()
