"""CLI: run every stage (use --offline to skip fetching)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.setup_logging()
    pipeline.run_all(fetch="--offline" not in sys.argv, force="--force" in sys.argv)
