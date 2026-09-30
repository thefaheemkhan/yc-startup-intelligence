"""CLI: load features into the relational database"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.setup_logging()
    print(pipeline.run_load())
