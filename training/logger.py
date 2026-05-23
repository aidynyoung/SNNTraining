"""
training/logger.py — lightweight JSONL experiment logger.

Writes one JSON record per log() call to a .jsonl file.
No external dependencies — no W&B, no MLflow.
Compatible with jq, pandas, and any line-delimited JSON reader.

Usage
-----
    from training.logger import ExperimentLogger, LogConfig

    logger = ExperimentLogger(LogConfig(path="results/run.jsonl", run_name="exp1"))
    logger.log({"step": 100, "pearson_r": 0.72, "mse": 0.04})
    logger.close()

CLI replay
----------
    python -m training.logger results/run.jsonl --tail 10
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LogConfig:
    path: str = "results/run.jsonl"
    run_name: str = "run"
    flush_every: int = 10       # flush to disk every N records
    print_every: int = 0        # 0 = silent; N = print every N records


class ExperimentLogger:
    """
    Append-only JSONL logger.

    Each record is a flat dict automatically augmented with:
        run_name   : str
        wall_time  : float (Unix timestamp)
    """

    def __init__(self, config: Optional[LogConfig] = None) -> None:
        self.cfg = config or LogConfig()
        self._path = Path(self.cfg.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._n = 0
        self._t0 = time.time()

    def log(self, record: Dict[str, Any]) -> None:
        entry = {
            "run": self.cfg.run_name,
            "wall_time": round(time.time() - self._t0, 3),
            **record,
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._n += 1
        if self._n % self.cfg.flush_every == 0:
            self._fh.flush()
        if self.cfg.print_every > 0 and self._n % self.cfg.print_every == 0:
            parts = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in record.items()
            )
            print(f"[{self.cfg.run_name} n={self._n}] {parts}")

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Static reader
    # ------------------------------------------------------------------

    @staticmethod
    def read(path: str) -> List[Dict[str, Any]]:
        """Load all records from a .jsonl file."""
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def tail(path: str, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last n records."""
        return ExperimentLogger.read(path)[-n:]


# ---------------------------------------------------------------------------
# CLI: python -m training.logger <path> [--tail N]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a .jsonl experiment log")
    parser.add_argument("path", help="Path to .jsonl file")
    parser.add_argument("--tail", type=int, default=0,
                        help="Show last N records (0 = all)")
    parser.add_argument("--keys", nargs="+", default=None,
                        help="Only print these keys")
    args = parser.parse_args()

    records = ExperimentLogger.read(args.path)
    if args.tail:
        records = records[-args.tail:]

    for r in records:
        if args.keys:
            r = {k: r[k] for k in args.keys if k in r}
        print(json.dumps(r))
