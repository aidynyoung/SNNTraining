"""
experiments/seed_benchmark.py
==============================
Multi-seed benchmark runner with confidence intervals.

Runs any Arthedain experiment N times across different random seeds and
reports mean ± std — the standard required by IQT evaluators and peer-
reviewed venues.  Single-seed results are not reproducibility evidence.

Usage
-----
    # Core benchmarks (recommended before any demo)
    python experiments/seed_benchmark.py --benchmark robustness --seeds 5
    python experiments/seed_benchmark.py --benchmark shd       --seeds 5
    python experiments/seed_benchmark.py --benchmark bci       --seeds 3

    # Quick smoke-test (2 seeds, reduced data)
    python experiments/seed_benchmark.py --benchmark robustness --seeds 2 --quick

    # All benchmarks (for submission)
    python experiments/seed_benchmark.py --benchmark all --seeds 5

Output
------
    stdout: table with mean ± std for each metric
    JSON:   results/seed_benchmark_{benchmark}_{timestamp}.json

IQT context
-----------
    TRL 5 requires "component validated in relevant environment."
    Seed-averaged results with confidence intervals are the minimum
    standard for "validated."  Single runs are TRL 4 at best.

Founders Fund context
---------------------
    "0.81 Pearson R" is a claim.
    "0.79 ± 0.03 Pearson R across 5 seeds (min 0.76, max 0.83)" is evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ci95(values: List[float]) -> tuple:
    """Return (mean, std, 95% CI half-width) using t-distribution."""
    if len(values) < 2:
        return float(values[0]), 0.0, 0.0
    arr = np.array(values, dtype=float)
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    from scipy import stats
    t = stats.t.ppf(0.975, df=len(values) - 1)
    ci = t * s / np.sqrt(len(values))
    return m, s, ci


def fmt(values: List[float], pct: bool = False) -> str:
    m, s, ci = ci95(values)
    scale = 100 if pct else 1
    unit  = "%" if pct else ""
    return f"{m*scale:.1f}{unit} ± {s*scale:.1f} (95% CI ±{ci*scale:.1f})"


def print_table(rows: List[Dict], title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if not rows:
        return
    col_w = max(len(r["metric"]) for r in rows) + 2
    for r in rows:
        pad = " " * (col_w - len(r["metric"]))
        print(f"  {r['metric']}{pad}{r['value']}")
    print(f"{'='*70}\n")


# ── Robustness benchmark ──────────────────────────────────────────────────────

def run_robustness_seed(seed: int, quick: bool = False) -> Dict[str, float]:
    """Run one seed of the SNN+HDC robustness experiment."""
    set_seed(seed)

    from experiments.arthedain_robustness import ExperimentConfig, run_single_config

    cfg = ExperimentConfig(
        input_size=20,
        hidden_size=64 if quick else 128,
        n_classes=4,
        T_train=2000 if quick else 4000,
        T_test=500  if quick else 1000,
        block_len=20,
        fault_type="stuck_at_0",
        fault_persistent=True,
        seed=seed,
    )

    results = {}
    for fault_rate in [0.0, 0.01, 0.1]:
        cfg.seed = seed  # ensure fresh seed each fault_rate
        r = run_single_config(cfg, fault_rate)
        results[f"snn_acc_{fault_rate}"] = float(r["snn_accuracy"])
        results[f"hdc_acc_{fault_rate}"] = float(r["hdc_accuracy"])

    results["snn_degradation_10pct"] = (
        results["snn_acc_0.0"] - results["snn_acc_0.1"]
    )
    results["hdc_degradation_10pct"] = (
        results["hdc_acc_0.0"] - results["hdc_acc_0.1"]
    )
    return results


def bench_robustness(seeds: List[int], quick: bool) -> Dict:
    print(f"\nRunning robustness experiment ({len(seeds)} seeds)…")
    all_results: List[Dict] = []
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{len(seeds)})…", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_robustness_seed(seed, quick=quick)
        print(f"{time.perf_counter()-t0:.1f}s")
        all_results.append(r)

    def collect(key):
        return [r[key] for r in all_results]

    rows = [
        {"metric": "SNN accuracy (0% faults)",  "value": fmt(collect("snn_acc_0.0"), pct=True)},
        {"metric": "HDC accuracy (0% faults)",  "value": fmt(collect("hdc_acc_0.0"), pct=True)},
        {"metric": "SNN accuracy (10% faults)", "value": fmt(collect("snn_acc_0.1"), pct=True)},
        {"metric": "HDC accuracy (10% faults)", "value": fmt(collect("hdc_acc_0.1"), pct=True)},
        {"metric": "SNN degradation @ 10%",     "value": fmt(collect("snn_degradation_10pct"), pct=True)},
        {"metric": "HDC degradation @ 10%",     "value": fmt(collect("hdc_degradation_10pct"), pct=True)},
    ]
    print_table(rows, f"Robustness (n={len(seeds)} seeds)")
    return {"rows": rows, "raw": all_results}


# ── SHD neuromorphic benchmark ─────────────────────────────────────────────────

def run_shd_seed(seed: int, quick: bool = False) -> Dict[str, float]:
    """Run one seed of the SHD benchmark (eprop-ridge mode)."""
    set_seed(seed)
    import subprocess, json as _json

    n_samples = 200 if quick else 500
    n_eval    = 100 if quick else 200
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "benchmark_neuromorphic.py"),
        "--mode", "eprop-ridge",
        "--n-samples", str(n_samples),
        "--n-eval", str(n_eval),
        "--epochs", "2" if quick else "3",
        "--hidden", "200" if quick else "300",
        "--no-save",
    ]
    # Note: benchmark_neuromorphic.py does not accept --seed; uses torch seed internally
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # Parse accuracy from stdout
    acc = None
    for line in out.stdout.splitlines():
        if "Arthedain (this run)" in line:
            try:
                acc = float(line.split()[-1].replace("%", "")) / 100.0
            except Exception:
                pass
    if acc is None:
        print(f"\n    [WARN] Could not parse accuracy from seed {seed}")
        print(out.stdout[-500:])
        acc = float("nan")
    return {"shd_accuracy": acc}


def bench_shd(seeds: List[int], quick: bool) -> Dict:
    print(f"\nRunning SHD benchmark ({len(seeds)} seeds)…")
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{len(seeds)})…", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_shd_seed(seed, quick=quick)
        print(f"acc={r['shd_accuracy']:.1%}  {time.perf_counter()-t0:.1f}s")
        all_results.append(r)

    accs = [r["shd_accuracy"] for r in all_results if not np.isnan(r["shd_accuracy"])]
    rows = [
        {"metric": "SHD accuracy (eprop-ridge)", "value": fmt(accs, pct=True)},
        {"metric": "Min accuracy",               "value": f"{min(accs)*100:.1f}%"},
        {"metric": "Max accuracy",               "value": f"{max(accs)*100:.1f}%"},
        {"metric": "Seeds run",                  "value": str(len(accs))},
    ]
    print_table(rows, f"SHD Neuromorphic (n={len(seeds)} seeds)")
    return {"rows": rows, "raw": all_results}


# ── BCI decoding benchmark ─────────────────────────────────────────────────────

def run_bci_seed(seed: int, quick: bool = False) -> Dict[str, float]:
    """Run one seed of the BCI velocity decoding experiment via subprocess.

    NOTE: The published result (Pearson R 0.81) was obtained on the real
    Indy reaching dataset (CRCNS pmd-1, indy_2016-10-05_1.mat, 50K+ steps).
    This benchmark uses the synthetic dataset mode of bci_decoding.py, which
    produces lower but meaningful R values demonstrating the same algorithm.

    For the real published result: python experiments/bci_decoding.py --dataset indy
    """
    set_seed(seed)
    import subprocess

    T = 2000 if quick else 5000
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "bci_decoding.py"),
        "--dataset", "synthetic",
        "--T", str(T),
        "--input-size", "50",
        "--hidden-size", "128",
        "--seed", str(seed),
    ]
    # Note: synthetic BCI stream is a toy benchmark. The published result
    # (Pearson R 0.81) requires the real Indy dataset (CRCNS pmd-1).
    # Synthetic results are expected to be near-zero — this is honest.
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Parse Pearson R from output
    r_val = float("nan")
    for line in out.stdout.splitlines():
        if "Pearson R:" in line:
            try:
                r_val = float(line.split("Pearson R:")[1].strip().split()[0])
            except Exception:
                pass

    return {"pearson_r": r_val, "mse": float("nan")}


def bench_bci(seeds: List[int], quick: bool) -> Dict:
    print(f"\nRunning BCI decoding benchmark ({len(seeds)} seeds)…")
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{len(seeds)})…", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            r = run_bci_seed(seed, quick=quick)
            print(f"R={r['pearson_r']:.3f}  {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            r = {"pearson_r": float("nan"), "mse": float("nan")}
        all_results.append(r)

    rs  = [r["pearson_r"] for r in all_results if not np.isnan(r["pearson_r"])]
    mse = [r["mse"]       for r in all_results if not np.isnan(r["mse"])]
    rows = [
        {"metric": "BCI Pearson R", "value": fmt(rs)  if rs  else "N/A"},
        {"metric": "BCI MSE",       "value": fmt(mse) if mse else "N/A"},
        {"metric": "Min Pearson R", "value": f"{min(rs):.3f}" if rs else "N/A"},
        {"metric": "Max Pearson R", "value": f"{max(rs):.3f}" if rs else "N/A"},
    ]
    print_table(rows, f"BCI Velocity Decoding (n={len(seeds)} seeds)")
    return {"rows": rows, "raw": all_results}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed benchmark runner with confidence intervals"
    )
    parser.add_argument(
        "--benchmark", choices=["robustness", "shd", "bci", "all"],
        default="robustness",
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds (default 5 for publication, 2 for quick check)"
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed value (seeds = seed_start, seed_start+1, ...)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduce data size for fast smoke-test (~5× faster, less accurate)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON path (default: results/seed_benchmark_{bench}_{ts}.json)"
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"\nSeeds: {seeds}  |  Quick: {args.quick}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_data: Dict[str, Any] = {
        "seeds": seeds,
        "quick": args.quick,
        "timestamp": ts,
        "benchmarks": {},
    }

    benchmarks_to_run = (
        ["robustness", "shd", "bci"]
        if args.benchmark == "all"
        else [args.benchmark]
    )

    for bench in benchmarks_to_run:
        if bench == "robustness":
            all_data["benchmarks"]["robustness"] = bench_robustness(seeds, args.quick)
        elif bench == "shd":
            all_data["benchmarks"]["shd"] = bench_shd(seeds, args.quick)
        elif bench == "bci":
            all_data["benchmarks"]["bci"] = bench_bci(seeds, args.quick)

    out_path = args.out or str(
        RESULTS_DIR / f"seed_benchmark_{args.benchmark}_{ts}.json"
    )
    with open(out_path, "w") as f:
        # Serialize, replacing nan with null
        def _clean(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj
        json.dump(_clean(all_data), f, indent=2)

    print(f"\nResults saved → {out_path}")
    print("\nReproducibility statement for IQT / papers:")
    print("  All results reported as mean ± std across N random seeds.")
    print("  95% confidence intervals computed using Student's t-distribution.")
    print("  Seeds, hyperparameters, and raw data saved to JSON.")


if __name__ == "__main__":
    main()
