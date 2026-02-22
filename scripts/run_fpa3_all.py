#!/usr/bin/env python3
"""
Re-run Article 3 experiments with files_per_activity=3 to match Article 1 configuration.
Output: results_gpu/maxwell_fpa3/
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

OUTPUT_DIR = PROJECT_ROOT / "results_gpu" / "maxwell_fpa3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = "benchmarks/wifi_csi_benchmark"
MODEL = "enhanced"
SEEDS = [0, 42, 100]
LABEL_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
FPA = 3

COMMON_ARGS = [
    sys.executable, "src/train_cross_domain.py",
    "--model", MODEL,
    "--benchmark_path", BENCHMARK,
    "--files_per_activity", str(FPA),
    "--batch_size", "128",
    "--lr", "1e-3",
    "--epochs", "50",
    "--patience", "10",
    "--min_epochs", "15",
    "--class_weight", "inv_freq",
    "--input_norm", "zscore",
    "--amp",
]


def run_experiment(extra_args, out_file, label):
    if out_file.exists():
        print(f"  [SKIP] {label}: {out_file.name} exists")
        return 0.0

    cmd = COMMON_ARGS + extra_args + ["--out", str(out_file)]
    print(f"  [RUN]  {label} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {label} ({elapsed:.0f}s)")
        err_lines = (result.stderr or "").strip().split("\n")[-5:]
        for line in err_lines:
            print(f"         {line}")
    else:
        print(f"  [OK]   {label} ({elapsed:.0f}s)")
    return elapsed


def run_loso():
    print("\n" + "=" * 60)
    print("PHASE 1/3: LOSO (Leave-One-Subject-Out) with fpa=3")
    print("=" * 60)
    total = 0.0
    for seed in SEEDS:
        out = OUTPUT_DIR / f"loso_{MODEL}_seed{seed}.json"
        extra = [
            "--protocol", "loso",
            "--seed", str(seed),
            "--loso_all_folds",
            "--checkpoint_dir", f"checkpoints/maxwell_fpa3",
        ]
        total += run_experiment(extra, out, f"LOSO seed={seed}")
    return total


def run_loro():
    print("\n" + "=" * 60)
    print("PHASE 2/3: LORO (Leave-One-Room-Out) with fpa=3")
    print("=" * 60)
    total = 0.0
    for seed in SEEDS:
        out = OUTPUT_DIR / f"loro_{MODEL}_seed{seed}.json"
        extra = [
            "--protocol", "loro",
            "--seed", str(seed),
            "--loro_all_folds",
            "--checkpoint_dir", f"checkpoints/maxwell_fpa3",
        ]
        total += run_experiment(extra, out, f"LORO seed={seed}")
    return total


def run_stea():
    print("\n" + "=" * 60)
    print("PHASE 3/3: STEA (Label Efficiency) with fpa=3")
    print("=" * 60)
    total = 0.0
    for seed in SEEDS:
        for ratio in LABEL_RATIOS:
            ratio_str = f"{ratio:.2f}"
            out = OUTPUT_DIR / f"stea_{MODEL}_ratio{ratio_str}_seed{seed}.json"
            extra = [
                "--protocol", "sim2real",
                "--seed", str(seed),
                "--label_ratio", str(ratio),
                "--transfer_method", "fine_tune",
            ]
            total += run_experiment(extra, out, f"STEA ratio={ratio_str} seed={seed}")
    return total


if __name__ == "__main__":
    print(f"Maxwell-PINN fpa=3 Experiment Suite")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"files_per_activity: {FPA}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    t_global = time.time()

    t_loso = run_loso()
    t_loro = run_loro()
    t_stea = run_stea()

    total_elapsed = time.time() - t_global

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"  LOSO:  {t_loso / 60:.1f} min")
    print(f"  LORO:  {t_loro / 60:.1f} min")
    print(f"  STEA:  {t_stea / 60:.1f} min")
    print(f"  TOTAL: {total_elapsed / 60:.1f} min")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Quick summary of results
    print("\n--- Quick Results Summary ---")
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        try:
            d = json.load(open(f))
            if "aggregate_stats" in d:
                f1 = d["aggregate_stats"]["macro_f1"]
                if isinstance(f1, dict):
                    f1_val = f1["mean"] * 100
                else:
                    f1_val = f1 * 100
                print(f"  {f.name}: F1={f1_val:.1f}%")
            elif "target_metrics" in d:
                f1_val = d["target_metrics"]["macro_f1"] * 100
                lr = d.get("label_ratio", "?")
                print(f"  {f.name}: F1={f1_val:.1f}% (ratio={lr})")
        except Exception:
            pass
