#!/usr/bin/env python3
"""
Full Maxwell-PINN experiment pipeline for Article 3.

Two-step pipeline:
  Step 1: Physics pre-training on synthetic data (train_eval.py with Helmholtz + optional gate)
  Step 2: Fine-tuning on real data (train_cross_domain.py with --d2_model_path)

Two configurations (for ablation table):
  Config A (nogate): lambda_phy_mse=1.0, lambda_phy_res=0.1, use_residual_gate=False
  Config B (gate):   lambda_phy_mse=0.5, lambda_phy_res=0.05, use_residual_gate=True

Output: results_gpu/maxwell_pinn_real/
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

OUTPUT_BASE = PROJECT_ROOT / "results_gpu" / "maxwell_pinn_real"
PRETRAIN_DIR = OUTPUT_BASE / "pretrain"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = "benchmarks/wifi_csi_benchmark"
MODEL = "enhanced"
SEEDS = [0, 42, 100]
LABEL_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
FPA = 3

CONFIGS = {
    "nogate": {
        "lambda_phy_mse": "1.0",
        "lambda_phy_res": "0.1",
        "use_residual_gate": "False",
    },
    "gate": {
        "lambda_phy_mse": "0.5",
        "lambda_phy_res": "0.05",
        "use_residual_gate": "True",
    },
}


def run_cmd(cmd, label, cwd=None):
    """Run a subprocess and return elapsed time in seconds."""
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAIL] {label} ({elapsed:.0f}s)")
        err_lines = (result.stderr or "").strip().split("\n")[-8:]
        for line in err_lines:
            print(f"         {line}")
    else:
        print(f"  [OK]   {label} ({elapsed:.0f}s)")
    return elapsed, result.returncode


# =============================================================================
# PHASE 0: Physics Pre-training on Synthetic Data
# =============================================================================
def run_pretrain():
    print("\n" + "=" * 70)
    print("PHASE 0: Physics Pre-training on Synthetic Data")
    print("=" * 70)
    total = 0.0
    ckpt_map = {}  # (config_name, seed) -> checkpoint_path

    for cfg_name, cfg in CONFIGS.items():
        for seed in SEEDS:
            out_name = f"enhanced_{cfg_name}_s{seed}"
            out_json = PRETRAIN_DIR / f"{out_name}.json"
            expected_ckpt = PRETRAIN_DIR / f"final_{out_name}.pth"

            if expected_ckpt.exists():
                print(f"  [SKIP] {cfg_name} seed={seed}: {expected_ckpt.name} exists")
                ckpt_map[(cfg_name, seed)] = str(expected_ckpt)
                continue

            cmd = [
                sys.executable, "src_maxwell_pinn/train_eval.py",
                "--model", MODEL,
                "--difficulty", "hard",
                "--seed", str(seed),
                "--epochs", "100",
                "--batch", "64",
                "--n_samples", "5000",
                "--patience", "10",
                "--ckpt_dir", str(PRETRAIN_DIR),
                "--out_json", str(out_json),
                "--save_ckpt", "final",
                "--lambda_phy_mse", cfg["lambda_phy_mse"],
                "--lambda_phy_res", cfg["lambda_phy_res"],
                "--use_residual_gate", cfg["use_residual_gate"],
            ]

            print(f"  [RUN]  Pretrain {cfg_name} seed={seed} ...")
            elapsed, rc = run_cmd(cmd, f"Pretrain {cfg_name} seed={seed}")
            total += elapsed

            if rc == 0 and expected_ckpt.exists():
                ckpt_map[(cfg_name, seed)] = str(expected_ckpt)
            else:
                print(f"  [WARN] Checkpoint not found: {expected_ckpt}")

    return total, ckpt_map


# =============================================================================
# Common args for fine-tuning on real data
# =============================================================================
def get_finetune_args(d2_model_path=None):
    args = [
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
    if d2_model_path:
        args += ["--d2_model_path", d2_model_path]
    return args


# =============================================================================
# PHASE 1: CDAE LOSO
# =============================================================================
def run_loso(ckpt_map):
    print("\n" + "=" * 70)
    print("PHASE 1: CDAE LOSO (Leave-One-Subject-Out) with physics pretrain")
    print("=" * 70)
    total = 0.0

    for cfg_name in CONFIGS:
        out_dir = OUTPUT_BASE / f"{cfg_name}_loso"
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            out_file = out_dir / f"loso_{MODEL}_seed{seed}.json"
            if out_file.exists():
                print(f"  [SKIP] {cfg_name} LOSO seed={seed}: exists")
                continue

            ckpt = ckpt_map.get((cfg_name, seed))
            if not ckpt:
                print(f"  [SKIP] {cfg_name} LOSO seed={seed}: no pretrain checkpoint")
                continue

            cmd = get_finetune_args(d2_model_path=ckpt) + [
                "--protocol", "loso",
                "--seed", str(seed),
                "--loso_all_folds",
                "--checkpoint_dir", f"checkpoints/maxwell_pinn_real/{cfg_name}",
                "--out", str(out_file),
            ]
            elapsed, _ = run_cmd(cmd, f"{cfg_name} LOSO seed={seed}")
            total += elapsed

    return total


# =============================================================================
# PHASE 2: CDAE LORO
# =============================================================================
def run_loro(ckpt_map):
    print("\n" + "=" * 70)
    print("PHASE 2: CDAE LORO (Leave-One-Room-Out) with physics pretrain")
    print("=" * 70)
    total = 0.0

    for cfg_name in CONFIGS:
        out_dir = OUTPUT_BASE / f"{cfg_name}_loro"
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            out_file = out_dir / f"loro_{MODEL}_seed{seed}.json"
            if out_file.exists():
                print(f"  [SKIP] {cfg_name} LORO seed={seed}: exists")
                continue

            ckpt = ckpt_map.get((cfg_name, seed))
            if not ckpt:
                print(f"  [SKIP] {cfg_name} LORO seed={seed}: no pretrain checkpoint")
                continue

            cmd = get_finetune_args(d2_model_path=ckpt) + [
                "--protocol", "loro",
                "--seed", str(seed),
                "--loro_all_folds",
                "--checkpoint_dir", f"checkpoints/maxwell_pinn_real/{cfg_name}",
                "--out", str(out_file),
            ]
            elapsed, _ = run_cmd(cmd, f"{cfg_name} LORO seed={seed}")
            total += elapsed

    return total


# =============================================================================
# PHASE 3: STEA (Label Efficiency)
# =============================================================================
def run_stea(ckpt_map):
    print("\n" + "=" * 70)
    print("PHASE 3: STEA (Label Efficiency) with physics pretrain")
    print("=" * 70)
    total = 0.0

    for cfg_name in CONFIGS:
        out_dir = OUTPUT_BASE / f"{cfg_name}_stea"
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            ckpt = ckpt_map.get((cfg_name, seed))
            if not ckpt:
                print(f"  [SKIP] {cfg_name} STEA seed={seed}: no pretrain checkpoint")
                continue

            for ratio in LABEL_RATIOS:
                ratio_str = f"{ratio:.2f}"
                out_file = out_dir / f"stea_{MODEL}_ratio{ratio_str}_seed{seed}.json"
                if out_file.exists():
                    print(f"  [SKIP] {cfg_name} STEA ratio={ratio_str} seed={seed}: exists")
                    continue

                cmd = get_finetune_args(d2_model_path=ckpt) + [
                    "--protocol", "sim2real",
                    "--seed", str(seed),
                    "--label_ratio", str(ratio),
                    "--transfer_method", "fine_tune",
                    "--out", str(out_file),
                ]
                elapsed, _ = run_cmd(cmd, f"{cfg_name} STEA ratio={ratio_str} seed={seed}")
                total += elapsed

    return total


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Maxwell-PINN Full Experiment Pipeline (Article 3)")
    print("=" * 70)
    print(f"Output:  {OUTPUT_BASE}")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Seeds:   {SEEDS}")
    print(f"FPA:     {FPA}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t_global = time.time()

    t_pre, ckpt_map = run_pretrain()
    print(f"\n  Pretrain checkpoints: {len(ckpt_map)}/{len(CONFIGS)*len(SEEDS)}")
    for k, v in sorted(ckpt_map.items()):
        print(f"    {k}: {Path(v).name}")

    t_loso = run_loso(ckpt_map)
    t_loro = run_loro(ckpt_map)
    t_stea = run_stea(ckpt_map)

    total_elapsed = time.time() - t_global

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"  Pretrain: {t_pre / 60:.1f} min")
    print(f"  LOSO:     {t_loso / 60:.1f} min")
    print(f"  LORO:     {t_loro / 60:.1f} min")
    print(f"  STEA:     {t_stea / 60:.1f} min")
    print(f"  TOTAL:    {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.1f} hours)")
    print(f"  Output:   {OUTPUT_BASE}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Quick results summary
    print("\n--- Quick Results Summary ---")
    for subdir in sorted(OUTPUT_BASE.iterdir()):
        if subdir.is_dir() and subdir.name != "pretrain":
            print(f"\n  [{subdir.name}]")
            for f in sorted(subdir.glob("*.json")):
                try:
                    d = json.load(open(f))
                    if "aggregate_stats" in d:
                        f1 = d["aggregate_stats"]["macro_f1"]
                        f1_val = f1["mean"] * 100 if isinstance(f1, dict) else f1 * 100
                        print(f"    {f.name}: F1={f1_val:.1f}%")
                    elif "target_metrics" in d:
                        f1_val = d["target_metrics"]["macro_f1"] * 100
                        lr = d.get("label_ratio", "?")
                        print(f"    {f.name}: F1={f1_val:.1f}% (ratio={lr})")
                except Exception:
                    pass

    # Pretrain summary
    print("\n  [pretrain]")
    for f in sorted(PRETRAIN_DIR.glob("*.json")):
        try:
            d = json.load(open(f))
            f1 = d.get("metrics", {}).get("macro_f1", 0)
            es = d.get("early_stop", {})
            print(f"    {f.name}: F1={f1*100:.1f}% (best_epoch={es.get('best_epoch')}, best={es.get('best_value', 0)*100:.1f}%)")
        except Exception:
            pass
