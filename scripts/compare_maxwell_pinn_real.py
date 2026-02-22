#!/usr/bin/env python3
"""Compare Maxwell-PINN (physics pretrained) vs EAN baseline (fpa=3)."""

import json
import os
import numpy as np

PINN_BASE = r"D:\phd_workspace\paperA\results_gpu\maxwell_pinn_real"
BASELINE_BASE = r"D:\phd_workspace\paperA\results_gpu\maxwell_fpa3"
SEEDS = [0, 42, 100]
RATIOS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]


def load_f1(path):
    d = json.load(open(path))
    if "aggregate_stats" in d:
        f1 = d["aggregate_stats"]["macro_f1"]
        return f1["mean"] if isinstance(f1, dict) else f1
    elif "target_metrics" in d:
        return d["target_metrics"]["macro_f1"]
    return None


def get_stea_f1(base_dir, seeds, ratios):
    """Returns dict {ratio: [f1_s0, f1_s42, f1_s100]}"""
    result = {}
    for ratio in ratios:
        vals = []
        for seed in seeds:
            ratio_str = f"{ratio:.2f}"
            path = os.path.join(base_dir, f"stea_enhanced_ratio{ratio_str}_seed{seed}.json")
            if os.path.exists(path):
                vals.append(load_f1(path))
        result[ratio] = vals
    return result


print("=" * 90)
print("Maxwell-PINN (Physics Pretrained) vs EAN Baseline Comparison")
print("=" * 90)

# CDAE Comparison
print("\n--- CDAE (Cross-Domain Adaptation Evaluation) ---")
print(f"{'Protocol':<12} {'Config':<12} {'Seed 0':>8} {'Seed 42':>8} {'Seed 100':>8} {'Mean':>8} {'vs Base':>8}")
print("-" * 76)

for protocol in ["loso", "loro"]:
    # Baseline
    base_vals = []
    for seed in SEEDS:
        path = os.path.join(BASELINE_BASE, f"{protocol}_enhanced_seed{seed}.json")
        if os.path.exists(path):
            base_vals.append(load_f1(path))
    base_mean = np.mean(base_vals) if base_vals else 0

    row = "%-12s %-12s" % (protocol.upper(), "Baseline")
    for v in base_vals:
        row += "%8.1f" % (v * 100)
    row += "%8.1f" % (base_mean * 100)
    row += "%8s" % "-"
    print(row)

    for cfg in ["nogate", "gate"]:
        vals = []
        for seed in SEEDS:
            path = os.path.join(PINN_BASE, f"{cfg}_{protocol}", f"{protocol}_enhanced_seed{seed}.json")
            if os.path.exists(path):
                vals.append(load_f1(path))
        mean = np.mean(vals) if vals else 0
        delta = (mean - base_mean) * 100

        row = "%-12s %-12s" % ("", cfg)
        for v in vals:
            row += "%8.1f" % (v * 100)
        row += "%8.1f" % (mean * 100)
        row += "%+8.1f" % delta
        print(row)

# STEA Comparison
print("\n--- STEA (Label Efficiency) ---")
print(f"{'Ratio':<8} {'Baseline':>10} {'Nogate':>10} {'Gate':>10} {'NoG-Base':>10} {'Gate-Base':>10}")
print("-" * 58)

base_stea = get_stea_f1(BASELINE_BASE, SEEDS, RATIOS)
nogate_stea = get_stea_f1(os.path.join(PINN_BASE, "nogate_stea"), SEEDS, RATIOS)
gate_stea = get_stea_f1(os.path.join(PINN_BASE, "gate_stea"), SEEDS, RATIOS)

for ratio in RATIOS:
    bm = np.mean(base_stea.get(ratio, [0])) * 100
    nm = np.mean(nogate_stea.get(ratio, [0])) * 100
    gm = np.mean(gate_stea.get(ratio, [0])) * 100
    print("%-8s %10.1f %10.1f %10.1f %+10.1f %+10.1f" % (
        f"{ratio:.0%}", bm, nm, gm, nm - bm, gm - bm))

# Pretrain Summary
print("\n--- Physics Pre-training Results (Synthetic Data) ---")
print(f"{'Config':<12} {'Seed 0':>8} {'Seed 42':>8} {'Seed 100':>8} {'Mean':>8}")
print("-" * 44)

for cfg in ["nogate", "gate"]:
    vals = []
    for seed in SEEDS:
        path = os.path.join(PINN_BASE, "pretrain", f"enhanced_{cfg}_s{seed}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            vals.append(d.get("metrics", {}).get("macro_f1", 0))
    mean = np.mean(vals) if vals else 0
    print("%-12s %8.1f %8.1f %8.1f %8.1f" % (cfg, vals[0]*100, vals[1]*100, vals[2]*100, mean*100))

print("\n" + "=" * 90)
