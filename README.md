# HelmholtzGatePinnResults

Experiment results and pretrained checkpoints for the **Maxwell-PINN (Physics-Informed Neural Network)** approach to WiFi CSI-based Human Activity Recognition (HAR).

This repository accompanies the paper:

> **Physics-Informed Neural Network with Helmholtz Wave Equation Constraints for WiFi CSI-based Human Activity Recognition**
>
> *Submitted to IEEE Internet of Things Journal*

## Overview

Maxwell-PINN integrates physical domain knowledge from the Helmholtz wave equation into a deep learning model for WiFi CSI sensing. The approach introduces three components on top of a standard EnhancedNet (EAN) backbone:

1. **Physics Head** — A linear projection layer that predicts wave frequency and temporal jitter from intermediate features.
2. **Helmholtz Loss** — A 1D wave equation residual loss enforcing that learned representations respect second-order temporal dynamics.
3. **Residual Gate** — A gating mechanism that modulates the task loss based on the physics residual magnitude.

The training pipeline consists of two phases:
- **Phase 1 (Pre-training)**: Train on physics-augmented synthetic CSI data with Helmholtz loss constraints.
- **Phase 2 (Fine-tuning)**: Transfer the pretrained backbone to real-world CSI data for downstream HAR tasks.

## Key Findings

| Finding | Detail |
|---|---|
| **+25.7 pp improvement at 1% labels** | Maxwell-PINN achieves 29.26% macro-F1 vs. EAN's 3.58% when only 1% of labels are available |
| **+21.9 pp improvement at 5% labels** | 62.63% vs. 40.76% macro-F1 |
| **Convergence at ≥10% labels** | Both models reach ~84% F1 with sufficient data |
| **Calibration improvement** | ECE reduced from 0.822 to 0.518 at 1% labels |
| **Cross-domain preservation** | LOSO/LORO performance maintained at ~84% F1 |

## Repository Structure

```
HelmholtzGatePinnResults/
├── README.md
├── results/
│   ├── maxwell_pinn_real/           # Core Maxwell-PINN experiment results
│   │   ├── pretrain/                # Phase 1: Synthetic pretraining
│   │   │   ├── enhanced_gate_s{0,42,100}.json      # Gate config results
│   │   │   ├── enhanced_nogate_s{0,42,100}.json    # No-gate config results
│   │   │   ├── final_enhanced_gate_s{0,42,100}.pth # Gate pretrained checkpoints
│   │   │   ├── final_enhanced_nogate_s{0,42,100}.pth # No-gate checkpoints
│   │   │   └── logs/               # Training logs
│   │   ├── gate_loso/              # CDAE LOSO (with residual gate)
│   │   ├── gate_loro/              # CDAE LORO (with residual gate)
│   │   ├── gate_stea/              # STEA label efficiency (with residual gate)
│   │   ├── nogate_loso/            # CDAE LOSO (Helmholtz only, no gate)
│   │   ├── nogate_loro/            # CDAE LORO (Helmholtz only, no gate)
│   │   └── nogate_stea/            # STEA label efficiency (Helmholtz only)
│   ├── ean_baseline_fpa3/          # EAN baseline results (files_per_activity=3)
│   ├── ablation_synthetic/         # Ablation study on synthetic data
│   ├── robustness/                 # Robustness evaluation
│   ├── srv_lite/                   # Synthetic Robustness Validation (SRV)
│   └── stea_ablation/              # STEA pretraining ablation
└── scripts/
    ├── run_maxwell_pinn_real.py     # Main orchestration script (full pipeline)
    ├── compare_maxwell_pinn_real.py # Result comparison and table generation
    └── run_fpa3_all.py             # EAN baseline runner (fpa=3)
```

## Experiment Results

### Cross-Domain Adaptation Evaluation (CDAE)

Three-seed evaluation (seeds 0, 42, 100) on the SenseFi WiFi CSI dataset.

| Model | Protocol | Seed 0 | Seed 42 | Seed 100 | Mean ± Std |
|---|---|---|---|---|---|
| EAN Baseline | LOSO | 84.15 | 84.11 | 84.12 | **84.13 ± 0.03** |
| Maxwell-PINN (gate) | LOSO | 84.08 | 84.02 | 84.02 | 84.04 ± 0.03 |
| Maxwell-PINN (nogate) | LOSO | 84.05 | 84.01 | 84.04 | 84.03 ± 0.03 |
| EAN Baseline | LORO | 84.25 | 84.07 | 84.25 | 84.19 ± 0.11 |
| Maxwell-PINN (gate) | LORO | 84.19 | 84.38 | 84.13 | **84.23 ± 0.13** |
| Maxwell-PINN (nogate) | LORO | 84.12 | 84.25 | 84.13 | 84.17 ± 0.09 |

**Takeaway**: Physics pre-training preserves baseline cross-domain generalization performance (~84% macro-F1).

### Sim2Real Transfer Efficiency Assessment (STEA)

Label efficiency evaluation across 6 label ratios, 3 seeds each.

| Label Ratio | EAN Baseline F1 (%) | Maxwell-PINN (gate) F1 (%) | EAN ECE | M-PINN ECE |
|---|---|---|---|---|
| **1%** | 3.58 ± 0.01 | **29.26 ± 10.61** | 0.822 | 0.518 |
| **5%** | 40.76 ± 3.49 | **62.63 ± 5.96** | 0.420 | 0.164 |
| 10% | 76.41 ± 1.41 | 75.50 ± 2.10 | 0.048 | 0.054 |
| 20% | 81.74 ± 1.40 | 80.44 ± 1.34 | 0.012 | 0.020 |
| 50% | 83.90 ± 0.08 | 83.05 ± 1.28 | 0.004 | 0.006 |
| 100% | 83.33 ± 0.00 | 83.33 ± 0.00 | 0.028 | 0.028 |

**Takeaway**: Maxwell-PINN provides substantial gains in **low-label regimes** (1%–5%), converging with the baseline when labels are abundant (≥10%).

### Ablation Study (Synthetic Data)

| Configuration | Macro-F1 (%) | ECE |
|---|---|---|
| EAN Baseline (no physics) | 95.35 | 0.034 |
| + Physics Head only | 95.01 | 0.032 |
| + Helmholtz Loss (nogate) | 72.59* | — |
| + Residual Gate (full) | 8.00* | — |

*\* Synthetic pretraining F1; the gate configuration shows instability on synthetic data but provides the best downstream low-label transfer.*

### Synthetic Robustness Validation (SRV)

| Condition | Macro-F1 (%) |
|---|---|
| Baseline (clean) | 7.82 |
| Gaussian Noise | 6.31 |
| Burst Interference | 1.13 |
| Overlap Corruption | 11.20 |

## Dataset

All experiments use the **SenseFi WiFi CSI Sensing Benchmark**:
- Source: [WiFi-CSI-Sensing-Benchmark](https://github.com/zhihaozhao/WiFi-CSI-Sensing-Benchmark)
- Input shape: 52 subcarriers × 32 time steps
- 8 activity classes
- `files_per_activity=3` for all experiments

## Pretrained Checkpoints

Six pretrained model checkpoints are provided in `results/maxwell_pinn_real/pretrain/`:

| Checkpoint | Config | Description |
|---|---|---|
| `final_enhanced_gate_s0.pth` | Gate, Seed 0 | Helmholtz + Residual Gate pretrained |
| `final_enhanced_gate_s42.pth` | Gate, Seed 42 | Helmholtz + Residual Gate pretrained |
| `final_enhanced_gate_s100.pth` | Gate, Seed 100 | Helmholtz + Residual Gate pretrained |
| `final_enhanced_nogate_s0.pth` | No-gate, Seed 0 | Helmholtz-only pretrained |
| `final_enhanced_nogate_s42.pth` | No-gate, Seed 42 | Helmholtz-only pretrained |
| `final_enhanced_nogate_s100.pth` | No-gate, Seed 100 | Helmholtz-only pretrained |

Each checkpoint is ~2.5 MB and contains the EnhancedNet backbone weights pretrained on physics-augmented synthetic CSI data.

## JSON Result Format

Each experiment produces a JSON file with structured results:

- **CDAE results** (`{protocol}_enhanced_seed{s}.json`): Contains `protocol`, `model`, `seed`, `aggregate_stats` (with `macro_f1.mean`, `macro_f1.std`, `ece`, etc.), and per-fold `fold_results`.
- **STEA results** (`stea_enhanced_ratio{r}_seed{s}.json`): Contains `label_ratio`, `zero_shot_metrics`, and `target_metrics` (with `macro_f1`, `ece`, per-class F1).
- **Pretrain results** (`enhanced_{config}_s{s}.json`): Contains `meta` (model, seed, git commit), `args` (full hyperparameters), `metrics` (macro_f1, ece, nll, brier), and `early_stop` info.

## Reproducibility

The orchestration scripts in `scripts/` can reproduce the full experiment pipeline:

```bash
# Full Maxwell-PINN pipeline (pretrain + CDAE + STEA, ~4 hours on RTX GPU)
python scripts/run_maxwell_pinn_real.py

# EAN baseline experiments (fpa=3)
python scripts/run_fpa3_all.py

# Generate comparison tables
python scripts/compare_maxwell_pinn_real.py
```

**Requirements**: PyTorch ≥ 2.0, CUDA-capable GPU, the SenseFi benchmark dataset.

## Hardware

All experiments were conducted on:
- **GPU**: NVIDIA GeForce RTX series
- **OS**: Windows 10/11
- **Python**: 3.11+, PyTorch 2.x

## License

This repository is provided for academic research purposes. Please cite the accompanying paper if you use these results.

## Citation

```bibtex
@article{zhao2026maxwell_pinn_har,
  title={Physics-Informed Neural Network with Helmholtz Wave Equation Constraints for WiFi CSI-based Human Activity Recognition},
  author={Zhao, Zhihao and others},
  journal={IEEE Internet of Things Journal},
  year={2026},
  note={Under Review}
}
```
