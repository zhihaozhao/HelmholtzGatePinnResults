# HelmholtzGatePinnResults

Experiment results and pretrained checkpoints for **Maxwell-PINN** (physics-informed learning with a supervised Physics Head, optional Helmholtz residual, and residual gating) for WiFi CSI-based human activity recognition (HAR).

This repository accompanies the manuscript work on Maxwell-PINN (journal submissions; see the current paper PDF for venue-specific wording and exact table numbering).

Repository home: [github.com/zhihaozhao/HelmholtzGatePinnResults](https://github.com/zhihaozhao/HelmholtzGatePinnResults)

## Overview

Maxwell-PINN builds on an **Enhanced Attention Network (EAN)** backbone and adds:

1. **Physics Head** — Supervised regression of interpretable physical targets (e.g., dominant Doppler frequency and frequency jitter) when synthetic supervision is available.
2. **Helmholtz residual (optional)** — A PDE-style residual term; naïve ungated residual training can be unstable on noisy CSI, which motivates gating and careful weighting.
3. **Residual gating** — Modulates how strongly physics objectives influence optimization when residual magnitudes are large.

Training is typically **two-stage**: synthetic pre-training with physics-aware objectives, then real-data fine-tuning for HAR.

## What is in this repository

| Area | Location | Notes |
|------|----------|--------|
| **Original end-to-end pipeline outputs** | `results/maxwell_pinn_real/`, `results/ean_baseline_fpa3/`, … | Pretrain, CDAE (LOSO/LORO), STEA JSONs and logs; includes `.pth` checkpoints where committed. |
| **Article-3 training-track mirror (JSON/CSV)** | `results/article3_results_gpu/` | Synced from the Maxwell-PINN article codebase (`results_gpu`): ablations, SRV-lite, robustness evals, ESTA, STEA (per-seed + aggregates), interpretability summaries. **No large checkpoints or ONNX in this subtree.** |
| **Article-3 edge profiling (JSON)** | `results/article3_edge_deployment/` | Latency / throughput / memory from the **compact deployment benchmark** (different parameterization than the full research model; see JSON `parameters` fields). |
| **Scripts** | `scripts/` | Orchestration and comparison utilities for the original pipeline. |

If you cite numbers, prefer the JSON files in `article3_results_gpu` for the latest article-aligned training track, and `article3_edge_deployment` only for on-device profiling tables.

## Key findings (article-aligned training track)

Summaries below follow `results/article3_results_gpu/stea_ablation/stea_results_ean.json` and `stea_results_maxwell.json` (**five seeds**, aggregated means ± standard deviations). Macro-F1 and ECE are reported as percentages (×100).

| Topic | Detail |
|-------|--------|
| **STEA @ 5%–20% labels** | Maxwell-PINN improves macro-F1 over the EAN baseline in the low-to-mid label regime; at **10%** labels the gap is about **+13.9 percentage points** (see aggregate JSON). |
| **STEA @ high label ratios** | Both models approach a high ceiling toward **50%–100%** labels; means converge within run-to-run variance. |
| **CDAE (LOSO / LORO)** | Physics-informed pre-training keeps cross-domain macro-F1 **near ~84%** and **statistically comparable** to the EAN baseline under the same protocol (see `results/maxwell_pinn_real/gate_*` and `ean_baseline_fpa3/` or the paper tables). |
| **Calibration (synthetic stress setting)** | In the ablation / trustworthiness table used in the paper, **residual-gated** Maxwell-PINN reaches **raw ECE ≈ 2.84%** vs **baseline EAN ≈ 3.39%** on the reported hard synthetic setting (single representative gated run; see manuscript). |
| **SRV / Hard+ robustness** | Under strong synthetic stress (**Hard+**, e.g. burst rate 0.3), macro-F1 remains low in absolute terms for both methods, but the physics-informed variant improves **relative** robustness (see `article3_results_gpu/robustness/eval_*_hard_plus.json`). |

## Repository structure (high level)

```
HelmholtzGatePinnResults/
├── README.md
├── results/
│   ├── article3_results_gpu/        # Mirror of article3 training-track JSON/CSV (no large weights)
│   │   ├── maxwell_ablation/
│   │   ├── srv_lite/
│   │   ├── robustness/
│   │   ├── stea_ablation/           # Per-seed STEA + stea_results_ean.json / stea_results_maxwell.json
│   │   ├── esta_ablation/
│   │   └── interpretability_analysis/
│   ├── article3_edge_deployment/    # Xavier, Nano, RDK X5 (BPU/CPU), Raspberry Pi 4 JSON benchmarks
│   ├── maxwell_pinn_real/           # Legacy full pipeline: pretrain, CDAE, STEA (gate / nogate)
│   ├── ean_baseline_fpa3/
│   ├── ablation_synthetic/
│   ├── robustness/
│   ├── srv_lite/
│   └── stea_ablation/
└── scripts/
    ├── run_maxwell_pinn_real.py
    ├── compare_maxwell_pinn_real.py
    └── run_fpa3_all.py
```

## Experiment results

### Cross-Domain Adaptation Evaluation (CDAE)

Three-seed evaluation (seeds **0, 42, 100**) on the SenseFi WiFi CSI benchmark, **files_per_activity = 3** (see `results/maxwell_pinn_real/` and `results/ean_baseline_fpa3/`).

| Model | Protocol | Seed 0 | Seed 42 | Seed 100 | Mean ± Std |
|-------|----------|--------|---------|----------|------------|
| EAN Baseline | LOSO | 84.15 | 84.11 | 84.12 | **84.13 ± 0.02** |
| Maxwell-PINN (gate) | LOSO | 84.08 | 84.02 | 84.02 | 84.04 ± 0.03 |
| Maxwell-PINN (nogate) | LOSO | 84.05 | 84.01 | 84.04 | 84.03 ± 0.03 |
| EAN Baseline | LORO | 84.25 | 84.07 | 84.25 | 84.19 ± 0.08 |
| Maxwell-PINN (gate) | LORO | 84.19 | 84.38 | 84.13 | **84.23 ± 0.11** |
| Maxwell-PINN (nogate) | LORO | 84.12 | 84.25 | 84.13 | 84.17 ± 0.09 |

**Takeaway:** Under abundant labeled real data, LOSO/LORO macro-F1 stays comparable to the EAN baseline; the physics-informed pipeline’s clearest gains in the article are under **label-scarce STEA** and **calibration / stress-test** settings.

### Sim2Real Transfer Efficiency Assessment (STEA)

Aggregated over **five seeds** (see per-seed files under `results/article3_results_gpu/stea_ablation/`). Values are **mean ± std** of macro-F1 (%) and ECE (%).

| Label ratio | EAN macro-F1 (%) | Maxwell-PINN macro-F1 (%) | EAN ECE (%) | M-PINN ECE (%) |
|-------------|------------------|---------------------------|-------------|----------------|
| **1%** | 8.96 ± 0.54 | 8.53 ± 0.97 | 70.0 ± 0.9 | 55.3 ± 9.2 |
| **5%** | 30.00 ± 2.04 | 40.84 ± 3.13 | 14.8 ± 1.3 | 7.9 ± 2.1 |
| **10%** | 47.32 ± 4.25 | 61.23 ± 6.21 | 6.1 ± 1.7 | 10.2 ± 4.8 |
| **20%** | 68.14 ± 1.88 | 77.81 ± 3.52 | 10.9 ± 2.6 | 5.9 ± 1.4 |
| **50%** | 94.00 ± 1.17 | 93.45 ± 3.94 | 7.6 ± 0.8 | 3.9 ± 0.9 |
| **100%** | 98.80 ± 0.38 | 98.65 ± 0.56 | 1.0 ± 0.2 | 1.5 ± 0.7 |

Source: `results/article3_results_gpu/stea_ablation/stea_results_ean.json`, `stea_results_maxwell.json`.

### Ablation study (synthetic, single-seed snapshots)

The `results/article3_results_gpu/maxwell_ablation/*.json` files record single-run metrics on synthetic data (macro-F1, raw/calibrated ECE, NLL, Brier, etc.). **Do not mix these seed-0 snapshots with multi-seed STEA aggregates without relabeling.** For the paper’s full ablation table (including residual-gated row with **ECE 2.84%**), follow the manuscript and the corresponding training logs.

### Synthetic Robustness Validation (SRV)

Condition-wise JSON summaries live under `results/article3_results_gpu/srv_lite/` (and the legacy `results/srv_lite/` tree). Macro-F1 values are protocol- and checkpoint-specific; inspect the JSON `metrics.macro_f1` fields for the exact numbers used in figures.

### Edge deployment benchmarks

JSON files under `results/article3_edge_deployment/` report **latency, throughput, and process-level memory** for the **compact** EAN vs Maxwell-PINN variants on Jetson Xavier AGX, Jetson Nano, Horizon RDK X5 (BPU and CPU paths), and Raspberry Pi 4. **Parameter counts here (≈204K / ≈256K) differ from the full research model** reported in the article’s architecture table; the README in the paper explains the two experimental tracks.

## Dataset

Experiments use the **SenseFi WiFi CSI Sensing Benchmark**:

- Source: [WiFi-CSI-Sensing-Benchmark](https://github.com/zhihaozhao/WiFi-CSI-Sensing-Benchmark)
- Typical tensor shape: **52 subcarriers × 32 time steps** (research track); deployment JSONs may use **128 × 52** windows as noted in each file.
- **8** activity classes where applicable.
- CDAE / baseline FPA3 runs use **files_per_activity = 3** unless a JSON `args` block states otherwise.

## Pretrained checkpoints

Checkpoints for the **original** `maxwell_pinn_real` pretrain sweep are under `results/maxwell_pinn_real/pretrain/` (e.g. `final_enhanced_gate_s{0,42,100}.pth`, `final_enhanced_nogate_s{0,42,100}.pth`). Large `.pth` / `.onnx` artifacts from the article-3 training tree are **not** duplicated under `article3_results_gpu/`; regenerate them from the article codebase if needed.

## JSON result format (quick reference)

- **CDAE:** `{loso,loro}_enhanced_seed{s}.json` — folds, aggregate macro-F1, ECE, etc.
- **STEA (per seed):** `stea_enhanced_ratio{r}_seed{s}.json` — zero-shot and fine-tuned metrics.
- **STEA (aggregated):** `stea_results_ean.json`, `stea_results_maxwell.json` — mean/std per label ratio.
- **Edge:** `*_results.json` — `test_config`, `system_info`, `results[]` per model variant.

## Reproducibility

```bash
# Original full pipeline (pretrain + CDAE + STEA; runtime depends on GPU)
python scripts/run_maxwell_pinn_real.py

# EAN baseline (files_per_activity = 3)
python scripts/run_fpa3_all.py

# Comparison tables
python scripts/compare_maxwell_pinn_real.py
```

Article-3 training and figure regeneration live in the separate **WiFi-CSI-Journal-Paper** / Maxwell-PINN codebase; this repo holds exported **results** for sharing and verification.

**Requirements (original scripts):** PyTorch ≥ 2.0, CUDA-capable GPU, SenseFi benchmark data paths as configured in your environment.

## Hardware

Reported training was performed on NVIDIA CUDA GPUs (Windows/Linux). Edge JSONs include per-board OS, PyTorch, and device metadata inside each file.

## License

Academic research use. Please cite the accompanying paper when using these results.

## Citation

```bibtex
@article{zhao2026maxwell_pinn_har,
  title={Physics-Informed Neural Network with Helmholtz Wave Equation Constraints for WiFi CSI-based Human Activity Recognition},
  author={Zhao, Zhihao and others},
  journal={to be updated},
  year={2026},
  note={Under review}
}
```
