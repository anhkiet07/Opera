# [ICML 2026] QuITE: Query-Based Irregular Time Series Embedding

[![Conference](https://img.shields.io/badge/ICML-2026-1f6feb?style=flat-square)](https://icml.cc/Conferences/2026)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation** of *QuITE: Query-Based Irregular Time Series Embedding* (ICML 2026) — 📄 [arXiv](https://arxiv.org/abs/2605.28166), 📑 [OpenReview](https://openreview.net/forum?id=ILQGHFvEoo).

A plug-and-play **input-embedding** module that lets any standard MTS backbone — PatchTST, PatchMixer, TMix, iTransformer, S-Mamba, TimeXer — handle **Irregular Multivariate Time Series (IMTS)** without architectural changes or artificial value generation.

---

## 🔥 News

- **2026-05-01** — *QuITE* is accepted by **ICML 2026**.

---

## Overview

Irregular Multivariate Time Series (IMTS) are common in healthcare, industrial monitoring, and climatology, yet they break the uniform-sampling assumption baked into standard MTS embeddings. Existing methods either redesign the architecture (giving up the rich MTS-backbone library) or interpolate IMTS onto a regular grid (introducing artificial values). QuITE attacks the problem at the **input-embedding** stage instead, getting the best of both:

**Table 1.** Comparison of approaches for IMTS modeling (paper Table 1).

| Approach                              | No Artificial Value | Model Flexibility |
| ------------------------------------- | :-----------------: | :---------------: |
| Architecture-based                    |         ✅          |        ❌         |
| Data-based                            |         ❌          |        ✅         |
| **Input-embedding-based (Ours)**      |         ✅          |        ✅         |

- **QuITE** *(paper §4)* — a plug-and-play module. A small set of learnable **query tokens** aggregates irregular observations through a single masked self-attention layer.
- **QuITE++** *(paper §5)* — a hierarchical extension: query-based patch embedding -> patch-level self-attention -> variable-level self-attention -> cross-attention decoder over future-time queries.

Across **7 benchmarks** and **6 MTS backbones**, plugging QuITE in yields average relative gains of **up to 54.7% in forecasting** and **up to 15.8% in classification**; QuITE++ achieves the **best performance on 20 of 24 forecasting settings** (paper Tables 2-4).

<p align="center">
  <img src="figs/forecasting.png" width="780"/>
  <br/>
  <em><b>Figure 1 (a).</b> Effectiveness of QuITE on forecasting — averaged over all datasets.</em>
</p>

<p align="center">
  <img src="figs/classification.png" width="780"/>
  <br/>
  <em><b>Figure 1 (b).</b> Effectiveness of QuITE on classification — averaged over all datasets.</em>
</p>

---

## Model

<p align="center">
  <img src="figs/QuITE.png" width="900"/>
  <br/>
  <em><b>Figure 2.</b> Overall framework of <b>QuITE</b>. Learnable query tokens aggregate irregular observations through a single self-attention layer.</em>
</p>

<p align="center">
  <img src="figs/model.jpg" width="900"/>
  <br/>
  <em><b>Figure 3.</b> Overall architecture of <b>QuITE++</b>. A hierarchical encoder models intra-variable patch-level (B) and inter-variable (C) interactions via learnable query tokens.</em>
</p>

---

## 📊 Datasets

We use **4 forecasting** + **3 classification** benchmarks, following [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN) for forecasting and [Raindrop](https://github.com/mims-harvard/Raindrop) for classification preprocessing. All data lives under `data/<dataset>/` inside this repository (override with the `QUITE_DATA_ROOT` environment variable).

### Forecasting (paper Appendix A.1)

| Dataset        | Samples | Variables | Avg. Length | Missing |
| -------------- | ------: | --------: | ----------: | ------: |
| Human Activity |   5,400 |        12 |         120 |  75.0 % |
| USHCN          |  26,736 |         5 |         163 |  77.9 % |
| PhysioNet      |  12,000 |        36 |          74 |  88.4 % |
| MIMIC-III      |  23,457 |        96 |          46 |  96.7 % |

### Classification (paper Appendix A.2)

| Dataset | Samples | Variables |    Classes | Missing |
| ------- | ------: | --------: | ---------: | ------: |
| P19     |  38,803 |        34 | 2 (binary) |  94.9 % |
| P12     |  11,988 |        36 | 2 (binary) |  88.4 % |
| PAM     |   5,333 |        17 | 8 classes  |  60.0 % |

### Acquisition

- **PhysioNet, Human Activity** — auto-downloaded by the code.
- **USHCN** — use `small_chunked_sporadic.csv` from [GRU-ODE-Bayes](https://github.com/edebrouwer/gru_ode_bayes).
- **MIMIC-III** — request raw data via [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (credentialed), then run the [Neural Flows preprocessing](https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/data_preproc/mimic_prep.ipynb).
- **P12, P19, PAM** — use the Raindrop-processed splits ([P19](https://doi.org/10.6084/m9.figshare.19514338.v1), [P12](https://doi.org/10.6084/m9.figshare.19514341.v1), [PAM](https://doi.org/10.6084/m9.figshare.19514347.v1)).

---

## ⚙️ Installation

Tested on **Python 3.10+** and **PyTorch 2.0+**.

```bash
git clone https://github.com/Meaningfull9502/QuITE.git
cd QuITE
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick your CUDA version
pip install -r requirements.txt
pip install mamba-ssm   # optional, only required for the S-Mamba backbone (Linux + CUDA)
```

PhysioNet and Human Activity are downloaded and preprocessed automatically into `data/physionet/` and `data/activity/` on first run.

### Weights & Biases logging

Every entrypoint logs to [W&B](https://wandb.ai) by default (project `quite`): per-epoch train loss, validation / test MSE-RMSE-MAE-MAPE (forecasting) or loss / accuracy / AUROC / AUPRC / F1 per fold (classification), epoch time, parameter count, and the final best-epoch test numbers in the run summary.

```bash
wandb login                       # once
python train_forecasting.py ... --wandb_project quite --wandb_name my-run
python train_forecasting.py ... --wandb_offline   # no network: later `wandb sync wandb/offline-run-*`
python train_forecasting.py ... --no_wandb        # disable entirely
```

---

## 🚀 Quick Start

```bash
# QuITE + iTransformer on PhysioNet (forecasting, 24 -> 24)
python train_forecasting.py --dataset physionet --history 24 \
    --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
    --batch_size 64 --lr 1e-3 --seed 1 --gpu 0 \
    --irr_emb --model itransformer --mode quite
```

```bash
# QuITE++ on the same setting
python train_quite_plus.py --dataset physionet --history 24 \
    --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 2 \
    --batch_size 64 --lr 1e-3 --seed 1 --gpu 0
```

```bash
# QuITE + PatchTST on P19 (classification)
python train_classification.py --dataset P19 \
    --patch_size 3.75 --stride 3.75 --hid_dim 64 --nhead 4 --nlayer 3 \
    --batch_size 64 --lr 1e-3 --epoch 1000 --gpu 0 \
    --irr_emb --model patchtst --mode quite
```

> Per-dataset hyperparameters follow paper Table C.1 and are encoded in `jobs/*.sh`. Run `python <script>.py --help` for the full argument list.

---

## 🧪 Sampling-Invariant QuITE (research extension, Phase 1)

`sampling_views.py` generates alternative *observation views* of the same trajectory by editing the observation mask (`random:p`, `bursty:p`, `variable:p`, `mixed`). `train_forecasting.py` and `train_classification.py` accept:

| Flag | Meaning |
| --- | --- |
| `--train_view mixed` | draw a fresh sampling view of every training batch (`none` = off) |
| `--n_views 2` | number of views per batch; the task loss is averaged over views |
| `--consistency_lambda 0.1` | weight of the cross-view consistency loss `D(z_A, z_B)` (`0` = augmentation only) |
| `--consistency_target repr\|pred` | apply consistency to the encoder representation or to the predictions |
| `--consistency_dist cos\|mse` | distance used for representation consistency |
| `--eval_views ...` | views used for the final cross-sampling robustness table on the test set |

Each run writes `results/<run_tag>.json` (test metrics, parameter count, time per epoch, peak GPU memory, inference cost, robustness table) and logs the same to W&B, including collapse diagnostics (`repr/pos_cos`, `repr/neg_cos`, `repr/std`).

```bash
bash jobs/run_phase01_all.sh          # Phase 0 reproduction + Phase 1 comparison & ablations
python summarize_results.py           # -> results/summary.md
```

---

## 🔁 Reproducing Paper Results

```bash
bash jobs/run_forecasting.sh      # Table 2  -- 6 backbones x 12 (dataset, horizon) x 5 seeds
bash jobs/run_quite_plus.sh       # Table 4  -- QuITE++ on 12 settings x 5 seeds
bash jobs/run_classification.sh   # Table 3  -- 6 backbones x 3 datasets
```

All runs use Adam, `lr = 1e-3`, **patience = 50**, **seeds {1, 2, 3, 4, 5}**, **MSE loss** (forecasting), **Cross-Entropy loss** (classification) — paper §6.1.

Supported backbones (selectable via `--model`): `patchtst`, `patchmixer`, `tmix`, `itransformer`, `s_mamba`, `timexer`. The per-backbone `--hid_dim`, `--nhead`, and `--nlayer` are configured automatically by the scripts in `jobs/`.

---

## Embedding Modes (`--mode`)

| `--mode`   | Meaning                                                       | Paper    | Use with `--irr_emb`? |
| ---------- | ------------------------------------------------------------- | -------- | :-------------------: |
| **`quite`**| QuITE -- query-based irregular embedding (paper main method)  | Eq. 5-13 |          ✅           |
| `mean`     | Mean Pooling baseline                                         | Table 5  |          ✅           |
| `mtand`    | mTAND attention baseline                                      | Table 5  |          ✅           |
| `add`      | value embedding + time embedding                              | Table 5  |          ❌           |
| `concat`   | value embedding concatenated with time embedding              | Table 5  |          ❌           |
| `False`    | vanilla backbone embedding (no time conditioning)             | --       |          ❌           |

---

## 📖 Citation

```bibtex
@inproceedings{lim2026quite,
  title         = {Qu{ITE}: Query-based Irregular Time-series Embedding},
  author        = {Lim, Junghoon},
  booktitle     = {Forty-third International Conference on Machine Learning},
  year          = {2026},
  url           = {https://openreview.net/forum?id=ILQGHFvEoo},
  eprint        = {2605.28166},
  archivePrefix = {arXiv}
}
```

---

## 🙏 Acknowledgements

Built on top of [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN), [Raindrop](https://github.com/mims-harvard/Raindrop), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba), [Hi-Patch](https://github.com/qianlima-lab/Hi-Patch), and [PyOmniTS](https://github.com/Ladbaby/PyOmniTS).
