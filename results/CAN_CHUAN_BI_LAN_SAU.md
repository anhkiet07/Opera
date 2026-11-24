# Còn thiếu gì để cập nhật kết quả mới (đối chiếu ngày 19/11/2026)

Nguồn đối chiếu: `results/tong_hop/per_run_merged.csv`, `results/sweep/`, `results/bao_cao/2026-11-18/excluded_runs.csv`.

## A. Chỉ cần EXPORT (đã chạy trên cluster, bị loại khỏi gói 18/11 vì "classification, out of scope")

| Gì | Run |
|---|---|
| Classification P12 / P19 / PAM × GRU-D, mTAND, Warpformer, iTransformer, PatchTST, TMix | 180 run (vanilla + opera, 5 seed) |
| Native shift `eicu_demo` | 30 run |

→ Export `runs/<run>/{results.json, config.yaml, train.log}`. Kiểm tra commit: nếu là commit cũ (`ce9a1cc`) thì đây chính là kết quả đang có trong `cu_ce9a1cc/`; nếu là `e606587` thì là bản mới, thay được §3.2. P19_hospB không có trong danh sách loại → kiểm tra có chạy chưa.

## B. Chạy SWEEP (đã có checkpoint, chỉ chưa sweep)

| Gì | Run |
|---|---|
| NOAA ISD × iTransformer, PatchTST, TMix | 30 run ("no severity sweep output") |

## C. Chạy SWEEP nhưng cần tìm checkpoint (máy export không có `model.pt`)

| Gì | Run |
|---|---|
| iTransformer, PatchTST, TMix × Activity 1s, Activity 2s, PhysioNet 12h, PhysioNet 36h, USHCN 6 tháng | 150 run |
| iTransformer, PatchTST, TMix × VitalDB | 30 run — **`model.pt` có sẵn ở `results/runs/bench_vitaldb_{itransformer,patchtst,tmix}_*`**, copy lên cluster là sweep được |

Nếu không tìm được checkpoint (commit `062d811` / `ce9a1cc`): train lại ở commit hiện tại rồi sweep, đồng thời thống nhất commit với các backbone khác.

## D. Chạy TRAIN (ô còn trống trên Hình 1.1), mỗi ô vanilla + opera × 5 seed, rồi sweep

| Backbone | Dataset còn thiếu |
|---|---|
| t-PatchGNN, GraFITi | NOAA ISD |
| DLinear | NOAA ISD, Activity 2s, CGMacros, PhysioNet 36h |
| CRU (chậm; `TONG_KET.md` ghi đã dừng) | NOAA, Activity 2s, CGMacros, PhysioNet 12h / 36h; VitalDB (thiếu cả bên OPERA) |

## E. Thêm SEED

| Gì | Seed cần thêm |
|---|---|
| SegRNN × NOAA ISD, Activity 2s, CGMacros, PhysioNet 36h | s3, s4 (đang có 3 seed) |
| t-PatchGNN × USHCN (tùy chọn: thua clean −4.2% chủ yếu do seed 0) | s5–s9 |

## F. Chạy LẠI (lỗi)

| Gì | Ghi chú |
|---|---|
| Warpformer vanilla: NOAA s4, Activity 2s s0, s2 (NaN khi train) | Chạy lại cả OPERA cùng seed vì teacher = checkpoint vanilla cùng seed |

## G. Thí nghiệm mới (cho các mục paper còn `[PLACEHOLDER]`)

| Gì | Vì sao |
|---|---|
| Thêm `identity` vào `train_processes` (SegRNN / DLinear trên VitalDB trước) | Kiểm giả thuyết chi phí clean của non-native (`TONG_KET.md` §4.1) và chỗ lạ của Warpformer |
| OPERA++ và ablation (`abl_mining`, `abl_opera++_scalar`, λ_d = 0 / V = 1) trên setting mới | Hiện chỉ có ở kết quả cũ; cần cho §6.2–6.5 |
| Classification với backbone mới (t-PatchGNN, GraFITi, DLinear, SegRNN) — tùy chọn | Hiện classification chỉ có 6 backbone cũ |

## Mỗi lần export cần kèm

1. `runs/<run>/{results.json, config.yaml, train.log}` cho mọi run mới.
2. `sweep/<run>.json` (đủ 20 điều kiện, đủ cặp vanilla / opera, khớp protocol với `results.json`) cho mọi run forecasting.
3. `slurm_logs/` của các job train và sweep.
4. `per_run.csv` tổng hợp (nếu có).

Lệnh sweep (trên cluster, ở repo gốc):

```bash
python scripts/severity_sweep.py --glob "results/raw/bench_*_<backbone>_*_s[0-9]" --out results/sweep
```
