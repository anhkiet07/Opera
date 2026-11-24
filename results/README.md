# results/: mọi kết quả chạy OPERA (mỗi thứ chỉ một bản)

| Thư mục / file | Nội dung |
|---|---|
| `tong_hop/` | **Bản gộp dùng cho phân tích** (sinh bởi `analysis/03_tong_ket.ipynb`). `per_run_merged.csv`: 778 run forecasting, MSE ở 5 điều kiện protocol, commit, cờ NaN, cột `src` = nguồn. `sweep_long.csv`: severity sweep, mỗi dòng = run × điều kiện (20 điều kiện) |
| `runs/<run>/` | Mỗi run một thư mục: `results.json`, `config.yaml`, `train.log` (một số run VitalDB của iTransformer / PatchTST / TMix có thêm `model.pt`). Tên run: `bench_<dataset>_<backbone>_<vanilla\|opera>_s<seed>` |
| `sweep/<run>.json` | Output của `scripts/severity_sweep.py`: MSE ở 20 điều kiện cho mỗi run có sweep |
| `slurm_logs/` | Log các job train và sweep (gộp từ mọi gói, không trùng) |
| `bao_cao/2026-11-18/` | Gói 18/11: `README.md`, `per_run.csv` (cột `arm` = method), `excluded_runs.csv` (run nào không export và vì sao) |
| `bao_cao/2026-11-15/` | Gói 15/11: `TONG_KET.md`, `BANG_SO_LIEU.md`, `per_run.csv` (có cờ `nan_during_training`), `summary_*.csv`, `code/` (code backbone mới + script sweep) |
| `bao_cao/2026-11-12_native/` | GRU-D / mTAND / Warpformer: `README.md`, `TONG_HOP.md`, `per_run.csv`, `summary_*.csv`, `nan_debug/` (lỗi NaN của Warpformer), `severity/` (`TOM_TAT_SEVERITY.md`, hình, script) |
| `bao_cao/plugin_vitaldb/`, `bao_cao/plugin_noaa_cgmacros/` | Bảng tóm tắt iTransformer / PatchTST / TMix trên VitalDB, NOAA, CGMacros (commit `062d811`) |
| `cu_ce9a1cc/` | Kết quả cũ (commit chủ yếu `ce9a1cc`): `per_seed_results.csv`, `forecast_*.csv`, `classify_*.csv`, `README.txt` (mô tả cột). Forecasting setting cũ, classification P12 / P19 / PAM, native shift, OPERA++ |
| `CAN_CHUAN_BI_LAN_SAU.md` | Những gì cần mang về từ cluster lần sau |

Quy ước dấu: trong các notebook, "OPERA hơn (%)" **dương = OPERA tốt hơn**. Các file `TONG_KET.md`, `TONG_HOP.md`, `README.md` trong `bao_cao/` dùng dấu ngược lại (Δ âm = OPERA tốt hơn).
