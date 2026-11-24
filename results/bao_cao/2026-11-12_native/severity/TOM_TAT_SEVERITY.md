# Robustness curve: MSE theo mức độ shift (12/11/2026)

## Cách làm

- **Không train lại.** Lấy 240 model đã train (8 dataset × 3 backbone × vanilla/OPERA × 5 seed) và đánh giá lại trên tập test, mỗi lần làm hỏng dữ liệu ở một mức khác nhau.
- **Seed cố định.** Việc làm hỏng dữ liệu dùng seed 12345, nên vanilla và OPERA được test trên cùng một input. Các điều kiện protocol cũ được chạy lại để kiểm tra, và khớp với `results.json` ở cả 240 run.
- **Hai job SLURM:** #18 cho 3 loại shift đầu, #19 cho loại composed. Cả hai đều 240/240 run ok, không có lỗi.
- **Cách đọc số:** "Mức giảm MSE" = (1 − MSE_OPERA / MSE_vanilla) × 100%. **Số dương nghĩa là OPERA tốt hơn.** (Chiều dấu ngược với Δ% trong `TONG_HOP.md`.)

| # | Loại shift | Các mức test | Mức OPERA đã thấy khi train |
|---|---|---|---|
| 1 | Random dropout: bỏ ngẫu nhiên từng điểm đo | 0 → 0.1 → … → 0.7 | ≤ 0.4 |
| 2 | Temporal stride: chỉ giữ 1 trong mỗi k mốc thời gian | k = 1, 2, 3, 4 | k = 2 |
| 3 | Variable dropout: mất hẳn từng biến | 0, 0.2, 0.3, 0.4, 0.5 | 0.2 |
| 4 | Composed: stride 3 cộng mất cả nhóm biến (4 nhóm liền nhau) | nhóm 0 → 0.1 → … → 0.5 | chưa thấy kiểu này |

Mức 0 của loại 4 là stride 3 đơn thuần, không phải dữ liệu sạch. Mức 0.3 của loại 4 chính là điều kiện composed trong protocol.

## Kết quả: mức giảm MSE của OPERA so với vanilla (%, trung bình 3 backbone)

### 1. Random dropout

| Dataset | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| Activity h1000 | 2 | 7 | 11 | 15 | 16 | 22 | 31 | **40** |
| NOAA ISD | 2 | 11 | 19 | 25 | 30 | 33 | 37 | **39** |
| VitalDB | 2 | 10 | 17 | 25 | 30 | 34 | 37 | **39** |
| Activity h2000 | 0 | 7 | 12 | 16 | 19 | 21 | 24 | 30 |
| CGMacros | 1 | 8 | 13 | 17 | 17 | 20 | 21 | 22 |
| PhysioNet h12 | 2 | 3 | 6 | 7 | 9 | 11 | 15 | 17 |
| PhysioNet h36 | 0 | 2 | 3 | 5 | 6 | 7 | 10 | 12 |
| USHCN 6mo | 0 | 0 | 0 | 1 | 2 | 4 | 7 | 9 |

### 2. Temporal stride

| Dataset | 1 (sạch) | 2 | 3 | 4 |
|---|---|---|---|---|
| Activity h1000 | 2 | 6 | 14 | **20** |
| USHCN 6mo | 0 | 4 | 8 | 9 |
| VitalDB | 2 | 4 | 6 | 9 |
| NOAA ISD | 2 | 3 | 7 | 8 |
| Activity h2000 | 0 | 2 | 4 | 7 |
| CGMacros | 1 | 4 | 5 | 7 |
| PhysioNet h12 | 2 | 4 | 6 | 7 |
| PhysioNet h36 | 0 | 2 | 4 | 4 |

### 3. Variable dropout

| Dataset | 0 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| NOAA ISD | 2 | 66 | 71 | 69 | **72** |
| VitalDB | 2 | 43 | 49 | 59 | **64** |
| Activity h2000 | 0 | 51 | 58 | 62 | **62** |
| Activity h1000 | 2 | 38 | 46 | 50 | 52 |
| CGMacros | 1 | 14 | 18 | 22 | 28 |
| PhysioNet h36 | 0 | 21 | 22 | 25 | 28 |
| PhysioNet h12 | 2 | 14 | 15 | 18 | 20 |
| USHCN 6mo | 0 | 2 | 2 | 3 | 4 |

### 4. Composed (stride 3 + mất nhóm biến)

| Dataset | 0 (stride 3) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| NOAA ISD | 7 | 60 | 63 | 71 | 71 | **73** |
| Activity h2000 | 4 | 37 | 42 | 48 | 50 | **53** |
| Activity h1000 | 14 | 32 | 37 | 43 | 45 | **48** |
| VitalDB | 6 | 30 | 34 | 39 | 41 | **47** |
| CGMacros | 5 | 12 | 13 | 16 | 18 | 24 |
| PhysioNet h36 | 4 | 9 | 11 | 15 | 19 | 23 |
| PhysioNet h12 | 6 | 9 | 9 | 11 | 12 | 14 |
| USHCN 6mo | 8 | 8 | 7 | 7 | 7 | 7 |

### Theo backbone, ở mức nặng nhất của mỗi loại (%)

| Dataset | Backbone | Random 0.7 | Stride 4 | Variable 0.5 | Composed 0.5 |
|---|---|---|---|---|---|
| Activity h1000 | grud | 30 | 22 | 66 | 56 |
| | mtand | 43 | 1 | 70 | 61 |
| | warpformer | 47 | 36 | 19 | 27 |
| Activity h2000 | grud | 25 | 9 | 84 | 80 |
| | mtand | 13 | 0 | 73 | 68 |
| | warpformer | 53 | 11 | 30 | 11 |
| CGMacros | grud | 24 | 11 | 38 | 32 |
| | mtand | 2 | 2 | 23 | 20 |
| | warpformer | 39 | 7 | 23 | 18 |
| NOAA ISD | grud | 40 | 5 | 83 | 80 |
| | mtand | 3 | 2 | 84 | 84 |
| | warpformer | 75 | 16 | 48 | 56 |
| PhysioNet h12 | grud | 26 | 15 | 30 | 22 |
| | mtand | 18 | 2 | 23 | 17 |
| | warpformer | 8 | 4 | 7 | 3 |
| PhysioNet h36 | grud | 15 | 9 | 39 | 34 |
| | mtand | 13 | 0 | 38 | 25 |
| | warpformer | 8 | 3 | 7 | 10 |
| USHCN 6mo | grud | 23 | 28 | 4 | 15 |
| | mtand | 2 | 0 | 5 | 3 |
| | warpformer | 1 | 0 | 2 | 3 |
| VitalDB | grud | 53 | 6 | 76 | 63 |
| | mtand | 2 | 2 | 66 | 36 |
| | warpformer | 62 | 18 | 49 | 43 |

## Nhận xét

1. **Drop càng nhiều thì vanilla thua OPERA càng nặng.** Ở cả 4 loại shift và 8 dataset, các đường cong đều đi lên. Trên dữ liệu sạch, OPERA không kém đi: chênh lệch nằm trong khoảng từ OPERA tốt hơn 6.4% đến OPERA kém hơn 1.6%.
2. **Mất biến là nơi OPERA mạnh nhất.** Đó là variable dropout và composed: 47–73% trên NOAA, VitalDB và hai bộ Activity. Random dropout xếp sau. **Temporal stride là loại yếu nhất** (4–9% ở stride 4), trừ Activity h1000 được 20%.
3. **Composed là kiểu shift OPERA chưa từng thấy khi train**, nhưng đường cong vẫn tăng đều. Điều này ủng hộ claim OPERA tổng quát hóa được sang shift mới. Bước nhảy lớn nhất nằm giữa mức 0 và 0.1, vì mất trọn một nhóm biến là sự cố nặng với vanilla.
4. **USHCN gần như không đổi** ở mọi loại shift. Bộ này chỉ có 5 biến.
5. **Hiệu quả khác nhau theo backbone:**
   - grud được lợi nhiều nhất.
   - mtand gần như hòa với random dropout trên NOAA, VitalDB và CGMacros (2–3%), nhưng giảm rất mạnh với variable dropout.
   - warpformer ngược lại: mạnh ở random dropout, yếu ở variable dropout và composed.

## Lưu ý trước khi đưa vào paper

- **warpformer + OPERA trên NOAA:** MSE *giảm* khi tăng random dropout (0.0032 → 0.0023). Hiện tượng này lạ, cần kiểm tra thêm.
- **Run vanilla warpformer bị NaN khi train** (NOAA s4, Activity h2000 s0 và s2) vẫn được tính trong trung bình. Các run này làm dải std rộng và có thể làm mức lợi của OPERA với warpformer trên 2 bộ này cao hơn thực tế. Xem mục 5 trong `TONG_HOP.md`.

## Files

| File | Nội dung |
|---|---|
| `figures/severity_<dataset>.{png,pdf}` | Curve MSE của vanilla (cam) và OPERA (xanh). Hàng là backbone, cột là loại shift, dải mờ là ±std qua 5 seed, vùng xám là các mức OPERA đã thấy khi train. |
| `figures/severity_gap.{png,pdf}` | Mức giảm MSE theo mức độ shift, mỗi dataset một đường, trung bình 3 backbone. |
| `severity_summary.csv` | Mọi điểm trên hình: MSE vanilla và OPERA (mean, std), mức giảm, cờ đã thấy khi train. |
| `reduction_by_dataset_mean_over_backbones.csv` | Các bảng ở trên, dạng CSV. |
| `sweep_json/` | Kết quả thô: mỗi run một file, 20 điều kiện test, mọi metric. |
| `scripts/` | `severity_sweep.py` (chạy đánh giá), `plot_severity.py` (vẽ hình), `reevaluate.py` (load model). |
| `slurm_logs/` | Log của job #18 và #19. |

Cách chạy lại (từ thư mục gốc repo):

```bash
python scripts/severity_sweep.py --glob 'results/raw/bench_<dataset>_<backbone>_*_s[0-9]'   # ghi vào results/sweep/
python scripts/plot_severity.py                                                          # ghi vào figures/severity/
```
