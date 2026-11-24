# Tổng hợp kết quả benchmark Opera: native backbones (10–11/11/2026)

8 bộ dữ liệu forecasting × 3 native backbone (grud, mtand, warpformer) × 2 phương pháp (vanilla, OPERA) × 5 seed = **240 run**. Tất cả đã chạy xong, không có stage nào failed.

- Chỉ số: MSE (càng thấp càng tốt).
- **Δ% = (OPERA − vanilla) / vanilla.** Số âm nghĩa là OPERA tốt hơn.
- **Shift MSE** = trung bình MSE trên 4 kiểu shift: random_dropout 0.7, temporal_stride 3, variable_dropout 0.5, composed (temporal_stride 3 + variable_group_dropout 0.3).
- **Độ suy giảm** = (shift MSE − clean MSE) / clean MSE.

## 1. Các job đã chạy

| Job | Dataset | Thời gian | Kết quả |
|---|---|---|---|
| #7 | CGMacros | 10/11 15:23 → 17:04 | 30/30 |
| #8 | NOAA ISD | 10/11 17:49 → 11/11 02:27 | 30/30 (1 run NaN) |
| #9 | USHCN 6 tháng | 11/11 02:30 → 03:14 | 30/30 |
| #10 | PhysioNet h12 | 11/11 03:37 → 08:13 | 30/30 |
| #11 | PhysioNet h36 | 11/11 08:40 → 15:31 | 30/30 |
| #12 | Tải dữ liệu VitalDB | xong 11/11 14:19 | 0 failed |
| #13 | VitalDB | 11/11 15:41 → 18:44 | 30/30 |
| #14 | Activity h2000 | 11/11 ~21:24 → 22:28 | 30/30 (2 run NaN) |
| #15 | Activity h1000 | 11/11 22:28 → 23:00 | 30/30 |

## 2. OPERA so với vanilla: Δ shift MSE

| Dataset | grud | mtand | warpformer | Δ clean (khoảng) |
|---|---|---|---|---|
| NOAA ISD | **−77.0%** | **−79.3%** | −52.9%* | −6.4% … +1.6% |
| Activity h2000 | **−72.4%** | **−57.6%** | −28.1%* | −1.5% … +0.4% |
| VitalDB | **−60.6%** | −38.3% | −44.3% | −5.7% … +0.4% |
| Activity h1000 | −49.6% | **−55.4%** | −30.5% | −3.8% … −0.4% |
| CGMacros | −25.3% | −12.4% | −21.0% | −2.4% … +1.2% |
| PhysioNet h36 | −24.4% | −20.9% | −6.1% | −0.4% … +0.1% |
| PhysioNet h12 | −22.8% | −14.8% | −5.2% | −2.6% … −1.0% |
| USHCN 6mo | −18.3% | −2.7% | −1.1% | −0.3% … +0.4% |

OPERA giảm shift MSE ở **24/24** cặp dataset × backbone. So theo từng seed thì thắng **119/120** cặp; cặp duy nhất thua là PhysioNet h36 với warpformer (4/5).
\* Vanilla có run bị NaN, xem mục 5.

Bảng đầy đủ có std nằm trong `summary_by_backbone.csv` và `README.md`.

## 3. Theo loại shift (gộp 3 backbone)

| Dataset | random_dropout 0.7 | temporal_stride 3 | variable_dropout 0.5 | composed | clean |
|---|---|---|---|---|---|
| NOAA ISD | −63.9% | −10.2% | −77.0% | −72.5% | −4.3% |
| VitalDB | −49.8% | −7.4% | −66.5% | −42.9% | −2.8% |
| Activity h2000 | −40.6% | −4.7% | −71.0% | −53.8% | −0.8% |
| Activity h1000 | −42.2% | −18.9% | −56.8% | −41.6% | −2.3% |
| CGMacros | −26.0% | −5.5% | −28.5% | −15.6% | −1.3% |
| PhysioNet h36 | −11.6% | −3.6% | −29.5% | −15.1% | −0.2% |
| PhysioNet h12 | −17.6% | −5.9% | −20.4% | −11.0% | −1.6% |
| USHCN 6mo | −10.3% | −9.9% | −3.7% | −8.0% | +0.1% |

## 4. Nhận xét

- **Lợi ích lớn nhất khi mất biến hoặc mất điểm đo** (variable_dropout, random_dropout). Với temporal_stride, OPERA chỉ cải thiện khoảng 4–10%; riêng Activity h1000 được −19%.
- **NOAA, VitalDB và Activity cải thiện mạnh nhất.** Ở NOAA, vanilla grud/mtand có lỗi tăng hơn 10 lần khi gặp shift; OPERA kéo xuống khoảng 1.5 lần.
- **grud được lợi nhiều nhất.** warpformer vốn đã bền với shift nên lợi ít, nhất là trên PhysioNet và USHCN.
- **USHCN với mtand/warpformer gần như hòa**: mức chênh nhỏ hơn độ lệch giữa các seed.
- **Không đánh đổi clean MSE**: thay đổi nằm trong khoảng −6.4% đến +1.6% (chưa loại các run NaN).

## 5. Vấn đề: warpformer bị NaN

**NaN** ("Not a Number") là giá trị máy tính trả về khi phép tính không ra số hợp lệ, ví dụ ∞ − ∞. Trong training, chỉ cần một bước ra NaN là nó lan sang toàn bộ trọng số và model hỏng hẳn.

**Run bị ảnh hưởng (đều là vanilla warpformer):** NOAA ISD s4, Activity h2000 s0 và s2.

**Nguyên nhân** (đã tái hiện trên Activity h2000 warpformer vanilla s0, log trong `nan_debug/`):
1. Bước warping trong `src/models/irregular_baselines.py` tính `gamma = cumsum(softplus(score)) / gmax`.
2. Phép chia cho `gmax` làm loss không phụ thuộc vào mức chung của `score`, nên không có gì giữ `score` lại. Adam đẩy dần `score` về phía âm: −12 (step 500), −47 (step 1000), −91 (step 1125).
3. `softplus(−91) ≈ 3e-40` thấp hơn ngưỡng float32 (khoảng 1.2e-38), nên `gmax ≈ 7e-37`. Gradient của phép chia tỉ lệ với 1/gmax² ≈ 1e72, tràn float32 thành ∞ rồi NaN.
4. Grad norm ngay trước bước hỏng vẫn rất nhỏ (0.06–0.14), nên **gradient clipping không chữa được**.

**Ảnh hưởng:** job không crash vì early stopping giữ checkpoint tốt nhất trước khi hỏng, nhưng các run này dừng học sớm (khoảng epoch 18 thay vì 35–40).

**Nếu loại các run NaN** (`summary_by_backbone_excl_nan.csv`):

| Dataset / warpformer | Δ clean | Δ shift | Seed thắng |
|---|---|---|---|
| NOAA ISD (còn 4 vanilla) | −6.4% → **−0.2%** | −52.9% → **−43.9%** | 4/4 |
| Activity h2000 (còn 3 vanilla) | −1.5% → **+10.5%** | −28.1% → **−29.5%** | 3/3 |

Ở Activity h2000, hai run NaN có clean MSE kém hơn hẳn (0.0117 và 0.0113, so với khoảng 0.0089 ở các seed còn lại). Khi loại chúng ra, **OPERA kém vanilla khoảng 10% trên dữ liệu sạch** với warpformer ở bộ này. Mức lợi khi gặp shift vẫn giữ nguyên. Con số này chỉ dựa trên 3 seed vanilla, nên cần chạy lại để xác nhận.

**Cách sửa (chưa áp dụng):**
- A (khuyên dùng): thay `softplus` + chia `gmax` bằng softmax có mask theo thời gian. Softmax đã chuẩn hóa sẵn và ổn định số học.
- B (nhanh): đặt sàn cho mẫu số, ví dụ `softplus(s) + 1e-4` hoặc clamp `gmax ≥ 1e-6`.

Sửa model sẽ làm đổi warpformer trên mọi dataset, nên để công bằng cần chạy lại toàn bộ warpformer (cả vanilla lẫn OPERA).

## 6. Việc còn lại

1. Sửa lỗi NaN của warpformer, rồi chạy lại phần warpformer.
2. GlucoBench: chưa có run nào (dữ liệu đã tải xong).
3. VitalDB với 3 backbone mặc định (itransformer, patchtst, tmix).

## 7. Robustness curve: MSE theo mức độ shift (bổ sung 12/11)

Chi tiết nằm trong `severity/TOM_TAT_SEVERITY.md`, hình nằm trong `severity/figures/`. Các model trên được đánh giá lại ở nhiều mức drop, không train lại. Số dưới đây là mức giảm MSE của OPERA so với vanilla ở mức nặng nhất, trung bình 3 backbone; **số dương nghĩa là OPERA tốt hơn**.

| Dataset | Random 0.7 | Stride 4 | Variable 0.5 | Composed (stride 3 + nhóm 0.5) |
|---|---|---|---|---|
| NOAA ISD | 39% | 8% | 72% | 73% |
| VitalDB | 39% | 9% | 64% | 47% |
| Activity h2000 | 30% | 7% | 62% | 53% |
| Activity h1000 | 40% | 20% | 52% | 48% |
| CGMacros | 22% | 7% | 28% | 24% |
| PhysioNet h36 | 12% | 4% | 28% | 23% |
| PhysioNet h12 | 17% | 7% | 20% | 14% |
| USHCN 6mo | 9% | 9% | 4% | 7% |

Ở cả 4 loại shift, drop càng nhiều thì vanilla thua càng nặng, còn trên dữ liệu sạch hai bên gần như bằng nhau.

## 8. Danh sách file

| File | Nội dung |
|---|---|
| `TONG_HOP.md` | File này |
| `README.md` | Bảng đầy đủ bằng tiếng Anh |
| `summary_by_backbone.csv` | Mean, std, Δ, độ suy giảm, seed thắng, run NaN |
| `summary_by_backbone_excl_nan.csv` | Như trên nhưng đã loại các run NaN |
| `summary_by_condition.csv` | Δ MSE theo từng loại shift |
| `per_run.csv` | Mỗi run một dòng, có cờ `nan_during_training` |
| `runs/<run>/` | `results.json`, `config.yaml`, `train.log` (không có `model.pt`) |
| `slurm_logs/` | Log của job #7–#15 và log tải dữ liệu |
| `nan_debug/` | Script và log tái hiện lỗi NaN |
| `severity/` | Robustness curve: hình, CSV, kết quả thô, script, log (xem `severity/TOM_TAT_SEVERITY.md`) |
