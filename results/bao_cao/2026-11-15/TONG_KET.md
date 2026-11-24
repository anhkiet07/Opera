# Tổng kết benchmark OPERA, 15/11/2026

Gói này gồm mọi run forecasting trên 8 bộ dữ liệu, với 11 backbone (6 native, 5 non-native). Mỗi cặp dataset × backbone có vanilla và OPERA, mỗi bên 5 seed. Toàn bộ bảng số nằm trong `BANG_SO_LIEU.md` và các file CSV.

- Chỉ số: MSE, càng thấp càng tốt.
- **Δ = (OPERA − vanilla) / vanilla.** Số âm nghĩa là OPERA tốt hơn.
- **Shift** = trung bình MSE trên 4 điều kiện: random_dropout 0.7, temporal_stride 3, variable_dropout 0.5, và stride 3 + variable_group_dropout 0.3.
- **Seed thắng** so từng cặp cùng seed. **p** là kết quả paired t-test trên 5 cặp. Với 5 seed và nhiều phép so sánh cùng lúc, p chỉ nên dùng để tham khảo.

## 1. Những gì đã chạy từ 13/11 đến 15/11

| Backbone | Loại | Cài thế nào | Bộ đã chạy |
|---|---|---|---|
| GraFITi (AAAI 2024) | native | Tự cài lại trong `irregular_baselines.py` | 7 bộ (không có NOAA) |
| t-PatchGNN (ICML 2024) | native | Tự cài lại, **bản đã sửa pooling** | 7 bộ (không có NOAA) |
| CRU (ICML 2022) | native | Tự cài lại | Activity h1000, USHCN. Đã dừng vì quá chậm |
| DLinear (AAAI 2023) | non-native | Qua QuITE (nhóm patch), file `token_backbones.py` | Activity h1000, USHCN, VitalDB, PhysioNet h12 |
| SegRNN (2023) | non-native | Qua QuITE (nhóm patch) | Activity h1000, USHCN, VitalDB, PhysioNet h12 |

grud, mtand, warpformer (native) và iTransformer, PatchTST, TMix (non-native) đã có từ trước. Các backbone này cũng có trong bảng số.

**Các sửa lỗi trong đợt này:**
- **t-PatchGNN:** bản đầu gộp patch bằng trung bình *chỉ trên các patch có quan sát*. Vì vậy, khi một biến bị mất hẳn, biểu diễn của nó thành vector 0 và model không dùng được các biến khác. Sau khi sửa để gộp mọi patch, VitalDB với variable dropout 0.5 chuyển từ −3% thành −83%. Kết quả bản cũ được để riêng ở `results/raw_superseded/` và không có trong gói này.
- **CRU:** khởi tạo ma trận chuyển trạng thái quanh −I để không bị nổ, và bỏ phép nhân thời gian với độ dài chuỗi.
- **GraFITi + OPERA trên PhysioNet h36:** bị hết bộ nhớ GPU khi chạy song song với t-PatchGNN, nên đã chạy lại riêng (job #37).

## 2. Kết quả chính

**Khi gặp shift: OPERA thắng ở mọi backbone.** Trung vị Δ shift nằm trong khoảng −22% đến −79%. Trên gần như toàn bộ seed, OPERA tốt hơn vanilla. Các bộ có lợi lớn nhất là VitalDB, NOAA và Activity. USHCN gần như hòa với mọi backbone.

**Trên clean: hai nhóm backbone khác hẳn nhau.**

| Nhóm | Vanilla kém đi khi gặp shift (trung vị) | OPERA thay đổi clean (trung vị) |
|---|---|---|
| Native (grud, mtand, warpformer, GraFITi, t-PatchGNN, CRU) | khoảng 0.4–1.4× | khoảng −2% đến 0% (riêng CRU +2%) |
| Non-native qua QuITE (iTransformer, PatchTST, TMix, DLinear, SegRNN) | khoảng 5–7.5× | khoảng +2.5% đến +6.3% |

Mức nhạy của bản vanilla tương quan với mức OPERA làm kém đi trên clean (Spearman ρ ≈ 0.55, p ≈ 10⁻⁶; xem Bảng A).

**Cách giải thích (giả thuyết, chưa kiểm chứng):**
- Backbone native được thiết kế cho dữ liệu thiếu. Với chúng, một chuỗi bị bỏ bớt điểm vẫn là "cùng loại input", nên train trên phiên bản bị hỏng vẫn có ích cho clean.
- Backbone non-native chỉ thấy token do QuITE tạo ra. Khi bỏ bớt điểm, các token này thay đổi mạnh. Trong khi đó, lúc train OPERA, student **không bao giờ thấy phiên bản sạch** (pool không có `identity`), nên model chuyên môn hóa cho dữ liệu bị hỏng và kém đi vài phần trăm trên clean.

## 3. Nên viết thế nào trong paper

- **Được claim:** OPERA bền với observation-process shift trên 11 backbone thuộc nhiều họ thiết kế, và trên 8 bộ dữ liệu. Với backbone native, OPERA **giữ nguyên** độ chính xác trên clean.
- **Phải nói rõ:** với backbone non-native qua QuITE, OPERA đổi khoảng 2–6% trên clean lấy mức lợi rất lớn khi gặp shift.
- **Không nên claim:** OPERA tốt hơn trên clean nói chung.

## 4. Việc còn mở

1. **Kiểm tra giả thuyết ở mục 2:** thêm phiên bản sạch (`identity`) vào pool khi train OPERA, chạy với SegRNN và DLinear trên VitalDB. Nếu mức kém đi trên clean biến mất, đó là cấu hình đề xuất cho mọi backbone.
2. **Sửa lỗi NaN của warpformer** (3 run vanilla: NOAA s4, Activity h2000 s0 và s2). Hiện các run này vẫn được tính, và đánh dấu `*` trong bảng.
3. **t-PatchGNN trên USHCN:** kém đi +4.3% trên clean, chủ yếu do 2 seed lệch (p = 0.19). Có thể chạy thêm seed để kiểm tra.
4. Chưa chạy: NOAA với GraFITi và t-PatchGNN (đã bỏ theo quyết định), CRU trên VitalDB (đã dừng).

## 5. Danh sách file

| File | Nội dung |
|---|---|
| `TONG_KET.md` | File này |
| `BANG_SO_LIEU.md` | Bảng A (theo backbone), B (Δ clean), C (Δ shift) |
| `summary_by_backbone.csv` | Mỗi cặp dataset × backbone: mean, std, Δ, seed thắng, p, mức nhạy của vanilla |
| `summary_per_backbone_aggregate.csv` | Gộp theo backbone |
| `summary_by_condition.csv` | Δ MSE theo từng điều kiện test |
| `per_run.csv` | Mỗi run một dòng, có cờ `nan_during_training` |
| `runs/<run>/` | `results.json`, `config.yaml`, `train.log` (không có `model.pt`) |
| `slurm_logs/` | Log của các job #20–#37 |
| `code/` | Code các backbone mới và script sweep, vẽ hình |
