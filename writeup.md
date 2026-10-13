# Lab 25 — GPU FinOps: Bài viết ngắn

**Học viên:** Lê Minh Hiếu — 2A202602848
**Số liệu:** sinh từ `python missions/run_all.py` (seed = 25, tháng 30 ngày). Chi tiết đầy đủ trong `outputs/report.md`.
**Kiểm tra:** `verify.py` 11/11 · `pytest` 19 passed (15 test gốc + 4 test mới trong `tests/test_extensions.py`).

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Giảm |
|---|---|---|---|
| Tổng chi phí/tháng (GPU fleet + API) | $27,133 | $14,660 | **$12,473 (46%)** |
| $/1M-token — chỉ inference API (M2) | $6.488 | $1.275 | -80% |
| $/1M-token — all-in (toàn bộ chi phí ÷ 226M token/tháng) | $120.06 | $64.87 | -46% |

Đạt mục tiêu cắt ≥40%. Dòng all-in là con số "blended": chia toàn bộ hóa đơn GPU (gồm cả training) cho token phục vụ, nên cao hơn nhiều so với giá API.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | $/tháng | Tỷ trọng |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | 80% |
| Inference (cascade/cache/batch) | $1,178 | 9% |
| Right-size util-lies | $655 | 5% |
| Kill idle GPUs | $600 | 5% |

**Purchasing thắng áp đảo** vì phần lớn tiền nằm ở GPU-giờ của fleet ($25,667/tháng on-demand), không phải ở API ($1,466/tháng). 5 job interruptible chuyển sang spot (rẻ hơn ~37% kể cả overhead checkpoint và rework, riêng A10G sandbox ~58%), 3 job inference chạy 18–24h/ngày vượt break-even 55% nên dùng reserved 3 năm. Inference giảm tới 80% `$/1M-token` nhưng trên một nền chi phí nhỏ, nên chỉ đóng góp 9% tổng tiền tiết kiệm. Bài học: % giảm cao chưa chắc là đòn bẩy lớn nhất. Phải nhân với quy mô chi tiêu.

## 3. GPU-Util Lie

`gpu-h100-4`: GPU-Util 98% nhưng MFU 0.19 và MBU 0.21. Các H100 khác cùng job training đạt MFU 0.40–0.43, nên mỗi FLOP hữu ích trên GPU này đắt **2.2×**. `nvidia-smi` chỉ đo "có ít nhất một kernel đang chạy", không đo SM/tensor core có được dùng hết hay không. Cả MFU lẫn MBU đều thấp nghĩa là GPU không bị giới hạn bởi compute hay băng thông mà đang **bị bỏ đói**: dataloader chậm, đồng bộ host–device, overhead launch kernel nhỏ, chờ NCCL. `gpu-a10g-1` (inference, util 97%, MBU 0.30) cũng bị gắn cờ, nhưng decode là memory-bound nên phải xét MBU. Ở đây MBU thấp là do batch nhỏ. Tác động tài chính: hạ mỗi GPU một bậc tiết kiệm $655/tháng, nhưng nên profile và sửa pipeline trước, vì có thể lấy lại throughput mà không phải đổi GPU.

## 4. Phần mở rộng đã làm

Kết quả trước khi sửa được lưu ở `outputs/before/` (M2, M5, report cũ).

**Ext 3 — `cache_is_worth_it()`** (`finops/pricing.py`, áp dụng trong M2). CSV không có prefix id nên tôi **giả định**: mỗi (team, tier) có một cache entry cho mỗi cửa sổ TTL 5 phút, và request đầu tiên trong cửa sổ trả phí ghi. Phí ghi = 0.25 × giá input (kiểu Anthropic) + $1/1M-tok/giờ lưu trữ (kiểu Gemini). Ngưỡng hòa vốn là 0.31 lần đọc/ghi cho model lớn và 0.74 cho model nhỏ (với model nhỏ, phí lưu trữ chiếm phần lớn).

| Chế độ cache | $/ngày | $/1M-token |
|---|---|---|
| Không cache | 9.66 | 1.282 |
| Trước: coi ghi miễn phí (lab gốc) | 8.48 | 1.126 |
| Cache tất cả, có trả phí ghi | 9.67 | 1.284 |
| **Sau: chỉ cache khi đáng** | **9.61** | **1.275** |

Insight: lab gốc **phóng đại** lợi ích cache ($1.18/ngày so với $0.05/ngày thực tế). Cả 4 nhóm model lớn chỉ có 0.18–0.25 lần đọc/ghi, dưới ngưỡng, nên cache chúng là lỗ. Bật cache cho tất cả còn tệ hơn không cache. Chính sách mới chỉ bật cache cho assistant/rag/search trên model nhỏ. Test kiểm tra cả trường hợp ít đọc (0 và 0.5 → không cache) và nhiều đọc (3 và 10 → cache). Muốn cache có lợi hơn, cần TTL dài hơn hoặc gom request theo prefix.

**Ext 4 — Ngân sách Reasoning** (M2 + M5, dùng `wh_per_query()`).

| Nhóm | Request | $ | Wh |
|---|---|---|---|
| Reasoning | 8% | 15% | **94%** |
| Non-reasoning | 92% | 85% | 6% |

Reasoning hiện là **8.4% traffic**, nên cap 10% **không đổi gì: tiết kiệm $0**. Quy tắc đề xuất: *chỉ cho reasoning khi cascade đã chuyển request sang model lớn* (router đánh giá là khó). Quy tắc này hạ cấp 162 request. Giả định: request bị hạ cấp có output giảm 6× (bỏ "reasoning tax" trong `data/generate.py`) và giữ nguyên model. Kết quả: chỉ tiết kiệm $3.16/tháng nhưng giảm **717 kWh/tháng (-75% Wh)**. Reasoning là đòn bẩy carbon nhiều hơn là đòn bẩy tiền, nên không cộng vào tổng $.

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên)

1. **Đổi chiến lược mua ngay** (spot + checkpoint cho training, reserved 3 năm cho inference 24/7): $10,040/tháng, không cần sửa code.
2. **Tuần này: tắt GPU idle và profile `gpu-h100-4`**: tự động tắt `gpu-h100-5` ban đêm ($600/tháng, gần như không tốn công). Thêm cảnh báo MFU/MBU vào dashboard, thay cho GPU-Util.
3. **Triển khai cascade + reasoning budget**: router model nhỏ trước, chỉ cho reasoning trên request khó, chỉ cache những nhóm vượt ngưỡng đọc. Chuyển training/batch interruptible sang **europe-north1** (carbon -92% so với us-east-1). Giữ chat gần người dùng vì latency.
