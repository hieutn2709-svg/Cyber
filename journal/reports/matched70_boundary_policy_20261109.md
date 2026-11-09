# CTI Q2 — triển khai bước can thiệp E2, ngày 09/11/2026

Đã triển khai chính sách `boundary_replace` để thay các cặp âm được lấy mẫu nhưng có cùng cặp biên có hướng với một cặp dương. Đây là kết quả triển khai và kiểm chứng kỹ thuật; chưa có kết quả huấn luyện pretrained mới để kết luận thay đổi F1.

## Cách xử lý

Chế độ mặc định `typed` giữ nguyên cách lấy mẫu trước đây. Khi bật can thiệp, chương trình thực hiện lượt lấy mẫu gốc, giữ mọi cặp âm không mâu thuẫn và thay riêng các cặp mâu thuẫn bằng ứng viên âm hợp lệ chưa được chọn. Việc thay thế dùng bộ sinh số ngẫu nhiên cục bộ, không lấy lặp chỉ số ứng viên. Nếu thiếu ứng viên hợp lệ, chương trình ghi rõ số thiếu thay vì tạo mẫu trùng hoặc đổi nhãn.

Biên được đối chiếu theo tài liệu, vị trí nguồn, vị trí đích và hướng; không gộp quan hệ đảo chiều. Tất cả cặp dương được giữ, kể cả cặp xa hơn giới hạn suy luận. Chính sách bỏ self-edge cũ được giữ nguyên. Kiến trúc, nhãn gold, trọng số loss, sinh ứng viên suy luận và quy tắc chọn checkpoint/ngưỡng không thay đổi.

Tôi chọn thay mẫu sau lượt lấy mẫu gốc thay vì lọc trước toàn bộ pool, nhằm giữ nguyên những batch không lấy trúng xung đột. Đây là lựa chọn thiết kế của đợt triển khai, không phải kết luận thực nghiệm rằng phương án này tốt hơn các phương án khác.

## Kiểm chứng đã thực hiện

| Kiểm tra | Kết quả |
|---|---|
| Bộ test tại mã gốc bất biến | 281 test, 96 subtest đạt |
| Bộ test sau triển khai | 290 test, 100 subtest đạt |
| Đối chiếu sampler mặc định trực tiếp với commit trước thay đổi | 504/504 trường hợp khớp toàn bộ batch và diagnostics |
| Can thiệp trên cùng 504 trường hợp tổng hợp | Không còn xung đột trong batch ở cả 504 trường hợp |
| Trường hợp vốn không lấy trúng xung đột | 358 batch giữ nguyên hoàn toàn |
| Trường hợp cần sửa mẫu | 146 batch được xử lý |
| Oracle thay mẫu độc lập | Đối chiếu cách chọn/thay mẫu trên 100 seed; pool đầu vào lấy từ sampler gốc |
| Loss/backward trên mô hình nhỏ | Gradient hữu hạn; giữ loss thực thể, loss loại quan hệ và số cặp dương; loss tồn tại quan hệ thay đổi khi bỏ nhãn âm mâu thuẫn |
| Lưu/khôi phục | Khớp chính xác trọng số, trạng thái AdamW, RNG và kết quả epoch kế tiếp; từ chối identity khác chính sách |
| Đường chạy smoke hoàn chỉnh trên fixture | Hai epoch cho mỗi chính sách; chạy thật training, inference, scoring và ghi artifact; kiểm tra policy trong config, recovery, checkpoint và summary |

Fixture smoke thay thế riêng bước tải dữ liệu/preflight và tạo encoder pretrained bằng mô hình nhỏ. Không gọi đây là smoke pretrained của RoBERTa/SecureBERT. Kiểm thử mới đã thất bại vì giao diện/chức năng chưa tồn tại trước khi triển khai, rồi đạt sau thay đổi. Rà soát độc lập không phát hiện lỗi Critical/Important; một mô tả oracle chưa chính xác đã được sửa. Đối chiếu mã gốc 504 trường hợp và test smoke xuyên suốt được bổ sung sau rà soát và đã chạy đạt.

## Truy vết và cách dùng

Thêm `--relation-negative-policy boundary_replace --train-pair-diagnostics` vào lệnh matched encoder hiện tại. Có thể dùng với overfit/smoke/dev; chế độ mặc định vẫn là `typed`. Hai wrapper matched encoder và seed43/44 đều truyền chính sách. Full/test không được mở cho can thiệp này.

`eligible` vẫn là pool gốc trước can thiệp, còn `sampled` là batch thực sự tính loss. Trường `negative_policy` ghi số âm gốc, số bỏ, số thay, số thiếu, kích thước pool âm hợp lệ và digest batch trước can thiệp. Không diễn giải xung đột còn trong `eligible` thành xung đột còn trong loss.

Chính sách được ghi vào cấu hình chạy, identity khôi phục, checkpoint được chọn, summary và hậu tố run ID. Chế độ mặc định không thêm trường identity mới. Các kiểm tra identity cũ, kể cả mã nguồn và môi trường, vẫn có hiệu lực; không sửa identity của checkpoint cũ để tiếp tục bằng chính sách mới.

Chạy lại kiểm toán tương thích từ gốc repository:

```bash
python -m journal.scripts.audit_boundary_policy_compatibility
python -m pytest -q
```

Commit đối chứng: `63a31ba67089ea2bb7bf0016ff91f9f0fa4e0921`. JSON kiểm toán đi kèm ghi hash mã cũ/mới; file verification ghi log và hash mã đã kiểm tra.

## Giao thức thí nghiệm chất lượng kế tiếp

Đã chốt thiết kế bốn lượt dev: RoBERTa và SecureBERT, mỗi encoder có `typed` và `boundary_replace`, seed42/fold1, 12 epoch, cùng phiên bản mã và môi trường CPU4 luồng/interop1. Cả bốn lượt khởi tạo từ pretrained đã khóa; không dùng các lượt khác môi trường của tháng trước làm đối chứng ghép cặp mới. Giữ matched70 gold v1, splitseed11800, ngân sách cặp và lưới chọn ngưỡng. Lưu cả trường hợp chất lượng giảm và mọi thiếu hụt ngân sách âm. So sánh precision/recall/F1, TP/FP/FN, độ phủ ứng viên và mức can thiệp thực tế; chưa trộn hard-negative mining.

**Bốn lượt pretrained này chưa được chạy trong đợt triển khai kỹ thuật.** Các số 504/358/146 là trường hợp tổng hợp kiểm thử, không phải tài liệu hoặc quan hệ của corpus. Chưa có bằng chứng cải thiện F1, ý nghĩa thống kê, khả năng khái quát, hiệu năng CUDA hoặc chi phí logging ở quy mô pretrained. Phân xử nhãn bởi người độc lập vẫn chờ. PR tiếp tục ở trạng thái nháp; chưa merge và chưa đánh giá full/test.
