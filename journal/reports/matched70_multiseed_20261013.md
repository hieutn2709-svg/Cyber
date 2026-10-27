# CTI Q2 — Báo cáo kiểm chứng ba seed

Ngày lập: 13/10/2026. Phạm vi: matched70, fold 1, validation.

## Công việc đã hoàn thành

Đã hoàn thành bốn lượt dev mới: RoBERTa và SecureBERT, mỗi mô hình chạy seed 43 và 44, mỗi lượt 12 epoch. Kết hợp với seed 42 đã hoàn thành trước đó, báo cáo gồm 12 trường hợp đánh giá (2 mô hình × 3 seed × 2 cap). Các lượt chạy dở ngày 10/10 không được tính vào kết quả.

Dữ liệu giữ nguyên 52 văn bản và 70 cửa sổ; chia theo văn bản với split seed 11800. Tập validation có 5 văn bản, 120 thực thể chính và 49 quan hệ gold. Tokenizer của từng mô hình được giữ nguyên phiên bản, xử lý cùng lát cắt ký tự.

## Thiết kế và kiểm soát so sánh

Mỗi lượt chỉ huấn luyện baseline cap 128 từ trọng số pretrained đã khóa. Chọn checkpoint theo relation F1 trước, primary entity F1 khi hòa; chọn ngưỡng theo lưới validation baseline đã có. Cap 96 dùng cùng checkpoint và cùng ngưỡng của baseline, không huấn luyện riêng và không chọn lại ngưỡng. Seed 43/44 không bị ép dùng ngưỡng 0,90 của seed 42.

## Kết quả từng lượt

| Mô hình | Seed | Cap | Epoch chọn | Ngưỡng | Entity F1 (%) | Relation F1 (%) | TP/FP/FN quan hệ |
|---|---:|---:|---:|---:|---:|---:|---|
| roberta | 42 | 128 | 12 | 0.90 | 38.722 | 25.381 | 25/123/24 |
| roberta | 42 | 96 | 12 | 0.90 | 44.017 | 26.042 | 25/118/24 |
| roberta | 43 | 128 | 10 | 0.90 | 42.266 | 31.579 | 15/31/34 |
| roberta | 43 | 96 | 10 | 0.90 | 43.304 | 31.579 | 15/31/34 |
| roberta | 44 | 128 | 12 | 0.90 | 39.841 | 32.787 | 20/53/29 |
| roberta | 44 | 96 | 12 | 0.90 | 45.662 | 32.787 | 20/53/29 |
| securebert | 42 | 128 | 8 | 0.90 | 31.447 | 20.438 | 14/74/35 |
| securebert | 42 | 96 | 8 | 0.90 | 37.037 | 21.212 | 14/69/35 |
| securebert | 43 | 128 | 11 | 0.90 | 32.357 | 23.529 | 14/56/35 |
| securebert | 43 | 96 | 11 | 0.90 | 36.590 | 23.529 | 14/56/35 |
| securebert | 44 | 128 | 10 | 0.90 | 30.088 | 28.846 | 15/40/34 |
| securebert | 44 | 96 | 10 | 0.90 | 34.722 | 28.846 | 15/40/34 |

## Trung bình và độ lệch chuẩn mẫu

Đơn vị: phần trăm; độ lệch chuẩn mẫu dùng mẫu số n−1. Đây là biến thiên theo seed trên cùng tập validation.

| Mô hình | Cap | Entity F1: TB ± SD (%) | Relation F1: TB ± SD (%) |
|---|---:|---:|---:|
| roberta | 128 | 40.276 ± 1.812 | 29.916 ± 3.973 |
| roberta | 96 | 44.328 ± 1.210 | 30.136 ± 3.597 |
| securebert | 128 | 31.298 ± 1.142 | 24.271 ± 4.253 |
| securebert | 96 | 36.116 ± 1.228 | 24.529 ± 3.914 |

## Chênh lệch ghép cặp: cap 96 trừ cap 128

| Mô hình | Seed | Δ Entity F1 (điểm %) | Δ Relation F1 (điểm %) | Δ TP | Δ FP | Δ cặp gold có thể dự đoán |
|---|---:|---:|---:|---:|---:|---:|
| roberta | 42 | +5.295 | +0.661 | +0 | -5 | +0 |
| roberta | 43 | +1.038 | +0.000 | +0 | +0 | +0 |
| roberta | 44 | +5.821 | +0.000 | +0 | +0 | +0 |
| securebert | 42 | +5.590 | +0.774 | +0 | -5 | +0 |
| securebert | 43 | +4.232 | +0.000 | +0 | +0 | -4 |
| securebert | 44 | +4.634 | +0.000 | +0 | +0 | -1 |

## Diễn giải kết quả

roberta: cap 96 tăng relation F1 ở 1/3 seed; chênh lệch trung bình +0.220 điểm phần trăm, độ lệch chuẩn mẫu 0.382. Cần xem đồng thời TP, FP và mức bao phủ cặp gold ở từng seed.
securebert: cap 96 tăng relation F1 ở 1/3 seed; chênh lệch trung bình +0.258 điểm phần trăm, độ lệch chuẩn mẫu 0.447. Cần xem đồng thời TP, FP và mức bao phủ cặp gold ở từng seed.

Trong cả hai mô hình, phần tăng relation F1 chỉ đến từ seed 42 với năm false positive bị loại; seed 43 và 44 giữ nguyên TP/FP/FN quan hệ. Entity F1 tăng ở cả sáu cặp so sánh. Tuy nhiên, SecureBERT mất bốn cặp gold có thể dự đoán ở seed 43 và một cặp ở seed 44; vì vậy chưa thể xem cap 96 là cải thiện đồng đều về độ bao phủ. Số cặp ứng viên trung bình giảm từ 19.724 xuống 14.661,33 ở RoBERTa và từ 37.081,33 xuống 25.957,33 ở SecureBERT. Đây là giảm khối lượng ứng viên, chưa phải số đo tăng tốc thực tế.

## Giới hạn và cách sử dụng trong bài báo

Ba seed trên năm văn bản validation chưa đủ để kết luận khả năng khái quát hoặc ý nghĩa thống kê. Tập validation được dùng để chọn checkpoint/ngưỡng và khảo sát cap, nên các số liệu chỉ mô tả giai đoạn phát triển. Biến thiên theo seed không đo được bất định do lấy mẫu văn bản. Không so sánh trực tiếp tọa độ token giữa hai tokenizer. Không dùng thời gian chạy CPU đồng thời làm bằng chứng tăng tốc.

Kết quả này có thể đưa vào phần thí nghiệm phát triển và phân tích độ ổn định của CTI Q2/chuyên đề 2. Chưa dùng làm kết quả test cuối cùng hoặc tuyên bố vượt trội tổng quát. Chưa mở rộng sang full/test; PR vẫn là draft, chưa merge.

## Kiểm chứng, phục hồi và tái lập

Đã khôi phục và xác minh 105 mục SHA-256 của bộ matched70 và 22 mục của bộ forward seed 42. Hai trọng số pretrained đúng SHA-256 đã khóa. 248 kiểm thử chạy đạt trong môi trường khôi phục. Mỗi checkpoint mới được nạp strict và đối chiếu mọi tensor; mỗi cặp cap được suy luận thật, đối chiếu khóa dự đoán, logits và điểm số với baseline/replay, rồi chấm lại tệp dự đoán đã xuất.

Kiểm tra độc lập cuối cùng chấm lại đủ 12 tệp dự đoán bằng bộ chấm điểm chính thức, đối chiếu trung bình/độ lệch chuẩn mẫu bằng NumPy và xác nhận lịch sử đủ 12 epoch của bốn lượt seed 43/44. Chạy lại toàn bộ 248 kiểm thử trước khi cập nhật báo cáo: đạt, thời gian 4,205 giây. Tại cap 96 của seed 44, sai khác tuyệt đối lớn nhất của existence logits so với replay là 1,90735×10⁻⁶ với RoBERTa và 1,43051×10⁻⁶ với SecureBERT; type logits và type argmax không đổi, dự đoán và điểm số khớp.

Các gate kỹ thuật preflight/overfit/smoke từ seed 42 được tái sử dụng có ghi nguồn, không trình bày như đã chạy mới cho seed 43/44. Nếu khác biệt tối ưu hóa xuất hiện ở seed mới, nó được ghi nhận trong lịch sử dev thực tế.

Phiên ngày 10/10 bị gián đoạn và mất checkpoint tạm sau khi môi trường đặt lại. Seed 43 hoàn thành ngày 12/10 và được giữ nguyên. Seed 44 chạy dở ngày 12/10 bị mất checkpoint tạm, nên được chạy lại từ pretrained ngày 13/10; không giả lập việc resume từ checkpoint đã mất.

Mã huấn luyện: commit `1e589a02bc77cef3af1dd27a09effce4e6c80f72`. Lệnh được hỗ trợ: `python -m journal.scripts.train_matched_seed`. Điểm nhỏ còn hoãn từ rà soát: gọi trực tiếp đường dẫn script thiếu bootstrap import.

## Công việc tiếp theo

Phân tích các lỗi lặp lại qua seed, chốt cấu hình phát triển và thiết kế đánh giá độc lập theo văn bản trước khi đưa ra kết luận cuối cùng. Cập nhật bảng thí nghiệm trong bản thảo CTI Q2 bằng kết quả có truy vết; các phần tổng quan và chuyên đề khác cần tiếp tục theo nguồn riêng của chúng.
