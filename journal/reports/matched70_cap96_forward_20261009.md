# CTI Q2 — Xác nhận suy luận trực tiếp với giới hạn 96 ứng viên

Ngày 09/10/2026. Phạm vi: matched-70, fold 1, seed 42, 5 tài liệu validation và 49 quan hệ chuẩn.

## Kết quả chính

Đã chạy mới bốn lượt suy luận trực tiếp từ checkpoint đã chọn: RoBERTa epoch 12 và SecureBERT epoch 8, mỗi mô hình với cấu hình baseline 128 ứng viên và cấu hình A2-96. Cả bốn lượt đều khớp hoàn toàn với kết quả replay trước đó về tập thực thể, tập quan hệ và strict micro metrics.

Đây là bước xác nhận bằng forward thật qua encoder và các đầu dự đoán. Không huấn luyện lại, không chọn checkpoint mới, không chọn lại ngưỡng quan hệ và không đánh giá test. Baseline trong dự án vẫn giữ nguyên để làm đối chứng.

| Mô hình | Giới hạn ứng viên | F1 thực thể chính | F1 quan hệ | TP / FP / FN quan hệ | Cặp chuẩn còn chấm /49 |
|---|---:|---:|---:|---|---:|
| RoBERTa | 128 | 38,72% | 25,38% | 25 / 123 / 24 | 37 |
| RoBERTa | 96 | 44,02% | 26,04% | 25 / 118 / 24 | 37 |
| SecureBERT | 128 | 31,45% | 20,44% | 14 / 74 / 35 | 30 |
| SecureBERT | 96 | 37,04% | 21,21% | 14 / 69 / 35 | 30 |

Với 96 ứng viên, mỗi mô hình giảm 5 FP quan hệ mà không mất TP. F1 quan hệ tăng 0,66 điểm phần trăm ở RoBERTa và 0,77 điểm phần trăm ở SecureBERT. Recall thực thể chính không đổi: 85,83% và 83,33%. Những mức tăng này vẫn chỉ là kết quả trên validation nhỏ, một seed, chưa xác nhận khả năng khái quát.

## Cách thực hiện

Hai checkpoint được tải lại và đối chiếu SHA-256 với báo cáo đã lưu. Kiến trúc encoder được khởi tạo từ config của đúng revision đã khóa; hash config phải khớp thông tin trong checkpoint. Sau đó nạp toàn bộ state_dict bằng strict=True và kiểm tra mọi tensor trong mô hình bằng đúng tensor checkpoint. Không có tham số khởi tạo ngẫu nhiên nào được giữ lại trong mô hình chạy.

Mô hình được đặt ở chế độ eval, chạy CPU, không tính gradient, 4 luồng PyTorch, OMP_NUM_THREADS=4 và MKL_NUM_THREADS=4. Giữ nguyên width cap 6, khoảng cách quan hệ tối đa 96 token, kích thước chunk quan hệ 512, ngưỡng thực thể 0,05 và ngưỡng quan hệ 0,90. Mỗi cấu hình chạy trong một tiến trình riêng; xử lý các tài liệu validation 365, 389, 395, 400, 420.

Đã sử dụng đúng mã suy luận, sinh cặp, giải mã và chấm điểm của dự án. Các tệp lõi đã đối chiếu ở lượt A1/A2 không thay đổi; runtime_model.py cũng được kiểm tra khớp bản công bố. Mã chạy và hash nguồn được bàn giao trong gói tái lập.

## Sai khác số học so với replay

| Kiểm tra | RoBERTa 128 | RoBERTa 96 | SecureBERT 128 | SecureBERT 96 |
|---|---:|---:|---:|---:|
| Tập dự đoán và metrics khớp | Có | Có | Có | Có |
| Sai khác lớn nhất của điểm thực thể | 0 | 0 | 0 | 0 |
| Sai khác lớn nhất của xác suất quan hệ đã xuất | 0 | 0 | 0 | 0 |
| Sai khác lớn nhất của logit tồn tại trên toàn bộ cặp còn giữ | 0 | 4,77×10⁻⁷ | 0 | 4,77×10⁻⁷ |
| Sai khác lớn nhất của logits loại quan hệ | 0 | 0 | 0 | 0 |
| Số cặp thay đổi argmax loại quan hệ | 0 | 0 | 0 | 0 |

Các sai khác rất nhỏ ở logit tồn tại không thay đổi quyết định qua ngưỡng. Đối chiếu đã bao gồm toàn bộ cặp còn được chấm, không chỉ những quan hệ dự đoán cuối cùng. Riêng baseline khớp cả raw logits của lượt dev đã lưu.

## Thời gian và bộ nhớ quan sát được

| Mô hình | Giới hạn ứng viên | Cặp được chấm | Tổng forward 5 tài liệu | Peak RSS tiến trình |
|---|---:|---:|---:|---:|
| RoBERTa | 128 | 22.458 | 2,571 giây | 1.286,84 MiB |
| RoBERTa | 96 | 15.592 | 2,306 giây | 1.287,03 MiB |
| SecureBERT | 128 | 34.136 | 3,322 giây | 1.286,89 MiB |
| SecureBERT | 96 | 22.804 | 2,532 giây | 1.286,88 MiB |

Thời gian là tổng các lần gọi suy luận từng cửa sổ, không gồm tải checkpoint, khởi tạo mô hình, kiểm chứng hoặc ghi tệp. Mỗi cấu hình chỉ chạy một lượt, chưa warmup và chưa lặp đo theo thiết kế benchmark. Vì vậy, không dùng bảng này để khẳng định tỷ lệ tăng tốc ổn định.

Peak RSS là mức cao nhất của toàn bộ tiến trình riêng, bao gồm nạp checkpoint và mô hình; không phải bộ nhớ riêng của phép forward. Các số gần như nhau không chứng minh cap96 giảm bộ nhớ tổng. Số cặp giảm 30,57% và 33,20% là số đếm chắc chắn; tác động hiệu năng cần benchmark riêng nếu trở thành nội dung công bố.

Môi trường: Python 3.12.14; PyTorch 2.14.1+cpu; Transformers 5.18.0; CPU 4 luồng; attention backend được ghi trong từng forward_result.json. Không có xác minh CUDA trong bước này.

## Trạng thái và bước nghiên cứu tiếp theo

Đã hoàn thành chuỗi: hai baseline dev → phân tích lỗi → A1/A2 trên điểm đã lưu → xác nhận forward thật cho baseline và cap96. Cấu hình cap96 đã có bằng chứng nhất quán để đưa vào vòng kiểm tra độ ổn định tiếp theo, chưa được tuyên bố là mô hình cuối cùng.

Vòng tiếp theo nên giữ cùng cap96 cho cả hai encoder, đối chiếu baseline và cap96 ở nhiều seed đã định trước. Với mục tiêu kiểm tra thay đổi chỉ tại suy luận, mỗi encoder/seed chỉ cần một lượt huấn luyện baseline, sau đó đánh giá cả cap128 và cap96 trên checkpoint được chọn theo quy tắc baseline. Cách này cô lập tác động của lọc ứng viên và không cần huấn luyện riêng một mô hình cap96. Seed42 đã có đầy đủ, chỉ các seed43/44 là mới nếu tiến hành. Không gộp thay đổi lấy mẫu âm B1 vào vòng này.

Các lượt nhiều seed chưa chạy. Full/test chưa thực hiện. Bốn lượt forward xác nhận trong báo cáo này không cung cấp ước lượng phương sai hoặc kiểm định ý nghĩa thống kê.

## Đoạn kết quả có thể dùng cho bản thảo chuyên đề 2

“Trên tập validation matched-70 ở fold 1 và seed 42, việc giảm số ứng viên thực thể tối đa từ 128 xuống 96 làm F1 quan hệ của RoBERTa tăng từ 25,38% lên 26,04% và của SecureBERT tăng từ 20,44% lên 21,21%. Mỗi cấu hình loại thêm năm quan hệ dương tính giả, trong khi giữ nguyên số quan hệ dương tính thật và độ bao phủ cặp thực thể chuẩn. Kết quả được kiểm chứng bằng suy luận trực tiếp từ các checkpoint đã chọn và trùng với phép tái tính trên logits đã lưu. Tuy nhiên, đánh giá hiện mới dựa trên năm tài liệu validation và một seed; do đó cần kiểm tra độ ổn định trước khi kết luận về hiệu quả khái quát của phương pháp lọc.”

## Bằng chứng và tệp bàn giao

Checkpoint RoBERTa SHA-256: d637294c72f159762c4606637706171fdb8799140d7b012be3c6652a1f56aa44.

Checkpoint SecureBERT SHA-256: 4e50005d62b3c6bc2f696d13a314d413204001124258ec003a947b718af70812.

Gói bàn giao gồm run_forward.py, forward_confirmation.json, bốn bộ config/forward_result/validation_predictions, raw logits nén, config encoder gốc, README và SHA256SUMS. Checkpoint lớn đã bàn giao trước đó, không chép lặp vào gói này. Báo cáo A1/A2 và kết quả dev gốc được giữ nguyên.
