# CTIQ2 E2 thử nghiệm thay thế mẫu âm trùng biên

Đã hoàn thành bốn lượt huấn luyện phát triển từ trọng số pretrained, mỗi lượt 12 epoch. So sánh dưới đây ghép cặp chính sách gốc typed và boundary_replace trong cùng bộ mã hóa, cùng nguồn mã và thông tin phiên bản môi trường được ghi nhận. Kết quả trên năm tài liệu validation chỉ mô tả thử nghiệm phát triển.

Sau can thiệp, xung đột trong batch sampled bằng 0 ở cả hai bộ mã hóa. F1 quan hệ thay đổi +1.520 điểm phần trăm với RoBERTa và -3.596 điểm với SecureBERT. Kết quả pilot này chưa đủ để chọn chính sách mặc định.

## Kết quả chất lượng

| Bộ mã hóa và chính sách | Epoch chọn | Ngưỡng | F1 thực thể chính % | F1 quan hệ % | Quan hệ TP FP FN |
|---|---:|---:|---:|---:|---|
| roberta typed | 12 | 0.90 | 38.722 | 25.381 | 25 / 123 / 24 |
| roberta boundary_replace | 11 | 0.90 | 38.372 | 26.901 | 23 / 99 / 26 |
| securebert typed | 8 | 0.90 | 31.447 | 20.438 | 14 / 74 / 35 |
| securebert boundary_replace | 8 | 0.90 | 28.402 | 16.842 | 16 / 125 / 33 |

Chênh lệch được tính bằng boundary_replace trừ typed, đơn vị điểm phần trăm.

- roberta: F1 quan hệ +1.520; F1 thực thể chính -0.350.
- securebert: F1 quan hệ -3.596; F1 thực thể chính -3.044.

Số đếm quan hệ cho thấy đánh đổi precision và recall:

- roberta: TP 25 → 23, FP 123 → 99, FN 24 → 26.
- securebert: TP 14 → 16, FP 74 → 125, FN 35 → 33.

## Xung đột trong các mẫu dùng để tính loss

| Bộ mã hóa và chính sách | Nhóm xung đột sampled | Tổng nhóm sampled | Mẫu âm loại | Mẫu bù | Thiếu hụt |
|---|---:|---:|---:|---:|---:|
| roberta typed | 3 | 32711 | 0 | 0 | 0 |
| roberta boundary_replace | 0 | 32717 | 1 | 1 | 0 |
| securebert typed | 11 | 32866 | 0 | 0 | 0 |
| securebert boundary_replace | 0 | 32864 | 4 | 4 | 0 |

Xung đột trước can thiệp và mẫu số eligible qua 12 epoch:

- roberta typed: 813 / 4824356 nhóm biên eligible.
- roberta boundary_replace: 813 / 4866898 nhóm biên eligible.
- securebert typed: 837 / 5601364 nhóm biên eligible.
- securebert boundary_replace: 880 / 5627392 nhóm biên eligible.

Mỗi hàng cộng 588 bản ghi cửa sổ qua 12 epoch trên 36 tài liệu train. Các lần lặp không phải các quan hệ độc lập. Eligible là tập ứng viên trước can thiệp; sampled là các cặp thực sự dùng để tính loss. Sau lần thay thế đầu tiên, trọng số và tập ứng viên của hai nhánh có thể khác nhau, nên số mẫu bị loại trong nhánh mới không cần bằng số xung đột của nhánh gốc.

## Độ bao phủ và điểm số đầy đủ

| Bộ mã hóa và chính sách | P thực thể % | R thực thể % | P quan hệ % | R quan hệ % | Cặp gold được phủ |
|---|---:|---:|---:|---:|---:|
| roberta typed | 25.000 | 85.833 | 16.892 | 51.020 | 37/49 |
| roberta boundary_replace | 25.000 | 82.500 | 18.852 | 46.939 | 33/49 |
| securebert typed | 19.380 | 83.333 | 15.909 | 28.571 | 30/49 |
| securebert boundary_replace | 17.266 | 80.000 | 11.348 | 32.653 | 27/49 |

Số đếm thực thể chính theo thứ tự TP / FP / FN:

- roberta typed: 103 / 309 / 17.
- roberta boundary_replace: 99 / 297 / 21.
- securebert typed: 100 / 416 / 20.
- securebert boundary_replace: 96 / 460 / 24.

Kiểm tra ghép cặp trước lần can thiệp đầu tiên:

- roberta: 367 cửa sổ liên tiếp có mã băm batch khớp. Lần thay thế đầu tiên: epoch 8, bước 25, tài liệu 370; batch đối chứng giả định tại bước này khớp nhánh gốc.
- securebert: 24 cửa sổ liên tiếp có mã băm batch khớp. Lần thay thế đầu tiên: epoch 1, bước 25, tài liệu 421; batch đối chứng giả định tại bước này khớp nhánh gốc.

## Điều kiện thực nghiệm và kiểm chứng

Nguồn mã cố định 0089d1bbe4fbce847f8ed26d9605ac5de4748907; seed 42, fold 1, split seed 11800. CPU với 4 luồng Torch và 1 luồng interop; Torch 2.14.1+cpu, Transformers 5.18.0. Thông tin phiên bản môi trường khớp giữa bốn lượt, được lưu trong paired_results.json. RoBERTa typed hoàn tất ngày 14/11, RoBERTa boundary_replace ngày 15/11; hai lượt này được khôi phục ngày 16/11 mà không huấn luyện lại. Hai lượt SecureBERT chạy ngày 16/11. Không khẳng định cùng máy vật lý qua các ngày hoặc so sánh tốc độ.

Giữ nguyên kiến trúc, dữ liệu gold, loss, optimizer, sinh ứng viên, lịch seed và quy tắc chọn checkpoint cùng lưới ngưỡng validation. Chỉ thay chính sách lấy mẫu âm. Các trọng số pretrained và 13 tệp dữ liệu khớp SHA-256 đã khóa. Gate preflight, overfit và smoke được tái sử dụng với nguồn gốc cũ; không trình bày thành các lượt chạy mới.

Toàn bộ 48 epoch và 2.352 bản ghi train được đối chiếu thứ tự cửa sổ, seed, tập tài liệu, số cặp và mã băm. Mỗi epoch giữ 386 cặp dương chọn được, gồm một cặp xa, và bỏ qua một self-edge theo hành vi gốc. Kiểm chứng checkpoint được chọn với bản nhúng trong recovery, trạng thái AdamW và metadata được lưu từ ngày hoàn tất của từng lượt. Với hai lượt RoBERTa, ngày 16/11 xác minh lại gói lưu, tensor được chọn và dự đoán; không nạp lại recovery vì gói không chứa các tensor recovery. Hai lượt SecureBERT được kiểm tra trực tiếp recovery trước khi lưu. Năm bản ghi validation của từng lượt được chấm lại bằng bộ chấm điểm chính thức.

Bộ kiểm thử tại nguồn mã đã khóa đạt 290 test và 100 subtest ngày 14/11; ba kiểm thử chặn khi lưu thất bại chạy lại đạt ngày 16/11. Các checkpoint được chọn cùng bằng chứng từng lượt được đóng gói riêng; các gói không chứa đầy đủ trạng thái optimizer để tiếp tục huấn luyện. Mã băm recovery và kết quả kiểm tra vẫn được giữ trong báo cáo máy đọc được.

Các lần chạy tạm chưa lưu ngày 13/11, nhánh RoBERTa thay thế chưa hoàn thành ngày 14/11 và lượt SecureBERT dở dang ngày 15/11 bị mất checkpoint không được dùng trong kết quả. Các lượt đã hoàn tất và lưu thành công được giữ nguyên.

## Giới hạn và bước tiếp theo

Đây là pilot một seed trên năm tài liệu validation đã dùng để chọn checkpoint và ngưỡng. Chưa có đánh giá full/test, chưa merge và chưa thay nhãn gold. Phân xử nhãn độc lập vẫn đang chờ. Không suy ra ý nghĩa thống kê, khả năng khái quát, thứ hạng bộ mã hóa hoặc mức tăng tốc từ các kết quả này.

Trước khi chọn chính sách mặc định, cần khóa trước protocol nhiều seed để kiểm tra độ ổn định và hoàn thành phân xử nhãn. Không điều chỉnh thêm ngưỡng hoặc sampler theo kết quả pilot này.
