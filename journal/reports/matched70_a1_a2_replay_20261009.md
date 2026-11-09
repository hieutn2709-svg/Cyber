# CTI Q2 — Kết quả A1/A2 trên validation

Ngày: 09/10/2026. Phạm vi: matched-70, fold 1, seed 42, 5 tài liệu validation, 49 quan hệ chuẩn. Thực hiện theo thiết kế đã trình bày trong báo cáo phân tích lỗi cùng ngày.

## 1. Phần việc vừa hoàn thành

Đã thực hiện 8 cấu hình thay đổi cùng 2 cấu hình baseline đối chứng: mỗi encoder có hai mức tăng ngưỡng thực thể và hai mức giảm số ứng viên. Toàn bộ phép đo sử dụng điểm thực thể và logits quan hệ đã lưu từ checkpoint RoBERTa epoch 12 và SecureBERT epoch 8. Cách chạy là **tái tính từ điểm đã lưu qua bộ lọc, sinh cặp, giải mã và chấm điểm chính thức của dự án** (verified saved-score replay).

Không có lượt forward mới qua checkpoint đã huấn luyện và không huấn luyện lại. Vì vậy, kết quả dưới đây là ablation hậu xử lý trên điểm đã lưu, không được mô tả thành kết quả huấn luyện mới hoặc thời gian suy luận thực đo. Chưa sử dụng tập test và chưa đổi cấu hình baseline trong dự án.

## 2. Thiết kế cố định

| Cấu hình | Ngưỡng giữ thực thể | Giới hạn ứng viên mỗi cửa sổ |
|---|---:|---:|
| Baseline | 0,05 | 128 |
| A1-0,10 | 0,10 | 128 |
| A1-0,20 | 0,20 | 128 |
| A2-96 | 0,05 | 96 |
| A2-64 | 0,05 | 64 |

Mỗi lần chỉ thay một tham số. Giữ nguyên checkpoint, tập validation, khoảng cách cặp tối đa 96 token, tập nhãn và ngưỡng quan hệ 0,90. Điểm lọc thực thể là P(entity) = 1 − P(NONE), không phải xác suất của riêng nhãn thực thể được chọn. Không kết hợp A1 với A2, không mở rộng lưới tìm kiếm sau khi thấy kết quả.

## 3. Toàn bộ kết quả

F1 thực thể chỉ tính 10 loại chính; F1 quan hệ tính strict micro trên tất cả quan hệ đánh giá được. Đơn vị F1 trong bảng là phần trăm.

| Encoder | Cấu hình | F1 thực thể | F1 quan hệ | TP | FP | FN | Cặp chuẩn còn chấm /49 |
|---|---|---:|---:|---:|---:|---:|---:|
| RoBERTa | Baseline | 38,72 | 25,38 | 25 | 123 | 24 | 37 |
| RoBERTa | A1-0,10 | 47,14 | 26,18 | 25 | 117 | 24 | 37 |
| RoBERTa | A1-0,20 | 57,64 | 27,06 | 23 | 98 | 26 | 34 |
| RoBERTa | A2-96 | 44,02 | 26,04 | 25 | 118 | 24 | 37 |
| RoBERTa | A2-64 | 51,26 | 27,62 | 25 | 107 | 24 | 36 |
| SecureBERT | Baseline | 31,45 | 20,44 | 14 | 74 | 35 | 30 |
| SecureBERT | A1-0,10 | 32,89 | 20,44 | 14 | 74 | 35 | 30 |
| SecureBERT | A1-0,20 | 36,56 | 21,05 | 14 | 70 | 35 | 30 |
| SecureBERT | A2-96 | 37,04 | 21,21 | 14 | 69 | 35 | 30 |
| SecureBERT | A2-64 | 44,09 | 20,47 | 13 | 65 | 36 | 28 |

Theo quy tắc ưu tiên F1 quan hệ, kết quả cao nhất trong lưới đã định là A2-64 với RoBERTa và A2-96 với SecureBERT. Đây là lựa chọn mô tả trên validation đã được dùng để chọn checkpoint trước đó; chưa phải bằng chứng khái quát hoặc cấu hình đã chốt để đánh giá test.

## 4. Diễn giải và đánh đổi

**RoBERTa A2-64:** F1 quan hệ tăng 2,24 điểm phần trăm, từ 25,38% lên 27,62%. Loại được 16 FP mà vẫn giữ 25 TP. Tuy nhiên, số cặp chuẩn còn được chấm giảm 37 xuống 36, và recall thực thể chính giảm từ 85,83% xuống 85,00%. Cặp chuẩn bị mất thêm vốn chưa được dự đoán đúng ở baseline; do đó recall quan hệ cuối cùng không giảm, nhưng tiềm năng thu hồi quan hệ đã giảm.

**SecureBERT A2-96:** F1 quan hệ tăng 0,77 điểm phần trăm, từ 20,44% lên 21,21%. Loại được 5 FP, giữ đủ 14 TP và 30 cặp chuẩn còn được chấm. Recall thực thể chính giữ ở 83,33%. Nếu giảm tiếp xuống 64 ứng viên, mô hình mất 1 TP và 2 cặp chuẩn, nên không dùng 64 làm lựa chọn chung cho hai encoder dựa trên lượt này.

**RoBERTa A1-0,20:** dù F1 thực thể cao nhất trong lưới (57,64%), mô hình mất 2 TP quan hệ và 3 cặp chuẩn còn được chấm. Kết quả này cho thấy không nên lựa chọn chỉ theo F1 thực thể khi mục tiêu chính là trích xuất quan hệ.

**Cấu hình chung đáng kiểm tra tiếp: A2-96.** Ở cả hai encoder, cấu hình này giảm FP mà giữ nguyên TP, recall thực thể chính và số cặp chuẩn còn được chấm so với baseline. Nó không phải cấu hình có F1 cao nhất riêng cho RoBERTa; lợi ích là giữ cùng một cách lọc để so sánh encoder có kiểm soát.

| A2-96 so với baseline | RoBERTa | SecureBERT |
|---|---:|---:|
| F1 quan hệ | 25,38% → 26,04% | 20,44% → 21,21% |
| Chênh lệch F1 quan hệ | +0,66 điểm % | +0,77 điểm % |
| FP quan hệ | 123 → 118 | 74 → 69 |
| TP quan hệ | 25 → 25 | 14 → 14 |
| Số cặp cần chấm | 22.458 → 15.592 | 34.136 → 22.804 |
| Giảm số cặp | 30,57% | 33,20% |

Giảm số cặp là số đếm cấu trúc, không đồng nghĩa giảm tương ứng thời gian chạy toàn hệ thống; encoder vẫn cần xử lý văn bản. Chưa có phép đo tốc độ hoặc bộ nhớ của forward mới để khẳng định mức tăng hiệu năng.

## 5. Tại sao có thể tái tính từ điểm đã lưu?

Bộ lọc sắp ứng viên theo điểm giảm dần cùng quy tắc phân xử hòa cố định. Khi chỉ tăng cutoff từ 0,05 hoặc giảm cap từ 128, tập được giữ là tập con của tập baseline. Không cần khôi phục proposal đã bị loại ở baseline cho đúng các thay đổi A1/A2 này. Điều này không áp dụng cho giảm cutoff dưới 0,05, tăng cap quá 128, thay cách xếp hạng hoặc thay mô hình.

Trong kiến trúc hiện tại, vector thực thể phụ thuộc token states và span; vector cặp phụ thuộc hai thực thể, ngữ cảnh giữa chúng và khoảng cách. Không có bước chuẩn hóa chung theo toàn bộ tập cặp. Do đó điểm đã lưu của một cặp còn giữ có thể dùng lại để đánh giá tác động của việc loại ứng viên ở cấp giải mã. Sai số dấu phẩy động của một lượt forward mới có thể khác do cách chia batch; chưa kiểm chứng numerical parity trên checkpoint thật.

Các kiểm chứng đã hoàn thành:

- 13 tệp nguồn liên quan khớp nội dung tại commit d0c7e456a7637a39c523a05513bdfd50754069c1 trên PR #5.
- Hai baseline tái tạo đúng toàn bộ khóa thực thể/quan hệ và điểm dự đoán đã lưu; số cặp sinh lại khớp tập logits gốc.
- Mỗi cặp trong từng cấu hình thay đổi đều có logits nguồn; không tự tạo điểm và không thay gold.
- Kết quả bộ giải mã chính thức khớp phép lọc độc lập trên dự đoán baseline ở cả 10 trường hợp.
- 400 trường hợp kiểm tra tính tương đương của lọc chặt hơn, bao gồm hòa điểm và bằng cutoff, đều đạt.
- Kiểm tra kiến trúc với đầu mạng tổng hợp đạt sai khác tối đa 3,13×10⁻⁷ khi lấy tập con; đây không phải phép chạy checkpoint đã huấn luyện.
- Chấm lại 10 tệp dự đoán đã xuất bằng scorer chính thức, đối chiếu tổng theo tài liệu, đều khớp.

## 6. Bước tiếp theo đã xác định

1. Chạy forward thực từ hai checkpoint đã chọn với baseline và A2-96, so sánh dự đoán với replay, ghi sai khác số học, thời gian và bộ nhớ. Việc này kiểm tra khả năng tái lập ở đường chạy mô hình thật, không phải huấn luyện lại 12 epoch.
2. Giữ RoBERTa A2-64 như một đối chứng riêng nếu kiểm tra sâu hơn; không thay cấu hình của riêng một encoder rồi trình bày như so sánh chỉ thay encoder.
3. Sau khi khóa cấu hình, kiểm tra độ ổn định nhiều seed của baseline và cấu hình chung. Hai baseline seed 42 đã hoàn thành được giữ lại; các seed 43/44 chưa chạy.
4. Chỉ triển khai thí nghiệm âm khó B1 như một ablation huấn luyện riêng, sau khi định nghĩa rõ nguồn mẫu âm trên train, giữ tỷ lệ âm 6 trong vòng đầu và bổ sung nhật ký proposal. Chưa triển khai B1 hoặc khẳng định tác dụng của nó.

Không mở full/test để lựa chọn A1/A2. Validation chỉ có 5 tài liệu và đã được dùng nhiều lần, nên những mức tăng trên là tín hiệu phát triển, chưa phải cải thiện đã xác nhận trên dữ liệu độc lập.

## 7. Tệp bàn giao

Gói kèm theo gồm mã chạy và kiểm chứng, hash nguồn, 10 bộ config/metrics/validation_predictions, tổng hợp ablation_results.json và danh sách các quan hệ bị loại trong removed_relations.json. Baseline gốc và báo cáo phân tích lỗi trước đó được giữ nguyên. Mọi trạng thái “đã chạy” trong báo cáo này chỉ đề cập 10 lần replay A1/A2 và các phép kiểm chứng nêu trên.
