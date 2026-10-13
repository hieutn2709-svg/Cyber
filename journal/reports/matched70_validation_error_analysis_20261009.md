# CTI Q2 — Phân tích lỗi validation và công việc tiếp theo

Ngày cập nhật: 09/10/2026. Phạm vi: matched-70, fold 1, seed 42; hai checkpoint đã chọn từ các lượt dev 12 epoch. Đây là phân tích bổ sung từ kết quả đã hoàn thành, không phải một lượt huấn luyện mới.

## 1. Tiến độ đã hoàn thành và điểm tiếp tục

Hai lượt dev RoBERTa và SecureBERT đều đã hoàn thành 12/12 epoch. RoBERTa chọn epoch 12; SecureBERT chọn epoch 8 theo quy tắc ưu tiên F1 quan hệ, sau đó F1 thực thể chính để phân xử hòa. Cả hai sử dụng ngưỡng quan hệ 0,90. Không chạy lại các lượt này.

Phần hoàn thành trong đợt tiếp tục này gồm: kiểm chứng bộ kết quả lưu; tính lại strict micro metrics; giải mã lại toàn bộ quan hệ từ logits; phân loại lỗi thực thể và quan hệ; đối chiếu từng quan hệ chuẩn giữa hai encoder bằng ID thực thể và tọa độ ký tự; lập kế hoạch thí nghiệm có tiêu chí đánh giá rõ ràng. Chưa chạy full/test, chưa huấn luyện cấu hình mới, chưa khẳng định cải thiện mô hình.

| Chỉ tiêu validation | RoBERTa | SecureBERT |
|---|---:|---:|
| Checkpoint được chọn | Epoch 12 | Epoch 8 |
| F1 thực thể chính | 38,72% | 31,45% |
| Precision thực thể chính | 25,00% | 19,38% |
| Recall thực thể chính | 85,83% | 83,33% |
| F1 quan hệ | 25,38% | 20,44% |
| Precision quan hệ | 16,89% | 15,91% |
| Recall quan hệ | 51,02% | 28,57% |
| Quan hệ TP / FP / FN | 25 / 123 / 24 | 14 / 74 / 35 |

Validation gồm 5 tài liệu, 120 thực thể chính, 2 thực thể phụ và 49 quan hệ. F1 thực thể trong bảng đầu chỉ tính 10 loại chính; phân tích lỗi thực thể bên dưới bao gồm cả hai thực thể phụ. Không dùng chỉ số của epoch khác để ghép thành một kết quả tổng hợp.

## 2. Quan hệ bị mất ở đâu?

Mỗi quan hệ chuẩn được gán đúng một trạng thái, theo thứ tự: dự đoán đúng; thiếu/sai kiểu ít nhất một thực thể đầu–cuối; cặp không được chấm điểm; toàn bộ điểm tồn tại thấp hơn ngưỡng; hoặc qua ngưỡng nhưng sai loại quan hệ. Đây là cách phân rã theo thứ tự xử lý, không phải chứng minh nguyên nhân duy nhất của mỗi lỗi.

| Trạng thái của 49 quan hệ chuẩn | RoBERTa | SecureBERT |
|---|---:|---:|
| Dự đoán đúng toàn bộ | 25 | 14 |
| Mất tại thực thể đầu–cuối | 12 | 19 |
| Đủ thực thể nhưng cặp không được chấm điểm | 0 | 0 |
| Điểm tồn tại thấp hơn 0,90 | 10 | 14 |
| Qua ngưỡng nhưng sai loại quan hệ | 2 | 2 |
| Tổng | 49 | 49 |

Bộ lọc khoảng cách 96 token không làm mất thêm quan hệ chuẩn nào trong các cặp đã giữ đúng hai đầu mút: RoBERTa vẫn có 37/49 cặp chuẩn trước và sau lọc; SecureBERT vẫn có 30/49. Do đó chưa có bằng chứng validation để ưu tiên tăng khoảng cách. Kết luận này chỉ áp dụng cho các cặp có đầu mút được giữ lại, không khẳng định mọi quan hệ trong toàn bộ dữ liệu đều gần nhau.

Trong nhóm bị ngưỡng loại, RoBERTa có 6/10 cặp đã dự đoán đúng loại quan hệ; SecureBERT có 11/14. Các trường hợp còn lại đồng thời sai loại. Vì vậy, giảm ngưỡng không tự động khôi phục toàn bộ số FN trong nhóm này và có thể tăng FP. Lưới ngưỡng cũ từ 0,10 đến 0,90 đã chọn 0,90; phân tích này không chọn thêm ngưỡng hay sửa kết quả baseline.

## 3. Quan hệ dự đoán sai có đặc điểm gì?

| Nhóm FP, phân loại theo thứ tự ưu tiên | RoBERTa | SecureBERT |
|---|---:|---:|
| Ít nhất một đầu mút không khớp thực thể chuẩn theo cả biên và loại | 82 | 50 |
| Hai đầu mút chuẩn, cặp có hướng đúng nhưng sai loại quan hệ | 2 | 2 |
| Đảo chiều một cặp chuẩn, không có cặp chuẩn cùng chiều | 9 | 7 |
| Hai đầu mút chuẩn nhưng cặp không được chú giải theo cả hai chiều | 30 | 15 |
| Tổng FP quan hệ | 123 | 74 |

82/123 = 66,67% FP của RoBERTa và 50/74 = 67,57% FP của SecureBERT liên quan đến đầu mút không khớp thực thể chuẩn. Kết quả này gợi ý ưu tiên kiểm nghiệm chất lượng và cách giữ ứng viên thực thể. Nó không chứng minh việc loại một nhóm ứng viên sẽ cải thiện F1, vì việc lọc cũng có thể loại đầu mút đúng và giảm recall.

“Không được chú giải” là mô tả theo gold hiện có, không khẳng định quan hệ đó sai về ngữ nghĩa. Những cặp này cần được xem lại cùng văn bản nguồn trước khi kết luận thiếu nhãn. Không sửa nhãn validation sau khi quan sát dự đoán nếu chưa có quy trình kiểm toán nhãn độc lập và phiên bản dữ liệu mới.

## 4. Lỗi thực thể

Quy tắc phân loại: ưu tiên cùng biên nhưng khác loại; tiếp đến chồng lấn với thực thể cùng loại; tiếp đến chồng lấn khác loại; cuối cùng không chồng lấn. Với FN, đối chiếu thực thể chuẩn với tập dự đoán; với FP, đối chiếu dự đoán với tập chuẩn. Các nhóm mô tả quan hệ hình học của span, không xác định riêng lỗi proposal, phân loại hay pruning.

| Loại lỗi | RoBERTa FP | SecureBERT FP | RoBERTa FN | SecureBERT FN |
|---|---:|---:|---:|---:|
| Cùng biên, sai loại | 12 | 8 | 12 | 8 |
| Chồng lấn, cùng loại | 137 | 180 | 1 | 0 |
| Chồng lấn, khác loại | 40 | 38 | 2 | 4 |
| Không chồng lấn | 120 | 190 | 4 | 10 |
| Tổng, gồm cả loại phụ | 309 | 416 | 19 | 22 |

Proposal ban đầu bao phủ 120/122 thực thể ở cả hai mô hình; sau phân loại và giữ ứng viên, số khớp đúng cả biên lẫn loại còn 103 và 100. Recall thực thể chính tương đối cao nhưng precision thấp. Cần đánh giá cách xếp hạng, lọc và xử lý các span chồng lấn, đồng thời bảo toàn thực thể lồng nhau hợp lệ. Không thể tách chính xác tác động riêng của từng khâu từ các artifact hiện có vì chưa lưu logits của tất cả proposal thực thể bị loại.

## 5. Hai mô hình có bổ sung cho nhau không?

Đối chiếu bằng document ID, source entity ID, target entity ID và loại quan hệ; xác minh tọa độ ký tự của gold khớp giữa hai bộ dữ liệu. Không so sánh trực tiếp token index của hai tokenizer.

| Kết quả trên cùng một quan hệ chuẩn | Số lượng |
|---|---:|
| Cả hai cùng đúng | 14 |
| Chỉ RoBERTa đúng | 11 |
| Chỉ SecureBERT đúng | 0 |
| Cả hai cùng sai | 24 |
| Tổng | 49 |

Ở đúng hai checkpoint và ngưỡng hiện tại, mọi quan hệ đúng của SecureBERT đều đã được RoBERTa dự đoán đúng. Vì vậy phép hợp các quan hệ đúng không tăng recall trong mẫu này. Chưa có cơ sở ưu tiên ensemble đơn giản; kết quả không loại trừ khả năng bổ sung ở seed, checkpoint, phương pháp kết hợp hoặc dữ liệu khác.

## 6. Mức độ tập trung theo tài liệu và loại quan hệ

| Tài liệu | Số quan hệ chuẩn | RoBERTa TP / FP / FN | SecureBERT TP / FP / FN |
|---|---:|---|---|
| 365 | 13 | 5 / 38 / 8 | 3 / 6 / 10 |
| 389 | 14 | 5 / 27 / 9 | 2 / 26 / 12 |
| 395 | 7 | 6 / 30 / 1 | 3 / 7 / 4 |
| 400 | 11 | 6 / 16 / 5 | 3 / 8 / 8 |
| 420 | 4 | 3 / 12 / 1 | 3 / 27 / 1 |

Hai tài liệu 365 và 389 đóng góp 17/24 FN của RoBERTa và 22/35 FN của SecureBERT. Đây là điểm bắt đầu phù hợp để đọc lỗi thủ công, nhưng không được tùy chỉnh quy tắc riêng cho hai tài liệu này.

| Nhãn quan hệ | Gold | RoBERTa TP / FP / FN | SecureBERT TP / FP / FN |
|---|---:|---|---|
| deliver | 1 | 0 / 0 / 1 | 0 / 0 / 1 |
| indicates | 2 | 0 / 0 / 2 | 0 / 0 / 2 |
| located-at | 1 | 0 / 0 / 1 | 0 / 0 / 1 |
| originates-from | 5 | 4 / 13 / 1 | 3 / 1 / 2 |
| targeted-by | 6 | 2 / 5 / 4 | 0 / 2 / 6 |
| targets | 12 | 7 / 40 / 5 | 3 / 32 / 9 |
| used-by | 4 | 2 / 20 / 2 | 2 / 14 / 2 |
| uses | 18 | 10 / 45 / 8 | 6 / 25 / 12 |

Nhãn targets và uses tạo 85/123 FP của RoBERTa và 57/74 FP của SecureBERT. Ba nhãn deliver, indicates, located-at chưa có TP, nhưng tổng cộng chỉ có 4 mẫu chuẩn nên chưa thể kết luận ổn định theo lớp.

## 7. Đối chiếu với chẩn đoán dùng thực thể chuẩn

Artifact cũ có phép đo riêng khi cung cấp thực thể chuẩn cho bộ dự đoán quan hệ. Đây là chẩn đoán khác với suy luận đầu–cuối; không được thay thế vào bảng baseline.

| Chẩn đoán đã lưu | RoBERTa | SecureBERT |
|---|---:|---:|
| F1 quan hệ khi dùng gold spans | 45,16% | 32,18% |
| TP / FP / FN với gold spans | 28 / 47 / 21 | 14 / 24 / 35 |
| Dự đoán đúng loại trên 49 cặp chuẩn | 37/49 | 37/49 |

Gold-span diagnostic cho thấy bộ dự đoán quan hệ vẫn còn lỗi ngay cả khi có thực thể chuẩn. Đây không phải cận trên lý thuyết, cũng không phải kết quả đạt được trong triển khai thực tế. Chênh lệch với pipeline đầu–cuối chỉ là bằng chứng chẩn đoán để thiết kế ablation.

## 8. Kế hoạch thí nghiệm tiếp theo

Trạng thái toàn bộ các thí nghiệm dưới đây: **đã thiết kế, chưa chạy**. Cần giữ baseline hiện tại bất biến; mỗi thay đổi là cấu hình và run ID mới, lưu đủ provenance. Chưa mở test để lựa chọn mô hình.

| Ưu tiên | Thí nghiệm đề xuất | Biến thay đổi và đối chứng | Kết quả cần theo dõi |
|---|---|---|---|
| 1 | A1 — Lọc ứng viên thực thể | Tại cùng checkpoint, so sánh min_entity_score 0,05 hiện tại với 0,10 và 0,20; giữ cap 128, khoảng cách 96 và ngưỡng quan hệ 0,90. Chạy cho cả hai encoder. | Strict relation F1; FP đầu mút; primary entity F1/recall; gold-pair recall trước và sau lọc. |
| 2 | A2 — Ngân sách ứng viên | Tách riêng cap 128 hiện tại với 96 và 64, giữ min score 0,05 và các cấu hình khác. Không trộn A1+A2 trong vòng đầu. | FP do chồng lấn; số cặp cần chấm; số quan hệ chuẩn mất thêm; thời gian suy luận. |
| 3 | B1 — Phân biệt quan hệ khó | Sau khi chốt thiết kế A, so sánh cách lấy mẫu âm hiện tại với mẫu âm khó lấy từ ứng viên dự đoán trên train. Giữ nguyên tỷ lệ 6 ở vòng đầu để cô lập cách chọn mẫu. | Precision/recall quan hệ; FP trên cặp đầu mút chuẩn; lỗi targets/uses; tác động lên nhãn ít mẫu. |
| 4 | C1 — Lặp nhiều seed | Kiểm tra baseline và cấu hình triển vọng trên cùng danh sách seed định trước, đề xuất 42/43/44; giữ fold và dữ liệu cố định. | Trung bình và độ lệch chuẩn F1; chênh lệch ghép cặp; độ ổn định lỗi. Seed 42 baseline đã có, không cần chạy lại vô cớ. |

A1/A2 có thể thực hiện như ablation suy luận trên checkpoint đã lưu, không nhất thiết huấn luyện lại. Nếu lọc trực tiếp artifact, phải đối chiếu một lần với decoder/inference để bảo đảm logic tương đương; không được gọi kết quả mô phỏng là một lượt suy luận đã xác minh khi chưa làm bước đó. B1 cần triển khai và huấn luyện mới, bao gồm lưu đủ thông tin proposal để phân rã lỗi rõ hơn.

Tiêu chí quyết định được đề xuất trước khi chạy: báo cáo đầy đủ mọi cấu hình; chỉ số chính là strict relation micro-F1, entity F1 dùng phân xử hòa theo quy tắc hiện có; kèm precision, recall và số cặp chuẩn còn lại. Mọi giảm recall phải trình bày rõ. Một cấu hình tốt hơn trên validation nhỏ chỉ được coi là tín hiệu để kiểm tra thêm, chưa phải cải thiện đã xác lập. Không tìm kiếm lặp vô hạn trên 5 tài liệu này; chốt phạm vi A1/A2, sau đó đánh giá tính ổn định trước khi xin mở giai đoạn full/test.

Không ưu tiên tăng khoảng cách ở vòng đầu vì chưa quan sát mất cặp chuẩn do khoảng cách. Không ưu tiên ensemble phép hợp đơn giản vì SecureBERT chưa bổ sung TP mới trong cặp kết quả hiện tại. Chưa ưu tiên thay schema chỉ để loại FP, vì cần kiểm tra độ bao phủ nhãn và đối chứng riêng.

## 9. Khả năng tái lập và giới hạn

Đã xác minh SHA-256 của 105 tệp có trong gói đầu vào. Một checkpoint bên ngoài được liệt kê trong manifest nhưng không có trong ZIP; phân tích này không cần tải hoặc nạp checkpoint đó. Đã giải mã lại 22.458 cặp ứng viên RoBERTa và 34.136 cặp SecureBERT; tập quan hệ dự đoán và xác suất khớp artifact với sai số cho phép 1e-12. Bộ scorer chính thức của dự án cho kết quả trùng hoàn toàn các chỉ số đã lưu. Các tổng theo tài liệu, nhãn, loại lỗi và đối chiếu 49 quan hệ đều khớp.

Mã phân tích chỉ dùng thư viện chuẩn Python; không huấn luyện, không chọn threshold mới, không đọc prediction test. Dataset gộp được mở để xác minh hash và ánh xạ, nhưng chỉ các hàng thuộc 5 tài liệu validation đóng góp vào phân tích. Token index là chỉ số native toàn cục theo từng encoder, cuối span tính inclusive; tọa độ ký tự gold có điểm cuối exclusive.

Các kết quả chỉ dựa trên một fold, một seed, 5 tài liệu validation. Chưa kiểm định độ ổn định, ý nghĩa thống kê hoặc khả năng khái quát. Các nhóm FP/FN là mô tả theo gold và thứ tự xử lý; thiếu raw entity proposal logits nên không gán riêng mọi lỗi cho pruning hoặc classifier. Không tuyên bố đã hoàn thiện bài báo Q2, các chuyên đề hoặc luận án từ bước phân tích này.

Tệp kèm theo: validation_error_analysis.json; roberta_error_inventory.jsonl; securebert_error_inventory.jsonl; paired_gold_relations.json; analyze_validation.py; verify_analysis.py; verification.txt. Nguồn là gói CTI_Q2_Matched70_Ket_qua_20261007.zip phiên bản 2; các SHA-256 đầu vào nằm trong JSON kết quả.
