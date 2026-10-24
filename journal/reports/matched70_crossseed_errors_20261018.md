# CTI Q2 Báo cáo phân tích lỗi qua ba seed

Ngày hoàn tất và tái kiểm chứng: 18/10/2026. Phạm vi: matched70, fold 1, validation. Bản này hoàn tất phần phân tích và cập nhật tài liệu bị gián đoạn ngày 15/10; không phát sinh huấn luyện hoặc suy luận mô hình mới.

## Kết quả chính

Có 22/49 quan hệ chuẩn chưa được dự đoán đúng trong bất kỳ lượt nào của sáu baseline (hai encoder × ba seed). Trong đó, 16 quan hệ thuộc hai tài liệu 365 và 389. Đây là nhóm ưu tiên rà soát ngữ cảnh, biên thực thể và hướng gán nhãn. Không được tự sửa gold chỉ vì mô hình dự đoán khác.

Giảm ngân sách đoạn từ 128 xuống 96 chỉ cải thiện relation F1 tại seed 42 của mỗi encoder. Nó không tạo thêm quan hệ đúng ở bất kỳ seed nào. SecureBERT mất thêm bốn cặp gold có thể chấm ở seed 43 và một cặp ở seed 44 dù F1 cuối không đổi.

## Công việc đã hoàn thành

- Khôi phục hai bộ kết quả đã lưu; xác minh 167 và 105 mục SHA-256 nội bộ tương ứng, cùng CRC của hai ZIP. Các trọng số được dẫn chiếu bên ngoài không được kiểm lại trong lần phân tích này.
- Chấm lại 12 tệp dự đoán bằng bộ chấm nghiêm ngặt của dự án; kiểm tra số liệu trùng các báo cáo đã công bố trong PR nháp.
- Giải mã logits của 12 trường hợp và đối chiếu chính xác tập dự đoán cùng xác suất; một trường hợp dùng logits baseline gốc theo ngoại lệ mô tả dưới đây.
- Đồng nhất 49 quan hệ gold bằng định danh đầu mút và biên ký tự giữa hai tokenizer; lập 588 kết quả quan hệ, 60 hàng theo tài liệu và 780 hàng theo nhãn–tài liệu.
- Tổng hợp lỗi lặp lại, tính bổ sung giữa hai encoder, và các chuyển trạng thái lỗi khi đổi cap.
- Cập nhật chuyên đề 2 và soạn phần Results and Discussion tiếng Anh. Các tổng quan/chuyên đề khác không được tính là hoàn thành trong mốc này.

## Thiết kế so sánh

Corpus có 52 tài liệu, 70 cửa sổ chung theo ký tự; 49 cửa sổ train, 5 validation và 16 test được giữ theo manifest. Không chạy test. Năm tài liệu validation là 365, 389, 395, 400 và 420, gồm 120 thực thể chính và 49 quan hệ gold. Ba seed huấn luyện là 42, 43, 44; seed chia dữ liệu là 11800.

Mỗi baseline được huấn luyện 12 epoch. Checkpoint được chọn theo relation F1, dùng primary entity F1 để phá hòa. Epoch chọn của RoBERTa là 12/10/12; SecureBERT là 8/11/10. Ngưỡng baseline được chọn từ lưới validation và thực tế đều bằng 0,90. Cap 96 dùng lại checkpoint/ngưỡng tương ứng; không huấn luyện riêng hoặc chọn lại ngưỡng.

## Kết quả tổng hợp

Đơn vị phần trăm; trung bình ± độ lệch chuẩn mẫu qua ba seed. Các lần chạy dùng cùng năm tài liệu validation.

| Mô hình | Cap | Entity F1 (%) | Relation F1 (%) |
|---|---:|---:|---:|
| roberta | 128 | 40.276 ± 1.812 | 29.916 ± 3.973 |
| roberta | 96 | 44.328 ± 1.210 | 30.136 ± 3.597 |
| securebert | 128 | 31.298 ± 1.142 | 24.271 ± 4.253 |
| securebert | 96 | 36.116 ± 1.228 | 24.529 ± 3.914 |

Chênh lệch relation F1 trung bình khi giảm cap là +0,220 điểm phần trăm với RoBERTa và +0,258 với SecureBERT. Đây là thay đổi nhỏ, chỉ xuất hiện ở seed 42. Không diễn giải thành cải thiện có ý nghĩa thống kê, khả năng khái quát hoặc tăng tốc suy luận.

## Phân rã 49 quan hệ gold tại cap 128

| Mô hình | Seed | Đúng | Mất đầu mút | Không có cặp | Dưới ngưỡng | Sai loại |
|---|---:|---:|---:|---:|---:|---:|
| roberta | 42 | 25 | 12 | 0 | 10 | 2 |
| roberta | 43 | 15 | 17 | 0 | 15 | 2 |
| roberta | 44 | 20 | 14 | 0 | 13 | 2 |
| securebert | 42 | 14 | 19 | 0 | 14 | 2 |
| securebert | 43 | 14 | 15 | 0 | 18 | 2 |
| securebert | 44 | 15 | 18 | 0 | 14 | 2 |

Thứ tự phân loại lỗi: dự đoán đúng; mất ít nhất một đầu mút đúng biên và kiểu; đầu mút còn nhưng cặp không được chấm; điểm tồn tại dưới ngưỡng; vượt ngưỡng nhưng sai loại. Mỗi quan hệ thuộc đúng một nhóm. Mất đầu mút ở đây là quan sát sau lọc ứng viên, không tự phân biệt được lỗi đề xuất, nhầm kiểu và loại bởi cap.

## Lỗi lặp lại và tính bổ sung

| Encoder | Đúng 0/3 seed | Đúng 1/3 | Đúng 2/3 | Đúng 3/3 |
|---|---:|---:|---:|---:|
| RoBERTa | 22 | 6 | 9 | 12 |
| SecureBERT | 30 | 6 | 2 | 11 |

Phân bố trên không đổi khi chuyển sang cap 96. Trong 22 quan hệ không encoder nào dự đoán đúng qua sáu baseline, số lượng theo tài liệu là 365: 8; 389: 8; 400: 4; 395: 1; 420: 1. Theo nhãn: uses 8; targets 5; targeted-by 4; indicates 2; deliver 1; located-at 1; originates-from 1. Hai tài liệu 365/389 chiếm 72,73% nhóm lỗi này, không phải 72,73% tổng quan hệ.

| Seed | Cả hai đúng | Chỉ RoBERTa đúng | Chỉ SecureBERT đúng | Cả hai sai |
|---|---:|---:|---:|---:|
| 42 | 14 | 11 | 0 | 24 |
| 43 | 11 | 4 | 3 | 31 |
| 44 | 14 | 6 | 1 | 28 |

Nhận định trước đây “SecureBERT không bổ sung TP” chỉ đúng cho seed 42. Ở seed 43 và 44, SecureBERT có lần lượt 3 và 1 TP riêng trong cặp cùng seed. Tuy nhiên, hợp các TP SecureBERT qua cả ba seed vẫn nằm trong hợp TP RoBERTa. Bảng dùng nhãn gold để phân tích, chưa phải đánh giá một ensemble có thể triển khai; chưa có precision/F1 của ensemble.

## Dương tính giả và tác động của cap

Trong ba baseline RoBERTa, 137/207 lượt FP có ít nhất một đầu mút không khớp gold về biên hoặc kiểu (66,18%). SecureBERT có 114/170 (67,06%). Đây là tổng lượt FP qua seed, không phải số lỗi độc lập. Có 13 FP RoBERTa và 7 FP SecureBERT lặp nguyên định danh trong cả ba seed; các số này không đổi ở cap 96.

Cap 96 loại đúng 5 FP ở seed 42 của từng encoder, đều thuộc nhóm đầu mút không khớp gold. Tập TP không đổi ở cả sáu cặp cap. Với SecureBERT, 4 quan hệ seed 43 và 1 quan hệ seed 44 chuyển từ dưới ngưỡng sang mất đầu mút: F1 bất biến che khuất suy giảm độ phủ. “Cặp chưa được gán nhãn” chỉ mô tả quan hệ không có trong gold; chưa đủ bằng chứng khẳng định nhãn nguồn thiếu hay mô hình đúng.

## Đính chính tính toàn vẹn logits

Tệp fresh-forward SecureBERT seed 42 cap 128 trong bộ ba seed bị cắt: đọc được 31.527 dòng JSON hoàn chỉnh trong 34.136 dòng dự kiến; còn thiếu 2.609 bản ghi hoàn chỉnh. SHA của một tệp lưu trữ khớp không chứng minh luồng gzip bên trong kết thúc hợp lệ.

Phân tích sử dụng logits của baseline dev gốc trong bộ matched70 ngày 07/10, phiên bản 2. SHA-256: `0cb2ca36a2195b1103abbe5e33e04480379293419066b5c6e32b3df115517467`. Tất cả 31.527 bản ghi fresh-forward còn đọc được trùng chính xác theo khóa cặp với nguồn này. Giải mã đầy đủ 34.136 bản ghi gốc tái tạo đúng tập dự đoán, xác suất và metrics đã lưu. Mười một tệp fresh-forward còn lại được đọc đầy đủ và kiểm chứng.

Đây là thay nguồn bằng chứng có khai báo cho một trường hợp, không phải đã phục hồi đuôi gzip hoặc chạy forward mới. Số liệu dự đoán/F1 không thay đổi. Tệp hỏng được giữ nguyên trong gói nguồn để truy vết.

## Giới hạn và công việc tiếp theo

588 dòng quan hệ là 49 quan hệ được quan sát lặp qua 12 cấu hình, không phải 588 quan hệ độc lập. Ba seed mô tả biến thiên tối ưu trên cùng năm tài liệu; không đo được bất định lấy mẫu văn bản. Validation đã tham gia chọn checkpoint, ngưỡng và khảo sát cap. Không dùng kết quả này làm ước lượng test độc lập hoặc xếp hạng encoder tổng quát.

1. Rà soát định tính 22 quan hệ khó theo văn bản gốc, hướng, biên và tính đầy đủ của gán nhãn; lưu phiếu nhận xét, không thay gold phục vụ nâng điểm.
2. Thiết kế đối chứng giữ độ phủ đầu mút và lấy mẫu âm khó; đăng ký trước thay đổi, số seed và chỉ số, sau đó mới triển khai huấn luyện khi phạm vi được mở.
3. Khóa cấu hình phát triển và thiết kế đánh giá độc lập theo tài liệu. Không dùng năm tài liệu hiện tại để xác nhận lại các lựa chọn đã tối ưu trên chúng.
4. Hoàn thiện ma trận hồ sơ cứng/xác suất, hiệu chỉnh và STIX ở các thí nghiệm riêng; chưa coi các tính năng này đã chứng minh lợi ích qua bảng encoder hiện tại.

Mã huấn luyện giữ commit `1e589a02bc77cef3af1dd27a09effce4e6c80f72`; mã chấm được lấy tại đầu nhánh báo cáo `4376c21963143e66720a0ea82fea4a02d6300fc5`. Lần này không huấn luyện mới, không đánh giá full/test và không merge. 248 kiểm thử hồi quy là kết quả đã ghi nhận ngày 13/10; kiểm chứng mới ngày 18/10 là chấm lại và giải mã dữ liệu đã lưu.
