# CTI Q 2 — Kết quả chẩn đoán D0

Ngày 22/10/2026. Trạng thái: hoàn tất ma trận D0 trong lượt chạy cuối;12/12 cấu hình hợp lệ. Đây là kết quả validation, không phải đánh giá độc lập hoặc cải tiến mô hình.

## Kết quả chính

D0 đã phân biệt được nguyên nhân của 90 lượt đầu mút không khớp nhãn chuẩn trong nhóm 22 quan hệ sai bền vững:52 lượt đúng biên nhưng sai kiểu,29 lượt bị loại bởi ngưỡng thực thể và 9 lượt bị loại bởi cap. Không có trường hợp thuộc nhóm ngoài miền đoạn, vượt độ rộng hoặc thiếu đề xuất trong nhóm quan sát này. Kết quả củng cố ưu tiên nghiên cứu bất định kiểu thực thể, đồng thời cho thấy tăng cap đơn thuần không giải quyết phần lớn lỗi.

## Phạm vi và kiểm chứng

- Sáu checkpoint đã chọn: RoBERTa/SecureBERT × seed 42/43/44; cap 128 và 96; fold 1; năm tài liệu 365/389/395/400/420. Giữ nguyên 120 thực thể chính,49 quan hệ chuẩn, độ rộng 6, ngưỡng thực thể 0,05, khoảng cách 96, ngưỡng quan hệ 0,90. Không huấn luyện lại, sửa gold, chạy full/test hoặc merge.
- SHA-256 của sáu checkpoint,167 tệp nguồn trong gói ba seed và cấu hình encoder đã được đối chiếu. Trọng số được nạp strict, mọi tensor khớp checkpoint.
- 12/12 cấu hình: bật/tắt chẩn đoán cho toàn bộ WindowInference giống hệt nhau, gồm thứ tự và logits quan hệ; trọng số và RNG không đổi.
- 12/12 cấu hình: định danh/thứ tự thực thể, quan hệ và xác suất xuất khớp chính xác bản dự đoán lưu trước. Sai lệch xác suất lớn nhất của lượt cuối bằng 0. Chấm lại các tệp dự đoán bằng scorer chính thức cho cùng TP/FP/FN và F 1. Điều này không khẳng định logits trung gian của mọi cặp trùng toàn bộ logits lịch sử.
- Đã kiểm tra hết 24 luồng gzip mới:97.848 đoạn đề xuất và 292.272 cặp quan hệ trên toàn bộ ma trận. Ghi đủ NONE +15 kiểu, biên token/ký tự, mask, thứ hạng, lý do giữ/loại và cặp được sinh.
- Bộ kiểm thử sau sửa mã:257 kiểm thử và 65 subtests vượt qua. Các kiểm thử mới đã được quan sát thất bại trước khi bổ sung chức năng.

## Phân rã đầu mút của 22 quan hệ tại cap 128

| Nguyên nhân | RoBERTa | SecureBERT | Tổng lượt |
|---|---:|---:|---:|
| Đúng biên và đúng kiểu, được giữ |88|86|174|
| Đúng biên, được giữ nhưng sai kiểu |26|26|52|
| Bị loại bởi ngưỡng thực thể |18|11|29|
| Vượt cap |0|9|9|
| Tổng |132|132|264|

Quy tắc quy nguyên nhân đi theo thứ tự thực thi: miền đoạn → độ rộng → thiếu đề xuất → ngưỡng → cap → kiểu. Một đoạn bị loại vì cap được tính vào cap, kể cả khi kiểu top 1 cũng sai. Vì vậy các nhóm loại trừ nhau và không được cộng thêm lỗi kiểu của đoạn đã loại.

Có 34 đề cập đầu mút duy nhất theo định danh/ký tự, tương ứng 68 khóa token khi tách theo hai encoder.264 là số lượt quan sát lặp qua quan hệ, đầu mút và seed, không phải 264 mẫu độc lập.

Các giai đoạn lỗi quan hệ vẫn khớp báo cáo trước:81/132 lượt mất đầu mút,39/132 dưới ngưỡng quan hệ và 12/132 sai loại. Tổng hợp cả hai cap khớp đủ 264 kết quả quan hệ của hồ sơ cũ.

## Chẩn đoán top 2

Trong 264 lượt đầu mút cap 128, kiểu chuẩn nằm trong top 2 ở đúng biên đề xuất trong 242 lượt; nếu chỉ xét biên còn được giữ thì còn 210 lượt. So với 174 lượt top 1 đúng, có thêm 36 lượt mà kiểu chuẩn đứng thứ hai ở biên được giữ:17 lượt RoBERTa và 19 lượt SecureBERT. Đây là 36/52 lượt nhầm kiểu ở biên được giữ.

Đây là độ bao phủ hồi cứu có dùng gold tại đúng biên, không phải recall/F 1 của hệ thống triển khai, không chứng minh khôi phục 36 quan hệ, và chưa tính số FP khi mở rộng kiểu. Mọi phương án E1 cần quy tắc giải mã trước khi xem gold và phải đo precision, recall, F 1 cùng chi phí cặp.

## Toàn bộ 120 thực thể chính của từng cấu hình

“Giữ đúng” trong bảng là số đề cập chuẩn có đúng biên và kiểu sau lọc, không phải số toàn bộ thực thể dự đoán hoặc precision.

| Encoder | Seed | Cap | Giữ đúng | Sai kiểu | Ngưỡng | Cap loại | Kiểu gold trong top 2 ở biên được giữ |
|---|---:|---:|---:|---:|---:|---:|---:|
|roberta|42|128|103|12|5|0|109|
|roberta|42|96|103|12|5|0|109|
|roberta|43|128|97|16|7|0|105|
|roberta|43|96|97|16|7|0|105|
|roberta|44|128|100|15|5|0|109|
|roberta|44|96|100|12|5|3|109|
|securebert|42|128|100|8|7|5|105|
|securebert|42|96|100|8|7|5|105|
|securebert|43|128|105|9|5|1|111|
|securebert|43|96|103|8|5|4|109|
|securebert|44|128|102|10|1|7|111|
|securebert|44|96|100|10|1|9|109|

Cap 96 không làm thay đổi số giữ đúng của RoBERTa, nhưng làm giảm hai đề cập đúng ở mỗi seed 43 và 44 của SecureBERT. Một số lỗi chuyển nhóm do thứ tự lọc: ví dụ RoBERTa 44 chuyển ba trường hợp từ sai kiểu sang cap, trong khi số giữ đúng vẫn 100. Không được diễn giải sự giảm số lỗi kiểu này là mô hình tốt hơn.

## Sai lệch và gián đoạn của các lượt trước

Các lượt tạm trước lần khôi phục cuối có sai lệch số học. SecureBERT 42 từng có sai lệch điểm thực thể lớn nhất khoảng 3,21× 10⁻⁵. SecureBERT 43 cap 96 từng thay ba ứng viên kiểu identity ở tài liệu 400, trong khi tập quan hệ dự đoán không đổi; điều kiện đối chiếu đã chặn lượt đó. Những quan sát này được ghi nhận trong nhật ký phiên, nhưng phần tệp chưa lưu đã mất khi phiên khởi tạo lại. Chúng không được nhập vào ma trận cuối hoặc coi là một thí nghiệm cải tiến.

Bản lưu tiến độ D0 phiên bản 0 còn có luồng logits RoBERTa 42 cap 96 bị cắt và không khớp SHA trong verification; kiểm tra sau đó cũng phát hiện luồng RoBERTa 43 cap 128 bị cắt. Mã xuất đã được sửa để nén hoàn chỉnh, ghi tệp tạm, đồng bộ, đọc kiểm tra rồi thay tên. Toàn bộ ma trận cuối được chạy mới từ các checkpoint gốc, đọc hết gzip và đối chiếu số bản ghi. Không sửa hash để hợp thức hóa tệp lỗi, không nối hoặc suy đoán phần đuôi thiếu.

Lượt cuối dùng Python 3.12.14, torch 2.14.1+cpu, transformers 5.18.0, CPU 4 luồng, SDPA; thông tin build nằm trong verification.json. Lượt này khớp chính xác mọi xác suất xuất lịch sử. Chưa có thí nghiệm cô lập đủ để quy nguyên nhân sai lệch của lượt trước cho phần cứng, thư viện hoặc lưu trữ; không tuyên bố đã tìm được nguyên nhân gốc.

Luồng SecureBERT 42 cap 128 trong gói ba seed lịch sử bị cắt vẫn được giữ nguyên như erratum trước. D0 tạo một luồng mới đầy đủ 34.136 cặp, không khôi phục hay thay thế âm thầm tệp lịch sử.

## Việc đã hoàn tất và bước tiếp theo

Đã hoàn tất: mã ghi chẩn đoán không đổi suy luận; kiểm tra cấu hình và định danh tệp; xuất tệp có kiểm chứng;12 lượt D0 cuối; phân rã 1.440 lượt thực thể chính,528 lượt đầu mút và 264 lượt quan hệ của nhóm 22; báo cáo cùng gói dữ liệu tái lập.

Bước tiếp theo: (1) rà soát độc lập nhãn của 22 hồ sơ, đặc biệt R17 và R22; (2) thiết kế E1 bảo toàn bất định kiểu trên biên đã giữ, khóa quy tắc phá hòa/gộp và chi phí trước chạy; (3) khảo sát độ ổn định số học của cap 96 bằng lặp lại có khóa môi trường trước khi dùng làm đối chứng khẳng định; (4) chuẩn bị tập đánh giá độc lập theo tài liệu. E1/E2, phân xử nhãn bởi con người và đánh giá ngoài chưa được thực hiện.

Rà soát mã độc lập trước sửa phát hiện ba vấn đề quan trọng: tệp bị cắt, thiếu ràng buộc cấu hình suy luận và thiếu kiểm tra định danh giữa thư mục/metadata. Hai ràng buộc đã được bổ sung và có kiểm thử; tệp lỗi được thay bằng lượt chạy mới đã kiểm chứng. Một góp ý nhỏ còn để lại: analyzer đối chiếu danh sách cặp với logits xuất nhưng chưa tự sinh lại toàn bộ cặp/khoảng cách bằng một đường độc lập. Đây là giới hạn của kiểm toán tính đầy đủ cặp, không phải phát hiện cặp bị thiếu trong lượt cuối.

## Đoạn có thể bổ sung vào thảo luận chuyên đề 2

Kết quả D0 cho thấy lỗi biểu diễn đầu mút cần được phân tích đồng thời ở cấp lựa chọn đoạn và cấp kiểu thực thể. Trong nhóm 22 quan hệ không được dự đoán đúng ở cả sáu đối chứng cap 128,52/90 lượt đầu mút không khớp vẫn giữ đúng biên nhưng chọn sai kiểu;29 lượt bị loại bởi ngưỡng và 9 lượt vượt ngân sách ứng viên. Việc giữ thêm kiểu có cơ sở chẩn đoán: ở 36 lượt nhầm kiểu, nhãn chuẩn nằm ở vị trí thứ hai tại biên còn được giữ. Tuy nhiên, đây là quan sát hồi cứu trên tập validation đã dùng cho lựa chọn mô hình, không phải bằng chứng cải thiện hiệu quả trích xuất. Hướng nghiên cứu tiếp theo cần kiểm tra phương án giữ bất định kiểu bằng giải mã không sử dụng gold và đánh giá đồng thời khả năng khôi phục quan hệ, số dương tính giả, chi phí tính toán và độ ổn định qua seed.
