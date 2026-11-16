# CTI Q2 Thiết kế thí nghiệm tiếp theo

Phiên bản thiết kế 0.1, ngày 19/10/2026. Cơ sở: so sánh matched70 ba seed và rà soát 22 quan hệ sai bền vững. Đây là thiết kế để triển khai giai đoạn tiếp theo, chưa phải kết quả thí nghiệm mới.

## Quyết định ưu tiên

Ưu tiên kiểm toán nhãn và đo hậu nghiệm kiểu thực thể trước khi tăng cap hoặc thay loss. Trong 132 lượt quan hệ của nhóm 22 tại cap128, có 81 lượt mất đầu mút, 39 lượt dưới ngưỡng và 12 lượt sai loại quan hệ. Trong 90 lượt đầu mút không khớp gold, 52 lượt vẫn giữ đúng biên nhưng sai kiểu (57,78%). Các lượt dùng lại một đề cập trong nhiều quan hệ và nhiều seed nên không độc lập.

Hai kết quả sai loại bền vững cần rà soát nguồn: R17 (TA459–Russia) và R22 (Gallmaker–Microsoft). R17 mâu thuẫn với vai trò Russia trong câu; R22 có direction=left trong nguồn, còn phép so sánh đóng băng dùng from_id→to_id. Các nghi vấn khác liên quan tới biên và mức chi tiết của nhãn. Chưa có kết luận phân xử cuối cùng và không thay đổi gold v1 để nâng điểm.

## 1. Phân xử nhãn có truy vết

Đầu vào là 22 hồ sơ trong `persistent22_review.json`, mỗi hồ sơ gồm định danh đề cập, biên ký tự, văn bản nguồn và trường hướng. Người gán nhãn đọc toàn bộ tài liệu liên quan cùng hướng dẫn hiện hành. Nhận xét máy là gợi ý kiểm tra, không phải đáp án chuẩn.

Để giảm thiên lệch, vòng đầu cung cấp văn bản và nhãn nguồn nhưng ẩn dự đoán, điểm số và tên encoder. Hai người gán nhãn đưa nhận định riêng về biên thực thể, kiểu, đầu mút quan hệ, hướng và nhãn. Một người phân xử bất đồng; nếu nguồn không đủ, giữ trạng thái chưa giải quyết. Chưa có vòng phân xử này được thực hiện trong báo cáo.

Mỗi quyết định cần: định danh hồ sơ, phiên bản hướng dẫn, người rà soát, quyết định giữ/sửa/chưa giải quyết, trích đoạn hỗ trợ, lý do, đề xuất thay đổi cụ thể, ngày và người phân xử. Mẫu JSONL đi kèm đã điền định danh và để trạng thái chờ; không tự điền tên người hoặc quyết định thay họ.

Nếu chốt sửa nhãn, tạo phiên bản corpus mới và báo cáo diff riêng. Giữ nguyên kết quả gold v1 và chạy lại mọi đối chứng trên cùng gold v2 khi được mở phạm vi. Không so mô hình cũ chấm bằng v1 với mô hình mới chấm bằng v2. Nhóm 22 là mẫu chọn theo lỗi, không dùng để ước lượng tỷ lệ lỗi nhãn toàn corpus; cần một mẫu ngẫu nhiên riêng để kiểm tra chất lượng tổng thể.

## 2. D0 Đo nguyên nhân mất đầu mút trên checkpoint đã chọn

Mục tiêu là phân biệt không đề xuất được đoạn, mất đoạn do lọc, nhầm kiểu và mất cặp do chính sách khoảng cách. Chạy lại suy luận dev có ghi chẩn đoán chỉ sau khi checkpoint/pretrained package và cấu hình được đối chiếu đúng SHA. Không huấn luyện ở bước này.

| Thành phần | Thiết kế cố định |
|---|---|
| Dữ liệu | matched70 gold v1; fold1; năm tài liệu validation hiện tại |
| Encoder và seed | RoBERTa/SecureBERT × 42/43/44 |
| Cap | 128 là chính; 96 là đối chứng đã có |
| Checkpoint và ngưỡng | Giữ đúng baseline-selected checkpoint; ngưỡng 0,90 |
| Mức ghi | Mọi đoạn hợp lệ trước lọc và các đoạn sau lọc, không chỉ đoạn có trong gold |
| Trường cần ghi | Biên token/ký tự, độ rộng, trạng thái mask, p(NONE), phân phối đầy đủ 15 kiểu, thứ hạng, lý do giữ/loại, cặp được sinh và khoảng cách |
| Kiểm chứng bắt buộc | Bật chẩn đoán không đổi dự đoán, xác suất xuất, TP/FP/FN hoặc thứ tự giải mã |

Phân loại nguyên nhân theo chuỗi loại trừ lẫn nhau: không nằm trong miền đoạn hợp lệ; bị loại bởi giới hạn độ rộng/mask; bị loại bởi ngưỡng; vượt cap; đúng biên nhưng top1 sai kiểu; không sinh cặp; dưới ngưỡng quan hệ; sai loại. Nếu thứ tự triển khai lọc khác, định nghĩa nhóm phải đi theo thứ tự thực tế và được khóa trước tổng hợp.

Đo cả top1 typed recall và tỷ lệ kiểu gold nằm trong top2 của phân phối tại đúng biên. Top2 là chỉ số chẩn đoán có sử dụng gold, chưa phải chất lượng hệ thống có thể triển khai. Không dùng nó để khẳng định khôi phục TP quan hệ. Ghi số đề cập duy nhất và số lượt đầu mút riêng để tránh đếm một đề cập nhiều lần như nhiều mẫu độc lập.

## 3. E1 Can thiệp kiểu thực thể có đối chứng

Chỉ chọn E1 sau D0 và vòng rà soát nhãn. Giả thuyết: bảo toàn bất định kiểu có thể giảm lỗi top1 ở đúng biên; tăng cap đơn thuần không trực tiếp sửa những trường hợp này.

Phương án chính để đánh giá khả thi là giữ tối đa hai kiểu ứng viên cho mỗi biên đã chọn, kèm xác suất và quy tắc loại NONE cố định. Đây là thay đổi thiết kế, chưa triển khai. Cần xác định rõ phần nào của head quan hệ dùng embedding kiểu, cách giải mã một nhãn cuối cùng và xử lý trùng; không được đổi nhãn đầu mút sau khi xem gold. Nếu kiến trúc chưa hỗ trợ đường này, dừng ở D0 để thiết kế cụ thể trước khi viết mã.

Đối chứng gồm baseline top1 và E1 top2 trên cùng checkpoint, cap biên 128, cùng ngưỡng quan hệ. Khóa quy tắc phá hòa và gộp dự đoán trước chạy. Báo cáo chi phí tăng số cặp cùng precision, recall và F1; mức tăng recall không được che giấu mức tăng FP. Chỉ khi cần huấn luyện lại mới mở một nhánh thí nghiệm riêng, không gộp kết quả suy luận và huấn luyện.

Tiêu chí tiếp tục: cải thiện quan hệ đúng hoặc giảm lỗi typed endpoint trong các cặp cùng seed, không chỉ tăng chỉ số oracle; đồng thời công bố FP và chi phí. Nếu chỉ đổi nhãn theo gold mới sửa được lỗi, không coi là thành công của E1. Trên validation hiện tại, mọi kết quả vẫn là khám phá.

## 4. E2 Lấy mẫu âm khó trong huấn luyện

E2 là nhánh độc lập sau kiểm toán, không chạy đồng thời với E1 ở lần so sánh đầu. Giữ kiến trúc, encoder, corpus, loss, cap và tổng ngân sách cặp âm như baseline; chỉ thay cách chọn cặp âm train. Tập âm khó có thể ưu tiên cặp gần biên hoặc thực thể thường bị nhầm, nhưng quy tắc phải được định nghĩa từ train và prediction ngoài mẫu của train, không từ 22 lỗi validation.

Khóa tỷ lệ âm khó và phần âm ngẫu nhiên, quy tắc không chọn nhầm quan hệ dương, số seed và ngân sách epoch trước khi chạy. Chưa ấn định tỷ lệ khi chưa đo số cặp đủ điều kiện; đó là tham số cần quyết định bằng một lượt kiểm kê train, không tìm kiếm liên tục trên validation. Ghi đủ lịch sử mọi lượt, kể cả suy giảm. Không đồng thời thay trọng số loss hoặc tăng ngân sách ứng viên rồi quy lợi ích cho lấy mẫu âm.

## 5. Đánh giá độc lập theo tài liệu

Năm tài liệu validation đã dùng để chọn checkpoint, ngưỡng, cap và ưu tiên thiết kế nên không phải tập xác nhận độc lập. Phần test của corpus lịch sử từng tham gia các nghiên cứu nguyên mẫu cũng không tự trở thành tập ngoài mới chỉ vì chưa chạy matched70 trên nó.

Thiết kế tập ngoài cần đóng băng nguồn, quy tắc chọn, loại trừ trùng và hướng dẫn gán nhãn trước khi xem kết quả mô hình. Chọn theo tài liệu, nhóm các bản sao/gần trùng và các biến thể cùng báo cáo vào cùng phần. Ghi nguồn phát hành, ngày, loại báo cáo, ngôn ngữ và nhóm chiến dịch nếu có. Kiểm tra trùng với toàn bộ dữ liệu phát triển, không chỉ train của fold1.

Ngân sách ban đầu có thể đặt mục tiêu 100 tài liệu mới đủ điều kiện, phân tầng theo nhóm nguồn và loại báo cáo. Đây là mục tiêu vận hành, không phải khẳng định đủ lực thống kê; cần tính lại cỡ mẫu dựa trên độ hỗ trợ nhãn và hiệu ứng tối thiểu quan tâm trước xác nhận. Nếu không đủ nguồn/nhãn, thu hẹp tuyên bố và báo cáo rõ mức hỗ trợ thay vì bù bằng chia một tài liệu thành nhiều mẫu độc lập.

Chỉ số chính: micro F1 quan hệ nghiêm ngặt tính từ tổng TP/FP/FN. Chỉ số phụ: primary entity F1, recall cặp gold còn khả thi, precision/recall theo nhãn với số hỗ trợ, số FP trên tài liệu và chi phí suy luận. Bootstrap ghép cặp ở cấp tài liệu hoặc nhóm gần trùng, tính lại F1 từ số đếm cho mỗi mẫu; ba seed phải được trình bày như biến thiên tối ưu, không coi mỗi quan hệ–seed là mẫu độc lập.

Đóng băng các mô hình so sánh trước khi mở nhãn tập xác nhận. Nếu chỉnh tiếp mô hình sau xem tập này, kết quả trở lại trạng thái phát triển và cần một tập xác nhận mới. Chưa chạy full/test trong mốc hiện tại.

## Trạng thái bàn giao

Đã hoàn tất: hồ sơ 22 quan hệ, phân rã lỗi trên 264 kết quả đã lưu, xác minh nguồn/biên và thiết kế D0–E1–E2 cùng đánh giá độc lập. Chưa hoàn tất: phân xử nhãn bởi người độc lập, instrument D0, chạy forward D0, triển khai E1/E2, huấn luyện mới hoặc đánh giá ngoài. PR nghiên cứu tiếp tục ở trạng thái nháp.
