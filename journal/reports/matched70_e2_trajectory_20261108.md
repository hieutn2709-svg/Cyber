# CTI Q2 — E2: Xung đột nhãn trong quá trình huấn luyện

Báo cáo cập nhật ngày 08/11/2026. Hoàn thành hai lần chạy dev, mỗi lần 12 epoch.

## Kết quả chính

| Mô hình | Lượt nhóm xung đột eligible | Lượt nhóm xung đột sampled |
|---|---:|---:|
| RoBERTa | 813 | 3 |
| SecureBERT | 837 | 11 |

Xung đột đã được ghi nhận trong quá trình huấn luyện. Các số trên là lượt nhóm cộng qua cửa sổ và epoch, không phải số quan hệ độc lập; chưa có bằng chứng nhân quả rằng xung đột làm giảm F1.

## Phạm vi thực nghiệm

Hai mô hình khởi tạo lại từ pretrained đã khóa, dùng fold 1, seed 42, 36 tài liệu/49 cửa sổ train và 5 tài liệu/5 cửa sổ validation. Giữ nguyên corpus, gold, sampler, loss, kiến trúc, optimizer và quy tắc chọn checkpoint/ngưỡng. Không thực hiện full/test. SecureBERT là lần chạy hoàn tất ngày 28/10, được khôi phục từ gói đã lưu. RoBERTa là lần chạy mới ngày 06/11: điểm phục hồi epoch 6 ngày 01/11 yêu cầu glibc 2.39, trong khi môi trường hiện tại dùng glibc 2.41 dù cùng kernel 6.18.57. Không sửa identity; RoBERTa được khởi tạo từ pretrained. Các phiên RoBERTa trước đó, gồm lần hoàn tất cục bộ ngày 05/11 nhưng chưa lưu đủ, đã mất khi thư mục tạm bị dọn; tất cả được loại khỏi báo cáo này. Lịch sử của các lần bị gián đoạn không được ghép vào kết quả mới.

Nguồn thực thi: b5ede38697c507d145b52d75b977ddb4dd17f87a. CPU 4 luồng, interop 1; Torch 2.14.1+cpu, Transformers 5.18.0, NumPy 2.3.5. Chế độ dev chạy đủ 12 epoch, không dừng sớm. Giao thức được chốt trước thực nghiệm tại docs/superpowers/plans/2026-10-27-e2-training-trajectory.md trong repository Cyber.

## Định nghĩa và mẫu số

Một nhóm xung đột là một cặp biên thực thể có hướng nhận cả nhãn tồn tại quan hệ 1 và 0 trong cùng lần gọi sampler, qua những biến thể kiểu thực thể. Eligible chỉ toàn bộ tập đủ điều kiện; sampled chỉ batch thực sự dùng để tính loss. Các tổng dưới đây cộng theo cửa sổ và epoch, không phải số quan hệ độc lập. Tỷ lệ trên toàn bộ nhóm biên khác với tỷ lệ cặp dương chịu nhãn âm mâu thuẫn.

| Mô hình | Epoch | Nhóm eligible | Xung đột eligible | Nhóm sampled | Xung đột sampled | Cửa sổ có xung đột sampled |
|---|---:|---:|---:|---:|---:|---:|
| RoBERTa | 1 | 674411 | 80 | 2750 | 0 | 0/49 |
| RoBERTa | 2 | 573267 | 104 | 2750 | 0 | 0/49 |
| RoBERTa | 3 | 478807 | 88 | 2741 | 0 | 0/49 |
| RoBERTa | 4 | 445049 | 107 | 2733 | 0 | 0/49 |
| RoBERTa | 5 | 416795 | 91 | 2731 | 0 | 0/49 |
| RoBERTa | 6 | 405841 | 83 | 2726 | 0 | 0/49 |
| RoBERTa | 7 | 387743 | 59 | 2718 | 0 | 0/49 |
| RoBERTa | 8 | 371105 | 65 | 2713 | 1 | 1/49 |
| RoBERTa | 9 | 322567 | 44 | 2712 | 1 | 1/49 |
| RoBERTa | 10 | 304743 | 29 | 2712 | 0 | 0/49 |
| RoBERTa | 11 | 256803 | 36 | 2711 | 1 | 1/49 |
| RoBERTa | 12 | 187225 | 27 | 2714 | 0 | 0/49 |
| SecureBERT | 1 | 633421 | 74 | 2748 | 1 | 1/49 |
| SecureBERT | 2 | 625595 | 69 | 2747 | 1 | 1/49 |
| SecureBERT | 3 | 515285 | 51 | 2750 | 0 | 0/49 |
| SecureBERT | 4 | 522107 | 63 | 2748 | 1 | 1/49 |
| SecureBERT | 5 | 489771 | 79 | 2749 | 1 | 1/49 |
| SecureBERT | 6 | 446829 | 110 | 2745 | 0 | 0/49 |
| SecureBERT | 7 | 438751 | 104 | 2734 | 2 | 2/49 |
| SecureBERT | 8 | 423353 | 82 | 2733 | 3 | 3/49 |
| SecureBERT | 9 | 411409 | 68 | 2737 | 1 | 1/49 |
| SecureBERT | 10 | 386455 | 62 | 2731 | 1 | 1/49 |
| SecureBERT | 11 | 368757 | 39 | 2720 | 0 | 0/49 |
| SecureBERT | 12 | 339631 | 36 | 2724 | 0 | 0/49 |

## Tổng hợp

- **RoBERTa:** eligible 813/4824356 nhóm (0.016852%); sampled 3/32711 nhóm (0.009171%). Xung đột sampled xuất hiện trong 3/12 epoch, liên quan 3/36 tài liệu. Trong 4632 lượt cặp dương, 813 lượt (17.5518%) có nhãn âm mâu thuẫn trong pool và 3 lượt (0.0648%) có nhãn âm mâu thuẫn trong batch.
- **SecureBERT:** eligible 837/5601364 nhóm (0.014943%); sampled 11/32866 nhóm (0.033469%). Xung đột sampled xuất hiện trong 8/12 epoch, liên quan 7/36 tài liệu. Trong 4632 lượt cặp dương, 837 lượt (18.0699%) có nhãn âm mâu thuẫn trong pool và 11 lượt (0.2375%) có nhãn âm mâu thuẫn trong batch.

Mỗi epoch giữ 386 cặp dương, bỏ qua một self-edge trong nguồn và giữ một cặp dương vượt khoảng cách 96. Các trường hợp vẫn giữ nguyên gold để chờ phân xử độc lập.

## Kết quả checkpoint được chọn bằng validation

| Mô hình | Epoch | Ngưỡng | Entity F1 (%) | Relation F1 (%) |
|---|---:|---:|---:|---:|
| RoBERTa | 12 | 0.90 | 38.721805 | 25.380711 |
| SecureBERT | 8 | 0.90 | 31.446541 | 20.437956 |

Đây là các chỉ số validation đã dùng để chọn mô hình, không phải kết quả test độc lập hoặc thí nghiệm can thiệp sampler.

## Kiểm chứng và giới hạn

- Ngày 08/11 kiểm tra CRC các gói nguồn và SHA-256 của 13 tệp corpus. Kiểm tra hai tệp pretrained RoBERTa và mã băm recovery SecureBERT là bằng chứng đã lưu từ trước. Tái sử dụng các điều kiện overfit/smoke đã hoàn tất với xuất xứ gốc; không báo các biên bản cũ thành kiểm thử mới.
- pytest -q ngày 08/11: 281 test và 96 subtest đạt trong môi trường khôi phục. Không sửa production code hoặc thêm test trong đợt chạy này.
- Đối chiếu 24 epoch và 1.176 bản ghi cửa sổ với tập train, thứ tự shuffle, seed từng bước, số lượng cặp và các bất biến sampler. Mã batch được giữ và kiểm tra định dạng, không tái dựng lại các pool chưa xuất.
- Biên bản kiểm chứng đã lưu đối chiếu 212 tensor mỗi mô hình với bản nhúng trong recovery. Ngày 08/11 kiểm tra lại tensor RoBERTa; kiểm chứng tensor SecureBERT được tái sử dụng từ biên bản cũ. Ngày 08/11 khôi phục recovery RoBERTa từ ba phần nén, kiểm tra mã băm và trạng thái đầy đủ. Tệp dự đoán RoBERTa chưa được lưu trước khi thư mục tạm bị dọn nên được xuất lại bằng forward chỉ trên 5 tài liệu validation, giữ checkpoint epoch 12 và ngưỡng 0,90; chỉ số khớp bản gốc đã lưu. Dự đoán SecureBERT được khôi phục và chấm lại. Không huấn luyện lại hoặc so khớp danh tính dự đoán với tệp RoBERTa đã mất. Bộ đếm AdamW có thể khác giữa các nhánh khi nhóm tích lũy không có cặp quan hệ; không giả định mọi tham số đều cập nhật 156 lần.
- Hai encoder chạy trên kernel và glibc khác nhau (SecureBERT 6.18.55/glibc 2.39, RoBERTa 6.18.57/glibc 2.41). Chưa kiểm chứng numerical parity xuyên môi trường; đây là một giới hạn của phép đối chiếu.
- Chưa có lần chạy logging-off có kiểm soát hoặc kiểm chứng độc lập từng forward/backward của pretrained. Bằng chứng fixture nhỏ và snapshot không được đổi tên thành kiểm chứng toàn bộ trajectory.
- Không xuất đầy đủ pool hoặc trạng thái từng bước; số đo tần suất không lượng hóa trực tiếp ảnh hưởng gradient hoặc chất lượng. Không kết luận nhân quả, ý nghĩa thống kê hay thứ hạng encoder tổng quát.
- Thời gian chạy có thể chịu ảnh hưởng của thao tác lưu tiến độ. Không dùng số thời gian này làm benchmark overhead.

## Bước nghiên cứu tiếp theo

Dữ liệu xác nhận xung đột tồn tại trong batch huấn luyện, nhưng mức phơi nhiễm thấp hơn số trường hợp tiềm tàng trong pool. Việc một snapshot có 0 xung đột sampled không chứng minh chúng vắng mặt xuyên suốt huấn luyện. Bước tiếp theo là thiết kế riêng ablation loại trừ nhãn âm cùng biên với cặp dương, tách khỏi hard-negative mining, giữ các điều kiện đối chứng và đo đồng thời chất lượng/độ phủ. Chưa triển khai can thiệp đó trong đợt này.

Phân xử nhãn bởi người còn chờ; chưa chạy full/test hoặc merge PR. Tiểu luận tổng quan và các chuyên đề dài chưa được cập nhật trong đợt thực nghiệm này.
