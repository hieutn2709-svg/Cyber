# CTI Q2 Rà soát 22 quan hệ khó

Ngày 19/10/2026. Đối chiếu văn bản gốc, định danh gán nhãn và dự đoán validation đã lưu.

## Kết luận chính

Phân tích cho thấy cần rà soát nhãn song song với cải thiện mô hình. Trường hợp TA459–Russia có nhãn quan hệ không phù hợp vai trò của Russia trong câu; trường hợp Gallmaker–Microsoft có trường direction=left cần kiểm tra với quy ước xuất dữ liệu. Đây là nhận xét bước đầu cần người gán nhãn phân xử độc lập; chưa sửa bất kỳ nhãn hoặc điểm số nào.

Tất cả 22 quan hệ được xác minh đúng định danh, biên ký tự, chuỗi văn bản và quan hệ nguồn. Đã đối chiếu 264 kết quả quan hệ (22 × 2 encoder × 3 seed × 2 cap) với phân tích trước. Riêng cap128 có 132 kết quả lặp và 264 quan sát đầu mút.

## Phân rã cơ học tại cap128

| Trạng thái quan hệ | Số lượt /132 |
|---|---:|
| Mất đầu mút | 81 |
| Dưới ngưỡng | 39 |
| Sai loại | 12 |

| Trạng thái đầu mút sau lọc | Số lượt /264 |
|---|---:|
| Không chồng lấn | 23 |
| Đúng | 174 |
| Sai kiểu | 52 |
| Sai biên và kiểu | 10 |
| Sai biên | 5 |

Phân loại đầu mút theo ưu tiên: đúng biên và kiểu; đúng biên sai kiểu; có chồng lấn đúng kiểu nhưng sai biên; chồng lấn sai cả biên và kiểu; không có ứng viên chồng lấn được giữ. Đây là mô tả tập ứng viên sau lọc, không phải bằng chứng nhân quả rằng cap, tokenizer hoặc đề xuất đoạn là nguyên nhân duy nhất. Tọa độ token chỉ được so trong cùng encoder.

Số quan hệ có cùng trạng thái ở cả sáu baseline: endpoint_loss: 9, mixed: 10, below_threshold: 1, wrong_type: 2.

## Hồ sơ từng quan hệ

Các nhận xét dưới đây phân biệt quan sát máy với đánh giá ngữ nghĩa bước đầu. Trích đoạn được lấy nguyên văn từ nguồn đã khóa, không tìm kiếm hay bổ sung sự kiện bên ngoài. Điểm tồn tại dưới 0,90 không tạo dự đoán, dù argmax loại có thể đúng.

### R01 — tài liệu 365

Gold: **C2 (attack-pattern) → targets → Twitter (identity)**.

Định danh nguồn/đích: `247O-jrKfP` / `QeuKdBEgZa`. Biên ký tự [bắt đầu, kết thúc): [1366, 1368) / [1332, 1339). direction nguồn: `right`.

> e its Raindrop malware. APT29 has acquired C2 domains, sometimes through . resellers and has registered algorithmically generated Twitter handles that are used for C2 by . malware, such as HAMMERTOSS. APT29 has used Rundll32.exe to execute payloads which calls out to . a remote network infrastructure using the domains avsvmc

Câu nói tài khoản Twitter được dùng cho C2; quan hệ C2 targets Twitter cần kiểm tra ý nghĩa targets trong hướng dẫn gán nhãn. Không suy ra Twitter là nạn nhân chỉ từ việc nền tảng bị sử dụng.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| roberta | 43 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| roberta | 44 | Mất đầu mút | Không chồng lấn | Sai kiểu | — | — |
| securebert | 42 | Mất đầu mút | Không chồng lấn | Sai kiểu | — | — |
| securebert | 43 | Mất đầu mút | Không chồng lấn | Sai kiểu | — | — |
| securebert | 44 | Mất đầu mút | Không chồng lấn | Sai kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R02 — tài liệu 365

Gold: **C2 (attack-pattern) → uses → HAMMERTOSS (malware)**.

Định danh nguồn/đích: `247O-jrKfP` / `wnf_UqG_6T`. Biên ký tự [bắt đầu, kết thúc): [1366, 1368) / [1391, 1401). direction nguồn: `right`.

> acquired C2 domains, sometimes through . resellers and has registered algorithmically generated Twitter handles that are used for C2 by . malware, such as HAMMERTOSS. APT29 has used Rundll32.exe to execute payloads which calls out to . a remote network infrastructure using the domains avsvmcloud.com. Observed malicious inst

Cụm C2 by malware, such as HAMMERTOSS mô tả mã độc dùng kênh C2. Cần kiểm tra hướng C2 uses HAMMERTOSS; không tự đảo đầu mút vì phải đối chiếu quy ước toàn bộ corpus.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| roberta | 43 | Mất đầu mút | Không chồng lấn | Sai kiểu | — | — |
| roberta | 44 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| securebert | 42 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| securebert | 43 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| securebert | 44 | Mất đầu mút | Không chồng lấn | Đúng | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R03 — tài liệu 365

Gold: **supply chain compromise (attack-pattern) → targets → SolarWinds (identity)**.

Định danh nguồn/đích: `CZybLw7CGD` / `_hcpxY0GU3`. Biên ký tự [bắt đầu, kết thúc): [663, 686) / [652, 662). direction nguồn: `right`.

> tarting in the summer of 2015. . Fig. 1: Example of figure inside text. . In April 2021, the US and UK governments attributed the SolarWinds supply chain compromise cyber . operation to the SVR; public statements included citations to APT29, Cozy Bear, and The Dukes. . Victims of this campaign included government, consulti

SolarWinds supply chain compromise là bằng chứng ngữ cảnh cho quan hệ tấn công–đối tượng. Ưu tiên kiểm tra biên và kiểu của cụm supply chain compromise; targets vẫn cần nhất quán với hướng dẫn tác vụ.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Sai biên và kiểu | Sai kiểu | — | — |
| roberta | 43 | Mất đầu mút | Sai biên và kiểu | Sai kiểu | — | — |
| roberta | 44 | Mất đầu mút | Sai biên và kiểu | Sai kiểu | — | — |
| securebert | 42 | Mất đầu mút | Sai biên và kiểu | Đúng | — | — |
| securebert | 43 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| securebert | 44 | Mất đầu mút | Không chồng lấn | Đúng | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R04 — tài liệu 365

Gold: **APT29 (intrusion-set) → uses → C2 (attack-pattern)**.

Định danh nguồn/đích: `JsV_ChNOmM` / `IfiHg8BcKu`. Biên ký tự [bắt đầu, kết thúc): [1226, 1231) / [1245, 1247). direction nguồn: `right`.

> has used WinRM via PowerShell to execute command and payloads on remote hosts. APT29 . used 7-Zip to decode its Raindrop malware. APT29 has acquired C2 domains, sometimes through . resellers and has registered algorithmically generated Twitter handles that are used for C2 by . malware, such as HAMMERTOSS. APT2

C2 nằm trong C2 domains. Cần xác định nhãn attack-pattern áp cho chức năng C2 hay hạ tầng tên miền; biên ngắn dễ mất ngữ cảnh phân loại.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Đúng | Sai biên | — | — |
| roberta | 43 | Mất đầu mút | Đúng | Sai biên | — | — |
| roberta | 44 | Mất đầu mút | Đúng | Sai biên | — | — |
| securebert | 42 | Mất đầu mút | Đúng | Không chồng lấn | — | — |
| securebert | 43 | Mất đầu mút | Đúng | Sai biên | — | — |
| securebert | 44 | Mất đầu mút | Đúng | Sai biên | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R05 — tài liệu 365

Gold: **APT29 (intrusion-set) → uses → PowerShell (tool)**.

Định danh nguồn/đích: `q1Y_SE_Oaj` / `QCZY8oPUmU`. Biên ký tự [bắt đầu, kết thúc): [1090, 1095) / [1115, 1125). direction nguồn: `right`.

> eporting referred to the . actors involved in this campaign as UNC2452, NOBELIUM, StellarParticle, and Dark Halo. In July 2020, . APT29 has used WinRM via PowerShell to execute command and payloads on remote hosts. APT29 . used 7-Zip to decode its Raindrop malware. APT29 has acquired C2 domains, sometimes through . reseller

APT29 has used WinRM via PowerShell hỗ trợ quan hệ sử dụng PowerShell. Kiểm tra ứng viên đúng biên bị phân loại thành attack-pattern hay loại khác trước khi thay ngân sách đoạn.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R06 — tài liệu 365

Gold: **APT29 (intrusion-set) → uses → Raindrop (malware)**.

Định danh nguồn/đích: `q_MhAP0kID` / `VO3Fr4N6mH`. Biên ký tự [bắt đầu, kết thúc): [1175, 1180) / [1208, 1216). direction nguồn: `right`.

> llarParticle, and Dark Halo. In July 2020, . APT29 has used WinRM via PowerShell to execute command and payloads on remote hosts. APT29 . used 7-Zip to decode its Raindrop malware. APT29 has acquired C2 domains, sometimes through . resellers and has registered algorithmically generated Twitter handles that are used for C2 by . ma

APT29 used 7-Zip to decode its Raindrop malware hỗ trợ liên hệ sử dụng mã độc ở mức tác nhân. Phân biệt lỗi endpoint của một số seed với điểm tồn tại thấp ở các lượt còn lại.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.8544 | uses |
| securebert | 42 | Dưới ngưỡng | Đúng | Đúng | 0.7919 | uses |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.6954 | uses |
| securebert | 44 | Dưới ngưỡng | Đúng | Đúng | 0.6436 | uses |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R07 — tài liệu 365

Gold: **APT29 (intrusion-set) → targets → NATO (identity)**.

Định danh nguồn/đích: `qrYEzoOgVl` / `TNX0Q8KwpM`. Biên ký tự [bắt đầu, kết thúc): [206, 211) / [393, 397). direction nguồn: `right`.

> …………… 1 . Section 2 ……………………………………………… 2 . Subsection 2.1 …………………………………….. 3 . Section 3 ……………………………………………… 4 . STIXnet Example . APT29 is threat group that has been attributed to Russia's Foreign Intelligence Service (SVR). They . have operated since at least 2008, often targeting government networks in Europe and NATO member . countries, research institutes, and think tanks. APT29 reportedly compromised the Democratic . National Committee starting in the summer of 2015. . Fi

Cụm NATO member countries nói về các quốc gia thành viên, không nhất thiết chính tổ chức NATO. Cần rà soát biên/kiểu và quy ước targets trước khi coi đây là ví dụ học đầu mút thuần túy.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R08 — tài liệu 365

Gold: **7-Zip (tool) → deliver → Raindrop (malware)**.

Định danh nguồn/đích: `wyXkQS7kDx` / `VO3Fr4N6mH`. Biên ký tự [bắt đầu, kết thúc): [1188, 1193) / [1208, 1216). direction nguồn: `right`.

>  and Dark Halo. In July 2020, . APT29 has used WinRM via PowerShell to execute command and payloads on remote hosts. APT29 . used 7-Zip to decode its Raindrop malware. APT29 has acquired C2 domains, sometimes through . resellers and has registered algorithmically generated Twitter handles that are used for C2 by . ma

Văn bản dùng decode, trong khi gold là deliver. Cần kiểm tra xem deliver có được định nghĩa rộng tới giải mã/giải nén hay không; không tự coi việc cùng xuất hiện là bằng chứng phát tán.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.8147 | uses |
| securebert | 42 | Dưới ngưỡng | Đúng | Đúng | 0.6931 | uses |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.6943 | uses |
| securebert | 44 | Dưới ngưỡng | Đúng | Đúng | 0.3832 | uses |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R09 — tài liệu 389

Gold: **Outlook (identity) → targeted-by → credentials (attack-pattern)**.

Định danh nguồn/đích: `249BVOjo-S` / `5RESWu9d5e`. Biên ký tự [bắt đầu, kết thúc): [902, 909) / [881, 892). direction nguồn: `right`.

>  "phish," and "malware" in a likely attempt to prevent organizations from communicating about their activities. FIN4 has captured credentials via fake Outlook Web App (OWA) login pages and has also used a .NET based keylogger. FIN4 has presented victims with spoofed Windows Authentication prompts to collect their cre

credentials là đối tượng thông tin bị thu thập trong câu, nhưng gold gán attack-pattern. Cần kiểm tra liệu span hành vi phải bao gồm captured credentials; Outlook Web App là giao diện giả được sử dụng.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.5477 | uses |
| roberta | 43 | Dưới ngưỡng | Đúng | Đúng | 0.3812 | uses |
| roberta | 44 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 42 | Dưới ngưỡng | Đúng | Đúng | 0.3109 | uses |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.3656 | uses |
| securebert | 44 | Dưới ngưỡng | Đúng | Đúng | 0.1585 | uses |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R10 — tài liệu 389

Gold: **Outlook (identity) → targeted-by → NET (tool)**.

Định danh nguồn/đích: `249BVOjo-S` / `qH7_nD5KdL`. Biên ký tự [bắt đầu, kết thúc): [902, 909) / [957, 960). direction nguồn: `right`.

> e" in a likely attempt to prevent organizations from communicating about their activities. FIN4 has captured credentials via fake Outlook Web App (OWA) login pages and has also used a .NET based keylogger. FIN4 has presented victims with spoofed Windows Authentication prompts to collect their credentials. FIN4 has used spearphishing emails contai

Hai mệnh đề mô tả trang đăng nhập Outlook giả và keylogger .NET. Câu không trực tiếp nói NET nhắm vào Outlook; cần kiểm tra liên kết quan hệ và biên NET thiếu dấu chấm.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.1643 | uses |
| roberta | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 44 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R11 — tài liệu 389

Gold: **credentials (attack-pattern) → uses → NET (tool)**.

Định danh nguồn/đích: `5RESWu9d5e` / `qH7_nD5KdL`. Biên ký tự [bắt đầu, kết thúc): [881, 892) / [957, 960). direction nguồn: `right`.

>  "phish," and "malware" in a likely attempt to prevent organizations from communicating about their activities. FIN4 has captured credentials via fake Outlook Web App (OWA) login pages and has also used a .NET based keylogger. FIN4 has presented victims with spoofed Windows Authentication prompts to collect their credentials. FIN4 has used spearphishing emails contai

credentials uses NET kết hợp một danh từ dữ liệu với nền tảng triển khai keylogger. Cần rà soát biên hành vi, loại thực thể và chủ thể của uses.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.8306 | uses |
| roberta | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.4993 | uses |
| securebert | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R12 — tài liệu 389

Gold: **email accounts (attack-pattern) → uses → Tor (tool)**.

Định danh nguồn/đích: `5XTBDlr4WD` / `Nay4UVDJtp`. Biên ký tự [bắt đầu, kết thúc): [1389, 1403) / [1363, 1366). direction nguồn: `right`.

> cious macros. FIN4 has used spearphishing emails (often sent from compromised accounts) containing malicious links. FIN4 has used Tor to log in to victims' email accounts. FIN4 has lured victims to click malicious links delivered via spearphishing emails (often sent from compromised accounts). FIN4 has lured victims to launch ma

FIN4 has used Tor to log in to victims' email accounts đặt FIN4 làm chủ thể sử dụng Tor. Hướng email accounts uses Tor và kiểu attack-pattern của email accounts cần kiểm tra.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| roberta | 43 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| roberta | 44 | Mất đầu mút | Không chồng lấn | Sai kiểu | — | — |
| securebert | 42 | Mất đầu mút | Không chồng lấn | Đúng | — | — |
| securebert | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R13 — tài liệu 389

Gold: **keylogger (indicator) → indicates → credentials (attack-pattern)**.

Định danh nguồn/đích: `NXrY2AgBGA` / `5RESWu9d5e`. Biên ký tự [bắt đầu, kết thúc): [967, 976) / [881, 892). direction nguồn: `right`.

>  "phish," and "malware" in a likely attempt to prevent organizations from communicating about their activities. FIN4 has captured credentials via fake Outlook Web App (OWA) login pages and has also used a .NET based keylogger. FIN4 has presented victims with spoofed Windows Authentication prompts to collect their credentials. FIN4 has used spearphishing emails containing attachments

keylogger được gán indicator và credentials được gán attack-pattern. Văn bản mô tả công cụ thu thập thông tin; cần đối chiếu hồ sơ nhãn riêng của dự án, không đồng nhất indicator của corpus với đối tượng STIX chuẩn.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| roberta | 43 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| roberta | 44 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 42 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 43 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 44 | Mất đầu mút | Sai kiểu | Đúng | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R14 — tài liệu 389

Gold: **keylogger (indicator) → indicates → FIN4 (intrusion-set)**.

Định danh nguồn/đích: `NXrY2AgBGA` / `6RHNLx43sd`. Biên ký tự [bắt đầu, kết thúc): [967, 976) / [863, 867). direction nguồn: `right`.

>  such as "hacked," "phish," and "malware" in a likely attempt to prevent organizations from communicating about their activities. FIN4 has captured credentials via fake Outlook Web App (OWA) login pages and has also used a .NET based keylogger. FIN4 has presented victims with spoofed Windows Authentication prompts to collect their credentials. FIN4 has used spearphishing emails containing attachments

Văn bản có FIN4 sử dụng keylogger, nhưng gold diễn đạt keylogger indicates FIN4. Cần xác minh quy tắc suy luận indicates và kiểu indicator có được hướng dẫn cho danh từ công cụ hay không.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| roberta | 43 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| roberta | 44 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 42 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 43 | Mất đầu mút | Sai kiểu | Đúng | — | — |
| securebert | 44 | Mất đầu mút | Sai kiểu | Đúng | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R15 — tài liệu 389

Gold: **FIN4 (intrusion-set) → uses → email accounts (attack-pattern)**.

Định danh nguồn/đích: `kirDJe0aoz` / `5XTBDlr4WD`. Biên ký tự [bắt đầu, kết thúc): [1349, 1353) / [1389, 1403). direction nguồn: `right`.

>  embedded malicious macros. FIN4 has used spearphishing emails (often sent from compromised accounts) containing malicious links. FIN4 has used Tor to log in to victims' email accounts. FIN4 has lured victims to click malicious links delivered via spearphishing emails (often sent from compromised accounts). FIN4 has lured victims to launch ma

Câu nêu truy cập tài khoản email bằng Tor; email accounts được gán attack-pattern. Cần kiểm tra granularity hành vi và vai trò đối tượng của accounts trước khi diễn giải lỗi mô hình.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Mất đầu mút | Đúng | Không chồng lấn | — | — |
| roberta | 43 | Mất đầu mút | Đúng | Không chồng lấn | — | — |
| roberta | 44 | Mất đầu mút | Đúng | Không chồng lấn | — | — |
| securebert | 42 | Mất đầu mút | Đúng | Không chồng lấn | — | — |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.5261 | uses |
| securebert | 44 | Dưới ngưỡng | Đúng | Đúng | 0.4408 | uses |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R16 — tài liệu 389

Gold: **Outlook (identity) → targeted-by → FIN4 (intrusion-set)**.

Định danh nguồn/đích: `uAA45pt9u0` / `gLHPJPyP3M`. Biên ký tự [bắt đầu, kết thúc): [669, 676) / [624, 628). direction nguồn: `right`.

> a dialog box and collect victim credentials. FIN4 has accessed and hijacked online email communications using stolen credentials. FIN4 has created rules in victims' Microsoft Outlook accounts to automatically delete emails containing words such as "hacked," "phish," and "malware" in a likely attempt to prevent organizations from communicati

FIN4 tạo quy tắc trong tài khoản Microsoft Outlook của nạn nhân. Có bằng chứng tác động lên tài khoản; kiểm tra phạm vi Outlook so với Outlook accounts và quy tắc targeted-by. Sáu baseline đều dưới ngưỡng.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.6032 | used-by |
| roberta | 43 | Dưới ngưỡng | Đúng | Đúng | 0.6079 | used-by |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.3346 | used-by |
| securebert | 42 | Dưới ngưỡng | Đúng | Đúng | 0.6954 | used-by |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.7600 | used-by |
| securebert | 44 | Dưới ngưỡng | Đúng | Đúng | 0.5994 | used-by |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R17 — tài liệu 395

Gold: **TA459 (intrusion-set) → originates-from → Russia (location)**.

Định danh nguồn/đích: `5lq92atdQi` / `Uygvc6SrAU`. Biên ký tự [bắt đầu, kết thúc): [0, 5) / [95, 101). direction nguồn: `right`.

> TA459 is a threat group believed to operate out of China that has targeted countries including Russia, Belarus, Mongolia, and others. TA459 has used PowerShell for execution of a payload. TA459 has a VBScript for execution. TA459 has exploited Microsoft Word vu

Câu phân biệt operate out of China và targeted countries including Russia. Gold TA459 originates-from Russia không phù hợp vai trò Russia trong câu này. Sáu baseline dự đoán targets cùng cặp; cần người gán nhãn xác nhận trước mọi thay đổi.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Sai loại | Đúng | Đúng | 0.9910 | targets |
| roberta | 43 | Sai loại | Đúng | Đúng | 0.9856 | targets |
| roberta | 44 | Sai loại | Đúng | Đúng | 0.9877 | targets |
| securebert | 42 | Sai loại | Đúng | Đúng | 0.9265 | targets |
| securebert | 43 | Sai loại | Đúng | Đúng | 0.9897 | targets |
| securebert | 44 | Sai loại | Đúng | Đúng | 0.9603 | targets |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R18 — tài liệu 400

Gold: **Andariel (intrusion-set) → uses → Flash Player (tool)**.

Định danh nguồn/đích: `4YIP0F-n7j` / `Gx166Hwfdl`. Biên ký tự [bắt đầu, kết thúc): [1340, 1348) / [1302, 1314). direction nguồn: `right`.

> ted a malicious script within compromised websites to collect potential victim information such as browser type, system language, Flash Player version, and other data. Andariel has limited its watering hole attacks to specific IP address ranges. Andariel has downloaded additional tools and malware onto compromised hosts. Andariel has 

Flash Player chỉ xuất hiện trong danh sách thông tin phiên bản bị thu thập; Andariel ở đầu câu kế tiếp nói giới hạn IP. Cần kiểm tra liên kết đề cập xuyên câu và bằng chứng uses Flash Player; không đủ căn cứ chỉ từ đồng xuất hiện.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.5444 | uses |
| roberta | 43 | Dưới ngưỡng | Đúng | Đúng | 0.0866 | targets |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.3309 | uses |
| securebert | 42 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 43 | Mất đầu mút | Đúng | Sai kiểu | — | — |
| securebert | 44 | Mất đầu mút | Đúng | Sai kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R19 — tài liệu 400

Gold: **Bureau (identity) → located-at → North Korea (location)**.

Định danh nguồn/đích: `aZS_FiUmNd` / `6nlTG2s7fP`. Biên ký tự [bắt đầu, kết thúc): [607, 613) / [570, 581). direction nguồn: `right`.

> lack Mine, Operation GoldenAxe, and Campaign Rifle. Andariel is considered a sub-set of Lazarus Group, and has been attributed to North Korea's Reconnaissance General Bureau. North Korean group definitions are known to have significant overlap, and some security researchers report all North Korean state-sponsored cyber activity und

Bureau là phần cuối tên North Korea's Reconnaissance General Bureau. located-at có thể được suy từ sở hữu/quốc gia nhưng cần quy ước rõ; rà soát việc cắt tên tổ chức còn một token.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.1600 | originates-from |
| roberta | 43 | Dưới ngưỡng | Đúng | Đúng | 0.1599 | originates-from |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.2256 | originates-from |
| securebert | 42 | Mất đầu mút | Sai biên và kiểu | Đúng | — | — |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.4597 | originates-from |
| securebert | 44 | Mất đầu mút | Sai biên và kiểu | Đúng | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R20 — tài liệu 400

Gold: **Andariel (intrusion-set) → targets → Bureau (identity)**.

Định danh nguồn/đích: `cT6Id2Ht7c` / `aZS_FiUmNd`. Biên ký tự [bắt đầu, kết thúc): [492, 500) / [607, 613). direction nguồn: `right`.

> and cryptocurrency exchanges. Andariel's notable activity includes Operation Black Mine, Operation GoldenAxe, and Campaign Rifle. Andariel is considered a sub-set of Lazarus Group, and has been attributed to North Korea's Reconnaissance General Bureau. North Korean group definitions are known to have significant overlap, and some security researchers report all North Korean state-sponsored cyber activity und

Câu dùng attributed to North Korea's Reconnaissance General Bureau, không mô tả Andariel tấn công cơ quan này. Rà soát nhãn targets và biên Bureau.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.3520 | targets |
| roberta | 43 | Dưới ngưỡng | Đúng | Đúng | 0.3086 | targets |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.6756 | targets |
| securebert | 42 | Mất đầu mút | Đúng | Sai biên và kiểu | — | — |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.1782 | targets |
| securebert | 44 | Mất đầu mút | Đúng | Sai biên và kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R21 — tài liệu 400

Gold: **Lazarus Group (intrusion-set) → targets → Bureau (identity)**.

Định danh nguồn/đích: `zxIoGVmYeL` / `aZS_FiUmNd`. Biên ký tự [bắt đầu, kết thúc): [528, 541) / [607, 613). direction nguồn: `right`.

> el's notable activity includes Operation Black Mine, Operation GoldenAxe, and Campaign Rifle. Andariel is considered a sub-set of Lazarus Group, and has been attributed to North Korea's Reconnaissance General Bureau. North Korean group definitions are known to have significant overlap, and some security researchers report all North Korean state-sponsored cyber activity und

Câu mô tả Andariel là sub-set của Lazarus Group và được quy thuộc một cơ quan. Quan hệ Lazarus Group targets Bureau không được khẳng định trực tiếp trong đoạn; cần kiểm tra chủ thể và nhãn.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Dưới ngưỡng | Đúng | Đúng | 0.2567 | targets |
| roberta | 43 | Dưới ngưỡng | Đúng | Đúng | 0.1360 | targets |
| roberta | 44 | Dưới ngưỡng | Đúng | Đúng | 0.7079 | originates-from |
| securebert | 42 | Mất đầu mút | Đúng | Sai biên và kiểu | — | — |
| securebert | 43 | Dưới ngưỡng | Đúng | Đúng | 0.1452 | originates-from |
| securebert | 44 | Mất đầu mút | Đúng | Sai biên và kiểu | — | — |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

### R22 — tài liệu 420

Gold: **Gallmaker (intrusion-set) → targeted-by → Microsoft (identity)**.

Định danh nguồn/đích: `N3J848oond` / `M-14tDy57X`. Biên ký tự [bắt đầu, kết thúc): [369, 378) / [400, 409). direction nguồn: `left`.

> inZip, likely to archive data prior to exfiltration. Gallmaker used PowerShell to download additional payloads and for execution. Gallmaker attempted to exploit Microsoft’s DDE protocol in order to gain access to victim machines and for execution. Gallmaker obfuscated shellcode used during execution. Gallmaker sent emails with m

Nguồn lưu direction=left, nhưng protocol đóng băng dùng from_id→to_id, tạo Gallmaker targeted-by Microsoft. Câu nói Gallmaker attempted to exploit Microsoft's DDE protocol. Sáu baseline dự đoán targets. Cần rà soát riêng hướng xuất dữ liệu và biên Microsoft so với Microsoft’s DDE protocol; không tự đảo trong phép so sánh encoder.

| Encoder | Seed | Lỗi cap128 | Đầu nguồn | Đầu đích | P(tồn tại) | Loại argmax |
|---|---:|---|---|---|---:|---|
| roberta | 42 | Sai loại | Đúng | Đúng | 0.9727 | targets |
| roberta | 43 | Sai loại | Đúng | Đúng | 0.9730 | targets |
| roberta | 44 | Sai loại | Đúng | Đúng | 0.9703 | targets |
| securebert | 42 | Sai loại | Đúng | Đúng | 0.9763 | targets |
| securebert | 43 | Sai loại | Đúng | Đúng | 0.9780 | targets |
| securebert | 44 | Sai loại | Đúng | Đúng | 0.9210 | targets |

Trạng thái phân xử: **chờ rà soát độc lập**. Giữ nguyên gold và kết quả đánh giá. Chi tiết ứng viên chồng lấn và cap96 nằm trong JSON đi kèm.

## Nguồn, kiểm chứng và phạm vi

SHA-256 văn bản gán nhãn: `dabd5c52c20c7aa912de65723699f246b0c12b3590289283b3c2a2fa6250e776`, trùng manifest matched70. Dữ liệu dự đoán lấy từ bộ ba seed ngày 13/10 và inventory phân tích ngày 18/10. Ngoại lệ logits SecureBERT seed42 cap128 tiếp tục dùng nguồn baseline gốc đã kiểm chứng, không diễn giải như forward mới.

Tập 22 quan hệ được chọn vì sai bền vững, không phải mẫu ngẫu nhiên để ước lượng tỷ lệ lỗi nhãn toàn corpus. Chưa có phân xử độc lập, chưa sửa nhãn, chưa tính điểm trên gold thay đổi, chưa chạy huấn luyện hay full/test. Không coi quan hệ mô hình dự đoán khác gold là mặc nhiên đúng.
