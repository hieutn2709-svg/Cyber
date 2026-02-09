# STIXnet-CyberEntRel

## 📋 Tổng quan

Dự án này trình bày việc phát triển và đánh giá một hệ thống trích xuất thông tin Cyber Threat Intelligence (CTI) tự động bằng cách tích hợp mô hình học sâu **CyberEntRel** vào kiến trúc **STIXnet**. Hệ thống kết hợp ưu điểm của trích xuất liên hợp (joint extraction) với tri thức chuyên gia để tự động nhận diện các thực thể STIX Domain Objects (SDOs) và quan hệ STIX Relationship Objects (SROs) từ báo cáo CTI.

### 🎯 Mục tiêu chính

- **Trích xuất tự động**: Nhận dạng đồng thời 18 loại thực thể STIX và hơn 100 loại quan hệ từ báo cáo CTI
- **Tích hợp thông minh**: Kết hợp mô hình học sâu end-to-end với tri thức chuyên gia (Rule-based/Knowledge Base)
- **Hiệu năng cao**: Đạt F1 ≈ 0.927 cho thực thể và 0.763 cho quan hệ
- **Chuẩn hóa STIX 2.1**: Xuất kết quả theo định dạng STIX 2.1 để chia sẻ và tích hợp

### ⚡ Điểm nổi bật

- **Joint Extraction**: Sử dụng kiến trúc RoBERTa-BiGRU-CRF để trích xuất đồng thời thực thể và quan hệ
- **Hybrid Approach**: Chiến lược "Tăng cường" kết hợp Rule-based (Precision) và Deep Learning (Recall)
- **BIEOS Tagging**: Sơ đồ nhãn mở rộng BIEOS để đánh dấu ranh giới thực thể và vai trò quan hệ
- **Multi-task Learning**: Học đa tác vụ với trọng số α=0.8 cho NER và 0.2 cho RE

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    CTI Reports (PDF/HTML/Text)                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Preprocessing Module                            │
│  ├─ Text Extraction (Apache Tika)                               │
│  ├─ Cleaning & Desanitizing                                     │
│  └─ Tokenization                                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌──────────────────┐      ┌──────────────────────┐
│  Joint Model     │      │  STIXnet Modules     │
│  (CyberEntRel)   │      │  (Parallel)          │
│                  │      │                      │
│ • RoBERTa-BiGRU  │      │ • IOC Finder (Regex) │
│ • Attention      │      │ • KB Matcher         │
│ • CRF Layer      │      │ • TTP Extractor      │
│ • BIEOS Tagging  │      │                      │
└────────┬─────────┘      └──────────┬───────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Integration & Merge Module                          │
│  ├─ Confidence Scoring                                          │
│  ├─ Conflict Resolution                                         │
│  └─ Duplicate Removal                                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                STIX 2.1 Output (JSON)                           │
│  ├─ 18 SDO Types (Entities)                                    │
│  └─ 100+ SRO Types (Relations)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
CTI_Integrated_Project/
│
├── configs/                          # Nền tảng cấu hình
│   ├── model_config.yaml             # Tham số AI (alpha, learning rate)
│   ├── stix_mapping.json             # Ánh xạ 18 SDO và >100 SRO
│   └── pipeline_settings.yaml        # Ngưỡng tin cậy (0.5)
│
├── data/                             # Quản lý dữ liệu đa nguồn
│   ├── raw/                          # Báo cáo CTI thô (PDF, HTML, Text)
│   ├── processed/                    # Dữ liệu BIEOS cho huấn luyện
│   └── bosch_annoCTR/                # Tập lệnh nhận diện TTP
│ 
│
│── preprocessing/                # Tiền xử lý
│      ├── text_extractor.py         # Trích xuất văn bản (Tika)
│      └── cleaner.py                # Làm sạch và desanitizing
│ 
│── joint_model/                  # Lõi AI CyberEntRel
│      ├── tagging_scheme.py         # Logic nhãn BIEOS-extended
│      ├── joint_model.py            # RoBERTa-BiGRU-Attention-CRF
│      ├── train_model.ipynb         # Huấn luyện & Fine-tuning
│      └── inference.py              # Service dự đoán
│   
│── extraction/                   # Module STIXnet bổ trợ
│      ├── ioc_finder.py             # Trích xuất bằng Regex
│      └── ttp_extractor.py          # Nhận diện TTP (Bosch)
│   
│── integration/                  # Hợp nhất và Chuẩn hóa
│      ├── merger.py                 # Merge & Confidence Scoring
│      └── stix_mapper.py            # Xuất JSON STIX 2.1
│   
│── utils/                        # Công cụ hỗ trợ
│       ├── data_converter.py         # Chuyển đổi sang BIEOS
│       └── metrics.py                # Tính P, R, F1
│
├── models_checkpoints/               # Lưu trữ trọng số mô hình
│   └── cyber_joint_v1/               # Mô hình fine-tuned
│
├── output/                           # Kết quả đầu ra
│    └─ stix_json/                    # Tệp JSON chuẩn STIX
│
├── main.py                           # Entry point chính
└── README.md                         # Tài liệu này
```


## 🔬 Phương pháp

### 1. Sơ đồ nhãn BIEOS-Extended

Mô hình sử dụng sơ đồ nhãn BIEOS để đánh dấu đồng thời:
- **Ranh giới thực thể**: B (Begin), I (Inside), E (End), S (Single), O (Outside)
- **Loại thực thể**: 18 loại SDO (Malware, Tool, Campaign, v.v.)
- **Vai trò quan hệ**: _1 (Subject), _2 (Object)

**Ví dụ:**
```
Token:  APT28   uses    a    variant   of    XYZ     malware
Label:  B-IS_1  O       O    O         O     B-Mal_2 I-Mal_2
```

### 2. Kiến trúc mô hình

```
Input Text
    ↓
RoBERTa Encoder (Pre-trained)
    ↓
BiGRU (Bidirectional Context)
    ↓
Self-Attention Layer
    ↓
CRF (Conditional Random Field)
    ↓
BIEOS Tag Sequence
    ↓
Relation Matching Algorithm
    ↓
(Entity_1, Relation, Entity_2) Triples
```

### 3. Chiến lược học đa tác vụ

```python
Total_Loss = α × Loss_NER + (1-α) × Loss_RE
# α = 0.8 (ưu tiên nhận dạng thực thể)
```

### 4. Cơ chế hợp nhất (Merge)

1. **Confidence Scoring**: Mỗi thực thể/quan hệ được gán điểm tin cậy
2. **Priority Rules**:
   - IOC Finder (Regex) > KB Matcher > Joint Model > TTP Extractor
   - Điểm tin cậy > 0.5 mới được chấp nhận
3. **Conflict Resolution**:
   - Nếu trùng lặp → chọn nguồn có độ tin cao hơn
   - Nếu mâu thuẫn → ưu tiên Rule-based

---

## 🛠️ Lộ trình phát triển

### Giai đoạn 1: Nền tảng cấu hình
- [x] Định nghĩa STIX mapping (18 SDO + 100+ SRO)
- [x] Thiết lập tham số mô hình
- [x] Cấu hình pipeline settings

### Giai đoạn 2: Tiền xử lý
- [x] Text extraction (Apache Tika)
- [x] Cleaning & desanitizing
- [x] Tokenization

### Giai đoạn 3: Lõi AI
- [x] BIEOS tagging scheme
- [x] RoBERTa-BiGRU-CRF architecture
- [x] Multi-task learning
- [x] Training pipeline

### Giai đoạn 4: Module bổ trợ
- [x] IOC Finder (Regex)
- [x] KB Matcher (Aho-Corasick)
- [x] TTP Extractor (Bosch)
- [x] Inference service

### Giai đoạn 5: Tích hợp
- [x] Merge algorithm
- [x] STIX 2.1 mapper
- [x] Confidence scoring
- [x] Evaluation metrics

### Giai đoạn 6: Hoàn thiện
- [x] Main pipeline
- [x] Documentation
- [ ] Web interface (Coming soon)
- [ ] API service (Coming soon)

---

## 📖 Tài liệu tham khảo

### Nghiên cứu chính

1. **STIXnet** (Marchiori et al., 2023)
   - Framework trích xuất thông tin CTI theo chuẩn STIX
   - Pipeline modular với nhiều module chuyên biệt
   - F1 ≈ 0.916 cho thực thể, 0.724 cho quan hệ

2. **CyberEntRel** (Ahmed et al., 2024)
   - Joint extraction với BIEOS tagging
   - RoBERTa-BiGRU-CRF architecture
   - F1 ≈ 0.86 trên dữ liệu CTI

### Dataset

- **STIXnet Dataset**: Báo cáo CTI đã gán nhãn (18 SDO types)
- **Bosch annoCTR**: Tập lệnh hướng dẫn nhận diện TTP

### Công cụ

- **Apache Tika**: Text extraction
- **spaCy**: NLP preprocessing
- **Transformers (HuggingFace)**: RoBERTa model
- **PyTorch**: Deep learning framework
- **torchcrf**: CRF implementation

