import os
from tika import parser
from typing import Optional

class TextExtractor:
    def __init__(self):
        pass

    def extract_from_tika(self, file_path: str) -> str:
        try:
            parsed_data = parser.from_file(file_path)
            content = parsed_data.get('content', '')
            
            if content:
                return str(content).strip()
            return ""
        except Exception as e:
            print(f"Lỗi Apache Tika khi xử lý {file_path}: {e}")
            return ""

    def extract_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # Xử lý fallback nếu gặp lỗi bảng mã khác,.
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Lỗi khi đọc tệp văn bản thô {file_path}: {e}")
            return ""

    def process_report(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            print(f"Tệp không tồn tại: {file_path}")
            return ""

        _, ext = os.path.splitext(file_path.lower())

        # Phân tách công cụ theo quy trình thực hiện đã thiết lập [1], [2],.
        if ext in ['.pdf', '.doc', '.docx']:
            return self.extract_from_tika(file_path)
        
        elif ext == '.txt':
            return self.extract_from_txt(file_path)
        
        else:
            print(f"Định dạng {ext} không được hỗ trợ chính thức trong prototype này.")
            return ""

if __name__ == "__main__":
    # Ví dụ vận hành theo lộ trình src/preprocessing/ [3],.
    extractor = TextExtractor()
    # raw_text = extractor.process_report("data/raw/apt_report.pdf")
    # print(raw_text)
