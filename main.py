import os
import json
import yaml
from Preprocessing.text_extractor import TextExtractor
from Preprocessing.cleaner import TextCleaner
from extraction.ioc_finder import IOCFinder
from extraction.ttp_extractor import TTPExtractor
from Joint_model.inference import InferenceService
from integration.merger import ResultMerger
from integration.stix_mapper import STIXMapper

class CTIPipeline:

    def __init__(self):
        print("🚀 Đang khởi tạo hệ thống tích hợp STIXnet + CyberEntRel...")
        
        self.extractor = TextExtractor()
        self.cleaner = TextCleaner()
        
        self.ioc_finder = IOCFinder()
        self.ttp_extractor = TTPExtractor()
        self.ai_inference = InferenceService() # Lõi RoBERTa-BiGRU-CRF
        
        self.merger = ResultMerger()
        self.stix_mapper = STIXMapper()

    def run(self, input_file_path: str):
        print(f"\n--- Bắt đầu xử lý báo cáo: {os.path.basename(input_file_path)} ---")
        
        raw_text = self.extractor.process_report(input_file_path)
        if not raw_text:
            print("❌ Lỗi: Không thể trích xuất văn bản.")
            return

        cleaned_text = self.cleaner.clean(raw_text)
        print(f"✅ Tiền xử lý hoàn tất. Độ dài văn bản: {len(cleaned_text)} ký tự.")

  
        ioc_results = self.ioc_finder.extract(cleaned_text)
        
        ttp_results = self.ttp_extractor.extract(cleaned_text)
        
        ai_output = self.ai_inference.predict(cleaned_text)
        ai_entities = ai_output.get("entities", [])
        ai_relations = ai_output.get("relations", [])


        all_entities_pool = {
            "ioc_finder": ioc_results,
            "ttp_extractor": ttp_results,
            "joint_model": ai_entities
        }
        
        final_entities = self.merger.merge_entities(all_entities_pool)
        final_relations = self.merger.merge_relations(ai_relations)
        
        print(f"📊 Kết quả trích xuất: {len(final_entities)} thực thể, {len(final_relations)} quan hệ.")

        bundle = self.stix_mapper.generate_bundle(final_entities, final_relations)
        output_path = self.stix_mapper.save_to_file(bundle)
        
        print(f"🏁 Hoàn thành! Báo cáo STIX đã được lưu tại: {output_path}")
        return output_path

# Test
if __name__ == "__main__":
    pipeline = CTIPipeline()
    
    # Giả sử có tệp báo cáo trong thư mục data/raw/
    report_path = "Data/Raw data/APT1.txt"
    
    if os.path.exists(report_path):
        pipeline.run(report_path)
    else:
        print(f"⚠️ Vui lòng đặt tệp báo cáo vào đường dẫn: {report_path}")

    
