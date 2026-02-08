import yaml
import json
from typing import List, Dict

class ResultMerger:
    """
    Module hợp nhất kết quả từ nhiều nguồn trích xuất (Regex, KB, Deep Learning).
    Sử dụng chiến lược Confidence Scoring và Priority Order để giải quyết xung đột [2, 3].
    """
    def __init__(self, config_path="Configs/pipeline_settings.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.settings = yaml.safe_load(f)['pipeline']
        
        self.threshold = self.settings['thresholds']['global_acceptance'] # 0.5
        self.priority_order = self.settings['merger']['priority_order'] # ioc -> kb -> joint -> ttp

    def merge_entities(self, all_extracted_entities: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Hợp nhất các thực thể và xử lý chồng lấn vị trí (overlapping).
        all_extracted_entities: Dict chứa danh sách thực thể từ từng module.
        """
        raw_pool = []
        for source, entities in all_extracted_entities.items():
            for ent in entities:
                if ent.get('confidence', 0) >= self.threshold:
                    ent['source'] = source  # Đảm bảo ghi nhận nguồn gốc
                    raw_pool.append(ent)

        raw_pool.sort(key=lambda x: (
            self.priority_order.index(x['source']),
            -x['confidence']
        ))

        final_entities = []
        
        for candidate in raw_pool:
            is_overlap = False
            for confirmed in final_entities:
                if self._is_overlapping(candidate, confirmed):
                    is_overlap = True
                    break
            
            if not is_overlap:
                final_entities.append(candidate)

        return final_entities

    def _is_overlapping(self, ent1: Dict, ent2: Dict) -> bool:
        return not (ent1['end'] <= ent2['start'] or ent1['start'] >= ent2['end'])

    def merge_relations(self, relations_list: List[Dict]) -> List[Dict]:
        """
        Hợp nhất các quan hệ trích xuất được.
        """
        unique_relations = {}
        for rel in relations_list:
            if rel.get('confidence', 0) < self.threshold:
                continue
            
            rel_key = f"{rel['source']}_{rel['target']}"
            
            if rel_key not in unique_relations:
                unique_relations[rel_key] = rel
            else:
                if rel['confidence'] > unique_relations[rel_key]['confidence']:
                    unique_relations[rel_key] = rel
                    
        return list(unique_relations.values())

if __name__ == "__main__":
    merger = ResultMerger()

    sample_data = {
        "ioc_finder": [
            {"type": "ipv4", "value": "192.168.1.1", "start": 10, "end": 21, "confidence": 1.0}
        ],
        "joint_model": [
            {"type": "indicator", "value": "192.168.1.1", "start": 10, "end": 21, "confidence": 0.88},
            {"type": "malware", "value": "UnknownTrojan", "start": 50, "end": 63, "confidence": 0.75}
        ]
    }

    merged = merger.merge_entities(sample_data)
    print(f"--- Kết quả hợp nhất thực thể ({len(merged)}) ---")
    for e in merged:
        print(f"[{e['source']}] {e['type']}: {e['value']} (Conf: {e['confidence']})")