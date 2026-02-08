import json
import os

class BIEOSScheme:
    def __init__(self, mapping_config_path="Configs/stix_mapping.json"):
 
        if not os.path.exists(mapping_config_path):
            self.sdo_types = [
                "attack-pattern", "campaign", "course-of-action", "grouping", 
                "identity", "indicator", "infrastructure", "intrusion-set", 
                "location", "malware", "malware-analysis", "note", 
                "observed-data", "opinion", "report", "threat-actor", 
                "tool", "vulnerability"
            ]
        else:
            with open(mapping_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.sdo_types = config.get("stix_domain_objects", [])

        self.tag_to_id = {}
        self.id_to_tag = {}
        self._build_labels()

    def _build_labels(self):

        labels = ["O"]
        for sdo in self.sdo_types:
            for role in ["1", "2"]:
                for prefix in ["B", "I", "E", "S"]:
                    labels.append(f"{prefix}-{sdo}_{role}")
        
        labels.append("OVE")
        
        self.tag_to_id = {tag: i for i, tag in enumerate(labels)}
        self.id_to_tag = {i: tag for tag, i in self.tag_to_id.items()}

    def get_num_labels(self):
        return len(self.tag_to_id)

    def decode_entities(self, tag_sequence):
        entities = []
        current_entity = None

        for i, tag_id in enumerate(tag_sequence):
            tag = self.id_to_tag.get(tag_id, "O")
            
            if tag == "O" or tag == "OVE":
                current_entity = None
                continue
            
            if tag.startswith("S-"):
                parts = tag[2:].split("_") 
                entities.append({
                    "type": parts, # Lấy loại thực thể (vị trí 0)
                    "role": parts[2], # Lấy vai trò 1 hoặc 2 (vị trí 1)
                    "start": i,
                    "end": i
                })
                current_entity = None
            
            elif tag.startswith("B-"):
                parts = tag[2:].split("_")
                current_entity = {
                    "type": parts, # Sửa index thành 0
                    "role": parts[2], # Sửa index thành 1
                    "start": i
                }
            
            elif tag.startswith("E-") and current_entity:
                parts = tag[2:].split("_")
                if parts == current_entity["type"] and parts[2] == current_entity["role"]:
                    current_entity["end"] = i
                    entities.append(current_entity)
                current_entity = None
                
            elif tag.startswith("I-") and current_entity:
                continue
            
            else:
                current_entity = None

        return entities

    def get_relation_triples(self, entities):

        subjects = [e for e in entities if e['role'] == "1"]
        objects = [e for e in entities if e['role'] == "2"]
        
        triples = []
        for sub in subjects:
            for obj in objects:
                triples.append({
                    "subject": sub,
                    "object": obj
                })
        return triples

if __name__ == "__main__":
    scheme = BIEOSScheme()
    print(f"✅ Đã khởi tạo {scheme.get_num_labels()} nhãn hợp lệ.")