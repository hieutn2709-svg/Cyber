import json
import uuid
import os
from datetime import datetime
from typing import List, Dict

class STIXMapper:

    def __init__(self, mapping_path="configs/stix_mapping.json"):
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Không tìm thấy file ánh xạ tại {mapping_path}")
            
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.stix_version = self.mapping_data.get("stix_version", "2.1")
        self.entity_map = self.mapping_data["label_mapping"]["entities"]

    def generate_bundle(self, entities: List[Dict], relations: List[Dict]) -> Dict:

        bundle_id = f"bundle--{uuid.uuid4()}"
        stix_objects = []
 
        value_to_id = {}

        for ent in entities:
            stix_type = self.entity_map.get(ent['type'], ent['type'])
            
            object_id = f"{stix_type}--{uuid.uuid4()}"
            
            sdo = {
                "type": stix_type,
                "spec_version": self.stix_version,
                "id": object_id,
                "created": self._get_timestamp(),
                "modified": self._get_timestamp(),
                "name": ent['value'],
                "confidence": int(ent.get('confidence', 0) * 100), # STIX dùng thang 0-100
                "external_references": [
                    {"source_name": ent.get('source', 'integrated_pipeline')}
                ]
            }

            if stix_type == "vulnerability" and "CVE-" in ent['value']:
                sdo["external_references"].append({
                    "source_name": "cve",
                    "external_id": ent['value']
                })
            
            stix_objects.append(sdo)
            value_to_id[ent['value']] = object_id

        for rel in relations:
            source_ref = value_to_id.get(rel['source'])
            target_ref = value_to_id.get(rel['target'])
            
            if source_ref and target_ref:
                sro = {
                    "type": "relationship",
                    "spec_version": self.stix_version,
                    "id": f"relationship--{uuid.uuid4()}",
                    "created": self._get_timestamp(),
                    "modified": self._get_timestamp(),
                    "relationship_type": rel['relationship'], # e.g., 'uses', 'targets' [8]
                    "source_ref": source_ref,
                    "target_ref": target_ref,
                    "confidence": int(rel.get('confidence', 0) * 100)
                }
                stix_objects.append(sro)

        return {
            "type": "bundle",
            "id": bundle_id,
            "objects": stix_objects
        }

    def save_to_file(self, bundle: Dict, output_dir="output/stix_json/"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = f"cti_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(output_dir, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=4, ensure_ascii=False)
        return path

    def _get_timestamp(self):
        return datetime.utcnow().isoformat() + "Z"

if __name__ == "__main__":
    mapper = STIXMapper()
    
    merged_entities = [
        {"type": "APT_group", "value": "OceanLotus", "confidence": 1.0, "source": "kb_matcher"},
        {"type": "vulnerability", "value": "CVE-2021-26855", "confidence": 1.0, "source": "ioc_finder"}
    ]
    merged_relations = [
        {"source": "OceanLotus", "target": "CVE-2021-26855", "relationship": "exploits", "confidence": 0.9}
    ]
    
    bundle = mapper.generate_bundle(merged_entities, merged_relations)
    output_path = mapper.save_to_file(bundle)
    
    print(f"✅ Đã xuất báo cáo STIX 2.1 thành công!")
    print(f"📍 Đường dẫn: {output_path}")
    print(f"📊 Tổng số đối tượng: {len(bundle['objects'])}")