import torch
import yaml
import json
import os
from transformers import RobertaTokenizerFast
from Joint_model.joint_model import CyberEntRelModel
from Joint_model.tagging_scheme import BIEOSScheme


class InferenceService:

    def __init__(
        self,
        config_path="Configs/model_config.yaml",
        mapping_path="Configs/stix_mapping.json",
        model_weights="models_checkpoints/cyber_joint_v1/best_model (10).pt",
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.model_config = yaml.safe_load(f)

        with open(mapping_path, "r", encoding="utf-8") as f:
            self.stix_mapping = json.load(f)

        self.scheme = BIEOSScheme(mapping_path)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            self.model_config["model"]["encoder"],
            add_prefix_space=True,
            use_fast=True,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_labels = self.scheme.get_num_labels()
        num_rel_types = len(
            self.stix_mapping.get("label_mapping", {}).get("entities", [])
        ) + 1

        self.model = CyberEntRelModel(
            num_labels=num_labels,
            num_rel_types=num_rel_types,
            model_config=self.model_config["model"],
        )

        if os.path.exists(model_weights):
            self.model.load_state_dict(
                torch.load(model_weights, map_location=self.device)
            )
            print(f"✅ Đã nạp trọng số mô hình từ: {model_weights}")
        else:
            print(f"⚠️ Không tìm thấy trọng số tại {model_weights}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offset_mapping = inputs["offset_mapping"][0].cpu()


        with torch.no_grad():
            tag_ids, rel_logits = self.model(input_ids, attention_mask)

        tag_ids = tag_ids[0]


        print("\n--- DEBUG: RAW TAG IDS ---")
        print(tag_ids)

        readable_tags = [
            self.scheme.id_to_tag.get(tid, "O") for tid in tag_ids
        ]
        print("\n--- DEBUG: READABLE TAGS ---")
        print(readable_tags)


        entities = self._decode_entities_safe(readable_tags)

        formatted_entities = []
        for ent in entities:
            tok_start = ent["start"]
            tok_end = ent["end"]

            char_start = offset_mapping[tok_start][0]
            char_end = offset_mapping[tok_end][1]

            if char_start >= char_end:
                continue

            ent_text = text[char_start:char_end]

            formatted_entities.append(
                {
                    "text": ent_text,
                    "type": ent["type"],
                    "role": ent["role"],
                    "start": char_start,
                    "end": char_end,
                    "confidence": 0.95,
                }
            )


        relations = self._build_relations(formatted_entities)

        return {
            "entities": formatted_entities,
            "relations": relations,
        }

    def _decode_entities_safe(self, tags):
        """
        Decode BIEOS tags:
        S-type_role
        B-type_role ... E-type_role
        """
        entities = []
        current = None

        for i, tag in enumerate(tags):
            if tag == "O":
                if current:
                    current["end"] = i - 1
                    entities.append(current)
                    current = None
                continue

            if "-" not in tag:
                continue

            prefix, rest = tag.split("-", 1)

            if "_" not in rest:
                continue

            ent_type, role = rest.rsplit("_", 1)

            if prefix == "S":
                entities.append(
                    {
                        "type": ent_type,
                        "role": role,
                        "start": i,
                        "end": i,
                    }
                )

            elif prefix == "B":
                current = {
                    "type": ent_type,
                    "role": role,
                    "start": i,
                }

            elif prefix == "I":
                continue

            elif prefix == "E":
                if current:
                    current["end"] = i
                    entities.append(current)
                    current = None

        return entities

    def _build_relations(self, entities):
        subjects = [e for e in entities if e["role"] == "1"]
        objects = [e for e in entities if e["role"] == "2"]

        relations = []

        for s in subjects:
            for o in objects:
                rel_type = self._match_relation(s["type"], o["type"])
                if rel_type:
                    relations.append(
                        {
                            "source": s["text"],
                            "target": o["text"],
                            "relationship": rel_type,
                            "confidence": 0.85,
                        }
                    )

        return relations

    def _match_relation(self, source_type, target_type):
        for rel in self.stix_mapping.get("label_mapping", {}).get("relations", []):
            if rel["source"] == source_type and rel["target"] == target_type:
                return rel["relationship"]
        return None



if __name__ == "__main__":
    service = InferenceService()

    sample_text = (
        "ALLANITE is ĠWord a suspected Russian cyber ĠApplication espionage group, that has primarily targeted the electric utility sector within the United States and United Kingdom. The group's tactics and techniques are reportedly similar to Dragonfly, although ALLANITEs technical capabilities have not exhibited disruptive or destructive abilities. It has been suggested that the group maintains a presence in ICS for the purpose of gaining understanding of processes and to maintain persistence. ALLANITE leverages watering hole attacks to gain access into electric utilities. ALLANITE has been identified to collect and distribute screenshots of ICS systems such as HMIs. ALLANITE utilized spear phishing to gain access into energy sector environments. ALLANITE utilized credentials collected through phishing and watering hole attacks. Gallmaker is a cyberespionage group that has targeted victims in the Middle East and has been active since at least December 2017. The group has mainly targeted victims in the defense, military, and government sectors. Gallmaker has used WinZip, likely to archive data prior to exfiltration. Gallmaker used PowerShell to download additional payloads and for execution. Gallmaker attempted to exploit Microsoft’s DDE protocol in order to gain access to victim machines and for execution. Gallmaker obfuscated shellcode used during execution. Gallmaker sent emails with malicious Microsoft Office documents attached. Gallmaker sent victims a lure document with a warning that asked victims to for execution. "
    )

    result = service.predict(sample_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
