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
        model_weights=None,
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

        self.model = CyberEntRelModel(
            num_labels=self.scheme.get_num_labels(),
            num_roles=self.scheme.get_num_role_labels(),
            model_config=self.model_config["model"],
        )

        if not model_weights:
            raise ValueError(
                "A reviewed checkpoint path is required; no V10-V13 checkpoint "
                "is bundled with this repository."
            )
        if not os.path.isfile(model_weights):
            raise FileNotFoundError(
                f"Reviewed checkpoint does not exist: {model_weights}"
            )
        self.model.load_state_dict(
            torch.load(model_weights, map_location=self.device)
        )

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
            tag_ids, role_ids = self.model(input_ids, attention_mask)

        tag_ids = tag_ids[0]
        role_ids = role_ids[0].tolist()
        entities = self.scheme.decode_entities(tag_ids, role_ids)

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
                }
            )


        relations = self._build_relations(formatted_entities)

        return {
            "entities": formatted_entities,
            "relations": relations,
        }

    def _build_relations(self, entities):
        subjects = [
            entity for entity in entities
            if entity["role"] in {"ROLE_1", "ROLE_BOTH"}
        ]
        objects = [
            entity for entity in entities
            if entity["role"] in {"ROLE_2", "ROLE_BOTH"}
        ]

        relations = []

        for s in subjects:
            for o in objects:
                if s is o:
                    continue
                rel_type = self._match_relation(s["type"], o["type"])
                if rel_type:
                    relations.append(
                        {
                            "source": s["text"],
                            "target": o["text"],
                            "relationship": rel_type,
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
