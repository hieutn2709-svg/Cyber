import json
import os
from typing import List, Tuple
from transformers import RobertaTokenizerFast
from Joint_model.tagging_scheme import BIEOSScheme


def fix_mojibake(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


class DataConverter:
    def __init__(
        self,
        mapping_path="Configs/stix_mapping.json",
        model_name="roberta-base",
        max_len=512,
    ):
        self.scheme = BIEOSScheme(mapping_path)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name,
            add_prefix_space=True,
            use_fast=True,
        )
        self.max_len = max_len

    @staticmethod
    def _overlap(a_start, a_end, b_start, b_end):
        return max(a_start, b_start) < min(a_end, b_end)

    def _char_to_token_span(
        self,
        offsets: List[Tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> List[int]:
        token_ids = []
        for i, (s, e) in enumerate(offsets):
            if s == e:  # special tokens
                continue
            if self._overlap(s, e, char_start, char_end):
                token_ids.append(i)
        return token_ids

    def convert_annotations(self, input_file, output_file):
        if not os.path.exists(input_file):
            raise FileNotFoundError(input_file)

        with open(input_file, "r", encoding="utf-8-sig") as f:
            tasks = json.load(f)

        processed = []

        for task in tasks:
            raw_text = task.get("data", {}).get("text", "")
            if not raw_text.strip():
                continue

            text = fix_mojibake(raw_text)

            annotations = task.get("annotations", [])
            if not annotations:
                continue

            results = annotations[0].get("result", [])
            if not results:
                continue


            entities = {}
            relations = []

            for r in results:
                r_type = r.get("type")
                v = r.get("value", {})

                if (
                    r_type == "labels"
                    and "labels" in v
                    and isinstance(v["labels"], list)
                    and v["labels"]
                    and "start" in v
                    and "end" in v
                ):
                    entities[r["id"]] = {
                        "start": v["start"],
                        "end": v["end"],
                        "type": v["labels"][0],
                        "role": "O",
                    }

                elif r_type == "relation":
                    relations.append(
                        {
                            "from": r.get("from_id"),
                            "to": r.get("to_id"),
                        }
                    )


            for rel in relations:
                if rel["from"] in entities:
                    entities[rel["from"]]["role"] = "1"
                if rel["to"] in entities:
                    entities[rel["to"]]["role"] = "2"

            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_len,
                add_special_tokens=True,
            )

            tokens = encoding.tokens()
            offsets = encoding["offset_mapping"]

            labels = ["O"] * len(tokens)

            for ent in entities.values():
                if ent["role"] == "O":
                    continue

                token_ids = self._char_to_token_span(
                    offsets, ent["start"], ent["end"]
                )

                if not token_ids:
                    continue

                suffix = f"{ent['type']}_{ent['role']}"

                if len(token_ids) == 1:
                    labels[token_ids[0]] = f"S-{suffix}"
                else:
                    labels[token_ids[0]] = f"B-{suffix}"
                    for i in token_ids[1:-1]:
                        labels[i] = f"I-{suffix}"
                    labels[token_ids[-1]] = f"E-{suffix}"


            self._validate_bieos(labels)

            processed.append(
                {
                    "tokens": tokens,
                    "labels": labels,
                    "label_ids": [
                        self.scheme.tag_to_id.get(l, 0) for l in labels
                    ],
                }
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"✅ Converted {len(processed)} samples → {output_file}")


    def _validate_bieos(self, labels: List[str]):
        prev = "O"
        for i, tag in enumerate(labels):
            if tag.startswith(("I-", "E-")) and not prev.startswith(
                ("B-", "I-")
            ):
                labels[i] = "O"
            prev = labels[i]


if __name__ == "__main__":
    converter = DataConverter()
    converter.convert_annotations(
        "Data/Annotations.json",
        "Data/processed/train_bieos.json",
    )
