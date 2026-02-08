import re
import json
from typing import List, Dict

class TTPExtractor:
    def __init__(self, model_path: str = None):
        self.system_prompt = "You are a Cyber Security Expert who finds MITRE ATT&CK framework concepts in CTI reports."
        self.user_instruction = (
            "List all MITRE ATT&CK (Sub)Technique IDs and Names that occur in the sentence "
            "from a CTI report, surrounded by grave accents (`). Do not give any assumptions, "
            "possibilities or reasons! Please provide concise, non-repetitive answers."
        )
        
        self.model_path = model_path
        self.is_model_ready = False if model_path is None else True

    def _format_instruction(self, text: str) -> str:
        return f"{self.user_instruction}\nSentence to analyze: `{text}`"

    def extract(self, text: str) -> List[Dict]:

        if not text:
            return []

        ai_response = self._call_model_inference(text)
        
        extracted_ttps = self._parse_ai_response(ai_response, text)
        
        return extracted_ttps

    def _call_model_inference(self, text: str) -> str:
        if "Invoice" in text or "malspam" in text:
            return "* T1566: Phishing"
        if "macros" in text or "Excel" in text:
            return "* T1059: Command and Scripting Interpreter"
        if "rundll32.exe" in text:
            return "* T1218: System Binary Proxy Execution"
        return "None"

    def _parse_ai_response(self, response: str, original_text: str) -> List[Dict]:
        results = []
        pattern = re.compile(r"(T\d{4}(?:\.\d{3})?|TA\d{4})")
        
        matches = pattern.findall(response)
        for mitre_id in set(matches):
            results.append({
                "type": "attack-pattern",
                "value": mitre_id,
                "confidence": 0.85, 
                "source": "ttp_extractor_bosch",
                "is_implicit": True
            })
        return results

if __name__ == "__main__":
    extractor = TTPExtractor()
    
    sample_cti = "Considering the nature of the documents usually named Invoice, we assess the goal was bank transfers."
    
    findings = extractor.extract(sample_cti)
    
    print(f"Văn bản: {sample_cti}")
    print(f"TTPs trích xuất được:")
    for f in findings:
        print(json.dumps(f, indent=2))