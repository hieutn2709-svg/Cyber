import re
from typing import List, Dict

class IOCFinder:
    def __init__(self):
        ipv4_segment = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
        
        self.patterns = {
            "vulnerability": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
            
            "ipv4": re.compile(
                rf"\b{ipv4_segment}\.{ipv4_segment}\.{ipv4_segment}\.{ipv4_segment}\b"
            ),
            
            "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
            "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
            "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
            
            "email": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),
            
            "attack-pattern": re.compile(r"\bT\d{4}(?:\.\d{3})?\b"),
            
            "file-path": re.compile(r"\b(?:[a-zA-Z]:\\|/)[\w\-.\\/ ]+")
        }

    def extract(self, text: str) -> List[Dict]:
        if not text:
            return []

        extracted_iocs = []

        for ioc_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                value = match.group().strip()

                if ioc_type == "file-path" and (len(value) < 5 or "." not in value):
                    continue

                extracted_iocs.append({
                    "type": ioc_type,
                    "value": value,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                    "source": "ioc_finder"
                })

        return extracted_iocs


if __name__ == "__main__":
    finder = IOCFinder()
    sample = """
    Mã độc Beacon (T1055) kết nối tới C2 tại 192.168.10.5.
    Tệp tin độc hại: C:\\Windows\\temp\\malware.exe
    MD5: e802c6b77dd5842906ed96ab1674c525
    Lỗ hổng CVE-2021-44228
    """

    results = finder.extract(sample)
    print(f"Tìm thấy {len(results)} thực thể:")
    for r in results:
        print(f"[{r['type']}] {r['value']} ({r['start']}-{r['end']})")
