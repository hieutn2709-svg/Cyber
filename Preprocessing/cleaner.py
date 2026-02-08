import re

class TextCleaner:
    def __init__(self):
        self.desanitize_patterns = [
            (re.compile(r'\[\.\]', re.IGNORECASE), '.'),    # [.] -> .
            (re.compile(r'\(\.\)', re.IGNORECASE), '.'),    # (.) -> .
            (re.compile(r'\{.\}', re.IGNORECASE), '.'),     # {.} -> .
            (re.compile(r'\[dot\]', re.IGNORECASE), '.'),   # [dot] -> .
            (re.compile(r'hxxp', re.IGNORECASE), 'http'),   # hxxp -> http
            (re.compile(r'\[:\]', re.IGNORECASE), ':'),     # [:] -> :
            (re.compile(r'\[/\]', re.IGNORECASE), '/'),     # [/] -> /
            (re.compile(r'\\\.', re.IGNORECASE), '.'),      # \. -> .
        ]

    def remove_noise(self, text: str) -> str:
        cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        return cleaned_text.strip()

    def fix_line_breaks(self, text: str) -> str:
        text = re.sub(r'(?<![.!?])\n', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text

    def desanitize(self, text: str) -> str:
        cleaned_text = text
        for pattern, replacement in self.desanitize_patterns:
            cleaned_text = pattern.sub(replacement, cleaned_text)
        return cleaned_text

    def clean(self, text: str) -> str:
        if not text:
            return ""
        
        text = self.fix_line_breaks(text)
        text = self.desanitize(text)
        text = self.remove_noise(text)
        
        return text

if __name__ == "__main__":
    cleaner = TextCleaner()
    # sample_text = "Báo cáo: hxxps[:]//malicious[.]com/path\nIP: 192.168[.]1[.]100"
    # print(cleaner.clean(sample_text))
