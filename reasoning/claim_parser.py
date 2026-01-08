import re
import nltk
from nltk.tokenize import sent_tokenize

# Download once safely
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")


class ClaimParser:
    def split_into_claims(self, text):
        sentences = sent_tokenize(text)
        return [self._clean(s) for s in sentences if len(s.strip()) > 5]

    def _clean(self, s):
        s = re.sub(r"\s+", " ", s.strip())
        return s.lower()
