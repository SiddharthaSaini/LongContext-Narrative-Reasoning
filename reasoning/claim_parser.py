import nltk
import re
from nltk.tokenize import sent_tokenize

# Download required resources (runs once, cached later)
nltk.download("punkt")
nltk.download("punkt_tab")

class ClaimParser:
    def __init__(self):
        pass

    def split_into_claims(self, text):
        sentences = sent_tokenize(text)
        claims = [self._clean_sentence(s) for s in sentences if len(s.strip()) > 5]
        return claims

    def _clean_sentence(self, sentence):
        sentence = sentence.strip()
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.lower()
