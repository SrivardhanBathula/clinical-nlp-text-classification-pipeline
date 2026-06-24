import re
import unicodedata
from typing import List, Optional
import spacy
import logging

logger = logging.getLogger(__name__)


class ClinicalTextPreprocessor:
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_sci_sm")
            except OSError:
                logger.warning("Sci-spaCy model not found. Using en_core_web_sm.")
                self.nlp = spacy.load("en_core_web_sm")

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\[\*\*.*?\*\*\]", "[REDACTED]", text)
        text = re.sub(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "[DATE]", text)
        text = re.sub(r"\d{3}[-.]?\d{3}[-.]?\d{4}", "[PHONE]", text)
        text = re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN]", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def batch_clean(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    def extract_sentences(self, text: str) -> List[str]:
        if self.use_spacy:
            doc = self.nlp(text[:100000])
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        return [s.strip() for s in text.split(".") if len(s.strip()) > 10]

    def segment_sections(self, text: str) -> dict:
        sections = {}
        section_headers = [
            "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY",
            "MEDICATIONS", "ALLERGIES", "PHYSICAL EXAMINATION", "ASSESSMENT", "PLAN"
        ]
        current_section = "GENERAL"
        current_text = []
        for line in text.split("\n"):
            matched = False
            for header in section_headers:
                if header in line.upper():
                    if current_text:
                        sections[current_section] = " ".join(current_text)
                    current_section = header
                    current_text = []
                    matched = True
                    break
            if not matched:
                current_text.append(line)
        if current_text:
            sections[current_section] = " ".join(current_text)
        return sections
