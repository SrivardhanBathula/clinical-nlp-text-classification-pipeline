import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict, Any
import torch


CLINICAL_ENTITY_TYPES = [
    "MEDICATION", "DOSAGE", "FREQUENCY", "ROUTE",
    "DIAGNOSIS", "SYMPTOM", "PROCEDURE", "ANATOMY",
    "VITAL_SIGN", "LAB_VALUE", "TEMPORAL", "NEGATION",
    "PATIENT_INFO", "PHYSICIAN", "FACILITY"
]


class ClinicalNERPipeline:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: int = 0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                           aggregation_strategy="simple", device=device)
        self.nlp = spacy.load("en_core_sci_lg")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        hf_entities = self.ner(text)
        spacy_doc = self.nlp(text)
        entities = []
        for ent in hf_entities:
            entities.append({
                "text": ent["word"],
                "label": ent["entity_group"],
                "score": round(ent["score"], 4),
                "start": ent["start"],
                "end": ent["end"],
                "source": "biobert"
            })
        for ent in spacy_doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "score": 1.0,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "scispacy"
            })
        return entities

    def batch_extract(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        return [self.extract_entities(text) for text in texts]
