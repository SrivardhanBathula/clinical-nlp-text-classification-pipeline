import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

CLINICAL_LABELS = [
    "cardiac_event", "respiratory_distress", "sepsis", "neurological",
    "gastrointestinal", "musculoskeletal", "endocrine", "psychiatric",
    "infectious_disease", "oncology", "renal_failure", "hepatic",
    "trauma", "post_operative", "chronic_condition"
]

LABEL2ID = {label: idx for idx, label in enumerate(CLINICAL_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


class ClinicalNotesDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 512
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.texts = texts
        self.label_ids = [LABEL2ID.get(l, 0) for l in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.label_ids[idx], dtype=torch.long)
        }

    @classmethod
    def from_csv(cls, path: str, text_col: str = "note_text", label_col: str = "diagnosis"):
        df = pd.read_csv(path)
        return cls(texts=df[text_col].tolist(), labels=df[label_col].tolist())
