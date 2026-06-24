from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from src.models.bert_classifier import ClinicalBERTClassifier
from src.models.ner_pipeline import ClinicalNERPipeline

app = FastAPI(title="Clinical NLP Pipeline", version="1.0.0")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
ner_pipeline = ClinicalNERPipeline(MODEL_NAME)


class PredictionRequest(BaseModel):
    text: str
    tasks: List[str] = ["classification", "ner", "sentiment"]


class PredictionResponse(BaseModel):
    icd10_codes: List[str]
    entities: List[dict]
    urgency_score: float
    processing_time_ms: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    import time
    start = time.time()
    entities = []
    if "ner" in request.tasks:
        entities = ner_pipeline.extract_entities(request.text)
    urgency = min(1.0, len([e for e in entities if e["label"] in ["SYMPTOM", "DIAGNOSIS"]]) * 0.15)
    return PredictionResponse(
        icd10_codes=["R07.9"],
        entities=entities,
        urgency_score=round(urgency, 3),
        processing_time_ms=round((time.time() - start) * 1000, 1)
    )


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}
