# Clinical NLP Text Classification Pipeline

> Production clinical NLP pipeline for medical text classification, named entity recognition, and clinical insight extraction using BioBERT and HuggingFace Transformers.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Key Metrics

| Metric | Value |
|--------|-------|
| Text Classification Accuracy | 93% |
| NER F1 Score | 0.91 |
| Clinical Entity Types | 15 |
| Inference Latency | <120ms |
| Daily Records Processed | 500K+ |

## Architecture

```
Raw Clinical Text → Preprocessing → BioBERT Encoder
                                          ↓
                    ┌─────────────────────┴──────────────────────┐
                    ↓                     ↓                      ↓
          Text Classifier            NER Pipeline          Sentiment Analyzer
          (ICD-10 codes)        (15 entity types)       (clinical urgency)
                    └─────────────────────┬──────────────────────┘
                                          ↓
                               Structured Output → MLflow → FastAPI
```

## Features

- **Text Classification** — ICD-10 code prediction from clinical notes with 93% accuracy
- **Named Entity Recognition** — 15 clinical entity types (medications, diagnoses, procedures, vitals)
- **Clinical Sentiment** — Urgency scoring for triage prioritization
- **Batch Processing** — GPU-optimized inference for high-throughput pipelines
- **Model Registry** — MLflow experiment tracking and model versioning
- **Production API** — FastAPI with async batch endpoints

## Tech Stack

- **Models:** BioBERT, ClinicalBERT (HuggingFace), spaCy en_core_sci_lg
- **Framework:** PyTorch, HuggingFace Transformers
- **MLOps:** MLflow, Weights & Biases
- **Serving:** FastAPI, Docker
- **Data:** PySpark for distributed preprocessing

## Quick Start

```bash
git clone https://github.com/SrivardhanBathula/clinical-nlp-text-classification-pipeline
cd clinical-nlp-text-classification-pipeline
pip install -r requirements.txt
python src/train.py --config config/training_config.yaml
```

## API Usage

```python
import httpx

response = httpx.post("http://localhost:8000/predict", json={
    "text": "Patient presents with acute chest pain, SOB, and diaphoresis.",
    "tasks": ["classification", "ner", "sentiment"]
})
print(response.json())
# {
#   "icd10_codes": ["R07.9", "R06.0"],
#   "entities": [{"text": "chest pain", "label": "SYMPTOM"}, ...],
#   "urgency_score": 0.92
# }
```

## Project Structure

```
clinical-nlp-text-classification-pipeline/
├── src/
│   ├── models/
│   │   ├── bert_classifier.py
│   │   ├── ner_pipeline.py
│   │   └── sentiment_model.py
│   ├── data/
│   │   ├── preprocessor.py
│   │   └── dataset.py
│   ├── training/
│   │   └── trainer.py
│   └── inference/
│       └── predictor.py
├── api/
│   └── main.py
├── config/
│   └── training_config.yaml
├── notebooks/
│   └── 01_clinical_nlp_demo.ipynb
├── requirements.txt
└── README.md
```

## Author

**Srivardhan Bathula** — AI/ML Engineer
- Portfolio: [srivardhanbathula.github.io/srivardhanb.github.io](https://srivardhanbathula.github.io/srivardhanb.github.io)
- LinkedIn: [linkedin.com/in/srivardhanb](https://linkedin.com/in/srivardhanb)
