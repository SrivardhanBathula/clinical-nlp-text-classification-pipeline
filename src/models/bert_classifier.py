import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any
import mlflow


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)
        return logits

    def predict(self, texts: List[str], tokenizer, device: str = "cuda") -> List[Dict[str, Any]]:
        self.eval()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.forward(**inputs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return [{"label_id": p.item(), "confidence": probs[i][p].item()} for i, p in enumerate(preds)]

    @classmethod
    def load_from_mlflow(cls, run_id: str, model_name: str, num_labels: int):
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        return model
