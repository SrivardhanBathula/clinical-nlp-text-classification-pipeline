import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
import mlflow
import mlflow.pytorch
import numpy as np
import logging
from typing import Optional
from ..models.bert_classifier import ClinicalBERTClassifier

logger = logging.getLogger(__name__)


class ClinicalNLPTrainer:
    def __init__(
        self,
        model: ClinicalBERTClassifier,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        batch_size: int = 32,
        warmup_steps: int = 500,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

    def train(self, train_loader: DataLoader, val_loader: DataLoader, experiment_name: str = "clinical-nlp"):
        mlflow.set_experiment(experiment_name)
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, total_steps)
        criterion = torch.nn.CrossEntropyLoss()
        best_f1 = 0.0

        with mlflow.start_run():
            mlflow.log_params({"lr": self.lr, "epochs": self.num_epochs, "batch_size": self.batch_size})
            for epoch in range(self.num_epochs):
                self.model.train()
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()

                val_f1 = self._evaluate(val_loader)
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} — Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")
                mlflow.log_metrics({"train_loss": avg_loss, "val_f1": val_f1}, step=epoch)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    mlflow.pytorch.log_model(self.model, "best_model",
                                            registered_model_name="clinical_bert_classifier")

            logger.info(f"Training complete. Best Val F1: {best_f1:.4f}")
            return best_f1

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        return f1_score(all_labels, all_preds, average="weighted", zero_division=0)
