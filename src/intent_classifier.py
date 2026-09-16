from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import ROOT


class IntentClassifier:
    def __init__(self, model_path: str | Path = None):
        """
        Loads the locally saved DistilBERT intent classifier.
        """
        if model_path is None:
            model_path = ROOT / "models" / "intent_classifier_distilbert"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        print(f"Loading Intent Classifier from local path: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.id2label = self.model.config.id2label
        print(f"Intent Classifier ready on {self.device} | Labels: {len(self.id2label)}")

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        Returns:
        {
            "label": "delivery_late",
            "confidence": 0.87,
            "all_scores": { ... }
        }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

        confidence, pred_id = torch.max(probs, dim=0)
        label = self.id2label[pred_id.item()]

        all_scores = {
            self.id2label[i]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "label": label,
            "confidence": float(confidence),
            "all_scores": all_scores
        }