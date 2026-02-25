"""
Utilities for NLI-based zero-shot category scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class NLICategory:
    name: str
    description: str
    hypothesis_template: str = "This article is about {label_description}."


class NLIClassifier:
    def __init__(
        self,
        model_name: str,
        max_length: int = 384,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.entailment_id, self.contradiction_id = self._resolve_nli_ids()

    def _resolve_nli_ids(self) -> tuple[int, int | None]:
        # Prefer id2label if available and descriptive.
        id2label = getattr(self.model.config, "id2label", None) or {}
        if id2label:
            normalized = {int(k): str(v).lower() for k, v in id2label.items()}
            entailment = next((idx for idx, label in normalized.items() if "entail" in label), None)
            contradiction = next((idx for idx, label in normalized.items() if "contrad" in label), None)
            if entailment is not None:
                return entailment, contradiction

        # Fallback to label2id map.
        label2id = getattr(self.model.config, "label2id", None) or {}
        if label2id:
            lowered = {str(k).lower(): int(v) for k, v in label2id.items()}
            entailment = next((idx for key, idx in lowered.items() if "entail" in key), None)
            contradiction = next((idx for key, idx in lowered.items() if "contrad" in key), None)
            if entailment is not None:
                return entailment, contradiction

        # Most MNLI checkpoints use index 2 for entailment and 0 for contradiction.
        return 2, 0

    def score_categories(
        self,
        text: str,
        categories: list[NLICategory],
        batch_size: int = 16,
    ) -> dict[str, float]:
        if not categories:
            return {}

        hypotheses = [
            category.hypothesis_template.format(label_description=category.description)
            for category in categories
        ]
        scores: dict[str, float] = {}

        for i in range(0, len(categories), batch_size):
            batch_categories = categories[i : i + batch_size]
            batch_hypotheses = hypotheses[i : i + batch_size]
            premises = [text] * len(batch_categories)

            encoded = self.tokenizer(
                premises,
                batch_hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits

            if self.contradiction_id is not None and logits.shape[1] > max(self.entailment_id, self.contradiction_id):
                pair_logits = logits[:, [self.contradiction_id, self.entailment_id]]
                entailment_probs = torch.softmax(pair_logits, dim=1)[:, 1]
            else:
                entailment_probs = torch.softmax(logits, dim=1)[:, self.entailment_id]

            for category, prob in zip(batch_categories, entailment_probs.tolist(), strict=True):
                scores[category.name] = float(prob)

        return scores
