"""
Lightweight FastAPI server for ONNX classification model.
Loads the model from onnx_model/ (produced by train.py --export-onnx) and exposes /v1/classify.
"""

import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "onnx_model"

# Loaded at startup
model = None
tokenizer = None
label_map = None  # dict str -> str (index -> category name)


class DocumentInput(BaseModel):
    id: str = Field(..., description="Client-provided document id")
    text: str = Field(..., description="Document text to classify")


class ClassifyRequest(BaseModel):
    documents: list[DocumentInput] = Field(..., min_length=1)
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_labels: int = Field(10, ge=1, le=200)


class LabelPrediction(BaseModel):
    category: str
    confidence: float


class DocumentPrediction(BaseModel):
    document_id: str
    labels: list[LabelPrediction]
    processing_time_ms: float


class ClassifyResponse(BaseModel):
    predictions: list[DocumentPrediction]
    model_version: str


def load_model():
    global model, tokenizer, label_map
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. Run: uv run train.py --max-steps 50 --export-onnx"
        )
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    with open(MODEL_DIR / "label_map.json") as f:
        label_map = json.load(f)


def predict_one(text: str, confidence_threshold: float, max_labels: int) -> tuple[list[dict], float]:
    """Run inference on one document. Returns (list of {category, confidence}, time_ms)."""
    start = time.perf_counter()
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    logits = model(**inputs).logits
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    probs = probs[0]  # (num_labels,)

    # Threshold, sort by confidence, take top max_labels
    indices = np.where(probs >= confidence_threshold)[0]
    sorted_indices = indices[np.argsort(-probs[indices])][:max_labels]

    labels = [
        {"category": label_map[str(i)], "confidence": float(probs[i])}
        for i in sorted_indices
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return labels, elapsed_ms


app = FastAPI(title="Reuters classifier", version="1.0")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/v1/health")
def health():
    """Model status, label count, and model directory."""
    return {
        "status": "ok",
        "model_version": "onnx_model",
        "num_labels": len(label_map) if label_map else 0,
        "model_dir": str(MODEL_DIR),
    }


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """Classify one or more documents; returns labels above threshold, up to max_labels per doc."""
    predictions = []
    for doc in req.documents:
        labels, elapsed_ms = predict_one(
            doc.text,
            req.confidence_threshold,
            req.max_labels,
        )
        predictions.append(
            DocumentPrediction(
                document_id=doc.id,
                labels=[LabelPrediction(**x) for x in labels],
                processing_time_ms=round(elapsed_ms, 2),
            )
        )
    return ClassifyResponse(predictions=predictions, model_version="onnx_model")
