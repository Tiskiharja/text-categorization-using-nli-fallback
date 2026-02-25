"""
Lightweight FastAPI server for ONNX classification model.
Loads the model from onnx_model/ (produced by train.py --export-onnx) and exposes /v1/classify.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from nli_fallback import NLICategory, NLIClassifier

MODEL_DIR = Path(__file__).resolve().parent / "onnx_model"
CATEGORIES_FILE = Path(__file__).resolve().parent / "categories.json"
MODEL_VERSION = "onnx_model+hybrid_nli"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configurable runtime knobs
NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
HYBRID_DISTIL_WEIGHT = float(os.getenv("HYBRID_DISTIL_WEIGHT", "0.6"))
HYBRID_NLI_WEIGHT = float(os.getenv("HYBRID_NLI_WEIGHT", "0.4"))
ENABLE_LOW_CONF_RESCORING = os.getenv("ENABLE_LOW_CONF_RESCORING", "false").lower() in {"1", "true", "yes"}
LOW_CONF_MIN = float(os.getenv("LOW_CONF_MIN", "0.35"))
LOW_CONF_MAX = float(os.getenv("LOW_CONF_MAX", "0.55"))

# Loaded at startup
model = None
tokenizer = None
label_map = None  # dict str -> str (index -> category name)
category_registry: dict[str, dict[str, str]] = {}
nli_classifier: NLIClassifier | None = None
nli_error: str | None = None


class DocumentInput(BaseModel):
    id: str = Field(..., description="Client-provided document id")
    text: str = Field(..., description="Document text to classify")


class ClassifyRequest(BaseModel):
    documents: list[DocumentInput] = Field(..., min_length=1)
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    nli_threshold: float = Field(0.4, ge=0.0, le=1.0)
    max_labels: int = Field(10, ge=1, le=200)
    enable_nli_fallback: bool = True
    include_new_categories: bool = True
    include_debug_scores: bool = False


class LabelPrediction(BaseModel):
    category: str
    confidence: float
    source: str
    debug_scores: dict[str, float] | None = None


class DocumentPrediction(BaseModel):
    document_id: str
    labels: list[LabelPrediction]
    processing_time_ms: float


class ClassifyResponse(BaseModel):
    predictions: list[DocumentPrediction]
    model_version: str


class CategoryInfo(BaseModel):
    name: str
    description: str
    status: str
    in_distilbert_model: bool
    hypothesis_template: str


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


def build_category_registry() -> dict[str, dict[str, str]]:
    # Start from training labels so existing behavior remains intact by default.
    registry = {}
    for label_name in label_map.values():
        registry[label_name] = {
            "name": label_name,
            "description": label_name,
            "status": "known",
            "hypothesis_template": "This article is about {label_description}.",
        }

    if not CATEGORIES_FILE.is_file():
        logger.warning("No categories file found at %s. Using label_map-derived categories only.", CATEGORIES_FILE)
        return registry

    with open(CATEGORIES_FILE) as f:
        data = json.load(f)

    categories = data.get("categories", data) if isinstance(data, dict) else data
    if not isinstance(categories, list):
        raise ValueError("categories.json must be a list or an object with a 'categories' list.")

    for item in categories:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        description = str(item.get("description") or name).strip()
        status = str(item.get("status") or ("known" if name in registry else "new")).strip().lower()
        template = str(item.get("hypothesis_template") or "This article is about {label_description}.").strip()
        registry[name] = {
            "name": name,
            "description": description,
            "status": status,
            "hypothesis_template": template,
        }
    return registry


def load_nli_model() -> None:
    global nli_classifier, nli_error
    try:
        nli_classifier = NLIClassifier(model_name=NLI_MODEL_NAME)
        nli_error = None
        logger.info("Loaded NLI fallback model: %s", NLI_MODEL_NAME)
    except Exception as exc:
        nli_classifier = None
        nli_error = str(exc)
        logger.warning("NLI model unavailable (%s). Running DistilBERT-only mode.", exc)


def _is_new_category(category_name: str) -> bool:
    category_meta = category_registry.get(category_name, {})
    status = category_meta.get("status", "").lower()
    if status == "new":
        return True
    if status == "known":
        return False
    return category_name not in label_map.values()


def _distilbert_probs(text: str) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    logits = model(**inputs).logits
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    return probs[0]  # (num_labels,)


def _build_nli_candidates(distil_probs: np.ndarray, include_new_categories: bool) -> list[NLICategory]:
    candidates: dict[str, NLICategory] = {}

    if include_new_categories:
        for name, meta in category_registry.items():
            if _is_new_category(name):
                candidates[name] = NLICategory(
                    name=name,
                    description=meta["description"],
                    hypothesis_template=meta["hypothesis_template"],
                )

    if ENABLE_LOW_CONF_RESCORING:
        for index, score in enumerate(distil_probs):
            if LOW_CONF_MIN <= score <= LOW_CONF_MAX:
                name = label_map[str(index)]
                meta = category_registry.get(name, {})
                candidates[name] = NLICategory(
                    name=name,
                    description=meta.get("description", name),
                    hypothesis_template=meta.get("hypothesis_template", "This article is about {label_description}."),
                )

    return list(candidates.values())


def predict_one(req: ClassifyRequest, text: str) -> tuple[list[dict[str, Any]], float]:
    """Run hybrid inference on one document. Returns (labels, elapsed_ms)."""
    start = time.perf_counter()
    distil_probs = _distilbert_probs(text)

    nli_scores: dict[str, float] = {}
    if req.enable_nli_fallback and nli_classifier is not None:
        candidates = _build_nli_candidates(distil_probs, req.include_new_categories)
        nli_scores = nli_classifier.score_categories(text=text, categories=candidates)

    predictions: list[dict[str, Any]] = []
    for i, distil_score in enumerate(distil_probs):
        label_name = label_map[str(i)]
        nli_score = nli_scores.get(label_name)
        if nli_score is None:
            final_score = float(distil_score)
            source = "distilbert"
        else:
            final_score = float((HYBRID_DISTIL_WEIGHT * distil_score) + (HYBRID_NLI_WEIGHT * nli_score))
            source = "hybrid"

        if final_score >= req.confidence_threshold:
            row: dict[str, Any] = {
                "category": label_name,
                "confidence": final_score,
                "source": source,
            }
            if req.include_debug_scores:
                row["debug_scores"] = {
                    "distilbert": float(distil_score),
                    "nli": float(nli_score) if nli_score is not None else -1.0,
                    "final": final_score,
                }
            predictions.append(row)

    for category_name, nli_score in nli_scores.items():
        if category_name in label_map.values():
            continue
        if nli_score < req.nli_threshold:
            continue
        row = {
            "category": category_name,
            "confidence": float(nli_score),
            "source": "nli",
        }
        if req.include_debug_scores:
            row["debug_scores"] = {
                "distilbert": -1.0,
                "nli": float(nli_score),
                "final": float(nli_score),
            }
        predictions.append(row)

    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    predictions = predictions[: req.max_labels]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return predictions, elapsed_ms


app = FastAPI(title="Reuters classifier", version="1.0")


@app.on_event("startup")
def startup():
    global category_registry
    load_model()
    category_registry = build_category_registry()
    load_nli_model()


@app.get("/v1/health")
def health():
    """Model status, label count, and model directory."""
    new_categories_count = sum(1 for name in category_registry if _is_new_category(name))
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "num_labels": len(label_map) if label_map else 0,
        "categories_count": len(category_registry),
        "new_categories_count": new_categories_count,
        "nli_loaded": nli_classifier is not None,
        "nli_model_name": NLI_MODEL_NAME,
        "nli_error": nli_error,
        "model_dir": str(MODEL_DIR),
    }


@app.get("/v1/categories")
def categories() -> dict[str, Any]:
    known = []
    new = []
    for name in sorted(category_registry.keys()):
        meta = category_registry[name]
        info = CategoryInfo(
            name=name,
            description=meta["description"],
            status="new" if _is_new_category(name) else "known",
            in_distilbert_model=name in label_map.values(),
            hypothesis_template=meta["hypothesis_template"],
        )
        if info.status == "new":
            new.append(info)
        else:
            known.append(info)

    return {
        "counts": {
            "total": len(category_registry),
            "known": len(known),
            "new": len(new),
        },
        "categories": [c.model_dump() for c in known + new],
    }


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """Classify one or more documents; returns labels above threshold, up to max_labels per doc."""
    predictions = []
    for doc in req.documents:
        labels, elapsed_ms = predict_one(
            req,
            doc.text,
        )
        predictions.append(
            DocumentPrediction(
                document_id=doc.id,
                labels=[LabelPrediction(**x) for x in labels],
                processing_time_ms=round(elapsed_ms, 2),
            )
        )
    return ClassifyResponse(predictions=predictions, model_version=MODEL_VERSION)
