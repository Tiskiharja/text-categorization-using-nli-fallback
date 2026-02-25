"""
Lightweight FastAPI server for ONNX classification model.
Loads the model from onnx_model/ (produced by train.py --export-onnx) and exposes /v1/classify.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import yaml

from nli_fallback import NLICategory, NLIClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.yaml"
MODEL_VERSION = "onnx_model+hybrid_nli"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class RuntimeConfig:
    model_dir: Path
    categories_file: Path
    nli_backend: str
    nli_model_dir: Path
    nli_model_name: str
    hybrid_distil_weight: float
    hybrid_nli_weight: float
    enable_low_conf_rescoring: bool
    low_conf_min: float
    low_conf_max: float

# Loaded at startup
app_config: RuntimeConfig | None = None
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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_config() -> RuntimeConfig:
    config_file = Path(os.getenv("APP_CONFIG_FILE", str(DEFAULT_CONFIG_FILE))).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_file}. "
            "Create it (example: config.yaml) or set APP_CONFIG_FILE=/path/to/config.yaml"
        )

    with open(config_file) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a mapping/object at the top level.")

    cfg = RuntimeConfig(
        model_dir=(PROJECT_ROOT / str(raw.get("model_dir", "onnx_model"))).resolve(),
        categories_file=(PROJECT_ROOT / str(raw.get("categories_file", "categories.json"))).resolve(),
        nli_backend=str(raw.get("nli_backend", "onnx")).strip().lower(),
        nli_model_dir=(PROJECT_ROOT / str(raw.get("nli_model_dir", "onnx_nli_model"))).resolve(),
        nli_model_name=str(raw.get("nli_model_name", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0")),
        hybrid_distil_weight=float(raw.get("hybrid_distil_weight", 0.6)),
        hybrid_nli_weight=float(raw.get("hybrid_nli_weight", 0.4)),
        enable_low_conf_rescoring=_to_bool(raw.get("enable_low_conf_rescoring", False)),
        low_conf_min=float(raw.get("low_conf_min", 0.35)),
        low_conf_max=float(raw.get("low_conf_max", 0.55)),
    )

    if cfg.low_conf_min > cfg.low_conf_max:
        raise ValueError(f"Invalid config: low_conf_min ({cfg.low_conf_min}) > low_conf_max ({cfg.low_conf_max})")
    if cfg.nli_backend not in {"onnx", "torch"}:
        raise ValueError(f"Invalid config: nli_backend must be 'onnx' or 'torch', got '{cfg.nli_backend}'")

    logger.info("Loaded runtime config from %s", config_file)
    return cfg


def get_config() -> RuntimeConfig:
    if app_config is None:
        raise RuntimeError("App config not loaded yet.")
    return app_config


def load_model():
    global model, tokenizer, label_map
    cfg = get_config()
    if not cfg.model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {cfg.model_dir}. Run: uv run train.py --max-steps 50 --export-onnx"
        )
    model = ORTModelForSequenceClassification.from_pretrained(cfg.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    with open(cfg.model_dir / "label_map.json") as f:
        label_map = json.load(f)


def build_category_registry() -> dict[str, dict[str, str]]:
    cfg = get_config()
    # Start from training labels so existing behavior remains intact by default.
    registry = {}
    for label_name in label_map.values():
        registry[label_name] = {
            "name": label_name,
            "description": label_name,
            "status": "known",
            "hypothesis_template": "This article is about {label_description}.",
        }

    if not cfg.categories_file.is_file():
        logger.warning("No categories file found at %s. Using label_map-derived categories only.", cfg.categories_file)
        return registry

    with open(cfg.categories_file) as f:
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
    cfg = get_config()
    try:
        nli_classifier = NLIClassifier(
            model_name=cfg.nli_model_name,
            backend=cfg.nli_backend,
            model_dir=cfg.nli_model_dir,
        )
        nli_error = None
        logger.info(
            "Loaded NLI fallback model: backend=%s, source=%s",
            nli_classifier.backend,
            cfg.nli_model_dir if nli_classifier.backend == "onnx" else cfg.nli_model_name,
        )
    except Exception as exc:
        if cfg.nli_backend == "onnx":
            logger.warning("ONNX NLI load failed (%s). Falling back to torch backend.", exc)
            try:
                nli_classifier = NLIClassifier(model_name=cfg.nli_model_name, backend="torch")
                nli_error = None
                logger.info("Loaded NLI fallback model: backend=torch, source=%s", cfg.nli_model_name)
                return
            except Exception as fallback_exc:
                nli_classifier = None
                nli_error = str(fallback_exc)
                logger.warning("NLI model unavailable (%s). Running DistilBERT-only mode.", fallback_exc)
                return

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
    cfg = get_config()
    candidates: dict[str, NLICategory] = {}

    if include_new_categories:
        for name, meta in category_registry.items():
            if _is_new_category(name):
                candidates[name] = NLICategory(
                    name=name,
                    description=meta["description"],
                    hypothesis_template=meta["hypothesis_template"],
                )

    if cfg.enable_low_conf_rescoring:
        for index, score in enumerate(distil_probs):
            if cfg.low_conf_min <= score <= cfg.low_conf_max:
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
    cfg = get_config()
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
            final_score = float((cfg.hybrid_distil_weight * distil_score) + (cfg.hybrid_nli_weight * nli_score))
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
    global app_config, category_registry
    app_config = load_runtime_config()
    load_model()
    category_registry = build_category_registry()
    load_nli_model()


@app.get("/v1/health")
def health():
    """Model status, label count, and model directory."""
    cfg = get_config()
    new_categories_count = sum(1 for name in category_registry if _is_new_category(name))
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "num_labels": len(label_map) if label_map else 0,
        "categories_count": len(category_registry),
        "new_categories_count": new_categories_count,
        "nli_loaded": nli_classifier is not None,
        "nli_backend": nli_classifier.backend if nli_classifier is not None else cfg.nli_backend,
        "nli_model_dir": str(cfg.nli_model_dir),
        "nli_model_name": cfg.nli_model_name,
        "nli_error": nli_error,
        "model_dir": str(cfg.model_dir),
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
