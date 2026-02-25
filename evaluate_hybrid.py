"""
Evaluate hybrid DistilBERT + NLI classification.

This script simulates newly added categories by holding out existing Reuters labels
and measuring NLI-only detection on those held-out labels.
It also tunes fusion weights on a known-label subset using DistilBERT + NLI scores.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer

from nli_fallback import NLICategory, NLIClassifier
from train import extract_reuters_data, parse_reuters_sgml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "onnx_model"
DEFAULT_CATEGORIES_FILE = PROJECT_ROOT / "categories.json"


def parse_list(arg: str) -> list[float]:
    return [float(x.strip()) for x in arg.split(",") if x.strip()]


def load_category_descriptions(categories_file: Path) -> dict[str, str]:
    if not categories_file.is_file():
        return {}
    with open(categories_file) as f:
        data = json.load(f)
    categories = data.get("categories", data) if isinstance(data, dict) else data
    if not isinstance(categories, list):
        return {}

    descriptions: dict[str, str] = {}
    for item in categories:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        descriptions[name] = str(item.get("description") or name).strip()
    return descriptions


def load_test_docs(max_docs: int = -1) -> list[dict]:
    sgm_dir = extract_reuters_data()
    docs = parse_reuters_sgml(sgm_dir)
    test_docs = [
        d for d in docs
        if d["topics_attr"] == "YES"
        and d["lewissplit"] == "TEST"
        and d["text"].strip()
        and d["topics"]
    ]
    if max_docs > 0:
        test_docs = test_docs[:max_docs]
    logger.info("Loaded %d test documents", len(test_docs))
    return test_docs


def distilbert_scores(
    texts: list[str],
    model: ORTModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="np",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        logits = model(**encoded).logits
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def nli_scores(
    texts: list[str],
    categories: list[NLICategory],
    nli: NLIClassifier,
    batch_size: int,
) -> np.ndarray:
    scores = np.zeros((len(texts), len(categories)), dtype=np.float32)
    for i, text in enumerate(texts):
        scored = nli.score_categories(text=text, categories=categories, batch_size=batch_size)
        for j, category in enumerate(categories):
            scores[i, j] = scored.get(category.name, 0.0)
        if (i + 1) % 100 == 0:
            logger.info("NLI scored %d/%d documents", i + 1, len(texts))
    return scores


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true.reshape(-1),
        y_pred.reshape(-1),
        average="binary",
        zero_division=0,
    )
    return float(p), float(r), float(f)


def pick_holdout_labels(
    test_docs: list[dict],
    label_names: list[str],
    holdout_k: int,
    min_support: int,
) -> list[str]:
    supports = {name: 0 for name in label_names}
    for doc in test_docs:
        topics = set(doc["topics"])
        for name in label_names:
            if name in topics:
                supports[name] += 1

    eligible = [(name, count) for name, count in supports.items() if count >= min_support]
    eligible.sort(key=lambda x: x[1], reverse=True)
    chosen = [name for name, _ in eligible[:holdout_k]]
    logger.info("Holdout labels: %s", chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid DistilBERT + NLI behavior.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--categories-file", type=Path, default=DEFAULT_CATEGORIES_FILE)
    parser.add_argument("--nli-model-name", default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
    parser.add_argument("--max-docs", type=int, default=800)
    parser.add_argument("--holdout-k", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--known-eval-k", type=int, default=20)
    parser.add_argument("--distil-threshold", type=float, default=0.5)
    parser.add_argument("--nli-thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--fusion-thresholds", default="0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--distil-weights", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--nli-weights", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--nli-batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    with open(args.model_dir / "label_map.json") as f:
        label_map = json.load(f)
    label_names = [label_map[str(i)] for i in range(len(label_map))]
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    category_descriptions = load_category_descriptions(args.categories_file)

    test_docs = load_test_docs(max_docs=args.max_docs)
    texts = [d["text"] for d in test_docs]
    topics_list = [set(d["topics"]) for d in test_docs]

    holdout_labels = pick_holdout_labels(
        test_docs=test_docs,
        label_names=label_names,
        holdout_k=args.holdout_k,
        min_support=args.min_support,
    )
    if len(holdout_labels) < args.holdout_k:
        raise ValueError(
            f"Only {len(holdout_labels)} labels met min support {args.min_support}. "
            f"Lower --min-support or --holdout-k."
        )

    holdout_set = set(holdout_labels)
    known_labels = [name for name in label_names if name not in holdout_set]

    model = ORTModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    nli = NLIClassifier(model_name=args.nli_model_name)

    logger.info("Scoring DistilBERT ONNX on %d docs", len(texts))
    distil_probs = distilbert_scores(texts=texts, model=model, tokenizer=tokenizer)

    true_holdout = np.zeros((len(texts), len(holdout_labels)), dtype=np.int64)
    for i, topics in enumerate(topics_list):
        for j, label in enumerate(holdout_labels):
            true_holdout[i, j] = int(label in topics)

    holdout_categories = [
        NLICategory(
            name=name,
            description=category_descriptions.get(name, name),
        )
        for name in holdout_labels
    ]
    logger.info("Scoring NLI on held-out labels")
    holdout_nli_scores = nli_scores(
        texts=texts,
        categories=holdout_categories,
        nli=nli,
        batch_size=args.nli_batch_size,
    )

    nli_thresholds = parse_list(args.nli_thresholds)
    best_new = None
    for thr in nli_thresholds:
        pred = (holdout_nli_scores >= thr).astype(np.int64)
        p, r, f = prf(true_holdout, pred)
        logger.info("NEW labels NLI-only @ thr=%.2f -> P=%.4f R=%.4f F1=%.4f", thr, p, r, f)
        candidate = {"threshold": thr, "precision": p, "recall": r, "f1": f}
        if best_new is None or candidate["f1"] > best_new["f1"]:
            best_new = candidate

    known_indices = [label_to_idx[name] for name in known_labels]
    true_known = np.zeros((len(texts), len(known_labels)), dtype=np.int64)
    for i, topics in enumerate(topics_list):
        for j, label in enumerate(known_labels):
            true_known[i, j] = int(label in topics)
    known_distil_scores = distil_probs[:, known_indices]
    known_distil_pred = (known_distil_scores >= args.distil_threshold).astype(np.int64)
    kp, kr, kf = prf(true_known, known_distil_pred)
    logger.info(
        "KNOWN labels DistilBERT-only @ thr=%.2f -> P=%.4f R=%.4f F1=%.4f",
        args.distil_threshold, kp, kr, kf
    )

    # Fusion tuning on a frequent known-label subset to keep NLI runtime practical.
    known_support = []
    for label in known_labels:
        support = sum(1 for topics in topics_list if label in topics)
        known_support.append((label, support))
    known_support.sort(key=lambda x: x[1], reverse=True)
    known_eval_labels = [name for name, _ in known_support[: args.known_eval_k]]
    logger.info("Fusion tuning labels (known subset): %s", known_eval_labels)

    known_eval_categories = [
        NLICategory(
            name=name,
            description=category_descriptions.get(name, name),
        )
        for name in known_eval_labels
    ]
    logger.info("Scoring NLI for known-label fusion tuning")
    known_nli_scores = nli_scores(
        texts=texts,
        categories=known_eval_categories,
        nli=nli,
        batch_size=args.nli_batch_size,
    )

    known_eval_indices = [label_to_idx[name] for name in known_eval_labels]
    known_eval_distil = distil_probs[:, known_eval_indices]
    true_known_eval = np.zeros((len(texts), len(known_eval_labels)), dtype=np.int64)
    for i, topics in enumerate(topics_list):
        for j, label in enumerate(known_eval_labels):
            true_known_eval[i, j] = int(label in topics)

    distil_weights = parse_list(args.distil_weights)
    nli_weights = parse_list(args.nli_weights)
    fusion_thresholds = parse_list(args.fusion_thresholds)

    best_fusion = None
    for dw in distil_weights:
        for nw in nli_weights:
            for thr in fusion_thresholds:
                fused = (dw * known_eval_distil) + (nw * known_nli_scores)
                pred = (fused >= thr).astype(np.int64)
                p, r, f = prf(true_known_eval, pred)
                logger.info(
                    "FUSION dw=%.2f nw=%.2f thr=%.2f -> P=%.4f R=%.4f F1=%.4f",
                    dw, nw, thr, p, r, f
                )
                candidate = {
                    "distil_weight": dw,
                    "nli_weight": nw,
                    "threshold": thr,
                    "precision": p,
                    "recall": r,
                    "f1": f,
                }
                if best_fusion is None or candidate["f1"] > best_fusion["f1"]:
                    best_fusion = candidate

    summary = {
        "docs_evaluated": len(texts),
        "holdout_labels": holdout_labels,
        "best_new_label_nli": best_new,
        "known_distil_baseline": {
            "threshold": args.distil_threshold,
            "precision": kp,
            "recall": kr,
            "f1": kf,
        },
        "fusion_tuning_subset_labels": known_eval_labels,
        "best_known_fusion": best_fusion,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
