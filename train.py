"""
Minimal Reuters-21578 training pipeline: DistilBERT multi-label topic classification.

For someone from recsys: think of this as N binary classifiers (one per topic) sharing
the same text encoder, trained with BCE loss — like conversion prediction per label.
"""

import argparse
import json
import logging
import tarfile
from pathlib import Path

import numpy as np
import torch
from bs4 import BeautifulSoup
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
REUTERS_ARCHIVE = PROJECT_ROOT / "reuters+21578+text+categorization+collection" / "reuters21578.tar.gz"
DATA_DIR = PROJECT_ROOT / "data"


def extract_reuters_data() -> Path:
    """
    Untar reuters21578.tar.gz into data/. Returns the directory containing .sgm files.
    The archive may put .sgm files at top level or in a subdir; we search for them.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extract_to = DATA_DIR / "reuters21578"
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(REUTERS_ARCHIVE, "r:gz") as tf:
        tf.extractall(extract_to)

    # Find any .sgm file to get the parent dir that contains all of them
    sgm_files = list(extract_to.rglob("*.sgm"))
    if not sgm_files:
        raise FileNotFoundError(f"No .sgm files found under {extract_to}")
    sgm_dir = sgm_files[0].parent
    logger.info("Extracted Reuters data: %d .sgm files in %s", len(sgm_files), sgm_dir)
    return sgm_dir


def parse_reuters_sgml(sgm_dir: Path) -> list[dict]:
    """
    Parse all .sgm files. Each document has LEWISSPLIT (TRAIN/TEST/NOT-USED),
    TOPICS attribute (YES/NO/BYPASS), and optional <TOPICS><D>...</D></TOPICS> and <TEXT>.
    Returns list of dicts: {"text", "topics", "lewissplit", "topics_attr"}.
    """
    docs = []
    for sgm_path in sorted(sgm_dir.glob("*.sgm")):
        with open(sgm_path, "r", encoding="latin-1") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, "lxml")
        for reuters in soup.find_all("reuters"):
            attrs = reuters.attrs
            lewissplit = (attrs.get("lewissplit") or "").upper()
            topics_attr = (attrs.get("topics") or "").upper()

            topic_tags = reuters.find("topics")
            topics = []
            if topic_tags:
                for d in topic_tags.find_all("d"):
                    if d.string and d.string.strip():
                        topics.append(d.string.strip())

            text_elem = reuters.find("text")
            parts = []
            if text_elem:
                title = text_elem.find("title")
                if title and title.string:
                    parts.append(title.string.strip())
                body = text_elem.find("body")
                if body and body.string:
                    parts.append(body.string.strip())
            text = " ".join(parts).strip() or ""

            docs.append({
                "text": text,
                "topics": topics,
                "lewissplit": lewissplit,
                "topics_attr": topics_attr,
            })

    logger.info("Parsed %d documents from SGML", len(docs))
    return docs


def build_datasets(docs: list[dict]) -> tuple[Dataset, Dataset, list[str]]:
    """
    ModApte split: only TOPICS="YES", split by LEWISSPLIT (TRAIN / TEST).
    Multi-hot encoding over labels that appear in the *training* set only,
    so train and test share the same label space (~90 labels).
    """
    train_docs = [d for d in docs if d["topics_attr"] == "YES" and d["lewissplit"] == "TRAIN"]
    test_docs = [d for d in docs if d["topics_attr"] == "YES" and d["lewissplit"] == "TEST"]

    # Build label vocabulary from training set only (standard practice)
    all_train_topics = set()
    for d in train_docs:
        all_train_topics.update(d["topics"])
    label_list = sorted(all_train_topics)
    label2id = {t: i for i, t in enumerate(label_list)}
    num_labels = len(label_list)
    logger.info("ModApte: %d train, %d test, %d topic labels", len(train_docs), len(test_docs), num_labels)

    def to_multi_hot(topics: list[str]) -> list[float]:
        vec = [0.0] * num_labels
        for t in topics:
            if t in label2id:
                vec[label2id[t]] = 1.0
        return vec

    def to_dataset(doc_list: list[dict]) -> Dataset:
        texts = [d["text"] for d in doc_list]
        labels = [to_multi_hot(d["topics"]) for d in doc_list]
        return Dataset.from_dict({"text": texts, "labels": labels})

    train_ds = to_dataset(train_docs)
    test_ds = to_dataset(test_docs)
    return train_ds, test_ds, label_list


def tokenize_dataset(
    train_ds: Dataset,
    test_ds: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> tuple[Dataset, Dataset]:
    """Tokenize text column; truncate to max_length. Leaves 'labels' unchanged."""

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    return train_ds, test_ds


def train_model(
    train_ds: Dataset,
    test_ds: Dataset,
    num_labels: int,
    output_dir: str = "output",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_steps: int = -1,
) -> DistilBertForSequenceClassification:
    """
    Fine-tune DistilBERT for multi-label classification with BCE loss (Trainer uses
    BCEWithLogitsLoss when problem_type is multi_label_classification).
    If max_steps > 0, training stops after that many steps (overrides num_epochs for a quick run).
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Using device: %s", device)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
    )
    model.config.problem_type = "multi_label_classification"

    use_max_steps = max_steps > 0
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if use_max_steps else num_epochs,
        max_steps=max_steps if use_max_steps else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=min(100, max_steps) if use_max_steps else 100,
        eval_strategy="steps" if use_max_steps else "epoch",
        eval_steps=max(1, max_steps // 2) if use_max_steps else None,
        save_strategy="steps" if use_max_steps else "epoch",
        save_steps=max(1, max_steps // 2) if use_max_steps else None,
        load_best_model_at_end=not use_max_steps,
        metric_for_best_model="eval_micro_f1",
        greater_is_better=True,
    )

    def compute_metrics(p: EvalPrediction):
        logits = np.asarray(p.predictions)
        preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.int64)
        labels = np.asarray(p.label_ids)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"micro_f1": micro_f1, "macro_f1": macro_f1}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model


def evaluate_model(
    model: DistilBertForSequenceClassification,
    test_ds: Dataset,
    label_list: list[str],
    device: str | None = None,
) -> None:
    """
    Run model on test set, threshold at 0.5, report Micro-F1, Macro-F1,
    and per-class precision/recall/F1 for the top-20 most frequent labels.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    batch_size = 16

    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i : i + batch_size]
        input_ids = torch.tensor(batch["input_ids"], device=device)
        attention_mask = torch.tensor(batch["attention_mask"], device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int64)
        all_preds.append(preds)
        all_labels.append(np.array(batch["labels"]))

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    logger.info("Test set — Micro-F1: %.4f, Macro-F1: %.4f", micro_f1, macro_f1)

    # Per-class report: one row per label (indices 0..num_labels-1)
    report = classification_report(
        labels,
        preds,
        target_names=label_list,
        zero_division=0,
        output_dict=True,
    )
    # Top-20 by support (number of true positives + false negatives)
    by_support = [
        (report[name]["support"], name, report[name]["precision"], report[name]["recall"], report[name]["f1-score"])
        for name in label_list
        if name in report
    ]
    by_support.sort(reverse=True, key=lambda x: x[0])
    logger.info("Top-20 categories by support (support, precision, recall, f1-score):")
    for support, name, p, r, f in by_support[:20]:
        logger.info("  %s: support=%d  P=%.3f R=%.3f F1=%.3f", name, int(support), p, r, f)


def export_to_onnx(
    model: DistilBertForSequenceClassification,
    tokenizer: AutoTokenizer,
    label_list: list[str],
    export_dir: str = "onnx_model",
) -> Path:
    """
    Export fine-tuned model to ONNX format for fast CPU serving (~20ms per doc).
    Uses HuggingFace Optimum which handles the graph tracing and optimizations.
    Also saves the tokenizer and label mapping alongside the ONNX file so the
    exported directory is self-contained for deployment.
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification

    export_path = Path(export_dir)

    # 1. Save the PyTorch model + tokenizer to a temp dir (Optimum exports from disk)
    tmp_pt = export_path / "_pytorch_tmp"
    model.save_pretrained(tmp_pt)
    tokenizer.save_pretrained(tmp_pt)

    # 2. Load as ONNX and re-save (this does the torch -> ONNX conversion)
    ort_model = ORTModelForSequenceClassification.from_pretrained(tmp_pt, export=True)
    ort_model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)

    # 3. Save label mapping for inference
    label_map = {str(i): name for i, name in enumerate(label_list)}
    with open(export_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Clean up temp pytorch dir
    import shutil
    shutil.rmtree(tmp_pt, ignore_errors=True)

    onnx_files = list(export_path.glob("*.onnx"))
    logger.info("ONNX export complete: %s (%d files)", export_path, len(onnx_files))
    for p in onnx_files:
        size_mb = p.stat().st_size / (1024 * 1024)
        logger.info("  %s (%.1f MB)", p.name, size_mb)

    # 4. Quick sanity check: run one dummy input through ONNX and compare to PyTorch
    dummy_text = "Gold prices rose sharply today in New York."
    pt_inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=512)
    model.eval()
    model_cpu = model.to("cpu")
    with torch.no_grad():
        pt_logits = model_cpu(**pt_inputs).logits.numpy()

    ort_inputs = tokenizer(dummy_text, return_tensors="np", truncation=True, max_length=512)
    ort_logits = ort_model(**ort_inputs).logits

    diff = np.abs(pt_logits - ort_logits).max()
    logger.info("ONNX vs PyTorch max logit diff: %.6f (should be < 1e-4)", diff)
    if diff > 1e-3:
        logger.warning("Large numerical difference between ONNX and PyTorch outputs!")

    return export_path


def export_nli_to_onnx(
    model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    export_dir: str = "onnx_nli_model",
) -> Path:
    """
    Export the NLI model used for zero-shot fallback to ONNX.
    Saves both ONNX graph and tokenizer into a self-contained folder.
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification

    export_path = Path(export_dir)
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    ort_model.save_pretrained(export_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(export_path)

    onnx_files = list(export_path.glob("*.onnx"))
    logger.info("NLI ONNX export complete: %s (%d files)", export_path, len(onnx_files))
    for p in onnx_files:
        size_mb = p.stat().st_size / (1024 * 1024)
        logger.info("  %s (%.1f MB)", p.name, size_mb)
    return export_path


def parse_args():
    p = argparse.ArgumentParser(description="Train DistilBERT on Reuters-21578 (multi-label topic classification).")
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        metavar="N",
        help="If set, train for at most N steps (quick run). Example: 50 for a few batches.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="Number of training epochs when not using --max-steps (default: 3).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Train batch size per device (default: 16).",
    )
    p.add_argument(
        "--export-onnx",
        action="store_true",
        default=False,
        help="Export trained model to ONNX format after training (saved to onnx_model/).",
    )
    p.add_argument(
        "--export-nli-onnx",
        action="store_true",
        default=False,
        help="Export NLI fallback model to ONNX format (saved to onnx_nli_model/).",
    )
    p.add_argument(
        "--nli-model-name",
        type=str,
        default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        help="Model name used for NLI ONNX export.",
    )
    p.add_argument(
        "--nli-export-dir",
        type=str,
        default="onnx_nli_model",
        help="Output directory for NLI ONNX export.",
    )
    p.add_argument(
        "--export-nli-onnx-only",
        action="store_true",
        default=False,
        help="Only export NLI ONNX model and skip Reuters training.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.export_nli_onnx_only:
        logger.info("Exporting NLI model to ONNX only")
        export_nli_to_onnx(
            model_name=args.nli_model_name,
            export_dir=args.nli_export_dir,
        )
        logger.info("Done.")
        return

    if args.max_steps > 0:
        logger.info("Quick run: training for at most %d steps", args.max_steps)

    logger.info("Step 1: Extract Reuters archive")
    sgm_dir = extract_reuters_data()

    logger.info("Step 2: Parse SGML")
    docs = parse_reuters_sgml(sgm_dir)

    logger.info("Step 3: Build ModApte train/test datasets with multi-hot labels")
    train_ds, test_ds, label_list = build_datasets(docs)
    num_labels = len(label_list)

    logger.info("Step 4: Load tokenizer and tokenize")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds, test_ds = tokenize_dataset(train_ds, test_ds, tokenizer)

    logger.info("Step 5: Train DistilBERT")
    model = train_model(
        train_ds,
        test_ds,
        num_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )

    logger.info("Step 6: Evaluate on test set")
    evaluate_model(model, test_ds, label_list)

    if args.export_onnx:
        logger.info("Step 7: Export to ONNX")
        export_to_onnx(model, tokenizer, label_list)

    if args.export_nli_onnx:
        logger.info("Step 8: Export NLI to ONNX")
        export_nli_to_onnx(
            model_name=args.nli_model_name,
            export_dir=args.nli_export_dir,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
