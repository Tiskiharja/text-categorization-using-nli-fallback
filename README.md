# DistilBERT + NLI Reuters Demo

Small end-to-end NLP demo: fine-tune DistilBERT for multi-label Reuters topic classification, export to ONNX, and serve local inference with FastAPI plus NLI fallback for new categories.

## What this project shows

- Fine-tuning `distilbert-base-uncased` for multi-label text classification.
- Exporting a trained Hugging Face model to ONNX for fast CPU inference.
- Serving ONNX model inference with a minimal FastAPI app.
- Hybrid classification where zero-shot NLI can score newly added categories without DistilBERT retraining.

## Quickstart

Install dependencies:

```bash
uv sync
```

Run a quick training + ONNX export:

```bash
uv run train.py --max-steps 50 --export-onnx
```

Run the API:

```bash
uv run uvicorn api:app --reload
```

Send a request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/classify" \
  -H "Content-Type: application/json" \
  --data @examples/request.json
```

Health check:

```bash
curl http://127.0.0.1:8000/v1/health
```

List loaded categories:

```bash
curl http://127.0.0.1:8000/v1/categories
```

## Response shape

`/v1/classify` returns:

- `predictions`: one entry per input document
- `document_id`
- `labels`: predicted categories above threshold
- `source` per label (`distilbert`, `hybrid`, `nli`)
- `processing_time_ms`
- `model_version`

## Category registry and NLI fallback

- Category metadata lives in `categories.json`.
- Categories that are not in the ONNX model label map are treated as new and scored via NLI.
- Default NLI model is `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`.
- If NLI model loading fails, the API degrades to DistilBERT-only mode and reports this in `/v1/health`.

Environment variables:

- `NLI_MODEL_NAME` (default: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`)
- `HYBRID_DISTIL_WEIGHT` (default: `0.6`)
- `HYBRID_NLI_WEIGHT` (default: `0.4`)
- `ENABLE_LOW_CONF_RESCORING` (default: `false`)
- `LOW_CONF_MIN` (default: `0.35`)
- `LOW_CONF_MAX` (default: `0.55`)

Request fields for `/v1/classify`:

- `enable_nli_fallback` (default: `true`)
- `include_new_categories` (default: `true`)
- `nli_threshold` (default: `0.4`)
- `include_debug_scores` (default: `false`)

## Hybrid evaluation and tuning

Run held-out-label simulation and tuning:

```bash
uv run python evaluate_hybrid.py --max-docs 800 --holdout-k 8 --min-support 20
```

The script prints:

- best NLI threshold for simulated new labels
- DistilBERT baseline on known labels
- best fusion weights/threshold on a known-label subset

Use the output to set:

- `nli_threshold` in requests (or service default)
- `HYBRID_DISTIL_WEIGHT` and `HYBRID_NLI_WEIGHT` in environment

## Project layout

- `train.py`: data extraction, Reuters SGML parsing, training, evaluation, ONNX export.
- `api.py`: FastAPI server with DistilBERT + NLI hybrid inference.
- `nli_fallback.py`: zero-shot NLI scoring utility.
- `evaluate_hybrid.py`: held-out-label simulation and threshold/weight tuning.
- `categories.json`: category registry (known + new categories).
- `examples/request.json`: sample inference request.

## Notes

- Dataset source: Reuters-21578 archive (`reuters21578.tar.gz`), expected under `reuters+21578+text+categorization+collection/`.
- Model artifacts (`onnx_model/`, `output/`) and dataset files are gitignored.
