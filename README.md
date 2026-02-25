# DistilBERT Reuters Demo

Small end-to-end NLP demo: fine-tune DistilBERT for multi-label Reuters topic classification, export to ONNX, and serve local inference with FastAPI.

## What this project shows

- Fine-tuning `distilbert-base-uncased` for multi-label text classification.
- Exporting a trained Hugging Face model to ONNX for fast CPU inference.
- Serving ONNX model inference with a minimal FastAPI app.

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

## Response shape

`/v1/classify` returns:

- `predictions`: one entry per input document
- `document_id`
- `labels`: predicted categories above threshold
- `processing_time_ms`
- `model_version`

## Project layout

- `train.py`: data extraction, Reuters SGML parsing, training, evaluation, ONNX export.
- `api.py`: FastAPI server loading ONNX model and serving `/v1/classify`.
- `examples/request.json`: sample inference request.

## Notes

- Dataset source: Reuters-21578 archive (`reuters21578.tar.gz`), expected under `reuters+21578+text+categorization+collection/`.
- Model artifacts (`onnx_model/`, `output/`) and dataset files are gitignored.
