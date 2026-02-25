# DistilBERT + NLI Reuters Demo

Small end-to-end NLP demo: fine-tune DistilBERT for Reuters multi-label topic classification, export to ONNX, and serve local inference with FastAPI + NLI fallback for new categories.

## Quickstart (Makefile)

```bash
make install
make download-data
make train-quick
make api
```

In another terminal:

```bash
make demo
```

Useful targets:

- `make train` - full training + ONNX export
- `make eval` - hybrid threshold/weight tuning
- `make health` / `make categories` / `make classify` - API smoke checks
- `make clean` - remove generated artifacts (`onnx_model`, `output`, caches)

## Python/uv interface (simple example)

```bash
uv sync
uv run python3 train.py --max-steps 50 --export-onnx
uv run uvicorn api:app --reload
curl -X POST "http://127.0.0.1:8000/v1/classify" \
  -H "Content-Type: application/json" \
  --data @examples/request.json
```

## Notes

- Reuters archive is expected at: `reuters+21578+text+categorization+collection/reuters21578.tar.gz`
- Category registry: `categories.json`
- Main scripts: `train.py`, `api.py`, `evaluate_hybrid.py`, `nli_fallback.py`
