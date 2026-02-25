from fastapi.testclient import TestClient

import api


def _disable_startup() -> list:
    original = list(api.app.router.on_startup)
    api.app.router.on_startup.clear()
    return original


def test_routes_health_categories_and_classify(monkeypatch) -> None:
    original_startup = _disable_startup()
    try:
        api.app_config = api.RuntimeConfig(
            model_dir=api.PROJECT_ROOT / "onnx_model",
            categories_file=api.PROJECT_ROOT / "categories.json",
            nli_backend="onnx",
            nli_model_dir=api.PROJECT_ROOT / "onnx_nli_model",
            nli_model_name="dummy",
            hybrid_distil_weight=0.6,
            hybrid_nli_weight=0.4,
            enable_low_conf_rescoring=False,
            low_conf_min=0.35,
            low_conf_max=0.55,
        )
        api.label_map = {"0": "earn", "1": "acq"}
        api.category_registry = {
            "earn": {"name": "earn", "description": "earn", "status": "known", "hypothesis_template": "This article is about {label_description}."},
            "crypto": {"name": "crypto", "description": "crypto", "status": "new", "hypothesis_template": "This article is about {label_description}."},
        }
        api.nli_classifier = None
        api.nli_error = None

        monkeypatch.setattr(
            api,
            "predict_one",
            lambda req, text: (
                [{"category": "earn", "confidence": 0.92, "source": "distilbert"}],
                12.34,
            ),
        )

        client = TestClient(api.app)

        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["num_labels"] == 2

        categories = client.get("/v1/categories")
        assert categories.status_code == 200
        assert categories.json()["counts"] == {"total": 2, "known": 1, "new": 1}

        classify = client.post(
            "/v1/classify",
            json={"documents": [{"id": "doc-1", "text": "hello world"}]},
        )
        assert classify.status_code == 200
        payload = classify.json()
        assert payload["model_version"] == api.MODEL_VERSION
        assert payload["predictions"][0]["document_id"] == "doc-1"
        assert payload["predictions"][0]["labels"][0]["category"] == "earn"
    finally:
        api.app.router.on_startup[:] = original_startup
