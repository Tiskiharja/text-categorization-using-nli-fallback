import api


def test_routes_health_categories_and_classify(
    monkeypatch,
    runtime_config_factory,
    base_label_map,
    client_without_startup,
) -> None:
    api.app_config = runtime_config_factory()
    api.label_map = base_label_map.copy()
    api.category_registry = {
        "earn": {"name": "earn", "description": "earn", "status": "known", "hypothesis_template": "This article is about {label_description}."},
        "crypto": {"name": "crypto", "description": "crypto", "status": "new", "hypothesis_template": "This article is about {label_description}."},
    }

    monkeypatch.setattr(
        api,
        "predict_one",
        lambda req, text: (
            [{"category": "earn", "confidence": 0.92, "source": "distilbert"}],
            12.34,
        ),
    )

    health = client_without_startup.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["num_labels"] == 2

    categories = client_without_startup.get("/v1/categories")
    assert categories.status_code == 200
    assert categories.json()["counts"] == {"total": 2, "known": 1, "new": 1}

    classify = client_without_startup.post(
        "/v1/classify",
        json={"documents": [{"id": "doc-1", "text": "hello world"}]},
    )
    assert classify.status_code == 200
    payload = classify.json()
    assert payload["model_version"] == api.MODEL_VERSION
    assert payload["predictions"][0]["document_id"] == "doc-1"
    assert payload["predictions"][0]["labels"][0]["category"] == "earn"
