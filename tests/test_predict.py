from pathlib import Path

import numpy as np

import api


class DummyNLI:
    def score_categories(self, text: str, categories: list[api.NLICategory]) -> dict[str, float]:
        return {"acq": 0.9, "crypto": 0.8}


def _config() -> api.RuntimeConfig:
    return api.RuntimeConfig(
        model_dir=Path("onnx_model"),
        categories_file=Path("categories.json"),
        nli_backend="onnx",
        nli_model_dir=Path("onnx_nli_model"),
        nli_model_name="dummy",
        hybrid_distil_weight=0.6,
        hybrid_nli_weight=0.4,
        enable_low_conf_rescoring=True,
        low_conf_min=0.4,
        low_conf_max=0.6,
    )


def test_predict_one_hybrid_and_new_category(monkeypatch) -> None:
    api.app_config = _config()
    api.label_map = {"0": "earn", "1": "acq"}
    api.category_registry = {
        "earn": {"description": "earn", "status": "known", "hypothesis_template": "This article is about {label_description}."},
        "acq": {"description": "acquisition", "status": "known", "hypothesis_template": "This article is about {label_description}."},
        "crypto": {"description": "digital assets", "status": "new", "hypothesis_template": "This article is about {label_description}."},
    }
    api.nli_classifier = DummyNLI()
    monkeypatch.setattr(api, "_distilbert_probs", lambda text: np.array([0.9, 0.45]))

    req = api.ClassifyRequest(
        documents=[api.DocumentInput(id="d1", text="x")],
        confidence_threshold=0.5,
        nli_threshold=0.4,
        max_labels=2,
        include_debug_scores=True,
    )

    labels, _elapsed = api.predict_one(req, "sample text")

    assert [row["category"] for row in labels] == ["earn", "crypto"]
    assert labels[0]["source"] == "distilbert"
    assert labels[1]["source"] == "nli"
    assert labels[0]["debug_scores"]["final"] == labels[0]["confidence"]


def test_predict_one_without_nli(monkeypatch) -> None:
    api.app_config = _config()
    api.label_map = {"0": "earn", "1": "acq"}
    api.category_registry = {}
    api.nli_classifier = None
    monkeypatch.setattr(api, "_distilbert_probs", lambda text: np.array([0.7, 0.3]))

    req = api.ClassifyRequest(
        documents=[api.DocumentInput(id="d1", text="x")],
        confidence_threshold=0.5,
        enable_nli_fallback=False,
    )

    labels, _elapsed = api.predict_one(req, "sample text")

    assert len(labels) == 1
    assert labels[0]["category"] == "earn"
    assert labels[0]["source"] == "distilbert"
