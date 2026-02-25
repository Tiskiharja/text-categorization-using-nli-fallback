import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api


@pytest.fixture(autouse=True)
def reset_api_globals():
    original = {
        "app_config": api.app_config,
        "label_map": api.label_map,
        "category_registry": api.category_registry,
        "nli_classifier": api.nli_classifier,
        "nli_error": api.nli_error,
    }
    api.app_config = None
    api.label_map = None
    api.category_registry = {}
    api.nli_classifier = None
    api.nli_error = None
    yield
    api.app_config = original["app_config"]
    api.label_map = original["label_map"]
    api.category_registry = original["category_registry"]
    api.nli_classifier = original["nli_classifier"]
    api.nli_error = original["nli_error"]


@pytest.fixture
def runtime_config_factory():
    def _make(**overrides) -> api.RuntimeConfig:
        base = {
            "model_dir": api.PROJECT_ROOT / "onnx_model",
            "categories_file": api.PROJECT_ROOT / "categories.json",
            "nli_backend": "onnx",
            "nli_model_dir": api.PROJECT_ROOT / "onnx_nli_model",
            "nli_model_name": "dummy",
            "hybrid_distil_weight": 0.6,
            "hybrid_nli_weight": 0.4,
            "enable_low_conf_rescoring": False,
            "low_conf_min": 0.35,
            "low_conf_max": 0.55,
        }
        base.update(overrides)
        return api.RuntimeConfig(**base)

    return _make


@pytest.fixture
def base_label_map() -> dict[str, str]:
    return {"0": "earn", "1": "acq"}


@pytest.fixture
def client_without_startup():
    original = list(api.app.router.on_startup)
    api.app.router.on_startup.clear()
    try:
        yield TestClient(api.app)
    finally:
        api.app.router.on_startup[:] = original
