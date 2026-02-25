from pathlib import Path

import api


def _cfg(backend: str = "onnx") -> api.RuntimeConfig:
    return api.RuntimeConfig(
        model_dir=Path("onnx_model"),
        categories_file=Path("categories.json"),
        nli_backend=backend,
        nli_model_dir=Path("onnx_nli_model"),
        nli_model_name="dummy-nli",
        hybrid_distil_weight=0.6,
        hybrid_nli_weight=0.4,
        enable_low_conf_rescoring=False,
        low_conf_min=0.35,
        low_conf_max=0.55,
    )


def test_load_nli_model_onnx_fallbacks_to_torch(monkeypatch) -> None:
    api.app_config = _cfg("onnx")
    api.nli_classifier = None
    api.nli_error = None
    calls: list[str] = []

    class DummyNLI:
        def __init__(self, model_name: str, backend: str, model_dir=None) -> None:
            calls.append(backend)
            if backend == "onnx":
                raise RuntimeError("no onnx model")
            self.backend = backend

    monkeypatch.setattr(api, "NLIClassifier", DummyNLI)
    api.load_nli_model()

    assert calls == ["onnx", "torch"]
    assert api.nli_classifier is not None
    assert api.nli_classifier.backend == "torch"
    assert api.nli_error is None


def test_load_nli_model_onnx_success(monkeypatch) -> None:
    api.app_config = _cfg("onnx")
    api.nli_classifier = None
    api.nli_error = None
    calls: list[str] = []

    class DummyNLI:
        def __init__(self, model_name: str, backend: str, model_dir=None) -> None:
            calls.append(backend)
            self.backend = backend

    monkeypatch.setattr(api, "NLIClassifier", DummyNLI)
    api.load_nli_model()

    assert calls == ["onnx"]
    assert api.nli_classifier is not None
    assert api.nli_classifier.backend == "onnx"
    assert api.nli_error is None
