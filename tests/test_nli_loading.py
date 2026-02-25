import pytest

import api


@pytest.mark.parametrize(
    ("onnx_fails", "expected_calls", "expected_backend"),
    [
        (True, ["onnx", "torch"], "torch"),
        (False, ["onnx"], "onnx"),
    ],
    ids=["fallback_to_torch", "onnx_success"],
)
def test_load_nli_model_onnx_paths(
    monkeypatch,
    runtime_config_factory,
    onnx_fails: bool,
    expected_calls: list[str],
    expected_backend: str,
) -> None:
    api.app_config = runtime_config_factory(nli_backend="onnx", nli_model_name="dummy-nli")
    calls: list[str] = []

    class DummyNLI:
        def __init__(self, model_name: str, backend: str, model_dir=None) -> None:
            calls.append(backend)
            if onnx_fails and backend == "onnx":
                raise RuntimeError("no onnx model")
            self.backend = backend

    monkeypatch.setattr(api, "NLIClassifier", DummyNLI)
    api.load_nli_model()

    assert calls == expected_calls
    assert api.nli_classifier is not None
    assert api.nli_classifier.backend == expected_backend
    assert api.nli_error is None
