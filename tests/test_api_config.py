from pathlib import Path

import pytest

import api


def test_load_runtime_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))

    cfg = api.load_runtime_config()
    assert cfg.model_dir == (api.PROJECT_ROOT / "onnx_model").resolve()
    assert cfg.categories_file == (api.PROJECT_ROOT / "categories.json").resolve()
    assert cfg.nli_backend == "onnx"
    assert cfg.nli_model_dir == (api.PROJECT_ROOT / "onnx_nli_model").resolve()
    assert cfg.nli_model_name == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


def test_load_runtime_config_rejects_bad_nli_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("nli_backend: invalid\n", encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))

    with pytest.raises(ValueError, match="nli_backend must be 'onnx' or 'torch'"):
        api.load_runtime_config()
