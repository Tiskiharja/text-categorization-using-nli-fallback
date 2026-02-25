from pathlib import Path

import pytest

import api


def test_to_bool_variants() -> None:
    assert api._to_bool(True) is True
    assert api._to_bool("true") is True
    assert api._to_bool("YES") is True
    assert api._to_bool("0") is False
    assert api._to_bool("no") is False


def test_load_runtime_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))

    cfg = api.load_runtime_config()

    assert cfg.model_dir == (api.PROJECT_ROOT / "onnx_model").resolve()
    assert cfg.categories_file == (api.PROJECT_ROOT / "categories.json").resolve()
    assert cfg.nli_backend == "onnx"
    assert cfg.nli_model_dir == (api.PROJECT_ROOT / "onnx_nli_model").resolve()
    assert cfg.nli_model_name == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert cfg.hybrid_distil_weight == pytest.approx(0.6)
    assert cfg.hybrid_nli_weight == pytest.approx(0.4)
    assert cfg.enable_low_conf_rescoring is False


def test_load_runtime_config_requires_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item\n", encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))

    with pytest.raises(ValueError, match="must be a mapping"):
        api.load_runtime_config()


def test_load_runtime_config_rejects_invalid_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("low_conf_min: 0.7\nlow_conf_max: 0.6\n", encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))

    with pytest.raises(ValueError, match="low_conf_min"):
        api.load_runtime_config()


def test_load_runtime_config_rejects_bad_nli_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("nli_backend: invalid\n", encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))

    with pytest.raises(ValueError, match="nli_backend must be 'onnx' or 'torch'"):
        api.load_runtime_config()
