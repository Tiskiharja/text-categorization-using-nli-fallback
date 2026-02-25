from pathlib import Path

import api


def _config(categories_file: Path) -> api.RuntimeConfig:
    return api.RuntimeConfig(
        model_dir=Path("onnx_model"),
        categories_file=categories_file,
        nli_backend="onnx",
        nli_model_dir=Path("onnx_nli_model"),
        nli_model_name="dummy",
        hybrid_distil_weight=0.6,
        hybrid_nli_weight=0.4,
        enable_low_conf_rescoring=False,
        low_conf_min=0.35,
        low_conf_max=0.55,
    )


def test_build_category_registry_without_categories_file(tmp_path: Path) -> None:
    api.app_config = _config(tmp_path / "missing.json")
    api.label_map = {"0": "earn", "1": "acq"}

    registry = api.build_category_registry()

    assert sorted(registry.keys()) == ["acq", "earn"]
    assert registry["earn"]["status"] == "known"


def test_build_category_registry_merges_custom_categories(tmp_path: Path) -> None:
    categories_file = tmp_path / "categories.json"
    categories_file.write_text(
        """
{
  "categories": [
    {"name": "earn", "description": "Earnings results"},
    {"name": "crypto", "description": "Digital assets", "status": "new"}
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    api.app_config = _config(categories_file)
    api.label_map = {"0": "earn", "1": "acq"}

    registry = api.build_category_registry()
    api.category_registry = registry

    assert registry["earn"]["description"] == "Earnings results"
    assert registry["crypto"]["status"] == "new"
    assert api._is_new_category("crypto") is True
    assert api._is_new_category("earn") is False
