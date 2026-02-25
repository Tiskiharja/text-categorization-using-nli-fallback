import api


def test_build_category_registry_without_categories_file(
    tmp_path,
    runtime_config_factory,
    base_label_map,
) -> None:
    api.app_config = runtime_config_factory(categories_file=tmp_path / "missing.json")
    api.label_map = base_label_map.copy()

    registry = api.build_category_registry()

    assert sorted(registry.keys()) == ["acq", "earn"]
    assert registry["earn"]["status"] == "known"


def test_build_category_registry_merges_custom_categories(
    tmp_path,
    runtime_config_factory,
    base_label_map,
) -> None:
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
    api.app_config = runtime_config_factory(categories_file=categories_file)
    api.label_map = base_label_map.copy()

    registry = api.build_category_registry()
    api.category_registry = registry

    assert registry["earn"]["description"] == "Earnings results"
    assert registry["crypto"]["status"] == "new"
    assert api._is_new_category("crypto") is True
    assert api._is_new_category("earn") is False
