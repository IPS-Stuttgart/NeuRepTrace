from __future__ import annotations

import neureptrace.bushmeg_all_protocols as all_protocols


def test_category2_autoencoder_loso_is_runnable_all_protocol_method() -> None:
    spec = all_protocols.method_registry()["category2_autoencoder_loso"]

    assert spec.protocol_category == 2
    assert spec.method_family == "category2_autoencoder"
    assert spec.runner == "category2_autoencoder_loso"
    assert spec.runnable is True
    assert spec.metadata()["inventory_only"] is False


def test_category2_autoencoder_loso_is_available_when_module_is_present(monkeypatch) -> None:
    spec = all_protocols.method_registry()["category2_autoencoder_loso"]
    monkeypatch.setattr(
        all_protocols,
        "_module_available",
        lambda module: module == "neureptrace.bushmeg_category2_autoencoder_loso",
    )

    available, skip_reason = all_protocols._method_availability(
        spec,
        {},
        settings={},
        include_heavy=False,
        max_folds=1,
    )

    assert available is True
    assert skip_reason == ""
