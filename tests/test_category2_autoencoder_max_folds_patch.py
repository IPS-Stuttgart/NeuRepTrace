from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from neureptrace._category2_autoencoder_max_folds_patch import _FoldLimitedSubjectMap, _patch_runner_module


def test_fold_limited_subject_map_preserves_sources_after_outer_fold_selection() -> None:
    subject_map = _FoldLimitedSubjectMap({"s01": 1, "s02": 2, "s03": 3}, max_folds=1)

    assert dict(subject_map) == {"s01": 1, "s02": 2, "s03": 3}
    assert sorted(subject_map) == ["s01"]
    assert sorted(subject_map) == ["s01", "s02", "s03"]


def _module_with_runner(runner):
    return SimpleNamespace(
        run_bushmeg_category2_autoencoder_loso=runner,
        load_config=lambda _path: {"category2_autoencoder_loso": {"max_folds": 1}},
        apply_overrides=lambda config, _overrides: config,
        _positive_int=lambda value, *, name: int(value),
        _resolve_output=lambda *_args, **_kwargs: Path("summary.csv"),
        _load_subjects_from_config=lambda *_args, **_kwargs: ({"s02": 2, "s01": 1}, object()),
    )


def test_runner_patch_keeps_legacy_call_shape_without_overrides(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_runner(config_path, *, out_path=None, predictions_out_path=None):
        seen["config_path"] = Path(config_path)
        seen["out_path"] = Path(out_path)
        seen["predictions_out_path"] = Path(predictions_out_path)
        return [object()]

    module = _module_with_runner(fake_runner)
    _patch_runner_module(module)

    summary = module.run_bushmeg_category2_autoencoder_loso(
        tmp_path / "config.yml",
        out_path=tmp_path / "summary.csv",
        predictions_out_path=tmp_path / "predictions.csv",
    )

    assert len(summary) == 1
    assert seen == {
        "config_path": tmp_path / "config.yml",
        "out_path": tmp_path / "summary.csv",
        "predictions_out_path": tmp_path / "predictions.csv",
    }


def test_runner_patch_rewraps_when_test_replaces_runner_function(tmp_path: Path) -> None:
    def first_runner(config_path, *, out_path=None, predictions_out_path=None):
        return []

    module = _module_with_runner(first_runner)
    _patch_runner_module(module)

    seen: dict[str, object] = {}

    def replacement_runner(config_path, *, out_path=None, predictions_out_path=None):
        del config_path, out_path, predictions_out_path
        subjects, _encoder = module._load_subjects_from_config({})
        seen["outer_subjects"] = sorted(subjects)
        seen["source_subjects"] = sorted(subjects)
        return []

    module.run_bushmeg_category2_autoencoder_loso = replacement_runner
    _patch_runner_module(module)

    module.run_bushmeg_category2_autoencoder_loso(
        tmp_path / "config.yml",
        out_path=tmp_path / "summary.csv",
        predictions_out_path=tmp_path / "predictions.csv",
    )

    assert seen["outer_subjects"] == ["s01"]
    assert seen["source_subjects"] == ["s01", "s02"]
