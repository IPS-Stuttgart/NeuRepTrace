from pathlib import Path

from neureptrace.validate_manifest import validate_manifest


def test_validate_manifest_rejects_default_n_splits_below_two(tmp_path: Path) -> None:
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text("subject,epochs\nsub-01,missing-epo.fif\n", encoding="utf-8")

    try:
        validate_manifest(manifest_csv, default_n_splits=1)
    except ValueError as exc:
        assert "default_n_splits must be at least 2" in str(exc)
    else:
        raise AssertionError("validate_manifest accepted default_n_splits=1")
