from pathlib import Path

from neureptrace.validate_manifest import validate_manifest


def test_validate_manifest_reports_fractional_n_splits(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("condition\nface\nobject\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "subject,epochs,metadata_csv,label_column,n_splits\nsub-01,missing-epo.fif,metadata.csv,condition,2.5\n",
        encoding="utf-8",
    )

    validation = validate_manifest(manifest_path)[0]

    assert not validation.ok
    assert "n_splits must be an integer >= 2, got '2.5'" in " ".join(validation.messages)
