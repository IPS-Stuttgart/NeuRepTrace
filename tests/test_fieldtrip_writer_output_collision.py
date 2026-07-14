from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.fieldtrip_mat import write_fieldtrip_raw_mat_epochs


def test_fieldtrip_writer_rejects_colliding_output_paths_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_called = False

    def fake_loader(mat_path: Path | str, **kwargs: object):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("loader must not run for colliding output paths")

    monkeypatch.setattr("neureptrace.fieldtrip_mat.load_fieldtrip_raw_mat_epochs", fake_loader)
    epochs_out = tmp_path / "shared-output.fif"
    metadata_alias = tmp_path / "nested" / ".." / "shared-output.fif"

    with pytest.raises(ValueError, match="epochs and metadata output paths must be distinct"):
        write_fieldtrip_raw_mat_epochs(
            tmp_path / "dummy.mat",
            epochs_out=epochs_out,
            metadata_out=metadata_alias,
            overwrite=True,
        )

    assert loader_called is False
    assert not epochs_out.exists()
