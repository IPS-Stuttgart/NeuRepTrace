from __future__ import annotations

import pytest

from neureptrace.bushmeg_data import expected_bushmeg_files, prepare_bushmeg_smoke_data


def test_expected_bushmeg_files_resolves_main_and_cue(tmp_path):
    files = expected_bushmeg_files(tmp_path, participants="2", roles=("main", "cue"))

    assert [file.relative_path for file in files] == ["Part2Data.mat", "Part2CueData.mat"]
    assert [file.role for file in files] == ["main", "cue"]
    assert all(not file.exists for file in files)


def test_prepare_bushmeg_smoke_data_reuses_existing_files_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("BUSHMEG_WEBDAV_URL", raising=False)
    monkeypatch.delenv("BUSHMEG_DATA_KEY", raising=False)
    monkeypatch.delenv("BUSHMEG_DATA_PASSWORD", raising=False)
    (tmp_path / "Part2Data.mat").write_bytes(b"main")
    (tmp_path / "Part2CueData.mat").write_bytes(b"cue")

    files = prepare_bushmeg_smoke_data(tmp_path, participants="2", roles=("main", "cue"))

    assert all(file.exists for file in files)


def test_prepare_bushmeg_smoke_data_requires_credentials_for_missing_files(tmp_path, monkeypatch):
    monkeypatch.delenv("BUSHMEG_WEBDAV_URL", raising=False)
    monkeypatch.delenv("BUSHMEG_DATA_KEY", raising=False)
    monkeypatch.delenv("BUSHMEG_DATA_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="Missing BUSH-MEG WebDAV credential"):
        prepare_bushmeg_smoke_data(tmp_path, participants="2", roles=("main",), max_files=1)


def test_prepare_bushmeg_smoke_data_can_allow_missing_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSHMEG_WEBDAV_URL", "https://example.invalid/data")
    monkeypatch.setenv("BUSHMEG_DATA_KEY", "user")
    monkeypatch.setenv("BUSHMEG_DATA_PASSWORD", "password")

    def fail_download(**_kwargs):
        raise FileNotFoundError("not available")

    monkeypatch.setattr("neureptrace.bushmeg_data._download_webdav_file", fail_download)

    files = prepare_bushmeg_smoke_data(
        tmp_path,
        participants="2",
        roles=("main",),
        max_files=1,
        allow_missing=True,
    )

    assert [file.relative_path for file in files] == ["Part2Data.mat"]
    assert not files[0].exists


def test_expected_bushmeg_files_honors_max_files(tmp_path):
    files = expected_bushmeg_files(tmp_path, participants="1-2", roles=("main", "cue"), max_files=2)

    assert [file.relative_path for file in files] == ["Part1Data.mat", "Part1CueData.mat"]
