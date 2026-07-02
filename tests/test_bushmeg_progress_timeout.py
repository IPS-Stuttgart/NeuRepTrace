from __future__ import annotations

import time

import pytest

import neureptrace.bushmeg_all_protocols as all_protocols
from neureptrace.bushmeg_all_protocols import MethodProgress, RunTimeoutError


def test_fold_done_checks_elapsed_timeout_without_signal_support(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(all_protocols, "_signal_timeouts_supported", lambda: False)

    progress = MethodProgress(tmp_path / "method", method="method", fold_timeout_seconds=0.001)
    progress.update("fold_start", outer_test_subject="sub-01", fold_index=1, n_folds=1)
    time.sleep(0.01)

    with pytest.raises(RunTimeoutError, match="fold timeout exceeded"):
        progress.update("fold_done", outer_test_subject="sub-01", fold_index=1, n_folds=1)
