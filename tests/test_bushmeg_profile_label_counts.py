from __future__ import annotations

import numpy as np

from neureptrace import _bushmeg_profile_label_counts_patch
import neureptrace.bushmeg_all_protocols as all_protocols


def test_package_import_installs_profile_class_count_patch() -> None:
    labels = np.asarray(
        [
            ["face", 1],
            ["face", 1],
            ["scene", 2],
        ],
        dtype=object,
    )

    assert getattr(all_protocols, "_neureptrace_bushmeg_profile_label_counts_patch_installed", False) is True
    assert all_protocols._class_count_dict(labels) == {"('face', 1)": 2, "('scene', 2)": 1}


def test_profile_class_counts_support_composite_label_rows() -> None:
    _bushmeg_profile_label_counts_patch.install()
    labels = np.asarray(
        [
            ["face", 1],
            ["face", 1],
            ["scene", 2],
        ],
        dtype=object,
    )

    assert all_protocols._class_count_dict(labels) == {"('face', 1)": 2, "('scene', 2)": 1}


def test_profile_class_counts_preserve_scalar_label_counts() -> None:
    _bushmeg_profile_label_counts_patch.install()
    labels = np.asarray(["face", "scene", "face"], dtype=object)

    assert all_protocols._class_count_dict(labels) == {"face": 2, "scene": 1}
