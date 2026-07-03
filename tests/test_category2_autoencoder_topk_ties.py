import numpy as np
import pytest

# Importing the package installs the Category-2 autoencoder patch module.
import neureptrace  # noqa: F401
from neureptrace.bushmeg_category2_autoencoder_loso import _top_k_accuracy


def test_category2_autoencoder_top_k_uses_stable_exact_k_tie_order():
    probabilities = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.9, 0.05, 0.03, 0.02],
        ]
    )
    labels = np.array([3, 0])

    assert _top_k_accuracy(probabilities, labels, k=1) == pytest.approx(0.5)
    assert _top_k_accuracy(probabilities, labels, k=2) == pytest.approx(0.5)
    assert _top_k_accuracy(probabilities, labels, k=3) == pytest.approx(0.5)
    assert _top_k_accuracy(probabilities, labels, k=4) == pytest.approx(1.0)
