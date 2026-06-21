import numpy as np
import pytest

from neureptrace.decoding.dann import CONDITIONAL_MMD_PROTOCOL, fit_dann_predict_proba


def test_conditional_mmd_loss_metadata():
    pytest.importorskip("torch")
    rng = np.random.default_rng(19)
    x_source = np.vstack([rng.normal(-1.0, 0.2, (8, 3)), rng.normal(1.0, 0.2, (8, 3))])
    y_source = np.repeat([0, 1], 8)
    x_target = np.vstack([rng.normal(-0.7, 0.2, (4, 3)), rng.normal(0.7, 0.2, (4, 3))])

    result = fit_dann_predict_proba(
        source_features=x_source,
        source_labels=y_source,
        target_features=x_target,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=2,
        batch_size=8,
        patience=1,
        domain_loss_weight=0.0,
        conditional_mmd_loss_weight=0.2,
        mmd_kernel_num=3,
        random_state=7,
        device="cpu",
    )

    assert result.probabilities.shape == (8, 2)
    assert result.metadata["dann_protocol"] == CONDITIONAL_MMD_PROTOCOL
    assert result.metadata["dann_mmd_adaptation"] is True
    assert result.metadata["dann_conditional_mmd_loss_weight"] == 0.2
