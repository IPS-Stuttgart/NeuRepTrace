import numpy as np

from neureptrace.decoding.source_vrex import _domain_balanced_batch


def test_vrex_batch_contains_each_source_domain():
    indices = np.arange(12)
    domains = np.repeat(np.arange(3), 4)
    batch = _domain_balanced_batch(indices, domains, batch_size=9, rng=np.random.default_rng(1))
    assert batch.shape == (9,)
    assert set(domains[batch].tolist()) == {0, 1, 2}
