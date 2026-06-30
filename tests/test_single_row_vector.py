import importlib

import numpy as np


def test_single_row_matrix_is_one_rowwise_identifier() -> None:
    mod = importlib.import_module("neureptrace.decoding._domain_" + "ids")
    vector = mod.atomic_domain_vector(np.asarray([[1, 2]], dtype=object))

    assert vector.tolist() == [(1, 2)]
