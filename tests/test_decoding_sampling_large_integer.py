from __future__ import annotations

import numpy as np

from neureptrace.decoding import normalize_class_limit_seed


def test_normalize_class_limit_seed_preserves_large_exact_integers() -> None:
    large_seed = 2**53 + 1

    assert normalize_class_limit_seed(large_seed) == large_seed
    assert normalize_class_limit_seed(np.uint64(large_seed)) == large_seed
    assert normalize_class_limit_seed(str(large_seed)) == large_seed
    assert normalize_class_limit_seed(large_seed) != normalize_class_limit_seed(large_seed - 1)


def test_normalize_class_limit_seed_keeps_integer_valued_string_aliases() -> None:
    assert normalize_class_limit_seed("1.0") == 1
    assert normalize_class_limit_seed("1e3") == 1000
