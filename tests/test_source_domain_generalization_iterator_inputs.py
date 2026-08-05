import numpy as np

from neureptrace.decoding.source_domain_generalization import _encode_inputs


def _items(values):
    return (value for value in values)


def test_source_domain_generalization_materializes_iterator_identifiers():
    features = np.arange(12, dtype=float).reshape(4, 3)
    source_labels = (
        _items(value)
        for value in [
            ("class", 0),
            ("class", 1),
            ("class", 0),
            ("class", 1),
        ]
    )
    source_domains = np.empty(4, dtype=object)
    source_domains[:] = [
        _items(("subject", 0)),
        _items(("subject", 0)),
        _items(("subject", 1)),
        _items(("subject", 1)),
    ]

    encoded_features, classes, labels, domain_names, domains = _encode_inputs(
        features,
        source_labels,
        source_domains,
        name="source_dg",
    )

    assert encoded_features.dtype == np.float32
    assert classes.tolist() == [("class", 0), ("class", 1)]
    assert labels.tolist() == [0, 1, 0, 1]
    assert domain_names.tolist() == [("subject", 0), ("subject", 1)]
    assert domains.tolist() == [0, 0, 1, 1]
