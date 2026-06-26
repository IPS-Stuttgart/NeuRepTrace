import numpy as np

from neureptrace.decoding.vrex import LinearVRExClassifier


def test_vrex_preserves_composite_labels_and_domain_rows():
    features = np.asarray([[10.0, 2.0], [11.0, 2.5], [12.0, 3.0], [13.0, 3.5]])
    labels = [["task", "left"], ["task", "right"], ["task", "left"], ["task", "right"]]
    domains = np.asarray([["s1", "r1"], ["s1", "r1"], ["s2", "r1"], ["s2", "r1"]], dtype=object)

    model = LinearVRExClassifier(max_iter=3, tol=1e-4)
    model.fit(features, labels, source_domains=domains)

    assert model.classes_.tolist() == [("task", "left"), ("task", "right")]
    assert model.source_domains_.tolist() == [("s1", "r1"), ("s2", "r1")]
    assert model.metadata()["vrex_n_source_domains"] == 2
