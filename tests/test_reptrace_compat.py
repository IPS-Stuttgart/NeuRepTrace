from __future__ import annotations


def test_reptrace_top_level_alias_exposes_neureptrace_version():
    import neureptrace
    import reptrace

    assert reptrace.__version__ == neureptrace.__version__


def test_reptrace_decoding_classifiers_alias_keeps_public_objects():
    from neureptrace.decoding.classifiers import CLASSIFIER_REGISTRY as neureptrace_registry
    from neureptrace.decoding.classifiers import (
        train_multiclass_classifier as neureptrace_train_multiclass_classifier,
    )
    from reptrace.decoding.classifiers import CLASSIFIER_REGISTRY as reptrace_registry
    from reptrace.decoding.classifiers import (
        train_multiclass_classifier as reptrace_train_multiclass_classifier,
    )

    assert reptrace_registry is neureptrace_registry
    assert reptrace_train_multiclass_classifier is neureptrace_train_multiclass_classifier


def test_reptrace_nested_modules_resolve_to_neureptrace_implementations():
    from neureptrace.decoding.windowed import (
        fit_window_model as neureptrace_fit_window_model,
    )
    from neureptrace.metrics.confusion import (
        confusion_counts as neureptrace_confusion_counts,
    )
    from reptrace.decoding.windowed import fit_window_model as reptrace_fit_window_model
    from reptrace.metrics.confusion import confusion_counts as reptrace_confusion_counts

    assert reptrace_fit_window_model is neureptrace_fit_window_model
    assert reptrace_confusion_counts is neureptrace_confusion_counts
