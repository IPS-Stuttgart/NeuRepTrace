import importlib


def test_placeholder() -> None:
    mod = importlib.import_module("neureptrace.decoding._domain_" + "ids")
    assert hasattr(mod, "atomic_domain_vector")
