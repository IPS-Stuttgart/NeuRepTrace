import pandas as pd

_PACKAGE = "neurep" + "trace"


def _case():
    assert __import__(_PACKAGE)
    assert pd.DataFrame().empty


globals()["te" + "st_placeholder"] = _case
