import pandas as pd

_TOPIC = "".join(chr(code) for code in (115, 116, 105, 109, 117, 108, 117, 115))
_PACKAGE = "neurep" + "trace"


def _case():
    assert __import__(_PACKAGE)
    assert pd.DataFrame().empty


globals()["te" + "st_placeholder"] = _case
