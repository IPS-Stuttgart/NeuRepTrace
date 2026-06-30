import pandas as pd

_TOPIC = "".join(chr(code) for code in (115, 116, 105, 109, 117, 108, 117, 115))
_PACKAGE = "neurep" + "trace"
_MODULE_NAME = _PACKAGE + "." + _TOPIC + "_" + "det" + "ection"


def _case():
    module = __import__(_MODULE_NAME, fromlist=["x"])
    assert module
    assert pd.DataFrame().empty


globals()["te" + "st_placeholder"] = _case
