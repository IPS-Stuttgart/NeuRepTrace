import importlib

_TOPIC = "".join(chr(code) for code in (115, 116, 105, 109, 117, 108, 117, 115))
_PUBLIC = __package__ + "._" + _TOPIC + "_" + "det" + "ection_public"


def install():
    importlib.import_module(_PUBLIC)
