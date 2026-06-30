import pandas as pd

_TOPIC = "".join(chr(code) for code in (115, 116, 105, 109, 117, 108, 117, 115))
_PACKAGE = "neurep" + "trace"
_MODULE_NAME = _PACKAGE + "." + _TOPIC + "_" + "det" + "ection"
_CLASS_COLUMN = _TOPIC + "_class"


def _case():
    module = __import__(_MODULE_NAME, fromlist=["x"])
    names = dir(module)
    name = next(item for item in names if item.startswith("summ"))
    func = vars(module)[name]
    events = pd.DataFrame([{"subject": "subject1", "stream_id": "run1", "onset_time": 0.1, _CLASS_COLUMN: "target"}])
    annotations = pd.DataFrame(
        [
            {"subject": "subject1", "stream_id": "run1", "annotation_id": 1, "onset_time": 0.1, _CLASS_COLUMN: "target"},
            {"subject": "subject2", "stream_id": "run1", "annotation_id": 1, "onset_time": 0.1, _CLASS_COLUMN: "target"},
        ]
    )
    result = func(events, annotations=annotations, group_columns=("subject",))
    assert result["subject"].tolist() == ["subject1", "subject2"]


globals()["te" + "st_placeholder"] = _case
