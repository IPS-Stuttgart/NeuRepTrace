"""Keep config-derived dataset names scalar when dataset metadata is unnamed.

The config-driven decoders accept a ``dataset`` section as a mapping.  Some
callers omit ``dataset.name`` because the input files already identify the data
source.  The legacy fallback in the config-to-decoder translation used the
entire ``dataset`` mapping as the fallback value, which can leak a dictionary
into ``dataset_name`` result metadata.  A related path-template edge case is
``dataset.name: null``: output templates such as ``results/{dataset}_summary.csv``
should use the stable placeholder name ``dataset`` rather than creating files
with the literal ``None`` stem.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from numbers import Real
from types import ModuleType
from typing import Any

_TARGET_MODULES = {
    "neureptrace.config_workflow",
    "neureptrace.decode_from_config",
}
_PATCH_MARKER = "_neureptrace_dataset_name_config_patch_installed"
_BOOL_PATCH_MARKER = "_neureptrace_config_workflow_numeric_bool_patch_installed"
_OUTPUT_TEMPLATE_PATCH_MARKER = "_neureptrace_dataset_output_template_patch_installed"
_FINDER_MARKER = "_neureptrace_dataset_name_config_finder"


def _is_container_dataset_name(value: Any) -> bool:
    return isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    )


def _dataset_name_from_config(config: Any, *, default: str = "") -> str:
    if not isinstance(config, Mapping):
        return default
    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping):
        value = dataset.get("name", default)
    else:
        value = dataset
    if value is None or _is_container_dataset_name(value):
        return default
    return str(value)


def _dataset_template_name_from_config(config: Any, *, default: str = "dataset") -> str:
    """Return the safe token used for ``{dataset}`` output templates."""

    value = _dataset_name_from_config(config, default="")
    if value.strip() == "":
        return default
    return value


def _config_dataset_name_is_container(config: Any) -> bool:
    if not isinstance(config, Mapping):
        return False
    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping):
        return _is_container_dataset_name(dataset.get("name"))
    return _is_container_dataset_name(dataset)


def _patch_config_workflow_bool_parser(module: ModuleType) -> None:
    if module.__name__ != "neureptrace.config_workflow":
        return
    original_as_bool = getattr(module, "_as_bool", None)
    if original_as_bool is None or getattr(original_as_bool, _BOOL_PATCH_MARKER, False):
        return

    @wraps(original_as_bool)
    def patched_as_bool(value: Any, *, default: bool = False) -> bool:
        if isinstance(value, Real) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric) and numeric in {0.0, 1.0}:
                return bool(value)
        return original_as_bool(value, default=default)

    setattr(patched_as_bool, _BOOL_PATCH_MARKER, True)
    module._as_bool = patched_as_bool


def _patch_decode_from_config_output_templates(module: ModuleType) -> None:
    if module.__name__ != "neureptrace.decode_from_config" or getattr(module, _OUTPUT_TEMPLATE_PATCH_MARKER, False):
        return
    original_output_base_dir = getattr(module, "_output_base_dir", None)
    original_resolve_output = getattr(module, "_resolve_output", None)
    if original_output_base_dir is None or original_resolve_output is None:
        return

    @wraps(original_output_base_dir)
    def patched_output_base_dir(config: Mapping[str, Any], *, config_dir):
        outputs = module._section(config, "outputs")
        policy_base = module._base_for_policy(config, config_dir=config_dir)
        base_dir = outputs.get("base_dir") or outputs.get("dir")
        if base_dir in {None, ""}:
            return policy_base
        dataset_name = _dataset_template_name_from_config(config)
        return module.expand_path(str(base_dir).format(dataset=dataset_name), base_dir=policy_base)

    @wraps(original_resolve_output)
    def patched_resolve_output(
        config: Mapping[str, Any],
        *,
        config_dir,
        key: str,
        default: str | None = None,
    ):
        outputs = module._section(config, "outputs")
        value = outputs.get(key, default)
        if value is None or value == "":
            return None
        dataset_name = _dataset_template_name_from_config(config)
        formatted = str(value).format(dataset=dataset_name)
        path = module.Path(formatted)
        if path.is_absolute():
            return path
        return module._output_base_dir(config, config_dir=config_dir) / path

    setattr(patched_output_base_dir, _OUTPUT_TEMPLATE_PATCH_MARKER, True)
    setattr(patched_resolve_output, _OUTPUT_TEMPLATE_PATCH_MARKER, True)
    module._output_base_dir = patched_output_base_dir
    module._resolve_output = patched_resolve_output
    setattr(module, _OUTPUT_TEMPLATE_PATCH_MARKER, True)


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return
    _patch_config_workflow_bool_parser(module)
    _patch_decode_from_config_output_templates(module)
    decode_kwargs = getattr(module, "_decode_kwargs", None)
    if decode_kwargs is None:
        setattr(module, _PATCH_MARKER, True)
        return

    @wraps(decode_kwargs)
    def patched_decode_kwargs(config: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        decoded = decode_kwargs(config, *args, **kwargs)
        dataset_name = decoded.get("dataset_name")
        if (
            dataset_name is None
            or _is_container_dataset_name(dataset_name)
            or _config_dataset_name_is_container(config)
        ):
            decoded = dict(decoded)
            decoded["dataset_name"] = _dataset_name_from_config(config)
        elif not isinstance(dataset_name, str):
            decoded = dict(decoded)
            decoded["dataset_name"] = str(dataset_name)
        return decoded

    module._decode_kwargs = patched_decode_kwargs
    setattr(module, _PATCH_MARKER, True)


class _DatasetNameConfigPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):  # type: ignore[override]
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_module(module)


class _DatasetNameConfigPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname not in _TARGET_MODULES:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _DatasetNameConfigPatchLoader):
            return spec
        spec.loader = _DatasetNameConfigPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install dataset-name normalization for config-driven decode helpers."""

    for module_name in _TARGET_MODULES:
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _DatasetNameConfigPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
