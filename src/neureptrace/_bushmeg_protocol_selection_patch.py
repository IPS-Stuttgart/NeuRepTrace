"""Respect explicit protocol filters when choosing default BUSH-MEG methods."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

_PATCH_MARKER = "_neureptrace_bushmeg_protocol_selection_patch_installed"


def _default_configured_methods(
    all_protocols_module: Any,
    registry: Mapping[str, Any],
    protocols: str | Sequence[str | int] | None,
) -> list[str]:
    requested_protocols = all_protocols_module._parse_protocols(protocols)
    default_protocols = requested_protocols if all_protocols_module._split_csv(protocols) else {1, 2}
    return [
        method
        for method, spec in registry.items()
        if spec.runnable and spec.protocol_category in default_protocols and spec.protocol_category != 4
    ]


def _expanded_configured_methods(
    all_protocols_module: Any,
    registry: Mapping[str, Any],
    configured: Sequence[str],
    groups: Mapping[str, Any],
) -> list[str]:
    expanded: list[str] = []
    for token in configured:
        if token.lower() == "all":
            expanded.extend(registry)
        elif token in groups:
            expanded.extend(all_protocols_module._split_csv(groups[token]))
        else:
            expanded.append(token)
    return list(dict.fromkeys(expanded))


def install() -> None:
    """Patch default method selection to honor explicit --protocols values."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    def _selected_methods(
        *,
        all_protocols: Mapping[str, Any],
        methods: str | Sequence[str] | None,
        protocols: str | Sequence[str | int] | None,
        include_oracle: bool,
        non_oracle: bool = False,
    ) -> list[Any]:
        registry = all_protocols_module.method_registry()
        configured = all_protocols_module._split_csv(methods) or all_protocols_module._split_csv(all_protocols.get("methods"))
        if not configured:
            configured = _default_configured_methods(all_protocols_module, registry, protocols)
        groups = all_protocols.get("method_groups", {}) or {}
        if not isinstance(groups, Mapping):
            raise ValueError("all_protocols.method_groups must be a mapping.")
        configured = _expanded_configured_methods(all_protocols_module, registry, configured, groups)
        unknown = sorted(set(configured).difference(registry))
        if unknown:
            raise ValueError(f"Unknown BUSH-MEG all-protocol method(s): {', '.join(unknown)}.")
        requested_protocols = all_protocols_module._parse_protocols(protocols)
        selected: list[Any] = []
        for method in configured:
            spec = registry[method]
            if spec.protocol_category not in requested_protocols:
                continue
            if non_oracle and spec.protocol_category == 4:
                continue
            if spec.protocol_category == 4 and not include_oracle:
                raise ValueError(f"Method {method!r} is Protocol 4 oracle/debug and requires --include-oracle.")
            all_protocols_module.validate_target_label_policy(
                spec.protocol,
                uses_target_labels_for_fitting=spec.protocol.uses_target_labels_for_fitting,
                include_oracle=include_oracle,
            )
            selected.append(spec)
        return selected

    def _configured_method_names(
        *,
        all_protocols: Mapping[str, Any],
        methods: str | Sequence[str] | None,
        protocols: str | Sequence[str | int] | None,
        include_oracle: bool,
        non_oracle: bool = False,
    ) -> set[str]:
        registry = all_protocols_module.method_registry()
        configured = all_protocols_module._split_csv(methods) or all_protocols_module._split_csv(all_protocols.get("methods"))
        if not configured:
            configured = _default_configured_methods(all_protocols_module, registry, protocols)
        groups = all_protocols.get("method_groups", {}) or {}
        if not isinstance(groups, Mapping):
            raise ValueError("all_protocols.method_groups must be a mapping.")
        configured = _expanded_configured_methods(all_protocols_module, registry, configured, groups)
        unknown = sorted(set(configured).difference(registry))
        if unknown:
            raise ValueError(f"Unknown BUSH-MEG all-protocol method(s): {', '.join(unknown)}.")
        requested_protocols = all_protocols_module._parse_protocols(protocols)
        return {
            method
            for method in configured
            if registry[method].protocol_category in requested_protocols
            and not (non_oracle and registry[method].protocol_category == 4)
            and (include_oracle or registry[method].protocol_category != 4)
        }

    all_protocols_module = all_protocols
    all_protocols._selected_methods = _selected_methods
    all_protocols._configured_method_names = _configured_method_names
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
