from __future__ import annotations

import ast
from pathlib import Path

import neureptrace


def _package_init_tree() -> ast.Module:
    return ast.parse(Path(neureptrace.__file__).read_text(encoding="utf-8"))


def _runtime_patch_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 1 or node.module is not None:
            continue
        names.extend(alias.asname or alias.name for alias in node.names if alias.name.startswith("_"))
    return names


def _install_call_targets(tree: ast.Module) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call = node.func
        if not isinstance(call, ast.Attribute) or call.attr != "install":
            continue
        if isinstance(call.value, ast.Name):
            targets.add(call.value.id)
    return targets


def test_imported_runtime_patch_modules_are_installed() -> None:
    tree = _package_init_tree()

    imported = _runtime_patch_imports(tree)
    installed = _install_call_targets(tree)
    missing = sorted(set(imported) - installed)

    assert missing == [], "Runtime patches imported by neureptrace.__init__ must also be installed. Missing: " + ", ".join(missing)
