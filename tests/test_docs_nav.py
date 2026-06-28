from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

NAV_ONLY_TOP_LEVEL_DOCS = {
    "index.md",
}


def _iter_nav_paths(entries: Iterable[object]) -> set[str]:
    paths: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            paths.add(entry)
        elif isinstance(entry, dict):
            for value in entry.values():
                if isinstance(value, str):
                    paths.add(value)
                elif isinstance(value, list):
                    paths.update(_iter_nav_paths(value))
    return paths


def test_top_level_docs_are_exposed_in_mkdocs_nav():
    config = yaml.safe_load(Path("mkdocs.yml").read_text(encoding="utf-8"))
    nav_paths = _iter_nav_paths(config["nav"])
    top_level_docs = {
        path.name
        for path in Path("docs").glob("*.md")
        if path.name not in NAV_ONLY_TOP_LEVEL_DOCS
    }

    missing_docs = sorted(top_level_docs - nav_paths)

    assert missing_docs == []


def test_top_level_nav_targets_exist():
    config = yaml.safe_load(Path("mkdocs.yml").read_text(encoding="utf-8"))
    nav_paths = _iter_nav_paths(config["nav"])
    missing_files = sorted(
        path
        for path in nav_paths
        if "/" not in path and not Path("docs", path).is_file()
    )

    assert missing_files == []
