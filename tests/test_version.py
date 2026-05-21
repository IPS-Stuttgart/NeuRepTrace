from pathlib import Path

import tomllib

import neureptrace


def test_package_version_matches_pyproject():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert neureptrace.__version__ == pyproject["tool"]["poetry"]["version"]
