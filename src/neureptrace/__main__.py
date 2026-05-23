"""Module entrypoint for ``python -m neureptrace``."""

from __future__ import annotations

from neureptrace.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
