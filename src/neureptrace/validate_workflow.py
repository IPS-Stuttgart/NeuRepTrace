"""Compatibility entry point for workflow-spec validation."""

from __future__ import annotations

from neureptrace.workflow import validate_main as main


if __name__ == "__main__":
    raise SystemExit(main())
