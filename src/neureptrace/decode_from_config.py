"""Console entry point for config-driven decoding."""

from __future__ import annotations

from neureptrace.config_workflow import decode_from_config_main


def main() -> int:
    return decode_from_config_main()


if __name__ == "__main__":
    raise SystemExit(main())
