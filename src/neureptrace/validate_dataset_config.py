"""Console entry point for dataset workflow config validation."""

from __future__ import annotations

from neureptrace.config_workflow import validate_dataset_config_main


def main() -> int:
    return validate_dataset_config_main()


if __name__ == "__main__":
    raise SystemExit(main())
