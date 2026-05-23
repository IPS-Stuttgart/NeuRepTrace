"""Compatibility entry point for synthetic FieldTrip fixture generation."""

from __future__ import annotations

from neureptrace.io.synthetic_fieldtrip import (  # noqa: F401
    SyntheticDataConfig,
    SyntheticDataOutput,
    SyntheticFieldTripConfig,
    SyntheticFieldTripOutput,
    build_parser,
    main,
    make_synthetic_fieldtrip_data,
    write_synthetic_dataset,
    write_synthetic_fieldtrip_dataset,
)


if __name__ == "__main__":
    raise SystemExit(main())
