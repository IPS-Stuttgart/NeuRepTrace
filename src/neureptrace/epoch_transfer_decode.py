"""Backward-compatible entry point for cross-epoch transfer decoding."""

from __future__ import annotations

from neureptrace.time_transfer_decode import main, run_time_transfer_decode

__all__ = ["main", "run_time_transfer_decode"]


if __name__ == "__main__":
    main()
