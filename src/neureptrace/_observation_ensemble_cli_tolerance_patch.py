"""Forward CLI probability tolerance into observation ensembling.

The observation-ensemble CLI validates input observation CSVs with the user-supplied
``--probability-tolerance`` but previously called ``ensemble_probability_observations``
without forwarding that value.  The second validation step therefore still used the
library default, so intentionally relaxed CLI runs could fail after passing input
schema validation.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_observation_ensemble_cli_tolerance_patch_installed"


def _cli_probability_tolerance(argv: Sequence[str] | None) -> float:
    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probability-tolerance", type=float, default=observation_ensemble.DEFAULT_PROBABILITY_TOLERANCE)
    parsed, _ = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    return float(parsed.probability_tolerance)


def install() -> None:
    """Patch ``neureptrace-observation-ensemble`` to respect CLI tolerance."""

    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    original_main = observation_ensemble.main
    if getattr(original_main, _PATCH_MARKER, False):
        return

    @wraps(original_main)
    def main(argv: Sequence[str] | None = None) -> int:
        tolerance = _cli_probability_tolerance(argv)
        original_ensemble = observation_ensemble.ensemble_probability_observations

        @wraps(original_ensemble)
        def ensemble_probability_observations(*args: Any, **kwargs: Any):
            kwargs.setdefault("probability_tolerance", tolerance)
            return original_ensemble(*args, **kwargs)

        observation_ensemble.ensemble_probability_observations = ensemble_probability_observations
        try:
            return original_main(argv)
        finally:
            observation_ensemble.ensemble_probability_observations = original_ensemble

    setattr(main, _PATCH_MARKER, True)
    observation_ensemble.main = main


__all__ = ["install"]
