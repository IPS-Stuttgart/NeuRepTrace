# Contributing to NeuRepTrace

Thank you for helping improve NeuRepTrace. This project is an early-stage Python toolkit for calibrated, time-resolved M/EEG decoding and downstream probability-trace analyses, so small reproducibility and workflow improvements are especially valuable.

## Development setup

NeuRepTrace supports Python 3.11 through 3.14. Use Poetry from a source checkout so local commands match the continuous-integration jobs:

```bash
poetry install --with dev,docs
poetry run neureptrace --list-commands
```

If you need only the runtime package for local experiments, editable pip installs are also supported:

```bash
python -m pip install -e .
```

## Checks before pushing

Run the same lightweight checks used by CI before submitting a change:

```bash
poetry run ruff check .
poetry run python -m pytest --rootdir . -v --strict-config ./tests
poetry run mkdocs build --strict
```

Use the focused test file or `-k` selector while iterating, then run the full suite before committing.

## Project boundaries

Keep reusable M/EEG decoding, calibration, temporal-generalization, probability-observation, onset/state-inference, and reporting code in the NeuRepTrace package. Dataset-specific filename conventions, paper-specific defaults, and one-off export scripts should stay in dataset specs, examples, or thin downstream wrappers unless they are generalized into reusable workflows.

When adding a console entry point, keep the grouped CLI and package metadata aligned:

1. Add the focused script to `pyproject.toml` when needed.
2. Add the grouped command alias in `src/neureptrace/cli.py`.
3. Extend `tests/test_grouped_cli.py` so aliases dispatch to the intended module.
4. Document the command in the README or the relevant page under `docs/`.

## Tests and fixtures

Prefer synthetic data generators or tiny fixtures over committed data extracts. Mark expensive or external-artifact checks with the existing `performance` or `parity` markers so the default test suite stays fast and deterministic.

For probability-trace and decoder changes, include tests that cover both the aggregate summary and row-level observation exports. For CLI changes, test the callable `main()` entry point directly where possible instead of spawning subprocesses.

## Data and generated artifacts

Do not commit raw M/EEG recordings, large downloaded datasets, private participant metadata, or bulky generated result folders. Put reproducible examples under `examples/`, dataset specifications under `configs/` or `examples/configs/`, and generated outputs under ignored result directories.

When sharing compact benchmark artifacts, remove direct participant identifiers and document the command used to recreate the artifact.

## Documentation

User-visible workflow changes should include documentation. Prefer concise command examples with explicit input and output paths, and keep docs buildable with:

```bash
poetry run mkdocs build --strict
```
