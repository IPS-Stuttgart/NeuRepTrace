# Diagnostics

Use the NeuRepTrace doctor command to check whether an installed environment can
run the core workflows and whether dataset configuration files are structurally
valid before launching long decoding jobs.

## Environment check

```bash
neureptrace doctor
```

The command reports the Python version, the installed NeuRepTrace version, and
the availability of required runtime dependencies. Optional machine-learning
extras such as XGBoost and PyTorch are reported as warnings when they are not
installed.

Use a quieter core-only check in continuous-integration jobs or lightweight
environments:

```bash
neureptrace doctor --skip-optional
```

For automation, request JSON output:

```bash
neureptrace doctor --skip-optional --json
```

## Dataset configuration check

Dataset YAML or JSON files can be validated without loading the full neural data:

```bash
neureptrace doctor \
  --dataset-config examples/configs/pymegdec_bushmeg.yml \
  --skip-optional
```

Add `--check-dataset-files` when the data directory is mounted and all referenced
files should already exist:

```bash
neureptrace doctor \
  --dataset-config configs/bush_meg/stimulus_decoding.yml \
  --check-dataset-files
```

## Requiring project-specific modules

Some benchmark workflows depend on optional project-level modules or adapters.
Use `--require-module` to make such imports fail the diagnostic explicitly:

```bash
neureptrace doctor \
  --require-module xgboost \
  --require-module torch
```

The command exits with status code `1` when required checks fail. It exits with
status code `0` when only optional dependency warnings are present, unless
`--fail-on-warnings` is supplied.
