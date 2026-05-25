# Validation checklist

Use this checklist before launching long NeuRepTrace decoding, transfer, or temporal-state workflows. The goal is to fail fast on malformed inputs and to keep generated artifacts reproducible.

## Dataset configuration

Validate dataset configs before running model code:

```bash
neureptrace-validate-dataset-config configs/example.yml --print-effective-config
```

Use `--check-files` once paths are staged on the execution machine. This catches missing epoch files, metadata CSVs, and participant-template mistakes before a decoder allocates memory or starts cross-validation.

## Dataset specs and manifests

For versioned dataset specs, validate and expand the spec through the grouped dataset CLI:

```bash
neureptrace dataset validate examples/configs/pymegdec_bushmeg.yml --require-files
neureptrace dataset manifest examples/configs/pymegdec_bushmeg.yml --workflow stimulus_transfer --out results/manifest.csv
```

Commit or archive the expanded manifest with the result bundle when the manifest defines a benchmark split or a paper-facing run.

## Probability observations

Validate probability-observation CSVs before feeding them into temporal modeling, stimulus detection, stacking, or reports:

```bash
neureptrace-validate-observations results/observations.csv --profile canonical --report-out results/observation_validation.csv
```

Use workflow-specific profiles when possible:

- `canonical` for paper-facing observation exports with reproducibility columns;
- `temporal-model` for sequence-based temporal-state workflows;
- `stimulus-detection` for continuous stream event detection.

Use `--require-normalized` when downstream code assumes each `prob_class_*` row sums to one. Use `--normalize-out` only when the normalization itself should be an explicit, saved preprocessing step.

## Environment diagnostics

Run the doctor command in every fresh environment and archive its output with large benchmark runs:

```bash
neureptrace-doctor --output-json results/neureptrace_doctor.json
```

This records required and optional dependency status, which makes later failures easier to distinguish from data or model issues.

## Result handoff

For each long run, keep the following together:

1. the original config or dataset spec;
2. the printed effective config or expanded manifest;
3. the observation-validation report;
4. the doctor report;
5. the exact command line used for the run.

This bundle is usually enough to reproduce path resolution, split expansion, probability-table assumptions, and environment state.