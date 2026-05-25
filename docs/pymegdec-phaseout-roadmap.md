# PyMEGDec phase-out roadmap

NeuRepTrace is the reusable decoding and probability-trace layer. PyMEGDec should remain a thin dataset- and paper-specific wrapper around BUSH-MEG conventions, legacy exports, and manuscript-specific analyses.

This page records the practical migration targets so future pull requests can move functionality in small, testable increments without blurring that boundary.

## Ownership rule

Move code to NeuRepTrace when it is independent of a single paper, subject naming convention, or local artifact layout. Keep code in PyMEGDec when it hard-codes BUSH-MEG filenames, CTF geometry assumptions, alpha-band paper defaults, or private result directory conventions.

## Migration candidates

| Capability | NeuRepTrace target | PyMEGDec residual role | Acceptance check |
| --- | --- | --- | --- |
| Feature-matrix decoding | `neureptrace.decoding` and config-driven CLIs | Build dataset config and pass project defaults | Same folds, labels, and chance level as the legacy command |
| MNE/FieldTrip loading | `neureptrace.io.fieldtrip_mat` and dataset specs | Provide BUSH-MEG path templates and metadata conventions | Manifest validates and all referenced files resolve |
| Onset and stimulus detection | `neureptrace.onset_detection`, `neureptrace.stimulus_detection`, and continuous-scan CLIs | Provide BUSH-MEG event annotations and expected windows | Event counts, latencies, and false-alarm summaries match legacy outputs within tolerance |
| Temporal generalization | `neureptrace.decoding.temporal_generalization` and config wrappers | Translate BUSH-MEG CLI arguments to NeuRepTrace configs | Output matrix shape, train/test-time grid, and balanced accuracy match reference artifacts |
| Hyperalignment/cross-subject alignment | `neureptrace.decoding.hyperalignment` and alignment utilities | Supply subject grouping, cue-task calibration inputs, and paper export names | LOSO split integrity is preserved and cue data never enter test classification labels |
| Probability observations | `neureptrace.observation_schema` and validation CLIs | Store project-specific result paths | Canonical validation passes before downstream state or event workflows |

## PR checklist

Before replacing a PyMEGDec-owned workflow with a NeuRepTrace-backed workflow, include the following in the PR description:

1. The legacy command and the new NeuRepTrace-backed command.
2. The dataset config or manifest used for the comparison.
3. The expected output files and their schemas.
4. A parity tolerance for numeric differences.
5. A smoke test that runs without private data, plus a documented command for the full BUSH-MEG validation run.

## Suggested migration order

1. Wrap the existing PyMEGDec command around a NeuRepTrace function without changing output files.
2. Add a parity test on a synthetic or tiny fixture.
3. Add a full-data validation command to documentation or CI artifacts.
4. Deprecate duplicated PyMEGDec implementation details after the parity gate is stable.
5. Remove the duplicate PyMEGDec code only after downstream scripts and paper artifacts use the NeuRepTrace-backed path.

## Validation commands

Dataset specs should be validated before any migration benchmark:

```bash
neureptrace dataset validate examples/configs/pymegdec_bushmeg.yml
neureptrace dataset manifest examples/configs/pymegdec_bushmeg.yml \
  --workflow stimulus_transfer \
  --out results/bushmeg_stimulus_manifest.csv
```

Probability-observation outputs should be checked before onset, stimulus, or temporal-state analyses:

```bash
neureptrace validate-observations results/observations.csv \
  --profile canonical \
  --report-out results/observation_validation.csv \
  --summary-out results/observation_summary.csv
```

For migrations that cannot run on public fixtures, commit the command line, expected artifact names, and a short interpretation of the result so the repository keeps a durable audit trail.
