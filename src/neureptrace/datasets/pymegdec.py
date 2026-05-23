"""PyMEGDec/BUSH-MEG dataset-spec compatibility helpers.

The historical PyMEGDec repository used private FieldTrip-style MATLAB files
named ``Part{subject}Data.mat`` and ``Part{subject}CueData.mat``.  NeuRepTrace
owns the reusable dataset/config machinery, so the small recipe that describes
those files lives here and PyMEGDec can import it as a compatibility shim.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

DEFAULT_PARTICIPANTS = "1-4,6,8,9,10,13-27"
DEFAULT_ENV_VAR = "PYMEGDEC_DATA_DIR"
DEFAULT_OUTPUT = Path("configs/bushmeg.yml")

TEMPLATE = """schema_version: neureptrace.dataset.v1
dataset_id: bushmeg
description: PyMEGDec-style MEG participant files described declaratively.

root:
{root_path_block}  env: {env_var}
  fallback_file: .pymegdec-data-dir

subjects:
  include: "{participants}"

splits:
  main:
    loader: matlab_fieldtrip
    path_template: "Part{{subject}}Data.mat"
    mat_key: data
    trial_key: trial
    time_key: time
    channel_key: label
    label_key: trialinfo
    label_index_base: 1
    trial_layout: channels_by_time

  cue:
    loader: matlab_fieldtrip
    path_template: "Part{{subject}}CueData.mat"
    mat_key: data
    trial_key: trial
    time_key: time
    channel_key: label
    label_key: trialinfo
    label_index_base: 1
    trial_layout: channels_by_time

labels:
  chance_classes: 16
  index_base: 1
  subtract_one_when_no_null_class: true

preprocessing_defaults:
  frequency_range_hz: [0.0, .inf]
  window_size_s: 0.1
  train_window_center_s: 0.2
  null_window_center_s: null
  resample_hz: null
  pca_components: 100

workflows:
  stimulus_transfer:
    split: main
    manifest:
      paired_split: cue
      transfer_direction: main-to-cue
      classifier: multiclass-svm
      chance: 0.0625
      window_start_s: -0.2
      window_stop_s: 0.6
      window_step_s: 0.05

  stimulus_transfer_reverse:
    split: cue
    manifest:
      paired_split: main
      transfer_direction: cue-to-main
      classifier: multiclass-svm
      chance: 0.0625
      window_start_s: -0.2
      window_stop_s: 0.6
      window_step_s: 0.05

outputs:
  default_dir: outputs
"""


def build_pymegdec_bushmeg_dataset_spec_text(
    *,
    participants: str = DEFAULT_PARTICIPANTS,
    env_var: str = DEFAULT_ENV_VAR,
    data_dir: str | Path | None = None,
) -> str:
    """Return a YAML NeuRepTrace dataset spec for PyMEGDec-style files."""

    root_path_block = ""
    if data_dir is not None:
        root_path_block = f"  path: {json.dumps(str(data_dir))}\n"
    return TEMPLATE.format(
        participants=participants,
        env_var=env_var,
        root_path_block=root_path_block,
    )


def write_pymegdec_bushmeg_dataset_spec_file(
    out: str | Path = DEFAULT_OUTPUT,
    *,
    participants: str = DEFAULT_PARTICIPANTS,
    env_var: str = DEFAULT_ENV_VAR,
    data_dir: str | Path | None = None,
) -> Path:
    """Write the canonical PyMEGDec/BUSH-MEG dataset spec and return its path."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_pymegdec_bushmeg_dataset_spec_text(
            participants=participants,
            env_var=env_var,
            data_dir=data_dir,
        ),
        encoding="utf-8",
    )
    return out


def add_pymegdec_bushmeg_dataset_spec_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add PyMEGDec/BUSH-MEG spec-writer arguments to ``parser``."""

    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output YAML path.")
    parser.add_argument("--participants", default=DEFAULT_PARTICIPANTS, help="Participant ids, for example 1-4,6,8.")
    parser.add_argument("--env-var", default=DEFAULT_ENV_VAR, help="Environment variable used by root.env.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional explicit data root to write into root.path. If omitted, the spec uses env/fallback root resolution.",
    )
    return parser


def write_pymegdec_bushmeg_dataset_spec_from_args(args: argparse.Namespace) -> int:
    """Write a PyMEGDec/BUSH-MEG spec from parsed CLI arguments."""

    out = write_pymegdec_bushmeg_dataset_spec_file(
        args.out,
        participants=args.participants,
        env_var=args.env_var,
        data_dir=args.data_dir,
    )
    print(f"Wrote {out}")
    print("Validate with: neureptrace dataset validate", out)
    return 0


def write_pymegdec_bushmeg_dataset_spec(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Write a NeuRepTrace YAML dataset spec for PyMEGDec/BUSH-MEG files."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Write a NeuRepTrace YAML dataset spec for the historical PyMEGDec Part*Data.mat convention.",
    )
    add_pymegdec_bushmeg_dataset_spec_arguments(parser)
    return write_pymegdec_bushmeg_dataset_spec_from_args(parser.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for writing the PyMEGDec/BUSH-MEG dataset spec."""

    return write_pymegdec_bushmeg_dataset_spec(argv)


if __name__ == "__main__":
    raise SystemExit(main())
