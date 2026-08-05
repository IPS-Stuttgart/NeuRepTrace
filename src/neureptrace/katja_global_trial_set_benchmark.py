"""Run the exact Katja global-finger benchmark with soft trial-set loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neureptrace import katja_global_physical_benchmark as benchmark
from neureptrace._katja_finger_sequence_support import (
    DEFAULT_PARTICIPANTS,
    _load_source_map,
    _parse_csv_values,
    katja_nested_trial_calibration_indices,
    load_katja_feature_cache,
)
from neureptrace.decoding.progressive_sequence_finetune import (
    NestedTrialCalibrationSplit,
)
from neureptrace.decoding.trial_set_sequence import (
    TorchTrialSetSequenceClassifier,
)


def _exact_splits(
    strata,
    calibration_counts=(20,),
    *,
    max_per_stratum=None,
    min_evaluation_per_stratum=1,
    seed=13,
    context=(),
):
    del context
    counts = tuple(int(value) for value in calibration_counts)
    if max_per_stratum is not None and int(max_per_stratum) != max(counts):
        raise ValueError(
            "Exact Katja split requires max_per_stratum=max(calibration_counts)."
        )
    if int(min_evaluation_per_stratum) != 1:
        raise ValueError("Exact Katja split requires evaluation rows per sequence.")
    calibration, evaluation, pool = katja_nested_trial_calibration_indices(
        strata,
        counts,
        seed=int(seed),
    )
    return {
        count: NestedTrialCalibrationSplit(
            calibration_indices=indices,
            evaluation_indices=evaluation.copy(),
            calibration_pool_indices=pool.copy(),
            per_stratum=int(count),
            max_per_stratum=max(counts),
            seed=int(seed),
        )
        for count, indices in calibration.items()
    }


def run_katja_global_trial_set_benchmark(
    cache,
    *,
    model_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Run global physical decoding with the pre-registered trial-set objective."""

    original_classifier = benchmark.TorchProgressiveSequenceClassifier
    original_splitter = benchmark.select_nested_trial_calibration_splits
    benchmark.TorchProgressiveSequenceClassifier = TorchTrialSetSequenceClassifier
    benchmark.select_nested_trial_calibration_splits = _exact_splits
    try:
        return benchmark.run_katja_global_physical_benchmark(
            cache,
            model_kwargs=model_kwargs,
            **kwargs,
        )
    finally:
        benchmark.TorchProgressiveSequenceClassifier = original_classifier
        benchmark.select_nested_trial_calibration_splits = original_splitter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--targets")
    parser.add_argument("--calibration-counts", default="20")
    parser.add_argument("--calibration-seeds", default="13")
    parser.add_argument("--n-source-participants", type=int, default=9)
    parser.add_argument("--source-selection-seed", type=int, default=2026)
    parser.add_argument("--source-map-json")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--hidden-units", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--source-max-epochs", type=int, default=120)
    parser.add_argument("--meta-epochs", type=int, default=2)
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--last-block-steps", type=int, default=60)
    parser.add_argument("--full-finetune-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trial-set-loss-weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cache = load_katja_feature_cache(args.feature_cache)
    per_seed, per_target, summary, metadata = run_katja_global_trial_set_benchmark(
        cache,
        participants=_parse_csv_values(args.participants),
        target_participants=(
            None if args.targets is None else _parse_csv_values(args.targets)
        ),
        calibration_counts=_parse_csv_values(args.calibration_counts, cast=int),
        calibration_seeds=_parse_csv_values(args.calibration_seeds, cast=int),
        n_source_participants=args.n_source_participants,
        source_selection_seed=args.source_selection_seed,
        source_map=_load_source_map(args.source_map_json),
        pca_components=args.pca_components,
        model_kwargs={
            "hidden_units": args.hidden_units,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "adapter_rank": args.adapter_rank,
            "source_max_epochs": args.source_max_epochs,
            "meta_epochs": args.meta_epochs,
            "adapter_steps": args.adapter_steps,
            "last_block_steps": args.last_block_steps,
            "full_finetune_steps": args.full_finetune_steps,
            "batch_size": args.batch_size,
            "trial_set_loss_weight": args.trial_set_loss_weight,
            "device": args.device,
        },
    )
    metadata.update(
        {
            "comparison_source_seed": int(args.source_selection_seed),
            "comparison_split_implementation": "recovered_sequential_rng",
            "spm_input_selection": "Bfc*ICA_corrected.mat",
            "trial_set_loss_weight": float(args.trial_set_loss_weight),
            "trial_set_weight_rationale": "matches_existing_assignment_loss_default",
            "screening_policy": "pre_registered_from_independent_vs_hungarian_gap",
        }
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    per_target.to_csv(output / "per_target.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "run_katja_global_trial_set_benchmark",
)
