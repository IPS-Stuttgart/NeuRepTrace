from pathlib import Path

from reptrace.nod_variant_sweep import DEFAULT_VARIANT_BY_NAME, DEFAULT_VARIANTS, build_commands, select_variants


def _argv_by_label(commands, label):
    return next(command.argv for command in commands if command.label == label)


def test_default_variants_cover_result_relevant_all_subject_manifests() -> None:
    manifests = {variant.manifest for variant in DEFAULT_VARIANTS}
    assert "nod_animate_all.csv" in manifests
    assert "nod_animate_logistic_tuned_pca_whiten_all.csv" in manifests
    assert "nod_animate_logistic_tuned_anova_select_all.csv" in manifests
    assert "nod_animate_shrinkage_lda_all.csv" in manifests
    assert "nod_animate_elastic_net_logistic_all.csv" in manifests
    assert "nod_animate_logistic_tuned_temporal_ensemble_all.csv" in manifests
    assert "nod_animate_logistic_temporal_smoothing_all.csv" in manifests
    assert len(DEFAULT_VARIANT_BY_NAME) == len(DEFAULT_VARIANTS)


def test_select_variants_preserves_requested_order() -> None:
    variants = select_variants(["shrinkage_lda", "baseline_logistic"])
    assert [variant.name for variant in variants] == ["shrinkage_lda", "baseline_logistic"]


def test_select_variants_rejects_unknown_names() -> None:
    try:
        select_variants(["not_a_variant"])
    except ValueError as exc:
        assert "not_a_variant" in str(exc)
        assert "baseline_logistic" in str(exc)
    else:
        raise AssertionError("select_variants should reject unknown variant names")


def test_build_commands_for_single_variant_includes_benchmark_outputs() -> None:
    commands = build_commands(
        [DEFAULT_VARIANT_BY_NAME["logistic_pca_whiten_tuned"]],
        benchmarks_dir=Path("benchmarks"),
        results_root=Path("results"),
        python_executable="python",
        chance=0.5,
        n_permutations=123,
        cluster_alpha=0.01,
        resume=True,
    )
    assert [command.label for command in commands] == [
        "validate:logistic_pca_whiten_tuned",
        "benchmark:logistic_pca_whiten_tuned",
        "report:logistic_pca_whiten_tuned",
        "inference:logistic_pca_whiten_tuned",
        "calibration:logistic_pca_whiten_tuned",
    ]

    benchmark = _argv_by_label(commands, "benchmark:logistic_pca_whiten_tuned")
    assert benchmark[:4] == ("python", "-m", "reptrace.benchmark", "benchmarks/nod_animate_logistic_tuned_pca_whiten_all.csv")
    assert "--resume" in benchmark
    assert "results/nod_animate_logistic_tuned_pca_whiten_all/summary.csv" in benchmark
    assert "results/nod_animate_logistic_tuned_pca_whiten_all/calibration" in benchmark


def test_temporal_smoothing_variant_adds_smoothing_benchmark_arguments() -> None:
    commands = build_commands(
        [DEFAULT_VARIANT_BY_NAME["logistic_temporal_smoothing"]],
        benchmarks_dir=Path("benchmarks"),
        results_root=Path("results"),
        python_executable="python",
    )
    benchmark = _argv_by_label(commands, "benchmark:logistic_temporal_smoothing")
    assert "--temporal-smoothing-dir" in benchmark
    assert "results/nod_animate_logistic_temporal_smoothing_all/temporal_smoothing" in benchmark
    assert "--temporal-smoothing-fit-window" in benchmark
    fit_window_index = benchmark.index("--temporal-smoothing-fit-window")
    assert benchmark[fit_window_index + 1 : fit_window_index + 3] == ("0.1", "0.8")
