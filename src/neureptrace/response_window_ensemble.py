from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from neureptrace.observations import stable_hash
from neureptrace.temporal_model import probability_columns, read_probability_observations
from neureptrace.temporal_smoothing import metrics_from_probability_observations, smooth_probability_observations

DEFAULT_RESPONSE_TIMES = (0.088, 0.136, 0.184, 0.232, 0.280)
ENSEMBLE_MODE_CHOICES = (
    "uniform",
    "source_oof_nonnegative",
    "decoder_source_oof_nonnegative",
    "response_window_poststimulus_forward",
)
COMBINE_CHOICES = ("log_probability_mean", "probability_mean")
OUTPUT_DECODER = "poststimulus_response_window_logit_ensemble"
OUTPUT_EMISSION_MODE = "response_window_logit_ensemble"
HYBRID_SMOOTHING_FIT_WINDOW = (0.10, 0.30)
HYBRID_SMOOTHING_STAY_GRID_SIZE = 200
HYBRID_SMOOTHING_EMISSION_SUFFIX = "response_window_poststimulus_forward"
EPSILON = 1e-12
TARGET_GROUP_COLUMNS = ("subject", "group", "outer_test_groups", "session", "fold")
SOURCE_HASH_COLUMNS = ("preprocessing_hash", "model_hash")
ROW_IDENTITY_COLUMNS = ("true_label", "true_class")
METRIC_PROVENANCE_COLUMNS = (
    "label_shuffle_control",
    "label_shuffle_seed",
    "class_prior_correction",
    "source_calibration",
    "source_time_selection",
)
SOURCE_PROTOCOL_COLUMNS = (
    *METRIC_PROVENANCE_COLUMNS,
    "alignment_method",
    "alignment_anchor_mode",
    "alignment_anchor_column",
    "alignment_repetition_cap",
    "alignment_components",
    "alignment_times",
    "alignment_window_mode",
    "alignment_same_decode_window",
    "alignment_target_projection",
    "alignment_strict_source_only",
    "alignment_uses_unlabeled_target_data",
    "alignment_uses_class_labels",
    "alignment_target_calibrated",
    "alignment_target_calibration_per_anchor",
    "alignment_target_calibration_seed",
    "alignment_oracle_target_calibrated",
    "alignment_debug_upper_bound",
    "alignment_valid_for_benchmark",
    "alignment_protocol",
)


def _normalize_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Response-window probabilities must be a two-dimensional array.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Response-window probabilities must be finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("Response-window probabilities must be non-negative.")
    probabilities = np.clip(probabilities, EPSILON, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Response-window probabilities must have positive row sums.")
    return probabilities / row_sums


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _time_label(value: float) -> str:
    return f"{float(value):.12g}"


def _nearest_times(available_times: np.ndarray, requested_times: tuple[float, ...]) -> tuple[float, ...]:
    if available_times.size == 0:
        raise ValueError("Observation table does not contain any time values.")
    selected = []
    for requested in requested_times:
        selected.append(float(available_times[np.argmin(np.abs(available_times - requested))]))
    selected_labels = [_time_label(time) for time in selected]
    if len(set(selected_labels)) != len(selected_labels):
        requested_labels = "|".join(_time_label(time) for time in requested_times)
        available_labels = "|".join(_time_label(time) for time in np.sort(available_times))
        raise ValueError(
            "Requested response-window times collapse to duplicate decoded time centers. "
            f"requested={requested_labels}; available={available_labels}; selected={'|'.join(selected_labels)}"
        )
    return tuple(selected)


def _has_nonempty_values(series: pd.Series) -> bool:
    if series.dropna().empty:
        return False
    values = series.dropna().astype(str).str.strip()
    values = values.loc[~values.str.lower().isin({"", "nan", "none", "nat"})]
    return not values.empty


def _subject_key_column(frame: pd.DataFrame) -> str | None:
    for column in TARGET_GROUP_COLUMNS:
        if column in frame.columns and _has_nonempty_values(frame[column]):
            return column
    return None


def _sequence_key_columns(frame: pd.DataFrame) -> list[str]:
    keys = []
    subject_column = _subject_key_column(frame)
    if subject_column is not None:
        keys.append(subject_column)
    if "fold" in frame.columns and "fold" not in keys:
        keys.append("fold")
    if "sample_index" in frame.columns:
        keys.append("sample_index")
    elif "sequence_id" in frame.columns:
        keys.append("sequence_id")
    else:
        raise ValueError("Observation table needs sample_index or sequence_id columns.")
    return keys


def _target_subject_values(base: pd.DataFrame, key_columns: list[str]) -> np.ndarray:
    subject_column = _subject_key_column(base)
    if subject_column is not None:
        return base[subject_column].astype(str).to_numpy()
    for column in TARGET_GROUP_COLUMNS:
        if column in key_columns:
            return base.index.get_level_values(column).astype(str).to_numpy()
    return np.full(len(base), "", dtype=object)


def _nonempty_unique_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(
        value
        for value in frame[column].dropna().astype(str).str.strip().unique().tolist()
        if value and value.lower() not in {"nan", "none", "nat"}
    )


def _check_single_plain_response_source(selected: pd.DataFrame) -> None:
    decoder_values = _nonempty_unique_values(selected, "decoder")
    if len(decoder_values) > 1:
        raise ValueError(
            "Plain response-window ensembling received multiple decoder values "
            f"{decoder_values}. Use mode='decoder_source_oof_nonnegative' or filter to one decoder."
        )
    emission_values = _nonempty_unique_values(selected, "emission_mode")
    if len(emission_values) > 1:
        raise ValueError(
            "Plain response-window ensembling received multiple emission_mode values "
            f"{emission_values}. Filter to one emission mode before response-window aggregation."
        )


def _check_unique_response_keys(frame: pd.DataFrame, key_columns: Sequence[str], *, context: str) -> None:
    duplicate_count = int(frame.duplicated(list(key_columns), keep=False).sum())
    if duplicate_count:
        examples = frame.loc[frame.duplicated(list(key_columns), keep=False), list(key_columns)].head(5).to_dict("records")
        raise ValueError(
            f"Response-window observations contain {duplicate_count} duplicate rows for {context}. "
            f"Key columns: {list(key_columns)}. Examples: {examples}"
        )


def _class_columns_for_probabilities(frame: pd.DataFrame, prob_columns: Sequence[str]) -> tuple[str, ...]:
    suffixes = tuple(column.removeprefix("prob_class_") for column in prob_columns)
    return tuple(column for column in (f"class_{suffix}" for suffix in suffixes) if column in frame.columns)


def _row_identity_columns(frame: pd.DataFrame, prob_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in (*ROW_IDENTITY_COLUMNS, *_class_columns_for_probabilities(frame, prob_columns)) if column in frame.columns)


def _source_protocol_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in SOURCE_PROTOCOL_COLUMNS if column in frame.columns)


def _integer_label_values(values: Sequence[object] | np.ndarray | pd.Series, *, n_classes: int | None = None) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any():
        raise ValueError("Response-window true_label values must be numeric and non-missing.")
    label_values = numeric.to_numpy(dtype=float)
    rounded = np.rint(label_values)
    if not bool(np.isclose(label_values, rounded, rtol=0.0, atol=1.0e-12).all()):
        raise ValueError("Response-window true_label values must be integer-valued.")
    labels = rounded.astype(int)
    if n_classes is not None and bool(((labels < 0) | (labels >= int(n_classes))).any()):
        raise ValueError("Response-window true_label values must index prob_class_* columns.")
    return labels


def _normalized_identity_values(values: Sequence[object] | np.ndarray | pd.Series, *, column: str) -> pd.Series:
    series = pd.Series(values).reset_index(drop=True)
    if column == "true_label":
        return pd.Series([str(int(value)) for value in _integer_label_values(series)], dtype=object)
    return series.where(~series.isna(), "").astype(str).str.strip()


def _check_row_identity_consistency(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    columns: Sequence[str],
    context: str,
) -> None:
    for column in columns:
        if column not in reference.columns or column not in candidate.columns:
            continue
        reference_values = _normalized_identity_values(reference[column], column=column)
        candidate_values = _normalized_identity_values(candidate[column], column=column)
        mismatch = reference_values != candidate_values
        if bool(mismatch.any()):
            examples = [
                {
                    "row": int(index),
                    "reference": reference_values.iloc[index],
                    context: candidate_values.iloc[index],
                }
                for index in np.flatnonzero(mismatch.to_numpy())[:5]
            ]
            raise ValueError(f"Response-window observations have inconsistent {column!r} values for {context}. Examples: {examples}")


def _candidate_weights(n_times: int, step: float) -> np.ndarray:
    if n_times < 1:
        raise ValueError("Need at least one response time.")
    if n_times == 1:
        return np.ones((1, 1), dtype=float)
    levels = np.arange(0.0, 1.0 + step / 2.0, step)
    weights = []
    for candidate in product(levels, repeat=n_times):
        candidate = np.asarray(candidate, dtype=float)
        total = float(candidate.sum())
        if abs(total - 1.0) <= step / 2.0 and total > 0.0:
            weights.append(candidate / total)
    if not weights:
        raise ValueError("Weight grid is empty; use a larger --weight-grid-step.")
    return np.unique(np.vstack(weights).round(12), axis=0)


def _combine_logits(probability_cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability_cube, EPSILON, 1.0))
    weighted_logits = np.tensordot(logits, weights, axes=([1], [0]))
    return _softmax(weighted_logits)


def _combine_probabilities(probability_cube: np.ndarray, weights: np.ndarray, *, combine: str) -> np.ndarray:
    if combine == "log_probability_mean":
        return _combine_logits(probability_cube, weights)
    if combine == "probability_mean":
        probabilities = np.tensordot(probability_cube, weights, axes=([1], [0]))
        return _normalize_rows(probabilities)
    raise ValueError(f"Unknown response-window combine mode '{combine}'.")


def _score_weights(probability_cube: np.ndarray, labels: np.ndarray, weights: np.ndarray, *, combine: str) -> float:
    probabilities = _combine_probabilities(probability_cube, weights, combine=combine)
    return float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))


def _source_hash_value(frame: pd.DataFrame, index_key: object, column: str) -> str:
    if column not in frame.columns:
        return ""
    value = frame.loc[index_key, column]
    if isinstance(value, pd.Series):
        values = value.drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"Source provenance column {column!r} is not unique for response-window key {index_key!r}.")
        value = values.iloc[0]
    return "" if pd.isna(value) else str(value)


def _source_hash_records(
    wide_provenance: Mapping[object, pd.DataFrame],
    source_keys: Sequence[object],
    row_index: pd.Index,
    *,
    column: str,
    labeler: Callable[[object], str],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index_key in row_index:
        records.append(
            {
                labeler(source_key): _source_hash_value(wide_provenance[source_key], index_key, column)
                for source_key in source_keys
            }
        )
    return records


def _has_nonempty_source_hash(records: Sequence[dict[str, str]]) -> bool:
    return any(any(value for value in record.values()) for record in records)


def _format_source_hash_record(record: Mapping[str, str]) -> str:
    return "|".join(f"{source}:{value}" for source, value in record.items())


def _apply_response_window_hashes(
    target_base: pd.DataFrame,
    *,
    source_preprocessing_records: Sequence[dict[str, str]],
    source_model_records: Sequence[dict[str, str]],
    model_payload: Mapping[str, object],
) -> None:
    has_source_preprocessing = _has_nonempty_source_hash(source_preprocessing_records)
    has_source_models = _has_nonempty_source_hash(source_model_records)
    if has_source_preprocessing:
        target_base["response_window_source_preprocessing_hashes"] = [
            _format_source_hash_record(record) for record in source_preprocessing_records
        ]
        target_base["preprocessing_hash"] = [
            stable_hash({"decoder": OUTPUT_DECODER, "response_window_source_preprocessing_hashes": record})
            for record in source_preprocessing_records
        ]
    if has_source_models:
        target_base["response_window_source_model_hashes"] = [
            _format_source_hash_record(record) for record in source_model_records
        ]
    if has_source_preprocessing or has_source_models:
        target_base["model_hash"] = [
            stable_hash(
                {
                    **dict(model_payload),
                    "response_window_source_preprocessing_hashes": source_preprocessing_records[row_index],
                    "response_window_source_model_hashes": source_model_records[row_index],
                }
            )
            for row_index in range(len(target_base))
        ]
    else:
        target_base["model_hash"] = stable_hash(dict(model_payload))


def _learn_weights(
    probability_cube: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    *,
    combine: str,
) -> tuple[np.ndarray, float]:
    best_weights = candidates[0]
    best_score = -np.inf
    for weights in candidates:
        score = _score_weights(probability_cube, labels, weights, combine=combine)
        if score > best_score:
            best_weights = weights
            best_score = score
    return best_weights, float(best_score)


def _response_window_rows(
    observations: pd.DataFrame,
    *,
    requested_times: tuple[float, ...],
    mode: str,
    combine: str,
    weight_grid_step: float,
    output_time: float | None,
) -> pd.DataFrame:
    if mode not in ENSEMBLE_MODE_CHOICES:
        raise ValueError(f"Unknown response-window ensemble mode '{mode}'.")
    if combine not in COMBINE_CHOICES:
        raise ValueError(f"Unknown response-window combine mode '{combine}'.")
    prob_columns = list(probability_columns(observations))
    times = _nearest_times(pd.to_numeric(observations["time"], errors="coerce").dropna().unique(), requested_times)
    selected = observations.loc[observations["time"].astype(float).isin(times)].copy()
    _check_single_plain_response_source(selected)
    key_columns = _sequence_key_columns(selected)
    metadata_columns = [
        column
        for column in selected.columns
        if column not in {*prob_columns, "time", "test_time", "window_start", "window_stop", "train_time"}
    ]
    time_labels = [_time_label(time) for time in times]
    candidates = _candidate_weights(len(times), weight_grid_step)
    subject_column = _subject_key_column(selected)
    target_subjects = selected[subject_column].dropna().astype(str).unique().tolist() if subject_column else [""]
    output_rows: list[pd.DataFrame] = []

    wide_probabilities = {}
    wide_provenance = {}
    wide_frames = {}
    identity_columns = _row_identity_columns(selected, prob_columns)
    protocol_columns = _source_protocol_columns(selected)
    for time in times:
        time_frame = selected.loc[selected["time"].astype(float) == float(time)].copy()
        _check_unique_response_keys(time_frame, key_columns, context=f"time {time}")
        time_frame = time_frame.sort_values(key_columns).drop_duplicates(key_columns, keep="first")
        indexed = time_frame.set_index(key_columns)
        wide_frames[time] = indexed
        wide_probabilities[time] = indexed[prob_columns]
        provenance_columns = [column for column in SOURCE_HASH_COLUMNS if column in indexed.columns]
        wide_provenance[time] = indexed.loc[:, provenance_columns] if provenance_columns else pd.DataFrame(index=indexed.index)
    common_index = None
    for frame in wide_probabilities.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    if common_index is None or len(common_index) == 0:
        raise ValueError("No observations contain all requested response-window times.")

    reference_time = times[0]
    base = wide_frames[reference_time].loc[common_index].copy()
    reference_identity = base.loc[:, [column for column in identity_columns if column in base.columns]]
    reference_protocol = base.loc[:, [column for column in protocol_columns if column in base.columns]]
    for time in times[1:]:
        _check_row_identity_consistency(
            reference_identity,
            wide_frames[time].loc[common_index],
            columns=identity_columns,
            context=f"time {time}",
        )
        _check_row_identity_consistency(
            reference_protocol,
            wide_frames[time].loc[common_index],
            columns=protocol_columns,
            context=f"time {time}",
        )
    labels = _integer_label_values(base["true_label"], n_classes=len(prob_columns))
    probability_cube = np.stack(
        [_normalize_rows(wide_probabilities[time].loc[common_index].to_numpy(dtype=float)) for time in times],
        axis=1,
    )
    subjects = _target_subject_values(base, key_columns)

    for target_subject in target_subjects:
        target_mask = subjects == str(target_subject)
        if not np.any(target_mask):
            continue
        if mode in {"uniform", "response_window_poststimulus_forward"}:
            weights = np.full(len(times), 1.0 / len(times), dtype=float)
            source_score = np.nan
        else:
            source_mask = ~target_mask
            if not np.any(source_mask):
                weights = np.full(len(times), 1.0 / len(times), dtype=float)
                source_score = np.nan
            else:
                weights, source_score = _learn_weights(
                    probability_cube[source_mask],
                    labels[source_mask],
                    candidates,
                    combine=combine,
                )

        probabilities = _combine_probabilities(probability_cube[target_mask], weights, combine=combine)
        target_index = common_index[target_mask]
        target_base = base.iloc[np.flatnonzero(target_mask)].reset_index()
        target_base = target_base[[column for column in metadata_columns if column in target_base.columns]].copy()
        predictions = probabilities.argmax(axis=1)
        for class_index, column in enumerate(prob_columns):
            target_base[column] = probabilities[:, class_index]
        class_names = []
        for class_index in range(len(prob_columns)):
            class_column = f"class_{class_index}"
            if class_column in target_base.columns and target_base[class_column].notna().any():
                class_names.append(str(target_base[class_column].dropna().iloc[0]))
            else:
                class_names.append(str(class_index))
        true_labels = _integer_label_values(target_base["true_label"], n_classes=len(prob_columns))
        target_base["predicted_label"] = predictions.astype(int)
        target_base["predicted_class"] = [class_names[int(label)] for label in predictions]
        target_base["probability_true_class"] = probabilities[np.arange(len(probabilities)), true_labels]
        target_base["confidence"] = probabilities.max(axis=1)
        target_base["is_correct"] = predictions == true_labels
        target_base["decoder"] = OUTPUT_DECODER
        target_base["emission_mode"] = f"{OUTPUT_EMISSION_MODE}_{mode}"
        target_base["time"] = float(np.mean(times) if output_time is None else output_time)
        target_base["test_time"] = target_base["time"]
        target_base["train_time"] = target_base["time"]
        target_base["window_start"] = float(min(times))
        target_base["window_stop"] = float(max(times))
        target_base["response_window_mode"] = mode
        target_base["response_window_requested_times"] = "|".join(_time_label(time) for time in requested_times)
        target_base["response_window_actual_times"] = "|".join(time_labels)
        target_base["response_window_weights"] = "|".join(f"{float(weight):.12g}" for weight in weights)
        target_base["response_window_combine"] = combine
        target_base["response_window_source_score"] = "" if not np.isfinite(source_score) else float(source_score)
        _apply_response_window_hashes(
            target_base,
            source_preprocessing_records=_source_hash_records(
                wide_provenance,
                times,
                target_index,
                column="preprocessing_hash",
                labeler=lambda source_time: _time_label(float(source_time)),
            ),
            source_model_records=_source_hash_records(
                wide_provenance,
                times,
                target_index,
                column="model_hash",
                labeler=lambda source_time: _time_label(float(source_time)),
            ),
            model_payload={
                "decoder": OUTPUT_DECODER,
                "mode": mode,
                "combine": combine,
                "requested_times": requested_times,
                "actual_times": times,
                "weights": tuple(float(weight) for weight in weights),
                "target_subject": target_subject,
            },
        )
        output_rows.append(target_base)

    if not output_rows:
        raise ValueError("No response-window ensemble rows were produced.")
    return pd.concat(output_rows, ignore_index=True)


def _decoder_source_oof_response_window_rows(
    observations: pd.DataFrame,
    *,
    requested_times: tuple[float, ...],
    combine: str,
    weight_grid_step: float,
    output_time: float | None,
) -> pd.DataFrame:
    if combine not in COMBINE_CHOICES:
        raise ValueError(f"Unknown response-window combine mode '{combine}'.")
    if "decoder" not in observations.columns:
        raise ValueError("Decoder-source OOF response-window ensembling requires a decoder column.")
    prob_columns = list(probability_columns(observations))
    times = _nearest_times(pd.to_numeric(observations["time"], errors="coerce").dropna().unique(), requested_times)
    selected = observations.loc[observations["time"].astype(float).isin(times)].copy()
    decoders = tuple(
        decoder
        for decoder in selected["decoder"].dropna().astype(str).drop_duplicates().tolist()
        if decoder.strip()
    )
    if len(decoders) < 2:
        raise ValueError("Decoder-source OOF response-window ensembling requires at least two decoder families.")

    key_columns = _sequence_key_columns(selected)
    metadata_columns = [
        column
        for column in selected.columns
        if column
        not in {
            *prob_columns,
            "time",
            "test_time",
            "window_start",
            "window_stop",
            "train_time",
        }
    ]
    time_labels = [_time_label(time) for time in times]
    time_weights = np.full(len(times), 1.0 / len(times), dtype=float)
    decoder_weight_candidates = _candidate_weights(len(decoders), weight_grid_step)

    wide_probabilities = {}
    wide_provenance = {}
    wide_frames = {}
    identity_columns = _row_identity_columns(selected, prob_columns)
    protocol_columns = _source_protocol_columns(selected)
    base_frame = None
    common_index = None
    for decoder in decoders:
        for time in times:
            time_frame = selected.loc[
                (selected["decoder"].astype(str) == decoder)
                & (selected["time"].astype(float) == float(time))
            ].copy()
            if time_frame.empty:
                raise ValueError(f"Missing observations for decoder {decoder!r} at response time {time}.")
            _check_unique_response_keys(time_frame, key_columns, context=f"decoder {decoder!r} time {time}")
            time_frame = time_frame.sort_values(key_columns).drop_duplicates(key_columns, keep="first")
            if base_frame is None:
                base_frame = time_frame
            indexed = time_frame.set_index(key_columns)
            wide_frames[(decoder, time)] = indexed
            wide_probabilities[(decoder, time)] = indexed[prob_columns]
            provenance_columns = [column for column in SOURCE_HASH_COLUMNS if column in indexed.columns]
            wide_provenance[(decoder, time)] = (
                indexed.loc[:, provenance_columns] if provenance_columns else pd.DataFrame(index=indexed.index)
            )
            common_index = indexed.index if common_index is None else common_index.intersection(indexed.index)
    if base_frame is None or common_index is None or len(common_index) == 0:
        raise ValueError("No observations contain all requested decoder/time response-window combinations.")

    base = base_frame.sort_values(key_columns).drop_duplicates(key_columns, keep="first").set_index(key_columns).loc[common_index]
    reference_key = (decoders[0], times[0])
    reference_identity = wide_frames[reference_key].loc[common_index, [column for column in identity_columns if column in wide_frames[reference_key].columns]]
    reference_protocol = wide_frames[reference_key].loc[common_index, [column for column in protocol_columns if column in wide_frames[reference_key].columns]]
    for decoder in decoders:
        for time in times:
            key = (decoder, time)
            if key == reference_key:
                continue
            _check_row_identity_consistency(
                reference_identity,
                wide_frames[key].loc[common_index],
                columns=identity_columns,
                context=f"decoder {decoder!r} time {time}",
            )
            _check_row_identity_consistency(
                reference_protocol,
                wide_frames[key].loc[common_index],
                columns=protocol_columns,
                context=f"decoder {decoder!r} time {time}",
            )
    labels = _integer_label_values(base["true_label"], n_classes=len(prob_columns))
    decoder_probabilities = []
    for decoder in decoders:
        time_cube = np.stack(
            [_normalize_rows(wide_probabilities[(decoder, time)].loc[common_index].to_numpy(dtype=float)) for time in times],
            axis=1,
        )
        decoder_probabilities.append(_combine_probabilities(time_cube, time_weights, combine=combine))
    probability_cube = np.stack(decoder_probabilities, axis=1)

    subject_column = _subject_key_column(selected)
    target_subjects = selected[subject_column].dropna().astype(str).unique().tolist() if subject_column else [""]
    subjects = _target_subject_values(base, key_columns)
    output_rows: list[pd.DataFrame] = []

    for target_subject in target_subjects:
        target_mask = subjects == str(target_subject)
        if not np.any(target_mask):
            continue
        source_mask = ~target_mask
        if not np.any(source_mask):
            decoder_weights = np.full(len(decoders), 1.0 / len(decoders), dtype=float)
            source_score = np.nan
        else:
            decoder_weights, source_score = _learn_weights(
                probability_cube[source_mask],
                labels[source_mask],
                decoder_weight_candidates,
                combine=combine,
            )

        probabilities = _combine_probabilities(probability_cube[target_mask], decoder_weights, combine=combine)
        target_index = common_index[target_mask]
        target_base = base.iloc[np.flatnonzero(target_mask)].reset_index()
        target_base = target_base[[column for column in metadata_columns if column in target_base.columns]].copy()
        predictions = probabilities.argmax(axis=1)
        for class_index, column in enumerate(prob_columns):
            target_base[column] = probabilities[:, class_index]
        class_names = []
        for class_index in range(len(prob_columns)):
            class_column = f"class_{class_index}"
            if class_column in target_base.columns and target_base[class_column].notna().any():
                class_names.append(str(target_base[class_column].dropna().iloc[0]))
            else:
                class_names.append(str(class_index))
        true_labels = _integer_label_values(target_base["true_label"], n_classes=len(prob_columns))
        target_base["predicted_label"] = predictions.astype(int)
        target_base["predicted_class"] = [class_names[int(label)] for label in predictions]
        target_base["probability_true_class"] = probabilities[np.arange(len(probabilities)), true_labels]
        target_base["confidence"] = probabilities.max(axis=1)
        target_base["is_correct"] = predictions == true_labels
        target_base["decoder"] = OUTPUT_DECODER
        target_base["emission_mode"] = f"{OUTPUT_EMISSION_MODE}_decoder_source_oof_nonnegative"
        target_base["time"] = float(np.mean(times) if output_time is None else output_time)
        target_base["test_time"] = target_base["time"]
        target_base["train_time"] = target_base["time"]
        target_base["window_start"] = float(min(times))
        target_base["window_stop"] = float(max(times))
        target_base["response_window_mode"] = "decoder_source_oof_nonnegative"
        target_base["response_window_requested_times"] = "|".join(_time_label(time) for time in requested_times)
        target_base["response_window_actual_times"] = "|".join(time_labels)
        target_base["response_window_weights"] = "|".join(f"{float(weight):.12g}" for weight in time_weights)
        target_base["response_window_combine"] = combine
        target_base["response_window_source_score"] = "" if not np.isfinite(source_score) else float(source_score)
        target_base["response_window_decoder_candidates"] = "|".join(decoders)
        target_base["response_window_decoder_weights"] = "|".join(f"{float(weight):.12g}" for weight in decoder_weights)
        source_keys = tuple((decoder, time) for decoder in decoders for time in times)
        _apply_response_window_hashes(
            target_base,
            source_preprocessing_records=_source_hash_records(
                wide_provenance,
                source_keys,
                target_index,
                column="preprocessing_hash",
                labeler=lambda source_key: f"{source_key[0]}@{_time_label(float(source_key[1]))}",
            ),
            source_model_records=_source_hash_records(
                wide_provenance,
                source_keys,
                target_index,
                column="model_hash",
                labeler=lambda source_key: f"{source_key[0]}@{_time_label(float(source_key[1]))}",
            ),
            model_payload={
                "decoder": OUTPUT_DECODER,
                "mode": "decoder_source_oof_nonnegative",
                "combine": combine,
                "requested_times": requested_times,
                "actual_times": times,
                "time_weights": tuple(float(weight) for weight in time_weights),
                "decoder_candidates": decoders,
                "decoder_weights": tuple(float(weight) for weight in decoder_weights),
                "target_subject": target_subject,
            },
        )
        output_rows.append(target_base)

    if not output_rows:
        raise ValueError("No decoder-source OOF response-window rows were produced.")
    return pd.concat(output_rows, ignore_index=True)


def run_response_window_ensemble(
    observation_csvs: list[Path],
    *,
    response_times: tuple[float, ...] = DEFAULT_RESPONSE_TIMES,
    mode: str = "uniform",
    combine: str = "log_probability_mean",
    weight_grid_step: float = 0.1,
    output_time: float | None = 0.184,
    smoothing_fit_window: tuple[float, float] = HYBRID_SMOOTHING_FIT_WINDOW,
    smoothing_apply_window: tuple[float, float] | None = None,
    smoothing_stay_grid_size: int = HYBRID_SMOOTHING_STAY_GRID_SIZE,
    ece_bins: int = 10,
    out_observations: Path | None = None,
    out_metrics: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "response_window_poststimulus_forward":
        if smoothing_apply_window is None:
            smoothing_apply_window = (min(response_times), max(response_times))
        observations, _ = smooth_probability_observations(
            observation_csvs,
            fit_window=tuple(float(value) for value in smoothing_fit_window),
            stay_grid_size=int(smoothing_stay_grid_size),
            mode="poststimulus_forward_only",
            apply_window=tuple(float(value) for value in smoothing_apply_window),
            emission_suffix=HYBRID_SMOOTHING_EMISSION_SUFFIX,
            ece_bins=ece_bins,
        )
    else:
        observations = read_probability_observations(observation_csvs).copy()
    if mode == "decoder_source_oof_nonnegative":
        ensembled = _decoder_source_oof_response_window_rows(
            observations,
            requested_times=tuple(float(time) for time in response_times),
            combine=combine,
            weight_grid_step=float(weight_grid_step),
            output_time=output_time,
        )
    else:
        ensembled = _response_window_rows(
            observations,
            requested_times=tuple(float(time) for time in response_times),
            mode=mode,
            combine=combine,
            weight_grid_step=float(weight_grid_step),
            output_time=output_time,
        )
    metrics = metrics_from_probability_observations(ensembled, ece_bins=ece_bins)
    metrics = _attach_response_window_provenance(metrics, ensembled)
    if out_observations is not None:
        out_observations.parent.mkdir(parents=True, exist_ok=True)
        ensembled.to_csv(out_observations, index=False)
    if out_metrics is not None:
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_metrics, index=False)
    return ensembled, metrics


def _attach_response_window_provenance(metrics: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Carry constant alignment/response-window provenance into metric rows."""

    if metrics.empty:
        return metrics
    enriched = metrics.copy()
    prefixes = ("alignment_", "response_window_")
    for column in observations.columns:
        if not column.startswith(prefixes) and column not in METRIC_PROVENANCE_COLUMNS:
            continue
        values = observations[column].drop_duplicates()
        if len(values) == 1:
            enriched[column] = values.iloc[0]
    return enriched


def _parse_times(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a post-stimulus response-window logit ensemble from probability observations.")
    parser.add_argument("observation_csv", nargs="+", type=Path)
    parser.add_argument("--mode", choices=ENSEMBLE_MODE_CHOICES, default="uniform")
    parser.add_argument("--combine", choices=COMBINE_CHOICES, default="log_probability_mean")
    parser.add_argument("--response-times", default=",".join(str(time) for time in DEFAULT_RESPONSE_TIMES))
    parser.add_argument("--output-time", type=float, default=0.184)
    parser.add_argument("--weight-grid-step", type=float, default=0.1)
    parser.add_argument("--smoothing-fit-window", nargs=2, type=float, default=HYBRID_SMOOTHING_FIT_WINDOW, metavar=("START", "STOP"))
    parser.add_argument(
        "--smoothing-apply-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help="Optional application window for response_window_poststimulus_forward; defaults to the requested response-time span.",
    )
    parser.add_argument("--smoothing-stay-grid-size", type=int, default=HYBRID_SMOOTHING_STAY_GRID_SIZE)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--out-observations", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    args = parser.parse_args(argv)

    _, metrics = run_response_window_ensemble(
        args.observation_csv,
        response_times=_parse_times(args.response_times),
        mode=args.mode,
        combine=args.combine,
        weight_grid_step=args.weight_grid_step,
        output_time=args.output_time,
        smoothing_fit_window=tuple(args.smoothing_fit_window),
        smoothing_apply_window=None if args.smoothing_apply_window is None else tuple(args.smoothing_apply_window),
        smoothing_stay_grid_size=args.smoothing_stay_grid_size,
        ece_bins=args.ece_bins,
        out_observations=args.out_observations,
        out_metrics=args.out_metrics,
    )
    print(f"Wrote response-window observations: {args.out_observations}")
    print(f"Wrote response-window metrics: {args.out_metrics}")
    if not metrics.empty:
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
