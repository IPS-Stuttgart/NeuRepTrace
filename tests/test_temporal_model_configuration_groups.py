from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.temporal_model import fit_temporal_models


def _configuration_rows(
    feature_preprocessor: str,
    pca_components: float,
    sequence_ids: range,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence_id in sequence_ids:
        for time, p0 in ((0.1, 0.85), (0.2, 0.80)):
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "feature_preprocessor": feature_preprocessor,
                    "pca_components": pca_components,
                    "time": time,
                    "sequence_id": sequence_id,
                    "sample_index": sequence_id,
                    "class_0": "left",
                    "class_1": "right",
                    "prob_class_0": p0,
                    "prob_class_1": 1.0 - p0,
                }
            )
    return rows


def test_temporal_models_keep_decoder_configurations_separate(tmp_path: Path):
    observations = pd.DataFrame(
        [
            *_configuration_rows("raw", np.nan, range(2)),
            *_configuration_rows("source_pca", 8.0, range(10, 13)),
        ]
    )
    csv_path = tmp_path / "observations.csv"
    states_path = tmp_path / "states.csv"
    observations.to_csv(csv_path, index=False)

    summary, states = fit_temporal_models(
        [csv_path],
        effect_window=(0.1, 0.2),
        baseline_window=None,
        n_permutations=0,
        stay_grid_size=10,
        out_states=states_path,
    )

    observed = summary.loc[summary["condition"] == "observed_effect"].sort_values("feature_preprocessor")
    assert observed["feature_preprocessor"].tolist() == ["raw", "source_pca"]
    assert observed["n_sequences"].tolist() == [2, 3]
    assert observed.loc[observed["feature_preprocessor"] == "raw", "pca_components"].item() == "nan"

    assert states is not None
    assert states_path.exists()
    assert set(states["feature_preprocessor"]) == {"raw", "source_pca"}
    assert states.groupby("feature_preprocessor", dropna=False).size().to_dict() == {"raw": 4, "source_pca": 6}
    assert states.loc[states["feature_preprocessor"] == "raw", "pca_components"].isna().all()
