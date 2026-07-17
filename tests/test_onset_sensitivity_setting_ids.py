from neureptrace.onset_sensitivity import build_sensitivity_settings


def test_setting_ids_preserve_compact_names_for_exact_grid_values():
    setting = build_sensitivity_settings(
        threshold_quantiles=(0.95,),
        min_duration_values=(0.002,),
    )[0]

    assert setting.setting_id == "point_q0950_c01_d0002ms_anypred"


def test_setting_ids_distinguish_quantiles_in_the_same_rounding_bucket():
    settings = build_sensitivity_settings(threshold_quantiles=(0.9504, 0.95049))
    setting_ids = [setting.setting_id for setting in settings]

    assert len(set(setting_ids)) == 2
    assert all(setting_id.startswith("point_q0950x") for setting_id in setting_ids)


def test_setting_ids_distinguish_durations_in_the_same_rounding_bucket():
    settings = build_sensitivity_settings(min_duration_values=(0.0014, 0.00149))
    setting_ids = [setting.setting_id for setting in settings]

    assert len(set(setting_ids)) == 2
    assert all("_d0001msx" in setting_id for setting_id in setting_ids)
