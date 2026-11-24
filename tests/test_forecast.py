import numpy as np
import pandas as pd
import pytest

from imbalance_forecast import (
    Config, align, backtest, build_features, forecast, metrics, published_through, stability_report,
)
from imbalance_forecast.features import _orthodox_easter
from imbalance_forecast.synthetic import make_synthetic

FAST = dict(max_iter=40, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=50, random_state=0)


@pytest.mark.parametrize(
    "issue, expected",
    [
        ("2026-11-22", "2026-11-15"),  # Κυριακή -> δημοσίευση Πέμ. 19/11 -> έως Κυρ. 15/11
        ("2026-11-20", "2026-11-15"),  # Παρασκευή
        ("2026-11-19", "2026-11-08"),  # Πέμπτη, πριν τη δημοσίευση της ημέρας
        ("2026-11-18", "2026-11-08"),  # Τετάρτη
    ],
)
def test_published_through(issue, expected):
    assert published_through(pd.Timestamp(issue)) == pd.Timestamp(expected)


def test_published_through_same_day():
    assert published_through(pd.Timestamp("2026-11-19"), same_day=True) == pd.Timestamp("2026-11-15")


def test_orthodox_easter():
    assert _orthodox_easter(2025) == pd.Timestamp("2025-04-20")
    assert _orthodox_easter(2026) == pd.Timestamp("2026-04-12")


@pytest.fixture(scope="module")
def data():
    cfg = Config(gbm_params=FAST, train_window_days=120, min_train_days=40)
    return cfg, align(make_synthetic(start="2025-06-01", end="2025-11-01"), cfg)


def test_no_leakage(data):
    """Αλλάζοντας imbalance ΜΕΤΑ το cutoff, τα features δεν πρέπει να αλλάζουν."""
    cfg, df = data
    f1, feats, _ = build_features(df, cfg)
    target_day = pd.Timestamp("2025-10-21")
    cutoff = f1.loc[f1["date"] == target_day, "cutoff"].iloc[0]
    assert cutoff == pd.Timestamp("2025-10-12")

    df2 = df.copy()
    after = df2.index.tz_convert(cfg.tz).tz_localize(None) >= cutoff + pd.Timedelta(days=1)
    df2.loc[after, "imbalance"] = 1e6
    f2, _, _ = build_features(df2, cfg)
    m = f1["date"] == target_day
    pd.testing.assert_frame_equal(f1.loc[m, feats], f2.loc[m, feats])


def test_pre_dam_defaults(data):
    """Πριν τη DAM: καμία IDA, η ευστάθεια της D δεν είναι feature (μόνο η πρόβλεψή της)."""
    cfg, df = data
    _, feats, stab_feats = build_features(df, cfg)
    assert not any(c.startswith("ida") for c in feats)
    assert "stab_prob" in feats and "stab_flag" not in feats and "stab_mw" not in feats
    assert "x_stability" not in feats and "x_curt_flag" not in feats and "curt" not in feats
    assert "p_curt" in feats and "curt_given_stab_m28" in feats
    assert "imbalance" not in feats and "spread" not in feats
    assert "stab_prob" not in stab_feats and "stab_h_m7" in stab_feats


def test_ida_when_available(data):
    cfg, df = data
    _, feats, _ = build_features(df, Config(**{**cfg.__dict__, "ida_available": ("ida1",)}))
    assert "ida1" in feats and "ida2" not in feats


def test_stability_history_lag(data):
    """Η ευστάθεια (ISP) της ίδιας της D δεν πρέπει να επηρεάζει τα features της D."""
    cfg, df = data
    f1, feats, _ = build_features(df, cfg)
    day = pd.Timestamp("2025-10-21")
    df2 = df.copy()
    loc = df2.index.tz_convert(cfg.tz).tz_localize(None).normalize()
    df2.loc[loc >= day, "stability"] = 5000.0
    f2, _, _ = build_features(df2, cfg)
    m = f1["date"] == day
    pd.testing.assert_frame_equal(f1.loc[m, feats], f2.loc[m, feats])


def test_backtest_and_forecast(data):
    cfg, df = data
    res = backtest(df, cfg, "2025-09-15", "2025-10-15", verbose=False)
    m = metrics(res)
    assert {"model_q50", "bl_dam", "bl_hist28"} <= set(m.index) and "bl_ida" not in m.index
    assert np.isfinite(m.loc["model_q50", "MAE"])
    assert (res["imb_q10"] <= res["imb_q90"]).all()
    assert res["stab_prob"].between(0, 1).all()
    # σημειακή: = DAM εκτός αν p_curt >= κατώφλι, και ποτέ πάνω από τη DAM
    assert "model_point" in m.index and (res["imb_point"] <= res["dam"] + 1e-9).all()
    low = res["p_curt"] < cfg.point_curt_threshold
    assert np.allclose(res.loc[low, "imb_point"], res.loc[low, "dam"])
    rep = stability_report(res)
    assert rep is not None and rep["n_15min"].sum() > 0

    # issue Πέμ. 30/10 (πριν τη δημοσίευση) -> γνωστά έως Κυρ. 19/10
    out = forecast(df, cfg, "2025-10-31")
    assert len(out) == 96
    assert out.attrs["imbalance_known_through"] == "2025-10-19"
    assert out.attrs["stability_known_through"] == "2025-10-30"

    # αν τα δεδομένα σταματούν νωρίτερα, δεσμεύει η τελευταία πλήρης ημέρα
    df_live = df.copy()
    df_live.loc[df_live.index >= pd.Timestamp("2025-10-15", tz=cfg.tz), "imbalance"] = np.nan
    out = forecast(df_live, cfg, "2025-10-31")
    assert out.attrs["imbalance_known_through"] == "2025-10-14"


def _with_stage_cols(df, cfg):
    d = df.copy()
    d["stab_isp2"] = d["stability"] * 1.1
    d["res_isp2"] = d["res_forecast"] * 1.05
    d["load_isp2"] = d["load_forecast"]
    pm = d.index.tz_convert(cfg.tz).hour >= 12
    for c in ("stab", "res", "load"):
        d[f"{c}_isp3"] = d[f"{c}_isp2"].where(pm)
    for k in (1, 2, 3):
        d[f"act_isp{k}"], d[f"pup_isp{k}"], d[f"pdn_isp{k}"] = 50.0 * k, 200.0, 0.0
    d["act_isp3"] = d["act_isp3"].where(pm)
    for k, mw in (("ua1", 6000.0), ("ua2", 5800.0), ("ua3", 5500.0)):
        d[f"{k}_thermal"], d[f"{k}_hydro"], d[f"{k}_n_out"], d[f"{k}_n_forced"] = mw, 2000.0, 5.0, 1.0
    return d


def test_unit_availability_by_stage(data):
    """Διαθεσιμότητα μονάδων: πριν τη DAM μόνο ISP1· ISP3 μόνο στο στάδιο isp3."""
    from imbalance_forecast.config import stage_config

    cfg, df = data
    d = _with_stage_cols(df, cfg)
    f0, feats0, _ = build_features(d, stage_config("pre_dam", use_unit_avail=True))
    assert "ua_margin" in feats0 and "sd_ua1_thermal" in feats0 and "ua_rev" not in feats0
    assert not any(c.startswith(("sd_ua2", "sd_ua3", "x_ua")) for c in feats0)
    assert (f0["ua_avail"] == 8000).all()
    f3, feats3, _ = build_features(d, stage_config("isp3", use_unit_avail=True))
    assert (f3["ua_avail"] == 7500).all() and (f3["ua_rev"] == -500).all()
    _, feats_nu, _ = build_features(d, stage_config("isp3"))  # προεπιλογή: χωρίς
    assert not any("ua" in c.split("_")[0] or c.startswith("sd_ua") for c in feats_nu)


def test_stage_columns_never_leak_pre_dam(data):
    """Τα ISP2/ISP3 της ίδιας ημέρας δεν είναι γνωστά πριν τη DAM -> κανένα feature από αυτά."""
    cfg, df = data
    _, feats, _ = build_features(_with_stage_cols(df, cfg), cfg)
    assert not any("isp2" in c or "isp3" in c for c in feats)
    assert not any(c.startswith("sd_") for c in feats)


def test_stage_isp2_isp3(data):
    from imbalance_forecast.config import stage_config

    cfg, df = data
    d = _with_stage_cols(df, cfg)
    c2 = stage_config("isp2", gbm_params=FAST, train_window_days=120, min_train_days=40)
    assert c2.issue_lead_days == 0 and c2.ida_available == ("ida1", "ida2")
    f2, feats2, _ = build_features(d, c2)
    assert {"sd_stab_isp2", "sd_stability", "sd_res_rev_isp2", "ida2"} <= set(feats2)
    assert "sd_stab_isp3" not in feats2 and "ida3" not in feats2

    c3 = stage_config("isp3", gbm_params=FAST, train_window_days=120, min_train_days=40)
    f3, feats3, _ = build_features(d, c3)
    assert {"sd_stab_isp3", "ida3"} <= set(feats3)
    assert f3["hour"].min() == 12  # μόνο 12:00-24:00
    res = backtest(d, c3, "2025-10-01", "2025-10-20", verbose=False)
    assert len(res) and (res.index.tz_convert(cfg.tz).hour >= 12).all()
    assert res["p_spike_up"].between(0, 1).all()
    assert np.allclose(res["imb_point"], res["imb_q50"])  # ίδια ημέρα: σημειακή = διάμεσος


def test_isp_activation_by_stage(data):
    """Σχεδιασμένη ενεργοποίηση ISP: όχι πριν τη DAM· στο isp3 η πιο πρόσφατη (ISP3, αλλιώς ISP2)."""
    from imbalance_forecast.config import stage_config

    cfg, df = data
    d = _with_stage_cols(df, cfg)
    _, feats0, _ = build_features(d, stage_config("pre_dam"))
    assert not any("act" in c or "pup" in c or "pdn" in c for c in feats0)
    f1, feats1, _ = build_features(d, stage_config("isp1"))
    assert {"act_last", "act_up_x_stab", "sd_act_isp1"} <= set(feats1) and "sd_act_isp2" not in feats1
    assert (f1["act_last"] == 50).all()
    f3, _, _ = build_features(d, stage_config("isp3"))
    assert (f3["act_last"] == 150).all()  # 12-24: ISP3
    _, feats_no, _ = build_features(d, stage_config("isp2", use_isp_act=False))
    assert "act_last" not in feats_no
