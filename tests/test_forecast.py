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
