"""Στάδιο πραγματικού χρόνου: τίποτα από το μέλλον του τ στα χαρακτηριστικά."""
import numpy as np
import pandas as pd

from imbalance_forecast.realtime import FEATURES, build_panel


def _frame(days=3):
    idx = pd.date_range("2026-11-20 00:00", periods=96 * days, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
    rng = np.random.default_rng(0)
    m = pd.DataFrame(index=idx)
    for c in ["dam", "ida1", "ida2", "stability", "stab_isp2", "act_isp1", "act_isp2", "isp_thermal", "isp_hydro",
              "isp2_thermal", "isp2_hydro", "isp_renewables", "res_isp2", "dev_n", "res_dev_n", "dev", "bess",
              "spread", "imbalance"]:
        m[c] = rng.normal(100, 30, len(idx))
    pm = idx.tz_convert("Europe/Berlin").hour >= 12
    for c, v in [("ida3", 3.0), ("stab_isp3", 3.0), ("act_isp3", 3.0), ("isp3_thermal", 3.0), ("isp3_hydro", 0.0),
                 ("res_isp3", 3.0)]:
        m[c] = np.where(pm, v, np.nan)
    m["act_isp1"], m["act_isp2"] = 1.0, 2.0
    return m


def test_no_future_actuals():
    m = _frame()
    p1 = build_panel(m, lag_min=30, every=1)
    row = p1[(p1["tau"] == pd.Timestamp("2026-11-21 10:00", tz="Europe/Berlin")) & (p1["k"] == 4)]
    m2 = m.copy()
    after = m2.index >= pd.Timestamp("2026-11-21 09:30", tz="Europe/Berlin")  # s = τ - 45' -> 09:15 το τελευταίο
    for c in ("dev", "dev_n", "res_dev_n", "bess"):
        m2.loc[after, c] = 1e6
    p2 = build_panel(m2, lag_min=30, every=1)
    row2 = p2[(p2["tau"] == row["tau"].iloc[0]) & (p2["k"] == 4)]
    pd.testing.assert_frame_equal(row[FEATURES].reset_index(drop=True), row2[FEATURES].reset_index(drop=True))


def test_isp3_only_after_publication():
    p = build_panel(_frame(), lag_min=30, every=1)
    loc_tau = pd.DatetimeIndex(p["tau"]).tz_convert("Europe/Berlin")
    loc_t = pd.DatetimeIndex(p["t"]).tz_convert("Europe/Berlin")
    same_day = loc_tau.normalize() == loc_t.normalize()
    early = same_day & (loc_tau.hour < 11) & (loc_t.hour >= 12)
    late = same_day & (loc_tau.hour >= 12) & (loc_t.hour >= 12)
    assert (p.loc[early, "isp_act_t"] == 2.0).all()  # ISP2 πριν το ISP3
    assert (p.loc[late, "isp_act_t"] == 3.0).all()   # ISP3 μετά τις 11:30 CET
