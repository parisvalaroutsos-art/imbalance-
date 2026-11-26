import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd
from imbalance_forecast.config import stage_config
from imbalance_forecast.data import load_data
from imbalance_forecast.pipeline import backtest
stage = sys.argv[1]
cfg = stage_config(stage, retrain_every_weeks=4)
df = load_data(["data/prices.csv", "data/admie.csv"], cfg)
res = backtest(df, cfg, "2026-01-01", verbose=True)
for c in ("ida1", "ida2", "ida3"):
    if c in df: res[c] = df[c].reindex(res.index)
res.to_csv(f"{sys.argv[2]}/bt_{stage}.csv")
print("done", stage, len(res), res.index.min(), res.index.max())
