import pandas as pd, numpy as np, sys
S = sys.argv[1]
MON = {1:"Ιαν",2:"Φεβ",3:"Μαρ",4:"Απρ",5:"Μαΐ",6:"Ιουν",7:"Ιουλ",8:"Αυγ",9:"Σεπ",10:"Οκτ",11:"Νοε",12:"Δεκ"}
def load(stage):
    r = pd.read_csv(f"{S}/bt_{stage}.csv", index_col=0); r.index = pd.to_datetime(r.index, utc=True)
    r = r[r.imbalance.notna()]
    r["p_up"] = 1 - r["p_imb_lt_dam"]
    r["m"] = r.index.tz_convert("Europe/Berlin").month
    return r
def strat(r, price, thr=0.5):
    d = r[r[price].notna()].copy()
    conf = np.maximum(d.p_up, 1 - d.p_up)
    pos = np.where(d.p_up > 0.5, 1, -1) * (conf >= thr)      # +1 long (αγορά στην αγορά, πώληση στο imbalance)
    d["pl"] = pos * (d.imbalance - d[price]); d["n"] = (pos != 0).astype(int)
    d["hit"] = np.where(pos != 0, (d.pl > 0).astype(float), np.nan)
    d["long_all"] = d.imbalance - d[price]
    d["perfect"] = (d.imbalance - d[price]).abs()
    return d
def monthly(d):
    g = d.groupby("m")
    t = pd.DataFrame({"Συναλλαγές": g.n.sum(), "P&L €": g.pl.sum(), "€/MWh": g.pl.sum() / g.n.sum(),
                      "Επιτυχία %": g.hit.mean() * 100, "Πάντα long €": g.long_all.sum(), "Τέλεια γνώση €": g.perfect.sum()})
    tot = pd.DataFrame({"Συναλλαγές": [d.n.sum()], "P&L €": [d.pl.sum()], "€/MWh": [d.pl.sum() / max(d.n.sum(), 1)],
                        "Επιτυχία %": [d.hit.mean() * 100], "Πάντα long €": [d.long_all.sum()], "Τέλεια γνώση €": [d.perfect.sum()]}, index=["Σύνολο"])
    t.index = [MON[i] for i in t.index]
    return pd.concat([t, tot])
pre, isp1 = load("pre_dam"), load("isp1")
print("pre_dam", pre.index.min(), pre.index.max(), len(pre))
out = {}
for name, r, price in [("A: imb−DAM (σήμα μετά τη DAM)", pre, "dam"),
                       ("B: imb−IDA1 (σήμα μετά τη DAM)", pre, "ida1"),
                       ("C: imb−IDA2 (σήμα μετά το ISP1)", isp1, "ida2")]:
    for thr in (0.5, 0.6, 0.7):
        t = monthly(strat(r, price, thr)); out[(name, thr)] = t
        print(f"\n=== {name} | κατώφλι {thr:.0%}"); print(t.round(1).to_string())
pd.to_pickle(out, f"{S}/pl_tables.pkl")
