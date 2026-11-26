import pandas as pd, numpy as np, sys
S=sys.argv[1]; stages=sys.argv[2:]
def load(st):
    r=pd.read_csv(f"{S}/bt_{st}.csv",index_col=0); r.index=pd.to_datetime(r.index,utc=True)
    r=r[r.imbalance.notna()].copy(); loc=r.index.tz_convert("Europe/Berlin")
    r["h"]=loc.hour; r["m"]=loc.month; r["day"]=loc.normalize(); r["p_up"]=1-r.p_imb_lt_dam; return r
def risk(pl, day, m):
    dd=pl.groupby(day).sum(); cum=dd.cumsum(); mdd=(cum-cum.cummax()).min()
    mm=pl.groupby(m).sum()
    return dict(PL=pl.sum(), n=(pl!=0).sum(), eur_mwh=pl.sum()/max((pl!=0).sum(),1), worst_day=dd.min(), worst_month=mm.min(),
                pos_months=f"{(mm>0).sum()}/{len(mm)}", maxDD=mdd, days_loss_gt5k=(dd<-5000).sum())
rows=[]
for st,price in [(s,{"pre_dam":"ida1","isp1":"ida2","isp2":"ida3"}[s]) for s in stages]:
    r=load(st); r=r[r[price].notna()]
    sp=r.imbalance-r[price]; conf=np.maximum(r.p_up,1-r.p_up); up=r.p_up>0.5
    # στατική κλίση ανά ώρα, walk-forward: πρόσημο μέσου spread της ώρας τις προηγούμενες 90 ημέρες γνωστού imbalance (με 10 ημ. καθυστέρηση)
    for lab,pos in [
        ("όλα",np.where(up,1,-1)),
        ("≥60%",np.where(up,1,-1)*(conf>=.6)),
        ("≥60% μόνο long",np.where(up&(conf>=.6),1,0)),
        ("≥60% μόνο short",np.where(~up&(conf>=.6),-1,0)),
        ("≥60%, short μόνο αν σπάικ<5%",np.where(up,1,np.where(r.p_spike_up<0.05,-1,0))*(conf>=.6)),
        ("≥70%",np.where(up,1,-1)*(conf>=.7)),
    ]:
        pl=pd.Series(pos*sp.values,index=r.index)
        rows.append(dict(strategy=f"{st}->{price}",rule=lab,**risk(pl,r.day,r.m)))
    # baselines
    for lab,pos in [("πάντα long",1),("πάντα short",-1)]:
        pl=pos*sp; rows.append(dict(strategy=f"{st}->{price}",rule=lab,**risk(pl,r.day,r.m)))
t=pd.DataFrame(rows); pd.set_option("display.width",250); print(t.round(1).to_string())
t.to_pickle(f"{S}/sup_{'_'.join(stages)}.pkl")
