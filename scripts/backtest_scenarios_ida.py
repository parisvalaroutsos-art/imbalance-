import pandas as pd, numpy as np, sys
S=sys.argv[1]; Q=2.5
r=pd.read_csv(f"{S}/bt_isp1.csv",index_col=0); r.index=pd.to_datetime(r.index,utc=True)
r=r[r.imbalance.notna()&r.dam.notna()].copy()
loc=r.index.tz_convert("Europe/Berlin"); r["m"]=loc.month; r["day"]=loc.normalize()
print("n",len(r),"ida1 missing",r.ida1.isna().sum(),"ida2 missing",r.ida2.isna().sum())
dam,i1,i2,imb=r.dam,r.ida1,r.ida2,r.imbalance
pu=1-r.p_imb_lt_dam; spk=r.p_spike_up
can2=i2.notna()
def long_close(cond):   # long στη DAM· αν cond (και υπάρχει IDA2) πουλάω στην IDA2, αλλιώς κλείνει στο imbalance
    c=cond&can2; return Q*np.where(c, i2-dam, imb-dam), c
def short_cover(cond):  # short στη DAM· αν cond αγοράζω πίσω στην IDA2, αλλιώς στο imbalance
    c=cond&can2; return Q*np.where(c, dam-i2, dam-imb), c
S_={}
S_["1. Long στη DAM, κράτα στο imbalance"]=(Q*(imb-dam),pd.Series(False,index=r.index))
S_["2. Short στη DAM, κράτα στο imbalance"]=(Q*(dam-imb),pd.Series(False,index=r.index))
S_["3. Long· αν IDA1<DAM πούλα στην IDA2 (το σενάριό σου)"]=long_close(i1<dam)
S_["4. Short· αν IDA1>DAM κάλυψε στην IDA2 (ανάποδο)"]=short_cover(i1>dam)
S_["5. Long· πούλα στην IDA2 αν το μοντέλο ISP1 δεν δίνει ↑≥60%"]=long_close(pu<0.6)
S_["6. Short· κάλυψε στην IDA2 αν το μοντέλο ISP1 δεν δίνει ↓≥60% ή σπάικ≥5%"]=short_cover(~((pu<=0.4)&(spk<0.05)))
S_["7. Long· πούλα στην IDA2 αν IDA1<DAM ΚΑΙ το μοντέλο δεν δίνει ↑≥60%"]=long_close((i1<dam)&(pu<0.6))
S_["8. Long· πούλα στην IDA2 αν IDA1<DAM Ή το μοντέλο δίνει ↓≥60%"]=long_close((i1<dam)|(pu<=0.4))
rows=[];mon={}
for k,(pl,c) in S_.items():
    pl=pd.Series(pl,index=r.index); dd=pl.groupby(r.day).sum(); cum=dd.cumsum()
    mm=pl.groupby(r.m).sum(); mon[k]=mm
    rows.append(dict(σενάριο=k,PL=pl.sum(),eur_MWh=pl.sum()/(Q*len(pl)),κλεισίματα_IDA2=f"{c.mean()*100:.0f}%",
        χειρ_μέρα=dd.min(),χειρ_μήνας=mm.min(),κερδ_μήνες=f"{(mm>0).sum()}/{len(mm)}",maxDD=(cum-cum.cummax()).min()))
t=pd.DataFrame(rows); pd.set_option("display.width",250); pd.set_option("display.max_colwidth",70)
print(t.round(0).to_string(index=False))
M=pd.DataFrame(mon).round(0); print(M.to_string())
pd.to_pickle((t,M),f"{S}/scen.pkl")
# έλεγχος σήματος: IDA1-DAM vs imb-DAM
x=pd.DataFrame({"s1":np.sign(i1-dam),"sp":imb-dam}).dropna()
print("P(imb>DAM | IDA1>DAM)=%.2f  P(imb>DAM | IDA1<DAM)=%.2f"%((x.sp[x.s1>0]>0).mean(),(x.sp[x.s1<0]>0).mean()))
print("mean(imb-DAM | IDA1>DAM)=%.1f  | IDA1<DAM=%.1f ; mean(IDA2-DAM|IDA1<DAM)=%.1f"%(x.sp[x.s1>0].mean(),x.sp[x.s1<0].mean(),(i2-dam)[i1<dam].mean()))
