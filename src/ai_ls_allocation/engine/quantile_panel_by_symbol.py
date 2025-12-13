from pathlib import Path
import pandas as pd, statsmodels.api as sm

DATA = Path("data/features/panel.parquet")
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)
TRAIN_END = pd.Timestamp("2017-12-31")
QUANTILES=[0.1,0.5,0.9]
FEATS=["vol20","vol60","ma20","ma60","trend_ma_diff","macd","macd_signal","macd_hist","spread_hyg_ief"]

def main():
    panel = pd.read_parquet(DATA).reset_index().dropna(subset=["r1"]+FEATS)
    panel["date"]=pd.to_datetime(panel["date"])
    train = panel[panel["date"]<=TRAIN_END].copy()
    test  = panel[panel["date"]> TRAIN_END].copy()

    Xtr = sm.add_constant(train[FEATS]); ytr=train["r1"]
    models = {q: sm.QuantReg(ytr,Xtr).fit(q=q) for q in QUANTILES}

    Xte = sm.add_constant(test[FEATS], has_constant="add")
    for q,res in models.items():
        test[f"q{int(q*100)}"] = res.predict(Xte)

    out=[]
    for (sym), g in test.groupby("symbol"):
        for q in QUANTILES:
            col=f"q{int(q*100)}"
            cov=float((g["r1"]<=g[col]).mean())
            out.append(dict(symbol=sym, quantile=q, coverage=cov, target=q, n=len(g)))
    pd.DataFrame(out).to_csv(REPORTS/"quantile_panel_by_symbol.csv", index=False)

if __name__=="__main__":
    main()
