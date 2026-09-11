"""Benchmark the linear reconstruction against open-source ML models.

Two questions:
  1. Does a gradient-boosted model recover more of ESPN QBR than the published
     linear-regression recipe? (Silver reports ~75% of the variance explained.)
  2. Does it project fantasy points better than the QBERT-based regression?
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
import qbert as Q

def r2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)

d = pd.read_parquet("data/qb_games_full.parquet")
g = pd.read_csv("data/games.csv")[["season", "week", "home_team", "away_team", "temp", "wind", "roof"]]
wx = pd.concat([g.rename(columns={"home_team": "team"}), g.rename(columns={"away_team": "team"})])
d = d.merge(wx[["season", "week", "team", "temp", "wind", "roof"]].drop_duplicates(
    ["season", "week", "team"]), on=["season", "week", "team"], how="left")

tr, te = d[d.season <= 2020], d[d.season > 2020]
Xtr, Xte = Q.rates(tr), Q.rates(te)
wtr, wte = tr.plays.clip(lower=1), te.plays.clip(lower=1)

print("=== Reconstructing ESPN QBR (holdout 2021-25, play-weighted R2) ===")
print(f"{'published QBERT (Silver)':32s} 0.750  (reported, in-sample basis unknown)")

# 1. The published recipe: linear regression on box-score rates.
y = tr.qbr_total.clip(0.5, 99.5)
lin = sm.WLS(np.log(y / (100 - y)), sm.add_constant(Xtr), weights=wtr).fit()
p_lin = 100 / (1 + np.exp(-lin.predict(sm.add_constant(Xte, has_constant="add"))))
print(f"{'replica: WLS on logit(QBR)':32s} {r2(te.qbr_total, p_lin, wte):.3f}")

# 2. Ridge, as a regularized linear control.
rg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xtr, tr.qbr_total, sample_weight=wtr)
print(f"{'ridge regression':32s} {r2(te.qbr_total, rg.predict(Xte), wte):.3f}")

# 3. Gradient boosting - can nonlinearity and interactions find more signal?
gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6,
                                   random_state=0).fit(Xtr, tr.qbr_total, sample_weight=wtr)
print(f"{'gradient boosting':32s} {r2(te.qbr_total, gb.predict(Xte), wte):.3f}")

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=8, n_jobs=-1,
                           random_state=0).fit(Xtr, tr.qbr_total, sample_weight=wtr)
print(f"{'random forest':32s} {r2(te.qbr_total, rf.predict(Xte), wte):.3f}")

# 4. Blend of the linear fit and gradient boosting.
print(f"{'linear + GB average':32s} {r2(te.qbr_total, (p_lin + gb.predict(Xte)) / 2, wte):.3f}")


# ---------------------------------------------------------------------------
# The ML models do not beat the linear fit, so the shortfall is missing inputs,
# not model class. QBERT is deliberately restricted to statistics available back
# to 1950. A modern-era build is not: EPA exists from 1999. Does it close the gap?
def rates_plus(x):
    f = Q.rates(x); p = x.plays.clip(lower=1)
    f["pass_epa"] = x.passing_epa.fillna(0) / p
    f["rush_epa"] = x.rushing_epa.fillna(0) / p
    f["comeback"] = ((x.net_q3.fillna(0) < 0) & (x.result.fillna(0) > 0)).astype(float)
    f["ot"] = (x.result.fillna(0).abs() <= 3).astype(float)
    return f

print("\n=== Same target, plus modern-era inputs QBERT forgoes for 1950 coverage ===")
Ptr, Pte = rates_plus(tr), rates_plus(te)
y2 = tr.qbr_total.clip(0.5, 99.5)
lin2 = sm.WLS(np.log(y2 / (100 - y2)), sm.add_constant(Ptr), weights=wtr).fit()
p2 = 100 / (1 + np.exp(-lin2.predict(sm.add_constant(Pte, has_constant="add"))))
print(f"{'linear + EPA + clutch':32s} {r2(te.qbr_total, p2, wte):.3f}")
gb2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6,
                                    random_state=0).fit(Ptr, tr.qbr_total, sample_weight=wtr)
print(f"{'gradient boosting + EPA':32s} {r2(te.qbr_total, gb2.predict(Pte), wte):.3f}")
print(f"{'blend':32s} {r2(te.qbr_total, (p2 + gb2.predict(Pte)) / 2, wte):.3f}")


# ---------------------------------------------------------------------------
# Fantasy side: does any of this beat a rolling average at projecting next-start
# fantasy points? Same holdout, same features, several model classes.
print("\n=== Projecting next-start QB fantasy points (holdout 2021-25) ===")
fp = pd.read_parquet("data/qbert_fantasy.parquet")
fp = fp[fp.plays >= 20]
ftr, fte = fp[fp.season <= 2020], fp[fp.season > 2020]

def fx(x):
    return pd.DataFrame({"proj_qbert": x.proj_qbert, "form": x.form,
                         "plays_form": x.plays_form, "rush_form": x.rush_form,
                         "fp_form": x.fp_form, "def_elo": x.def_elo,
                         "home": x.home.fillna(0.5), "age": x.age}, index=x.index)

cands = {
    "QBERT regression": sm.OLS(ftr.fantasy_points, sm.add_constant(fx(ftr))).fit()
        .predict(sm.add_constant(fx(fte), has_constant="add")),
    "gradient boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
        max_depth=4, random_state=0).fit(fx(ftr), ftr.fantasy_points).predict(fx(fte)),
    "random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
        n_jobs=-1, random_state=0).fit(fx(ftr), ftr.fantasy_points).predict(fx(fte)),
    "rolling fantasy PPG": fte.fp_form,
    "season-to-date mean": np.full(len(fte), ftr.fantasy_points.mean()),
}
for name, p in cands.items():
    rmse = np.sqrt(np.mean((fte.fantasy_points - p) ** 2))
    mae = np.mean(np.abs(fte.fantasy_points - p))
    corr = np.corrcoef(fte.fantasy_points, p)[0, 1]
    print(f"  {name:22s} RMSE {rmse:5.2f}   MAE {mae:5.2f}   corr {corr:.3f}")
print(f"  n = {len(fte)} starts;  spread of actual points sd = {fte.fantasy_points.std():.2f}")
