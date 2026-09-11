"""Project a quarterback's fantasy points for his next start from his QBERT projection."""
import numpy as np, pandas as pd, statsmodels.api as sm

def X(d):
    return pd.DataFrame({
        "proj_qbert": d.proj_qbert - 80,
        "plays_form": d.plays_form - 38,
        "rush_form": d.rush_form - 0.09,
        "qbert_x_plays": (d.proj_qbert - 80) * (d.plays_form - 38) / 10,
        "rush_x_plays": (d.rush_form - 0.09) * (d.plays_form - 38),
        "def_elo": d.def_elo,                # defense he is about to face
        "home": d.home.fillna(0.5),
    }, index=d.index)

import sys
MODERN = "--modern" in sys.argv
d = pd.read_parquet("data/qbert_projected_modern.parquet" if MODERN else "data/qbert_projected.parquet")
d = d[d.plays >= 20]                         # actual starts only
tr, te = d[d.season <= 2020], d[d.season > 2020]

m = sm.OLS(tr.fantasy_points, sm.add_constant(X(tr))).fit()
print(m.params.round(3).to_string())

pred = m.predict(sm.add_constant(X(te), has_constant="add"))
rmse = lambda p: np.sqrt(np.mean((te.fantasy_points - p) ** 2))
mae = lambda p: np.mean(np.abs(te.fantasy_points - p))
base = te.fp_form                            # rolling fantasy PPG, the usual baseline
flat = np.full(len(te), tr.fantasy_points.mean())
print(f"\nholdout 2021-25, n={len(te)} QB starts")
for name, p in (("QBERT projection", pred), ("rolling fantasy PPG", base), ("league average", flat)):
    print(f"  {name:22s} RMSE {rmse(p):5.2f}   MAE {mae(p):5.2f}   corr {np.corrcoef(te.fantasy_points, p)[0,1]:.3f}")

d["proj_fp"] = m.predict(sm.add_constant(X(d), has_constant="add"))
d.to_parquet("data/qbert_fantasy_modern.parquet" if MODERN else "data/qbert_fantasy.parquet")

last = d.sort_values(["season", "week"]).groupby("player_display_name").last()
last = last[(last.season == 2025) & (last.plays_form > 25)]
print("\n=== Next-start fantasy projection, as of end of 2025 ===")
print(last.sort_values("proj_fp", ascending=False)[
    ["age", "form", "proj_qbert", "proj_fp"]].head(16).to_string(
    float_format=lambda v: f"{v:.1f}"))
