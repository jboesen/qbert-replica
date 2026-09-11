"""Build the full QBERT-replica ratings table and print leaderboards."""
import numpy as np, pandas as pd, statsmodels.api as sm
import qbert as Q

d = pd.read_parquet("data/qb_games_full.parquet")
g = pd.read_csv("data/games.csv")[["season", "week", "home_team", "away_team", "temp", "wind", "roof"]]
wx = pd.concat([g.rename(columns={"home_team": "team"})[["season","week","team","temp","wind","roof"]],
                g.rename(columns={"away_team": "team"})[["season","week","team","temp","wind","roof"]]])
wx["team"] = wx.team.replace({"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"})
d = d.merge(wx.drop_duplicates(["season","week","team"]), on=["season","week","team"], how="left")

import sys
MODERN = "--modern" in sys.argv
feats = Q.rates_modern if MODERN else Q.rates
m = Q.fit_raw(d[d.season <= 2020], feats=feats)
d["qbr_hat"] = Q.predict_qbr(m, d, feats=feats)
print("variant:", "modern (1999+, EPA allowed)" if MODERN else "historical (QBERT feature set)")
d = Q.to_scale(Q.adjust(d))

# Regressing points-added on the rating attenuates the slope - the rating is
# measured with error - and undershoots elite seasons by roughly a factor of three.
# Calibrate on the structural claim instead: after a replacement-level baseline,
# quarterbacks take about 30% of the remaining wins.
REPL_WIN_PCT = 0.20                       # a team of replacement players
QB_SHARE = 0.30
games_per_season = d.groupby("season").apply(
    lambda g: g[g.week <= 18].shape[0] / 2, include_groups=False)
league_war = (games_per_season * (1 - REPL_WIN_PCT) * QB_SHARE).mean()

raw_par = (d.qbert - Q.REPL) * d.plays
scale = league_war / (raw_par.groupby(d.season).sum() / Q.PTS_PER_WIN).mean()
d["war"] = raw_par * scale / Q.PTS_PER_WIN
d["paa"] = (d.qbert - Q.AVG) * d.plays * scale / Q.PTS_PER_WIN
print(f"league QB WAR per season = {league_war:.1f}; scale = {scale:.4f}")
d.to_parquet("data/qbert_games_modern.parquet" if MODERN else "data/qbert_games.parquet")

season = d.groupby(["player_display_name", "season"]).agg(
    plays=("plays", "sum"), qbert=("qbert", lambda s: np.average(s, weights=d.loc[s.index, "plays"])),
    war=("war", "sum"), g=("week", "count")).reset_index()
season = season[season.plays >= 250]
season.to_parquet("data/qbert_seasons_modern.parquet" if MODERN else "data/qbert_seasons.parquet")

print("\n=== Top 15 QB seasons since 2006, by QBERT-replica rating (min 250 plays) ===")
print(season.sort_values("qbert", ascending=False).head(15).to_string(index=False,
      float_format=lambda v: f"{v:.1f}"))
print("\n=== 2025 season ===")
print(season[season.season == 2025].sort_values("qbert", ascending=False).head(12).to_string(
      index=False, float_format=lambda v: f"{v:.1f}"))
