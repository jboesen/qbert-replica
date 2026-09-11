"""Triangulate the replica's QB WAR against an independent open-source construction.

The reference is the nflWAR approach (Yurko, Ventura & Horowitz, JQAS 2019): value a
quarterback by his expected points added over a replacement-level baseline, then divide
by points per win. Built here from public EPA rather than from the QBR reconstruction,
so it shares no inputs with the QBERT replica beyond the underlying games.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PTS_PER_WIN = 35.0

d = pd.read_parquet("data/qbert_games.parquet")
d["epa_total"] = d.passing_epa.fillna(0) + d.rushing_epa.fillna(0)

# Replacement level, nflWAR style: the pooled per-play rate of the bottom tier of QBs.
season_plays = d.groupby(["player_id", "season"]).plays.sum().rename("season_plays")
d = d.join(season_plays, on=["player_id", "season"])
repl_pool = d[d.season_plays < 150]                 # backups and spot starters
repl_rate = repl_pool.epa_total.sum() / repl_pool.plays.sum()
print(f"replacement-level EPA/play = {repl_rate:.3f}  (n={len(repl_pool)} games)")

d["epa_war"] = (d.epa_total - repl_rate * d.plays) / PTS_PER_WIN

s = d.groupby(["player_display_name", "season"]).agg(
    plays=("plays", "sum"), epa_war=("epa_war", "sum"), qbert_war=("war", "sum"),
    qbert=("qbert", lambda x: np.average(x, weights=d.loc[x.index, "plays"]))).reset_index()
s = s[s.plays >= 250]

r = spearmanr(s.qbert_war, s.epa_war)
print(f"\nQBERT-replica WAR vs nflWAR-style EPA WAR: Spearman {r.statistic:.3f} "
      f"(n={len(s)} seasons)")
print(f"Pearson {np.corrcoef(s.qbert_war, s.epa_war)[0, 1]:.3f}")

print("\n=== Top 10 seasons by the independent EPA-based WAR ===")
top = s.sort_values("epa_war", ascending=False).head(10).reset_index(drop=True)
top.index += 1
print(top[["player_display_name", "season", "epa_war", "qbert_war", "qbert"]].to_string(
    float_format=lambda v: f"{v:.1f}"))

for name, yr, claim in [("Tom Brady", 2007, "published: #1 season by WAR, all time"),
                        ("Lamar Jackson", 2024, "published: 10th all time by WAR")]:
    row = s[(s.player_display_name == name) & (s.season == yr)]
    if len(row):
        er = int((s.epa_war > row.epa_war.iloc[0]).sum()) + 1
        qr = int((s.qbert_war > row.qbert_war.iloc[0]).sum()) + 1
        print(f"{name} {yr}: EPA-WAR rank {er}, replica QBERT-WAR rank {qr}   ({claim})")
