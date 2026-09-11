"""Check the replica against every published QBERT figure it can be compared to.

Only one published number is used to build the model - Rodgers' 2011 rating, which
fixes the spread of the scale in qbert.to_scale. Everything below is therefore a test,
not a fit.
"""
import numpy as np, pandas as pd
import anchors as A

d = pd.read_parquet("data/qbert_games.parquet")
s = pd.read_parquet("data/qbert_seasons.parquet")
players = pd.read_parquet("data/players.parquet")[["gsis_id", "draft_pick"]].rename(
    columns={"gsis_id": "player_id"})

print("=" * 72)
print("USED TO FIT: Rodgers 2011 = 108.6  ->  replica",
      f"{s[(s.player_display_name == 'Aaron Rodgers') & (s.season == 2011)].qbert.iloc[0]:.1f}")
print("=" * 72)

# 1. Replacement level. Silver defines it as an undrafted rookie's first start and
#    puts it at 68. Nothing in the build targets that number.
g = d.merge(players, on="player_id", how="left")
g["start_no"] = g.sort_values(["season", "week"]).groupby("player_id").cumcount()
und = g[(g.draft_pick.isna()) & (g.start_no == 0) & (g.plays >= 20)]
print(f"\n1. Replacement level: undrafted QBs' first starts (n={len(und)}) average "
      f"{np.average(und.qbert, weights=und.plays):.1f}   published 68")

# 2. League average must sit at 80 by construction of the centering; confirm.
print(f"2. League average across all starts: {np.average(d.qbert, weights=d.plays):.1f}"
      "   published 80")

# 3. Brady 2007 is the top season by WAR, all time.
w = s.sort_values("war", ascending=False).reset_index(drop=True); w.index += 1
br = w[(w.player_display_name == "Tom Brady") & (w.season == 2007)]
print(f"3. Brady 2007 WAR rank in 2006-2025: {br.index[0]}   published #1 all time")

# 4. Lamar Jackson 2024 ranks 10th all time in seasonal WAR.
lj = w[(w.player_display_name == "Lamar Jackson") & (w.season == 2024)]
print(f"4. Lamar 2024 WAR rank in 2006-2025: {lj.index[0]}   published 10th all time")

# 5. Preseason 2025 projections.
f = pd.read_parquet("data/qbert_fantasy.parquet")
pre = f[f.season <= 2024].sort_values(["season", "week"]).groupby("player_display_name").last()
print("\n5. Preseason 2025 projected ratings   [NOT a clean test: fantasy.py's")
print("   constant offset was fit on these three, so read the leave-one-out line]")
errs = []
for name, pub in A.PROJECTED_2025.items():
    rep = pre.loc[name].proj_qbert
    errs.append(rep - pub)
    print(f"   {name:18s} published {pub:6.1f}   replica {rep:6.1f}   {rep - pub:+.1f}")
pub_order = list(A.PROJECTED_2025.values())
rep_order = [pre.loc[n].proj_qbert for n in A.PROJECTED_2025]
same = list(np.argsort(pub_order)) == list(np.argsort(rep_order))
print(f"   mean error {np.mean(errs):+.1f}, mean absolute error {np.mean(np.abs(errs)):.1f}, "
      f"rank order {'matches' if same else 'differs'}")
raw = np.array(rep_order) - 4.4                    # undo the offset
loo = [pub_order[i] - (raw[i] + np.mean(np.delete(np.array(pub_order) - raw, i)))
       for i in range(3)]
print(f"   leave-one-out error of the offset: {np.round(loo, 1)} "
      f"(mean abs {np.mean(np.abs(loo)):.1f}); uncorrected it is "
      f"{np.mean(np.abs(np.array(pub_order) - raw)):.1f}")

# 6. Super Bowl LX single-game ratings.
print("\n6. Super Bowl LX game ratings")
sb = d[(d.season == 2025) & (d.week == 22)]
for name, pub in [("Sam Darnold", 82.3), ("Drake Maye", 75.9)]:
    r = sb[sb.player_display_name == name]
    if len(r):
        print(f"   {name:18s} published {pub:6.1f}   replica {r.qbert.iloc[0]:6.1f}   "
              f"{r.qbert.iloc[0] - pub:+.1f}   (raw ESPN QBR was {r.qbr_total.iloc[0]:.1f})")

# 7. Career WAR. Only partial windows are available, so this bounds rather than matches.
print("\n7. Career WAR (replica covers 2006+ only, so these are partial)")
for name, pub in A.CAREER_WAR.items():
    x = d[d.player_display_name == name]
    print(f"   {name:18s} published {pub:6.1f} full career   replica {x.war.sum():6.1f} "
          f"over {x.season.min()}-{x.season.max()}")

# 8. 2025 season standings.
print("\n8. 2025 season (published: Maye is MVP; Darnold 14th projected, 8th by WAR)")
s25 = s[s.season == 2025].sort_values("qbert", ascending=False).reset_index(drop=True)
s25.index += 1
w25 = s[s.season == 2025].sort_values("war", ascending=False).reset_index(drop=True)
w25.index += 1
for who in ("Drake Maye", "Sam Darnold"):
    rr = s25.index[s25.player_display_name == who]
    wr = w25.index[w25.player_display_name == who]
    if len(rr):
        print(f"   {who:18s} rating rank {rr[0]:2d}   WAR rank {wr[0]:2d}")
