"""Does the trade evaluator predict which side of a trade actually came out ahead?

For each holdout season, twelve rosters are drafted from the preseason board, then a few
hundred random trades between them are valued three ways at week 1:

  - lineup-marginal: the simulator in trade.py (change in each side's expected lineup
    points over weeks 1-17, byes and injuries included);
  - naive points: projected season points received minus points sent;
  - naive VBD: the same with value over replacement instead of points.

Each trade is then played out on what really happened. Every week both rosters, with and
without the trade, start their best lineup by projection among players who actually
played, and score what those starters actually scored. The question is which valuation
tracks the realized change.

    .venv/bin/python draft/trade_backtest.py
"""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
from vbd import LEAGUE, add_vbd
from backtest import run_draft
import trade as T

SLOTS, FLEX = LEAGUE["starters"], list(LEAGUE["flex"])
TRADES_PER_SEASON = 400


def actual_weeks(season):
    """Player x week actual PPR; NaN where he didn't play (bye, injury, inactive)."""
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/stats/w*.parquet"))])
    d = d[(d.season == season) & (d.season_type == "REG") & (d.week <= T.LAST_WEEK)]
    pts = d.pivot_table(index="player_id", columns="week", values="fantasy_points_ppr",
                        aggfunc="sum")
    team = d.groupby("player_id").team.agg(lambda t: t.mode().iloc[0])
    return pts, team


def realized(roster, v, pts, waiver):
    """Season points from lineups set on projections among players who played."""
    total = 0.0
    for w in pts.columns:
        played = [x for x in roster if x in pts.index and not np.isnan(pts.at[x, w])]
        used, week = set(), 0.0
        for p, k in SLOTS.items():
            c = sorted([x for x in played if v.at[x, "position"] == p],
                       key=lambda x: -v.at[x, "ppg"])[:k]
            used.update(c)
            week += sum(pts.at[x, w] for x in c) + waiver[p] * (k - len(c))
        rest = [x for x in played if x not in used and v.at[x, "position"] in FLEX]
        rest = sorted(rest, key=lambda x: -v.at[x, "ppg"])
        week += pts.at[rest[0], w] if rest else waiver["WR"]
        total += week
    return total


def random_trade(rng, rosters, sim):
    a, b = rng.choice(len(rosters), 2, replace=False)
    mine, theirs = rosters[a], rosters[b]
    ra, rb = T.relevant(sim, mine, 12), T.relevant(sim, theirs, 12)
    ng, nr = [(1, 1), (1, 1), (2, 1), (1, 2)][rng.integers(4)]
    if len(ra) < ng or len(rb) < nr:
        return None
    return a, b, list(rng.choice(ra, ng, replace=False)), list(rng.choice(rb, nr, replace=False))


def main():
    s = pd.read_parquet("data/draft_projected.parquet")
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2021, 2026):
        y = s[(s.season == season) & s.proj_points.notna()].set_index("player_id")
        pts, team = actual_weeks(season)
        board, _, _ = add_vbd(y)
        prev = s[s.season == season - 1].set_index("player_id").ppr
        board["market_rank"] = board.index.to_series().map(prev).fillna(0).rank(ascending=False)

        v = board[["player_display_name", "position", "vbd", "proj_points"]].copy()
        v["ppg"] = board.proj_ppg
        v["avail"] = (board.proj_games / 17).clip(0.3, 0.98)
        v["team"] = team.reindex(v.index)
        v["status"] = None
        v = v[v.team.notna()]                     # never played that season: no byes to set
        sim = T.Season(v, season, sims=500, seed=season, start=1, live=False)

        drafted = run_draft(board.loc[board.index.isin(v.index)],
                            ["vbd", "market", "dp"] * 4)
        rosters = [list(r) for r in drafted.values()]
        base_real = {i: realized(r, v, pts, sim.waiver) for i, r in enumerate(rosters)}

        done = 0
        while done < TRADES_PER_SEASON:
            t = random_trade(rng, rosters, sim)
            if t is None:
                continue
            a, b, give, get = t
            r = T.evaluate(sim, rosters[a], rosters[b], give, get, exact=False)
            after_a, _ = T.apply(sim, rosters[a], give, get, exact=False)
            after_b, _ = T.apply(sim, rosters[b], get, give, exact=False)
            rows.append({
                "season": season,
                "model": r["me"]["pts"][0] + r["me"]["playoff_pts"][0],
                "model_them": r["them"]["pts"][0] + r["them"]["playoff_pts"][0],
                "naive": v.loc[get, "proj_points"].sum() - v.loc[give, "proj_points"].sum(),
                "vbd": v.loc[get, "vbd"].sum() - v.loc[give, "vbd"].sum(),
                "real": realized(after_a, v, pts, sim.waiver) - base_real[a],
                "real_them": realized(after_b, v, pts, sim.waiver) - base_real[b],
                "shape": f"{len(give)}-for-{len(get)}",
            })
            done += 1
        print(f"{season}: {done} trades", flush=True)

    d = pd.DataFrame(rows)
    d.to_parquet("data/trade_backtest.parquet")
    print(f"\n=== {len(d)} random trades, holdout 2021-25: predicted vs realized change "
          f"in season lineup points for the side receiving ===")
    print(f"{'valuation':16s} {'corr':>6s} {'right side':>11s}   by shape (corr)")
    for m in ("model", "naive", "vbd"):
        by = "  ".join(f"{k} {g[m].corr(g.real):.2f}" for k, g in d.groupby("shape"))
        print(f"{m:16s} {d[m].corr(d.real):6.3f} {100 * (np.sign(d[m]) == np.sign(d.real)).mean():10.1f}%   {by}")
    boot = []
    for _ in range(1000):
        x = d.sample(len(d), replace=True, random_state=rng.integers(1 << 31))
        boot.append(x.model.corr(x.real) - x.naive.corr(x.real))
    print(f"\nmodel minus naive, corr: {np.mean(boot):+.3f} "
          f"(95% interval {np.percentile(boot, 2.5):+.3f} to {np.percentile(boot, 97.5):+.3f})")
    both = d[(d.model > 0) & (d.model_them > 0)]
    print(f"trades the model called good for both sides: {len(both)}; both actually "
          f"gained in {100 * ((both.real > 0) & (both.real_them > 0)).mean():.0f}% "
          f"(all trades: {100 * ((d.real > 0) & (d.real_them > 0)).mean():.0f}%)")


if __name__ == "__main__":
    main()
