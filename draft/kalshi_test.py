"""Do Kalshi's 2025 prop prices know anything the consensus ranking doesn't?

`kalshi.py` pulls every pregame price for receptions, receiving yards, passing yards and
anytime touchdowns. Each stat is a ladder of "X or more" markets, so the prices are the
market's survival curve for that player-game, and the expected value is the area under
it. From those this builds a market projection:

    receptions + receiving yards / 10 + passing yards / 25 + 6 x P(touchdown)

That misses rushing yards, passing touchdowns and interceptions, so it is close to a
full PPR score for receivers and tight ends and incomplete for backs and quarterbacks.
The tests are built around that:

  1. are the prices sane: implied stat against the actual stat, touchdown calibration;
  2. does the market add to consensus: leave one week out, consensus alone against
     consensus plus the market's pieces, every position;
  3. can the market replace consensus where it prices nearly everything (WR/TE):
     within a week and position, which ranks players closer to how they scored.

Consensus is the weekly positional rank published before the slate, priced on a curve
fit to seasons before 2025. One season, so a screen for whether markets deserve a place
in the harness, not a harness verdict.

    .venv/bin/python draft/kalshi_test.py
"""
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "draft")
import props_test as P

SEASON = 2025
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
# Per-position folds drop columns a week doesn't vary; statsmodels warns on every fit.
warnings.filterwarnings("ignore", module="statsmodels")


def norm(s):
    s = s.str.lower().str.replace(r"[.'`]", "", regex=True)
    return s.str.replace(SUFFIX, "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def expected(strikes, probs, discrete):
    """Area under the market's survival curve.

    Prices on a ladder should fall as the bar rises; thin books don't always oblige, so
    they are forced non-increasing first, after averaging any rung listed twice. Below
    the lowest rung the curve runs to 1 at zero; above the highest it continues at its
    last slope down to 0, but never more than three rung-widths out. Without that cap two
    nearly equal top prices make the slope almost flat and the tail almost endless.
    """
    s = pd.Series(np.clip(np.asarray(probs, float), 0, 1)).groupby(
        np.asarray(strikes, float)).mean().sort_index()
    x = s.index.values + (0.5 if discrete else 0.0)
    p = np.minimum.accumulate(s.values)
    xs, ps = [0.0] + list(x), [1.0] + list(p)
    if len(x) >= 2 and p[-1] > 0:
        width = x[-1] - x[-2]
        slope = (p[-2] - p[-1]) / width
        step = min(p[-1] / slope, 3 * width) if slope > 0 else width
        xs.append(x[-1] + step)
        ps.append(0.0)
    return float(np.trapezoid(ps, xs))


def market_table():
    d = pd.read_parquet("data/kalshi_2025.parquet")
    d = d[d.p_yes.notna() & (d.player != "")]
    rows = []
    for (gid, wk, player, kind), g in d.groupby(["game_id", "week", "player", "kind"]):
        if kind == "td":
            v = float(g.p_yes.iloc[0])
        else:
            v = expected(g.strike.values, g.p_yes.values, discrete=(kind == "rec"))
        rows.append({"game_id": gid, "week": wk, "player": player, "kind": kind,
                     "value": v, "volume": g.volume.sum()})
    t = pd.DataFrame(rows).pivot_table(index=["game_id", "week", "player"],
                                       columns="kind", values="value").reset_index()
    for k in ("rec", "rec_yds", "pass_yds", "td"):
        if k not in t:
            t[k] = np.nan
    return t


def attach_ids(t):
    """Match names within the two teams in that game, from that week's rosters, so
    inactive players still match and a shared name on another team can't."""
    g = pd.read_csv("data/games.csv")
    g = g[g.season == SEASON][["game_id", "away_team", "home_team"]]
    ro = pd.read_parquet(f"data/roster_weekly_{SEASON}.parquet")
    ro = ro[ro.position.isin(["QB", "RB", "WR", "TE"])][["week", "team", "full_name",
                                                         "gsis_id", "position"]]
    ro["k"] = norm(ro.full_name)
    t = t.merge(g, on="game_id", how="left")
    t["k"] = norm(t.player)
    long = pd.concat([t.assign(team=t.away_team), t.assign(team=t.home_team)])
    m = long.merge(ro, on=["week", "team", "k"], how="inner").drop_duplicates(
        ["game_id", "player"])
    print(f"market player-games: {len(t)}, matched to ids: {len(m)} "
          f"({100 * len(m) / max(len(t), 1):.0f}%)")
    return m.rename(columns={"gsis_id": "player_id", "position": "pos_ro"})


def validity(d):
    st = pd.read_parquet(f"data/stats/w{SEASON}.parquet")
    st = st[st.season_type == "REG"]
    a = st.groupby(["player_id", "week"])[["receptions", "receiving_yards", "passing_yards",
                                           "rushing_tds", "receiving_tds"]].sum().reset_index()
    j = d.merge(a, on=["player_id", "week"], how="left").fillna(
        {"receptions": 0, "receiving_yards": 0, "passing_yards": 0,
         "rushing_tds": 0, "receiving_tds": 0})
    print("\n1. are the prices sane (implied against actual):")
    for k, col in (("rec", "receptions"), ("rec_yds", "receiving_yards"),
                   ("pass_yds", "passing_yards")):
        x = j[j[k].notna()]
        if len(x):
            print(f"  {k:9s} n={len(x):4d}  implied {x[k].mean():6.1f}  actual "
                  f"{x[col].mean():6.1f}  corr {x[k].corr(x[col]):.3f}")
    x = j[j.td.notna()].copy()
    x["scored"] = ((x.rushing_tds + x.receiving_tds) > 0).astype(float)
    for lo, hi in ((0, .15), (.15, .3), (.3, .45), (.45, 1.01)):
        b = x[(x.td >= lo) & (x.td < hi)]
        if len(b):
            print(f"  P(TD) {lo:.2f}-{min(hi, 1):.2f}: priced {b.td.mean():.2f}, "
                  f"scored {b.scored.mean():.2f}  (n={len(b)})")


def main():
    t = attach_ids(market_table())
    weeks = sorted(t.week.unique())
    cons = P.consensus_points(weeks)
    d = t.merge(cons, on=["player_id", "week"], how="inner")
    d = d.merge(P.actuals(weeks), on=["player_id", "week"], how="left")
    d["ppr"] = d.ppr.fillna(0.0)
    validity(d)

    for k in ("rec", "rec_yds", "pass_yds", "td"):
        d[f"has_{k}"] = d[k].notna().astype(float)
        d[k] = d[k].fillna(0.0)
    d["market"] = d.rec + d.rec_yds / 10 + d.pass_yds / 25 + 6 * d.td
    print(f"\nplayer-weeks with a market price and a consensus rank: {len(d)}, "
          f"weeks {weeks[0]}-{weeks[-1]}")
    print(d.groupby("pos").size().to_dict())

    print("\n2. does the market add to consensus (leave one week out):")
    base = ["cons"]
    full = base + ["rec", "rec_yds", "pass_yds", "td", "has_rec", "has_rec_yds",
                   "has_pass_yds", "has_td"]
    full = [c for c in full if d[c].std() > 0]
    for label, sub in (("all", d),) + tuple((p, d[d.pos == p]) for p in ("QB", "RB", "WR", "TE")):
        res = []
        for w in weeks:
            tr, te = sub[sub.week != w], sub[sub.week == w]
            if len(tr) < 40 or len(te) < 5:
                continue
            cols = [c for c in full if tr[c].std() > 0]
            pb, _ = P.fit_predict(tr, te, base)
            pf, _ = P.fit_predict(tr, te, cols)
            res.append((len(te), np.abs(te.ppr - pb).sum(), np.abs(te.ppr - pf).sum(),
                        te.ppr.values, pb.values, pf.values))
        if not res:
            continue
        n = sum(r[0] for r in res)
        y = np.concatenate([r[3] for r in res])
        b = np.concatenate([r[4] for r in res])
        f = np.concatenate([r[5] for r in res])
        print(f"  {label:4s} n={n:5d}  MAE {sum(r[1] for r in res) / n:5.2f} -> "
              f"{sum(r[2] for r in res) / n:5.2f}   corr {np.corrcoef(y, b)[0, 1]:.3f} -> "
              f"{np.corrcoef(y, f)[0, 1]:.3f}")

    print("\n3. market alone against consensus alone, ranking within week and position:")
    for p in ("WR", "TE", "RB", "QB"):
        s = d[d.pos == p]
        if len(s) < 30:
            continue
        rc = s.groupby("week").apply(lambda g: g.ppr.corr(g.cons, method="spearman"),
                                     include_groups=False)
        rm = s.groupby("week").apply(lambda g: g.ppr.corr(g.market, method="spearman"),
                                     include_groups=False)
        better = (rm > rc).sum()
        print(f"  {p}: consensus {rc.mean():.3f}  market {rm.mean():.3f}   market ahead in "
              f"{better} of {len(rc)} weeks  (n={len(s)})")
    print("\nOne season. A screen for whether markets earn a harness test, not a verdict.")


if __name__ == "__main__":
    main()
