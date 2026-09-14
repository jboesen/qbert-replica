"""Do prop prices carry anything the consensus ranking doesn't?

The narrow question `props.py` exists to answer. Prop coverage is too thin to rebuild a
fantasy score (about ten players a game, no receptions market, four weeks of one
season), so this does not ask props to beat consensus on their own. It asks whether
adding them to consensus predicts a player's week better than consensus alone.

Both sides are pregame. Consensus is the weekly positional rank published before the
slate, priced through the same rank-to-points curve the harness uses, fit here on
seasons before 2025 so the season being scored never touches it. Prop prices are read
at least an hour before kickoff.

Scored by leave-one-week-out: fit on three weeks, predict the fourth, rotate. With four
weeks and one season this cannot carry a preregistered verdict, and is not presented as
one. It is a screen: if the market adds nothing here, the avenue is closed cheaply.

    .venv/bin/python draft/props_test.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, "draft")
import consensus as C

SEASON = 2025
# A prop line sits near the middle of the outcome, so read the price as a normal whose
# spread scales with the line. The constant only has to be about right: it sets how far
# a price away from even money moves the implied yardage, not the ranking.
SPREAD = 0.55


def implied_yards(line, p_over):
    """Market-implied mean yards from one line and the price of going over it."""
    p = np.clip(p_over, 0.02, 0.98)
    return line + SPREAD * line * norm.ppf(p)


def load():
    d = pd.read_parquet("data/props_2025.parquet")
    ids = pd.read_csv("data/playerids.csv")
    ids = ids[ids.gsis_id.notna() & ids.name.notna()].drop_duplicates("name")
    key = lambda s: s.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    ids["k"] = key(ids.name)
    d["k"] = key(d.player)
    d = d.merge(ids[["k", "gsis_id"]], on="k", how="left")
    miss = d.gsis_id.isna().mean()
    print(f"props: {len(d)} rows, {d.player.nunique()} players, "
          f"{100 * miss:.0f}% unmatched to ids")
    d = d[d.gsis_id.notna()].rename(columns={"gsis_id": "player_id"})

    # One row per player-week: the pieces the market priced.
    f = d.pivot_table(index=["player_id", "week"], columns="kind",
                      values=["line", "p_over"], aggfunc="first")
    f.columns = [f"{b}_{a}" for a, b in f.columns]
    f = f.reset_index()
    for kind in ("rec_yds", "rush_yds", "pass_yds"):
        lo, po = f.get(f"{kind}_line"), f.get(f"{kind}_p_over")
        f[kind] = implied_yards(lo, po) if lo is not None else np.nan
    f["p_td"] = f.get("td_p_over", np.nan)
    return f


def consensus_points(weeks):
    """The weekly consensus rank, priced on a curve fit to seasons before 2025."""
    crv = C.weekly_curve(range(2020, SEASON))
    w = pd.read_parquet("data/ecr_weekly.parquet")
    w = w[(w.season == SEASON) & w.week.isin(weeks)].copy()
    w["rank"] = w.groupby(["week", "pos"]).ecr.rank(method="first")
    w["cons"] = C.rank_points(crv, w.pos, w["rank"])
    return w[["player_id", "week", "pos", "cons"]]


def actuals(weeks):
    st = pd.read_parquet(f"data/stats/w{SEASON}.parquet")
    st = st[(st.season_type == "REG") & st.week.isin(weeks)]
    return st.groupby(["player_id", "week"]).fantasy_points_ppr.sum().rename("ppr").reset_index()


def fit_predict(tr, te, cols):
    m = sm.OLS(tr.ppr, sm.add_constant(tr[cols])).fit()
    return m.predict(sm.add_constant(te[cols], has_constant="add")), m


def main():
    f = load()
    weeks = sorted(f.week.unique())
    d = f.merge(consensus_points(weeks), on=["player_id", "week"], how="inner")
    d = d.merge(actuals(weeks), on=["player_id", "week"], how="left")
    d["ppr"] = d.ppr.fillna(0.0)                 # priced but did not play: a real zero
    for c in ("rec_yds", "rush_yds", "pass_yds", "p_td"):
        d[c] = d[c].fillna(0.0)
    print(f"\nplayer-weeks with both a prop price and a consensus rank: {len(d)}")
    print(d.groupby("week").size().rename("rows").to_frame().T.to_string())

    # A market that never listed (no passing-yards props in this window) is a column of
    # zeros, which says nothing and makes the fit singular, so it is left out.
    base = ["cons"]
    full = base + [c for c in ("rec_yds", "rush_yds", "pass_yds", "p_td") if d[c].std() > 0]
    rows = []
    for w in weeks:                               # leave one week out
        tr, te = d[d.week != w], d[d.week == w]
        if len(tr) < 30 or not len(te):
            continue
        pb, _ = fit_predict(tr, te, base)
        pf, _ = fit_predict(tr, te, full)
        rows.append({"week": w, "n": len(te),
                     "mae_cons": np.mean(np.abs(te.ppr - pb)),
                     "mae_both": np.mean(np.abs(te.ppr - pf)),
                     "r_cons": np.corrcoef(te.ppr, pb)[0, 1],
                     "r_both": np.corrcoef(te.ppr, pf)[0, 1]})
    r = pd.DataFrame(rows)
    print("\nleave-one-week-out, consensus alone vs consensus + market:")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\npooled MAE {r.mae_cons.mean():.2f} -> {r.mae_both.mean():.2f}   "
          f"corr {r.r_cons.mean():.3f} -> {r.r_both.mean():.3f}")

    _, m = fit_predict(d, d, full)
    print("\nin-sample coefficients (does the market add on top of consensus?):")
    for name in full:
        print(f"  {name:10s} {m.params[name]:+7.3f}   t {m.tvalues[name]:+5.2f}   "
              f"p {m.pvalues[name]:.3f}")
    print(f"  R2 {m.rsquared:.3f} vs consensus alone "
          f"{sm.OLS(d.ppr, sm.add_constant(d[base])).fit().rsquared:.3f}")
    print("\nFour weeks of one season, about ten players a game, no receptions market.\n"
          "A screen, not a verdict: it cannot pass the harness rule and is not claimed to.")


if __name__ == "__main__":
    main()
