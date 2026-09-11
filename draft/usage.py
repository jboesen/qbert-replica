"""Usage-based breakout detector for the waiver wire.

Consensus rest-of-season ranks are slow on role changes. A back-up who just took over
a backfield, or a receiver whose snap share jumped when the man ahead of him got hurt,
scores like a starter this week but still carries last month's rank. Box scores show
that a week before the rankings do, so this reads role directly: how much of the
offense a player has touched lately against his own earlier baseline, scaled by how
much his team actually runs, plus whether the player ahead of him on his own depth
chart is Out.

The signal has to be comparable with a consensus rank, so it is not a score: it is
fitted to the same quantity a rank stands for, mean PPR per week over the rest of the
season, counting weeks he does not play as zero. The fit uses only seasons strictly
before the one being decided.

    .venv/bin/python draft/usage.py          # fit and score the holdout seasons
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "draft")

POS = ["QB", "RB", "WR", "TE"]
WEEKS = 17
FIRST_FIT = 2015            # snap counts start here; earlier seasons add little
RECENT = 3                  # games in the "lately" window, about a month of football
MIN_GAMES = 2               # below this there is no baseline to compare against
OUT = {"Out", "Doubtful"}
FEATURES = ["opp_g", "d_opp", "snap", "d_snap", "tgt_share", "rz_g", "opp_share",
            "ppg", "vacancy"]


# ---------------------------------------------------------------- the panel

def snaps(y):
    """Snap share per player-week, keyed to nflverse ids.

    nflverse publishes snap counts against pro-football-reference ids, so they come
    across the dynastyprocess id bridge. A player who misses the map is simply left
    without a snap share rather than dropped, since his touches still count.
    """
    path = f"data/snaps_{y}.parquet"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player_id", "week", "snap_pct"])
    d = pd.read_parquet(path)
    d = d[(d.game_type == "REG") & d.position.isin(POS) & (d.week <= WEEKS)]
    ids = pd.read_csv("data/playerids.csv")
    ids = ids[ids.gsis_id.notna() & ids.pfr_id.notna()].drop_duplicates("pfr_id")
    m = ids.set_index("pfr_id").gsis_id
    d = d.assign(player_id=d.pfr_player_id.map(m), snap_pct=d.offense_pct)
    return d[d.player_id.notna()][["player_id", "week", "snap_pct"]]


def panel(y):
    """Every skill player-game of season y with the usage behind the points."""
    st = pd.read_parquet(f"data/stats/w{y}.parquet")
    st = st[(st.season_type == "REG") & (st.week <= WEEKS) & st.position.isin(POS)]
    d = pd.DataFrame({
        "player_id": st.player_id, "pos": st.position, "team": st.team, "week": st.week,
        "ppr": st.fantasy_points_ppr.fillna(0.0),
        "targets": st.targets.fillna(0.0), "carries": st.carries.fillna(0.0),
        "tgt_share": st.target_share.fillna(0.0),
    })
    o = pd.read_parquet("data/pbp_opps.parquet")
    o = o[(o.season == y) & (o.week <= WEEKS)]
    o = o.assign(rz=o.rz_tgt + o.gl_tgt + o.rz_car + o.gl_car,
                 team_vol=o.team_pass + o.team_rush)
    d = d.merge(o[["player_id", "week", "rz", "team_vol"]], on=["player_id", "week"],
                how="left")
    d = d.merge(snaps(y), on=["player_id", "week"], how="left")
    d["opp"] = d.targets + d.carries
    # A missing play-by-play row means no red-zone work, not unknown work. Team volume
    # is the one thing worth a league-average fill, since it only scales the share.
    d["rz"] = d.rz.fillna(0.0)
    d["team_vol"] = d.team_vol.fillna(d.team_vol.median())
    return d.sort_values(["player_id", "week"])


def vacancies(y, upto):
    """Players whose position-mate with the most work so far is Out this week.

    The injury report for week w is published before week w is played, so this is
    pregame information. "The man ahead of him" is read off usage rather than a depth
    chart: the team-mate at his position with the most opportunities to date.
    """
    path = f"data/injuries_{y}.parquet"
    if not os.path.exists(path):
        return set()
    inj = pd.read_parquet(path)
    hurt = set(inj[(inj.week == upto) & inj.report_status.isin(OUT)].gsis_id)
    return hurt


def features(y, d, upto):
    """One row per player for a decision made before week `upto`, from weeks before it."""
    past = d[d.week < upto]
    if not len(past):
        return pd.DataFrame(columns=["player_id", "pos"] + FEATURES)
    g = past.groupby("player_id")
    recent = past.groupby("player_id").tail(RECENT).groupby("player_id")
    f = pd.DataFrame({
        "pos": g.pos.last(), "team": g.team.last(), "n": g.size(),
        "opp_g": recent.opp.mean(), "base_opp": g.opp.mean(),
        "snap": recent.snap_pct.mean(), "base_snap": g.snap_pct.mean(),
        "tgt_share": recent.tgt_share.mean(), "rz_g": recent.rz.mean(),
        "team_vol": recent.team_vol.mean(), "ppg": g.ppr.mean(),
    })
    f = f[f.n >= MIN_GAMES].copy()
    # The break-out case is the jump, not the level, so both go in and the fit decides.
    f["d_opp"] = f.opp_g - f.base_opp
    f["snap"] = f.snap.fillna(0.0)
    f["d_snap"] = f.snap - f.base_snap.fillna(0.0)
    f["opp_share"] = f.opp_g / f.team_vol.replace(0, np.nan)
    f["opp_share"] = f.opp_share.fillna(0.0)

    hurt = vacancies(y, upto)
    lead = past[past.player_id.isin(f.index)].groupby(["team", "pos", "player_id"]).opp.sum()
    ahead = {}
    for (team, pos), grp in lead.groupby(level=[0, 1]):
        ranked = grp.sort_values(ascending=False)
        ids = [i[2] for i in ranked.index]
        # Anyone below the leader inherits work when the leader is out.
        if ids and ids[0] in hurt:
            for p in ids[1:]:
                ahead[p] = 1.0
    f["vacancy"] = pd.Series(ahead).reindex(f.index).fillna(0.0)
    return f.reset_index()[["player_id", "pos"] + FEATURES]


def rest_of_season(d, upto):
    """Mean PPR per week from `upto` to the end, weeks he did not play counting zero.

    This is what a rest-of-season rank is a statement about, so fitting to it is what
    puts the signal and the consensus rank on one scale.
    """
    pts = d.pivot_table(index="player_id", columns="week", values="ppr",
                        aggfunc="sum").reindex(columns=range(1, WEEKS + 1)).fillna(0.0)
    left = pts.loc[:, upto:]
    return left.mean(axis=1)


def rows(y):
    """Fitting rows for one season: features at each decision point and what followed."""
    d = panel(y)
    out = []
    for w in range(2, WEEKS + 1):
        f = features(y, d, w)
        if not len(f):
            continue
        f["target"] = f.player_id.map(rest_of_season(d, w)).fillna(0.0)
        f["week"], f["season"] = w, y
        out.append(f)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- fit and apply

def fit_before(y, first=FIRST_FIT):
    """Per position, rest-of-season ppg on usage, fit only on seasons before y."""
    t = pd.concat([rows(s) for s in range(first, y)], ignore_index=True)
    fits = {}
    for p in POS:
        x = t[t.pos == p]
        fits[p] = sm.OLS(x.target, sm.add_constant(x[FEATURES])).fit()
    return fits


def predict(fits, f):
    out = pd.Series(np.nan, index=f.index)
    for p in POS:
        x = f[f.pos == p]
        if len(x):
            out.loc[x.index] = fits[p].predict(
                sm.add_constant(x[FEATURES], has_constant="add")).clip(lower=0)
    return out


def season_values(y, fits):
    """Rest-of-season ppg for every player at every decision point of season y.

    Returned as (player_id, week, value) where week is the upcoming week, so the row
    uses only games played strictly before it.
    """
    d = panel(y)
    out = []
    for w in range(2, WEEKS + 1):
        f = features(y, d, w)
        if not len(f):
            continue
        f["value"] = predict(fits, f)
        f["week"] = w
        out.append(f[["player_id", "week", "value"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["player_id", "week", "value"])


def main():
    for y in range(2021, 2026):
        fits = fit_before(y)
        d = panel(y)
        v = season_values(y, fits)
        truth = pd.concat([rest_of_season(d, w).rename("t").reset_index().assign(week=w)
                           for w in range(2, WEEKS + 1)], ignore_index=True)
        j = v.merge(truth, on=["player_id", "week"], how="left")
        print(f"{y}: n={len(j)}  corr {j.value.corr(j.t):.3f}  "
              f"MAE {np.mean(np.abs(j.value - j.t)):.2f}  "
              f"mean pred {j.value.mean():.2f} vs actual {j.t.mean():.2f}")
        for p in POS:
            print(f"   {p}: " + "  ".join(f"{n} {c:+.2f}"
                                          for n, c in fits[p].params.items()))


if __name__ == "__main__":
    main()
