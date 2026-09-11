"""Depth-chart role adjustment for the season projections.

The hierarchical projection knows how good a player has been, not whether he'll be on
the field. A backup quarterback with a good history projects like a starter; a receiver
who fell to fourth on the chart projects off last year's role. The week-1 depth chart
(draft/depth_role.py) answers that, so the projection is re-weighted by role:

    games = a_role + b_role * projected games           (per position)
    ppg   = c_role + d_role * projected points per game (per position, weighted by games)

Kept as two parts because they mean different things downstream: a backup quarterback
scores fine when he plays but rarely plays, and the trade simulator needs to know which.
Fit on training seasons only.

    .venv/bin/python draft/role.py     # holdout check, 2021-25 against a <=2020 fit
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

POSITIONS = ["QB", "RB", "WR", "TE"]
FIRST = 2012            # first season nflverse has depth charts; earlier rows are left be


def attach(s):
    """Add the player's week-1 depth bucket: d1, d2, d3 (third or lower), or none."""
    role = pd.read_parquet("data/depth_role.parquet")[["player_id", "season", "depth"]]
    s = s.drop(columns=[c for c in ("depth", "role") if c in s]).merge(
        role, on=["player_id", "season"], how="left")
    s["role"] = np.select([s.depth == 1, s.depth == 2, s.depth >= 3],
                          ["d1", "d2", "d3"], "none")
    # A rookie's projection is the population prior alone, fit on veterans, and it
    # undershoots the rookies who win jobs; they get their own coefficients per role.
    s["rookie"] = np.where(s.own_ppg.isna(), "rookie", "vet")
    return s


def fit(s, train_mask):
    t = s[train_mask & (s.season >= FIRST) & s.proj_points.notna() & s.games.notna()
          & (s.games > 0)]
    fits = {}
    g = "C(role):C(rookie)"
    for p in POSITIONS:
        x = t[t.position == p]
        fits[p] = (
            smf.ols(f"games ~ 0 + {g} + proj_games:{g}", x).fit(),
            smf.wls(f"ppr_pg ~ 0 + {g} + proj_ppg:{g}", x, weights=x.games).fit(),
        )
    return fits


def apply(s, fits):
    """Role-adjusted proj_games, proj_ppg and proj_points; the originals kept as *_base."""
    s = s.copy()
    for c in ("proj_games", "proj_ppg", "proj_points"):
        s[f"{c}_base"] = s[c]
    for p, (g, q) in fits.items():
        x = s[(s.position == p) & s.proj_points.notna() & (s.season >= FIRST)]
        if not len(x):
            continue
        s.loc[x.index, "proj_games"] = g.predict(x).clip(0.5, x.season_games)
        s.loc[x.index, "proj_ppg"] = q.predict(x).clip(lower=0)
    s["proj_points"] = s.proj_ppg * s.proj_games
    return s


if __name__ == "__main__":
    s = pd.read_parquet("data/draft_projected.parquet")
    for c in ("proj_games", "proj_ppg", "proj_points"):   # start from the pre-role numbers
        if f"{c}_base" in s:
            s[c] = s[f"{c}_base"]
    s = attach(s)
    s = s[s.season >= FIRST]
    out = apply(s, fit(s, s.season <= 2020))
    te = out[(out.season > 2020) & out.own_ppg.notna() & out.ppr.notna()]
    N = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
    pool = pd.concat([g[g.proj_points.rank(ascending=False).le(N[p])
                        | g.proj_points_base.rank(ascending=False).le(N[p])]
                      for (_, p), g in te.groupby(["season", "position"])])
    for tag, d in (("all player-seasons", te), ("fantasy-relevant pool", pool)):
        print(f"\nholdout 2021-25, {tag}: base -> role-adjusted")
        for p, x in d.groupby("position"):
            e0, e1 = x.ppr - x.proj_points_base, x.ppr - x.proj_points
            print(f"  {p}  n={len(x):4d}  MAE {e0.abs().mean():5.1f} -> {e1.abs().mean():5.1f}   "
                  f"RMSE {np.sqrt((e0 ** 2).mean()):5.1f} -> {np.sqrt((e1 ** 2).mean()):5.1f}   "
                  f"r {x.ppr.corr(x.proj_points_base):.3f} -> {x.ppr.corr(x.proj_points):.3f}")
