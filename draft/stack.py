"""Combine our season projection with expert consensus.

Neither source wins alone: in the league backtest our board loses title odds to
consensus, and consensus is only a rank. Forecasts that are wrong for different reasons
average out each other's errors (preseason ADP and early-season results each correlate
about .6 with the rest of the season, their average about .68). So rather than choosing
one, we weight them. Per position,

    actual season points ~ a * our projection + b * consensus-implied points + c

fit on 2020-22 (consensus history starts in 2020) and applied unchanged afterwards.
Consensus arrives as a positional rank; it becomes points through a curve fit on the
same seasons: what players at each preseason rank went on to score, smoothed and made
non-increasing. Players who never played that season count, at zero, since a draft
has to price the busts as well.

    .venv/bin/python draft/stack.py     # fit on 2020-22, check on 2023-25
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "draft")
import board as B

POS = ["QB", "RB", "WR", "TE"]
FIT = (2020, 2021, 2022)
MAX_RANK = 150
# The draftable pool per position, where the weights are fit and scored: anyone either
# source ranks inside it. Beyond it the question is moot.
POOL = {"QB": 30, "RB": 70, "WR": 80, "TE": 30}


def actual(y):
    d = pd.read_parquet(f"data/stats/w{y}.parquet")
    d = d[(d.season_type == "REG") & (d.week <= 17)]
    return d.groupby("player_id").fantasy_points_ppr.sum()


def model_board(y):
    p = B.project_upcoming(y, rookies=True)
    p = p[(p.own_w > 4) | ((p.exp == 0) & p.role.isin(["d1", "d2"]))]
    return p.drop_duplicates("player_id").set_index("player_id")


def ecr_rank(y):
    e = pd.read_parquet("data/ecr_preseason.parquet")
    e = e[e.season == y].drop_duplicates("player_id").copy()
    e["rank"] = e.groupby("pos").ecr.rank(method="first")
    return e.set_index("player_id")[["pos", "rank"]]


def table(y, board=None):
    """Every player either source rates, with both forecasts and what he scored."""
    board = model_board(y) if board is None else board
    e = ecr_rank(y)
    ids = board.index.union(e.index)
    t = pd.DataFrame(index=ids)
    t["pos"] = board.position.reindex(ids).fillna(e.pos.reindex(ids))
    t = t[t.pos.isin(POS)].copy()
    t["model"] = board.proj_points.reindex(t.index)
    worst = e.groupby("pos")["rank"].max()
    t["rank"] = e["rank"].reindex(t.index).fillna(t.pos.map(worst) + 1)
    t["actual"] = actual(y).reindex(t.index).fillna(0.0) if y < 2026 else np.nan
    t["season"] = y
    return t


def curve(tables):
    t = pd.concat(tables)
    out = {}
    for p in POS:
        x = t[t.pos == p]
        by = x.groupby(x["rank"].clip(upper=MAX_RANK)).actual.mean()
        by = by.reindex(range(1, MAX_RANK + 1)).interpolate().bfill().ffill()
        sm_ = by.rolling(7, center=True, min_periods=1).mean()
        out[p] = np.minimum.accumulate(sm_.values)
    return out


def ecr_points(crv, pos, rank):
    r = np.clip(np.asarray(rank, float), 1, MAX_RANK).astype(int) - 1
    return np.array([crv[p][i] for p, i in zip(pos, r)])


def in_pool(t):
    m = t.pos.map(POOL)
    model_rank = t.groupby(["season", "pos"]).model.rank(ascending=False)
    return (t["rank"] <= m) | (model_rank <= m)


def fit(years=FIT, tables=None):
    tables = tables or [table(y) for y in years]
    crv = curve(tables)
    t = pd.concat(tables)
    t["ecr_pts"] = ecr_points(crv, t.pos, t["rank"])
    t["model"] = t.model.fillna(t.ecr_pts)
    t = t[in_pool(t)]
    coefs = {}
    for p in POS:
        x = t[t.pos == p]
        coefs[p] = sm.OLS(x.actual, sm.add_constant(x[["model", "ecr_pts"]])).fit().params
    return crv, coefs


_TABLES = {}


def cached_table(y):
    if y not in _TABLES:
        _TABLES[y] = table(y)
    return _TABLES[y]


def fit_before(y):
    """Curve and weights from the seasons before y only (2020 through y-1)."""
    return fit(tables=[cached_table(s) for s in range(FIT[0], y)])


def apply(t, crv, coefs):
    """Stacked season points for every row of a table()."""
    e = pd.Series(ecr_points(crv, t.pos, t["rank"]), index=t.index)
    model = t.model.fillna(e)
    out = pd.Series(0.0, index=t.index)
    for p, c in coefs.items():
        m = t.pos == p
        out[m] = c["const"] + c["model"] * model[m] + c["ecr_pts"] * e[m]
    return out.clip(lower=0)


if __name__ == "__main__":
    crv, coefs = fit()
    print("weights, fit on 2020-22 (actual ~ const + a*model + b*consensus points):")
    for p, c in coefs.items():
        print(f"  {p}: const {c['const']:+6.1f}   model {c['model']:.2f}   consensus {c['ecr_pts']:.2f}")
    te = pd.concat([table(y) for y in (2023, 2024, 2025)])
    te["ecr_pts"] = ecr_points(crv, te.pos, te["rank"])
    te["stack"] = apply(te, crv, coefs)
    te["model_f"] = te.model.fillna(te.ecr_pts)
    te = te[in_pool(te)]
    print("\nchecked on 2023-25, draftable pool, busts included (MAE / corr):")
    for p, x in te.groupby("pos"):
        cells = "   ".join(f"{n} {np.mean(np.abs(x.actual - x[c])):5.1f} / {x.actual.corr(x[c]):.3f}"
                           for n, c in (("model", "model_f"), ("consensus", "ecr_pts"),
                                        ("stacked", "stack")))
        print(f"  {p} n={len(x):3d}   {cells}")
