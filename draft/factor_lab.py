"""Rank fantasy factors by what they add after expert consensus.

This is intentionally a research gate, not a projection replacement.  It asks a
harder question than ordinary feature importance: after a rolling, out-of-sample
consensus model has explained the easy part of weekly PPR, does a factor still
contain information about the residual?  It also measures what remains after
the factors already selected, so a second version of the same signal cannot win
by being correlated with the first.

The input is the decision-time player-week panel built by source_value_weekly.py.
No same-season training is allowed.  The report is the shortlist for expensive
league-simulation experiments, not evidence that a factor wins leagues.

    .venv/bin/python draft/factor_lab.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd


TEST_SEASONS = range(2021, 2026)
BASE = ["w_rank", "w_sd", "w_best", "w_worst"]
GROUPS = {
    "availability": ["inj_q", "inj_d", "prac_dnp", "prac_lim", "prac_full", "on_report", "ina"],
    "role_usage": ["tgt_rec", "tgt_base", "car_rec", "car_base", "rz_rec", "rz_base",
                   "snap_rec", "snap_base", "vacancy", "use_miss"],
    "game_environment": ["implied", "spread", "total", "home"],
    "season_to_date": ["ppg", "games", "missed", "ytd_miss"],
    "our_model": ["proj"],
    "consensus_ros": ["r_rank", "r_sd"],
}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_CANDIDATES = [
    os.path.join(HERE, "data", "source_value_weekly_panel.parquet"),
    os.path.join(os.path.dirname(HERE), "qbert-replica", "data", "source_value_weekly_panel.parquet"),
]
OUT = os.path.join(HERE, "data", "factor_lab.json")


def panel_path():
    for path in PANEL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Build data/source_value_weekly_panel.parquet first")


def fill_fit(train, test):
    """Median fill and standardize using training data only."""
    med = train.median(axis=0).fillna(0.0)
    scale = train.fillna(med).std(axis=0).replace(0, 1.0)
    return (train.fillna(med) - med) / scale, (test.fillna(med) - med) / scale


def ridge(train_x, train_y, test_x, lam=10.0):
    """Small deterministic ridge fit, with an unpenalized intercept."""
    x, z = fill_fit(train_x, test_x)
    x = np.column_stack([np.ones(len(x)), x.values])
    z = np.column_stack([np.ones(len(z)), z.values])
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train_y.values)
    return z @ beta


def residuals(d, features):
    """Rolling-origin residuals for each position, never fitting the scored season."""
    out = pd.Series(np.nan, index=d.index)
    for pos, g in d.groupby("pos"):
        for season in TEST_SEASONS:
            train = g[g.season < season]
            test = g[g.season == season]
            if len(train) < 30 or len(test) == 0:
                continue
            out.loc[test.index] = test.ppr - ridge(train[features], train.ppr, test[features])
    return out


def binned_mi(x, y, bins=8):
    """Distribution-free plug-in mutual information, in nats."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < bins * bins:
        return 0.0
    xb = pd.qcut(pd.Series(x).rank(method="first"), bins, labels=False, duplicates="drop")
    yb = pd.qcut(pd.Series(y).rank(method="first"), bins, labels=False, duplicates="drop")
    joint = pd.crosstab(xb, yb).to_numpy(dtype=float).copy()
    joint /= joint.sum()
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def conditional_mi(d, candidate, target_residual):
    """MI(candidate residual; PPR residual | consensus), season/position weighted."""
    x = d[["season", "pos", *BASE, candidate]].copy()
    x["ppr"] = x.pop(candidate)
    x_res = residuals(x, BASE)
    rows = []
    for (_, _), g in d.assign(x_res=x_res, y_res=target_residual).groupby(["season", "pos"]):
        if g.season.iloc[0] in TEST_SEASONS:
            rows.append((len(g), binned_mi(g.x_res, g.y_res)))
    return float(np.average([v for n, v in rows], weights=[n for n, v in rows]))


def incremental_rmse(d, selected, candidate):
    """Out-of-sample RMSE reduction from adding one factor group to the selected set."""
    before = residuals(d, BASE + selected)
    after = residuals(d, BASE + selected + candidate)
    keep = after.notna()
    return float(np.sqrt(np.mean(before[keep] ** 2)) - np.sqrt(np.mean(after[keep] ** 2)))


def main():
    d = pd.read_parquet(panel_path())
    d = d[(d.season.isin(TEST_SEASONS)) & d.pos.isin(["QB", "RB", "WR", "TE"])].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    y_res = residuals(d, BASE)
    factors = []
    for group, cols in GROUPS.items():
        usable = [c for c in cols if c in d]
        gain = incremental_rmse(d, [], usable)
        mi = np.mean([conditional_mi(d, c, y_res) for c in usable])
        factors.append({"factor": group, "conditional_mi_nats": round(float(mi), 5),
                        "incremental_rmse_points": round(gain, 4), "columns": usable})
    factors.sort(key=lambda x: (-x["conditional_mi_nats"], -x["incremental_rmse_points"]))

    selected, ladder = [], []
    remaining = {x["factor"]: x["columns"] for x in factors}
    while remaining:
        scored = []
        for name, cols in remaining.items():
            gain = incremental_rmse(d, selected, cols)
            scored.append((gain, name, cols))
        gain, name, cols = max(scored)
        ladder.append({"add": name, "remaining_incremental_rmse_points": round(gain, 4),
                       "model_columns": selected + cols})
        selected += cols
        del remaining[name]

    report = {"purpose": "factor research gate, not a league-win verdict",
              "baseline": "weekly expert consensus only",
              "timing": "rolling origin: train seasons before the scored season",
              "factor_screen": factors, "greedy_diversification_ladder": ladder,
              "next_action": "Only run the first factor that improves an out-of-sample league policy and is available at the decision time."}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
