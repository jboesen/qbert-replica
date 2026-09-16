"""Out-of-sample information value of each preseason source, on top of consensus.

One row per player-season (2020-25) for everyone consensus or our board puts inside the
draftable pool (QB 30, RB 70, WR 80, TE 30). Eight feature groups, all knowable before
week 1:

  G1 consensus   positional ECR, positional rank, overall ECR, sd/best/worst
  G2 projection  board.project_upcoming(y), fit on seasons before y
  G3 stack       stack.py stacked points, weights fit on seasons before y
  G4 ADP         Fantasy Football Calculator ADP (2021 on)
  G5 platform    ESPN and Sleeper ADP ranks (2021 on)
  G6 role        week-1 depth-chart slot
  G7 profile     age, experience, draft pick, rookie
  G8 prior       last season's PPR per game, games, target share, carry share

Targets: T1 season PPR (weeks 1-17, busts 0); T2 top-12 QB/TE, top-24 RB/WR.
Rolling origin: test y in 2021-25, fit on 2020..y-1, per position. Ridge (alpha 10)
on standardized features with mean imputation and missing indicators, L2 logistic
(lambda 2) for T2; a small fixed gradient-boosting model as a robustness check.

    .venv/bin/python draft/source_value_season.py build     # features (cached)
    .venv/bin/python draft/source_value_season.py subsets   # all 255 subsets + null
    .venv/bin/python draft/source_value_season.py gbm       # GBM robustness, all subsets
    .venv/bin/python draft/source_value_season.py perm      # permutation test
    .venv/bin/python draft/source_value_season.py report
    .venv/bin/python draft/source_value_season.py leak      # walk-forward checks
"""
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "draft")

POS = ["QB", "RB", "WR", "TE"]
POOL = {"QB": 30, "RB": 70, "WR": 80, "TE": 30}
TOPN = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
SEASONS = range(2020, 2026)
TEST = [2021, 2022, 2023, 2024, 2025]
GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
NAMES = {"G1": "consensus", "G2": "projection", "G3": "stack", "G4": "market ADP",
         "G5": "platform ADP", "G6": "role", "G7": "profile", "G8": "prior prod"}
ALPHA = 10.0        # ridge penalty, standardized features
LAMBDA = 2.0        # logistic L2 penalty, standardized features
NPERM = 200
SEED = 20260914

FEAT = "data/source_value_season_features.parquet"
OUT = "data/source_value_season.parquet"
OUT_GBM = "data/source_value_season_gbm.parquet"
OUT_PERM = "data/source_value_season_perm.parquet"


# ---------------------------------------------------------------- features

def log1(x):
    return np.log(np.asarray(x, float))


def prior_season(y):
    """Last season's production, from season y-1 only."""
    d = pd.read_parquet(f"data/stats/w{y - 1}.parquet")
    d = d[(d.season_type == "REG") & d.position.isin(POS)].copy()
    team_car = d.groupby(["team", "week"]).carries.transform("sum")
    d["carry_share"] = np.where(team_car > 0, d.carries / team_car, np.nan)
    g = d.groupby("player_id").agg(prev_games=("week", "count"),
                                   prev_ppr=("fantasy_points_ppr", "sum"),
                                   prev_tgt_share=("target_share", "mean"),
                                   prev_carry_share=("carry_share", "mean"))
    g["prev_ppr_pg"] = g.prev_ppr / g.prev_games
    return g.drop(columns="prev_ppr")


def outcomes(y):
    d = pd.read_parquet(f"data/stats/w{y}.parquet")
    d = d[(d.season_type == "REG") & (d.week <= 17)]
    tot = d.groupby("player_id").fantasy_points_ppr.sum()
    pos = d.groupby("player_id").position.first()
    cut = {p: tot[pos == p].sort_values(ascending=False).iloc[TOPN[p] - 1] for p in POS}
    return tot, cut


def build():
    import board  # noqa: F401  (imported via stack)
    import stack as S

    boards = {y: S.model_board(y) for y in SEASONS}
    for y in SEASONS:                           # stack.fit_before reads these
        S._TABLES[y] = S.table(y, board=boards[y])
    ecr = pd.read_parquet("data/ecr_preseason.parquet")
    ovr = pd.read_parquet("data/ecr_preseason_overall.parquet")
    adp = pd.read_parquet("data/adp_ffc.parquet")
    plat = pd.read_parquet("data/platform_ranks.parquet")
    depth = pd.read_parquet("data/depth_role.parquet")
    pl = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")

    rows = []
    for y in SEASONS:
        t = S._TABLES[y].copy()                 # board U ECR ids, pos, model, rank, actual
        b = boards[y]
        t["board_rank"] = t.groupby("pos").model.rank(ascending=False, method="first")
        e = ecr[ecr.season == y].drop_duplicates("player_id").copy()
        e["prank"] = e.groupby("pos").ecr.rank(method="first")
        e = e.set_index("player_id")
        pool = (e.prank.reindex(t.index) <= t.pos.map(POOL)) | (t.board_rank <= t.pos.map(POOL))
        t = t[pool].copy()
        ids = t.index
        f = pd.DataFrame(index=ids)
        f["season"], f["pos"] = y, t.pos
        tot, cut = outcomes(y)
        f["y_pts"] = tot.reindex(ids).fillna(0.0)
        f["y_top"] = (f.y_pts >= f.pos.map(cut)).astype(float)
        f["name"] = b.player_display_name.reindex(ids).fillna(e.player.reindex(ids))

        # G1 consensus
        f["G1_ecr"] = e.ecr.reindex(ids)
        f["G1_rank_log"] = log1(e.prank.reindex(ids))
        f["G1_sd"] = e.sd.reindex(ids)
        f["G1_best_log"] = log1(e.best.reindex(ids))
        f["G1_worst_log"] = log1(e.worst.reindex(ids))
        o = ovr[ovr.season == y].drop_duplicates("player_id").set_index("player_id")
        f["G1_ovr"] = o.ecr.reindex(ids)
        f["G1_ovr_log"] = log1(o.ecr.reindex(ids))
        # G2 projection
        f["G2_points"] = b.proj_points.reindex(ids)
        f["G2_games"] = b.proj_games.reindex(ids)
        f["G2_rank_log"] = log1(t.board_rank)
        # G3 stack, weights from seasons before y (none exist for 2020)
        if y > 2020:
            crv, coefs = S.fit_before(y)
            f["G3_stack"] = S.apply(t, crv, coefs)
        else:
            f["G3_stack"] = np.nan
        # G4 market ADP
        a = adp[(adp.season == y) & adp.player_id.notna()].drop_duplicates("player_id").copy()
        a["prank"] = a.groupby("position").adp.rank(method="first")
        a = a.set_index("player_id")
        f["G4_adp_log"] = log1(a.adp.reindex(ids))
        f["G4_prank_log"] = log1(a.prank.reindex(ids))
        f["G4_stdev"] = a.stdev.reindex(ids)
        # G5 platform ADP
        p = plat[(plat.season == y) & plat.player_id.notna()].drop_duplicates("player_id").copy()
        for s in ("espn", "sleeper"):
            p[f"{s}_prank"] = p.groupby("position")[f"{s}_rank"].rank(method="first")
        p = p.set_index("player_id")
        for s in ("espn", "sleeper"):
            f[f"G5_{s}_log"] = log1(p[f"{s}_rank"].reindex(ids))
            f[f"G5_{s}_prank_log"] = log1(p[f"{s}_prank"].reindex(ids))
        # G6 role: week-1 depth slot, no chart = all zero
        dp = depth[depth.season == y].drop_duplicates("player_id").set_index("player_id").depth
        dd = dp.reindex(ids)
        f["G6_d1"] = (dd == 1).astype(float)
        f["G6_d2"] = (dd == 2).astype(float)
        f["G6_d3"] = (dd >= 3).astype(float)
        # G7 profile
        birth = pd.to_datetime(pl.birth_date.reindex(ids), errors="coerce")
        f["G7_age"] = (pd.Timestamp(f"{y}-09-01") - birth).dt.days / 365.25
        exp = y - pl.rookie_season.reindex(ids)
        f["G7_exp"] = exp.clip(lower=0)
        f["G7_pick_log"] = log1(pl.draft_pick.reindex(ids).fillna(260))
        f["G7_rookie"] = (exp == 0).astype(float)
        # G8 prior production, season y-1
        ps = prior_season(y).reindex(ids)
        for c in ("prev_ppr_pg", "prev_games", "prev_tgt_share", "prev_carry_share"):
            f[f"G8_{c}"] = ps[c]
        f["G8_prev_games"] = f.G8_prev_games.fillna(0)
        rows.append(f.reset_index(names="player_id"))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(FEAT)
    print(out.groupby(["season", "pos"]).size().unstack())
    print(out.filter(regex="^G").notna().groupby(out.season).mean().round(2).T.to_string())
    return out


# ---------------------------------------------------------------- fitting

def group_cols(f):
    return {g: [c for c in f.columns if c.startswith(g + "_")] for g in GROUPS}


def design(tr, te):
    """Mean-impute on training rows, add missing indicators, standardize on training."""
    tr_, te_ = [], []
    for c in tr.columns:
        a, b = tr[c].to_numpy(float), te[c].to_numpy(float)
        ma, mb = np.isnan(a), np.isnan(b)
        if ma.all():
            continue
        mu = a[~ma].mean()
        cols = [(np.where(ma, mu, a), np.where(mb, mu, b))]
        if ma.any():
            cols.append((ma.astype(float), mb.astype(float)))
        for u, v in cols:
            sd = u.std()
            if sd < 1e-9:
                continue
            m = u.mean()
            tr_.append((u - m) / sd)
            te_.append((v - m) / sd)
    if not tr_:
        return np.zeros((len(tr), 0)), np.zeros((len(te), 0))
    return np.column_stack(tr_), np.column_stack(te_)


def ridge(Xtr, ytr, Xte):
    mu = ytr.mean()
    if Xtr.shape[1] == 0:
        return np.full(len(Xte), mu)
    A = Xtr.T @ Xtr + ALPHA * np.eye(Xtr.shape[1])
    beta = np.linalg.solve(A, Xtr.T @ (ytr - mu))
    return mu + Xte @ beta


def logistic(Xtr, ytr, Xte, iters=25):
    n, k = Xtr.shape
    p0 = np.clip(ytr.mean(), 1e-3, 1 - 1e-3)
    X1 = np.column_stack([np.ones(n), Xtr])
    w = np.zeros(k + 1)
    w[0] = math.log(p0 / (1 - p0))
    pen = np.full(k + 1, LAMBDA)
    pen[0] = 0.0
    for _ in range(iters):
        z = np.clip(X1 @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X1.T @ (p - ytr) + pen * w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(pen) + 1e-8 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    z = np.clip(np.column_stack([np.ones(len(Xte)), Xte]) @ w, -30, 30)
    return 1 / (1 + np.exp(-z))


def blocks(f, cols, y, pos):
    """Per-group design blocks for one rolling origin and position."""
    tr = f[(f.season < y) & (f.pos == pos)]
    te = f[(f.season == y) & (f.pos == pos)]
    out = {g: design(tr[cols[g]], te[cols[g]]) for g in GROUPS}
    return out, tr, te


def masks():
    return list(range(256))


def members(mask):
    return [g for i, g in enumerate(GROUPS) if mask >> i & 1]


def label(mask):
    return "+".join(members(mask)) or "null"


def score(te, pts, prob):
    """MSE, log loss, and per-position Spearman for one season's predictions."""
    yv, tv = te.y_pts.to_numpy(), te.y_top.to_numpy()
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    rho = []
    for p in POS:
        m = (te.pos == p).to_numpy()
        x = pts[m]
        rho.append(0.0 if np.ptp(x) < 1e-9 else spearmanr(x, yv[m]).statistic)
    return dict(mse=float(np.mean((yv - pts) ** 2)),
                logloss=float(-np.mean(tv * np.log(prob) + (1 - tv) * np.log(1 - prob))),
                rho=float(np.mean(rho)), **{f"rho_{p}": r for p, r in zip(POS, rho)})


def fit_subsets(f, subset_masks, fitter="ridge"):
    cols = group_cols(f)
    res = []
    for y in TEST:
        pb = {p: blocks(f, cols, y, p) for p in POS}
        te_all = pd.concat([pb[p][2] for p in POS])
        for mask in subset_masks:
            gs = members(mask)
            pts, prob = [], []
            for p in POS:
                bl, tr, te = pb[p]
                Xtr = np.column_stack([bl[g][0] for g in gs]) if gs else np.zeros((len(tr), 0))
                Xte = np.column_stack([bl[g][1] for g in gs]) if gs else np.zeros((len(te), 0))
                if fitter == "ridge":
                    pts.append(ridge(Xtr, tr.y_pts.to_numpy(), Xte))
                    prob.append(logistic(Xtr, tr.y_top.to_numpy(), Xte))
                else:
                    a, b = gbm(Xtr, tr.y_pts.to_numpy(), tr.y_top.to_numpy(), Xte)
                    pts.append(a)
                    prob.append(b)
            s = score(te_all, np.concatenate(pts), np.concatenate(prob))
            res.append(dict(season=y, mask=mask, subset=label(mask), n_groups=len(gs),
                            model=fitter, **{g: g in gs for g in GROUPS}, **s))
    return pd.DataFrame(res)


def gbm(Xtr, ytr, ttr, Xte):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    if Xtr.shape[1] == 0:
        p0 = np.clip(ttr.mean(), 1e-3, 1 - 1e-3)
        return np.full(len(Xte), ytr.mean()), np.full(len(Xte), p0)
    kw = dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
              min_samples_leaf=10, random_state=0)
    r = GradientBoostingRegressor(**kw).fit(Xtr, ytr).predict(Xte)
    if ttr.min() == ttr.max():
        c = np.full(len(Xte), ttr.mean())
    else:
        c = GradientBoostingClassifier(**kw).fit(Xtr, ttr).predict_proba(Xte)[:, 1]
    return r, c


def run_gbm(f):
    from joblib import Parallel, delayed
    chunks = [list(range(i, 256, 2)) for i in range(2)]
    parts = Parallel(n_jobs=2)(delayed(fit_subsets)(f, c, "gbm") for c in chunks)
    return pd.concat(parts, ignore_index=True).sort_values(["season", "mask"])


# ---------------------------------------------------------------- Shapley

def shapley(v):
    """Exact Shapley values from a value per mask (array of 256)."""
    n = len(GROUPS)
    phi = np.zeros(n)
    for i in range(n):
        for mask in range(256):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            phi[i] += w * (v[mask | 1 << i] - v[mask])
    return phi


def value(df, metric):
    """Gain over the null per season: loss reduction, or rho gain."""
    out = {}
    for y, g in df.groupby("season"):
        s = g.set_index("mask")[metric].reindex(range(256)).to_numpy()
        out[y] = (s - s[0]) if metric.startswith("rho") else (s[0] - s)
    return out


# ---------------------------------------------------------------- permutation

def permute_group(f, cols, g, rng):
    f = f.copy()
    for _, idx in f.groupby(["season", "pos"]).groups.items():
        idx = np.asarray(idx)
        f.loc[idx, cols[g]] = f.loc[rng.permutation(idx), cols[g]].to_numpy()
    return f


def perm_one(f, g, mask, seed):
    """Model `mask` (contains g) with g's features permuted, NPERM times."""
    cols = group_cols(f)
    rng = np.random.default_rng(seed)
    res = []
    for r in range(NPERM):
        fp = permute_group(f, cols, g, rng)
        d = fit_subsets(fp, [mask])
        d["group"], d["draw"] = g, r
        res.append(d)
    return pd.concat(res, ignore_index=True)


def run_perm(f, sub):
    from joblib import Parallel, delayed
    best = best_mask(sub)
    jobs = []
    for i, g in enumerate(GROUPS):
        m = best | 1 << i
        jobs.append((g, m, SEED + i))
    parts = Parallel(n_jobs=2)(delayed(perm_one)(f, g, m, s) for g, m, s in jobs)
    return pd.concat(parts, ignore_index=True)


def best_mask(sub):
    pooled = sub[sub["mask"] > 0].groupby("mask")[["rho", "mse"]].mean()
    return int(pooled.sort_values(["rho", "mse"], ascending=[False, True]).index[0])


# ---------------------------------------------------------------- report

def raw_ecr_rho(f):
    out = {}
    for y in TEST:
        te = f[f.season == y]
        r = []
        for p in POS:
            x = te[te.pos == p]
            r.append(spearmanr(-x.G1_rank_log.fillna(99), x.y_pts).statistic)
        out[y] = np.mean(r)
    return out


def report(f, sub, gbm_df=None, perm=None):
    pd.set_option("display.width", 200)
    for model, df in (("ridge", sub), ("gbm", gbm_df)):
        if df is None:
            continue
        print(f"\n===== {model}: Shapley values vs null (pooled mean, [min, max] over seasons)")
        rows = {}
        for metric, scale in (("rho", 1), ("mse", 1), ("logloss", 1000)):
            v = value(df, metric)
            ph = pd.DataFrame({y: shapley(v[y]) for y in v}, index=GROUPS) * scale
            rows[metric] = ph.mean(axis=1).map("{:+.3f}".format) + " [" + \
                ph.min(axis=1).map("{:+.3f}".format) + "," + ph.max(axis=1).map("{:+.3f}".format) + "]"
        t = pd.DataFrame(rows)
        t.index = [f"{g} {NAMES[g]}" for g in GROUPS]
        print("(mse in PPR pts^2 reduction; logloss x1000 reduction)")
        print(t.to_string())

        piv = {m: df.pivot(index="mask", columns="season", values=m) for m in ("rho", "mse", "logloss")}
        print(f"\n----- {model}: adds to consensus: metric(G1+Gk) - metric(G1); mse/logloss sign flipped so + = better")
        rr = []
        for i, g in enumerate(GROUPS[1:], start=1):
            m = 1 | 1 << i
            d_rho = piv["rho"].loc[m] - piv["rho"].loc[1]
            d_mse = piv["mse"].loc[1] - piv["mse"].loc[m]
            d_ll = (piv["logloss"].loc[1] - piv["logloss"].loc[m]) * 1000
            rr.append(dict(group=f"{g} {NAMES[g]}", d_rho=d_rho.mean(), rho_by_season=" ".join(f"{x:+.3f}" for x in d_rho),
                           rho_pos=int((d_rho > 0).sum()), d_mse=d_mse.mean(), mse_pos=int((d_mse > 0).sum()),
                           d_ll_x1000=d_ll.mean(), ll_pos=int((d_ll > 0).sum())))
        print(pd.DataFrame(rr).round(3).to_string(index=False))

        pooled = df.groupby(["mask", "subset"])[["rho", "mse", "logloss"]].mean().reset_index()
        g1 = pooled[pooled["mask"] == 1].iloc[0]
        print(f"\n----- {model}: top 5 subsets by pooled rank correlation (consensus alone rho {g1.rho:.4f}, "
              f"mse {g1.mse:.0f}, logloss {g1.logloss:.4f})")
        top = pooled.sort_values("rho", ascending=False).head(5).copy()
        top["d_rho_vs_G1"] = top.rho - g1.rho
        top["seasons_beat_G1"] = [int((piv["rho"].loc[m] > piv["rho"].loc[1]).sum()) for m in top["mask"]]
        top["rho_by_season"] = [" ".join(f"{x:+.3f}" for x in piv["rho"].loc[m] - piv["rho"].loc[1]) for m in top["mask"]]
        print(top.round(4).to_string(index=False))
        withg1 = pooled[(pooled["mask"] & 1) == 1].sort_values("rho", ascending=False).iloc[0]
        print(f"best subset containing G1: {withg1.subset}  rho {withg1.rho:.4f}  mse {withg1.mse:.0f}  "
              f"logloss {withg1.logloss:.4f}")
        bm = pooled.sort_values("mse").iloc[0]
        bl = pooled.sort_values("logloss").iloc[0]
        print(f"best by mse: {bm.subset} {bm.mse:.0f};  best by logloss: {bl.subset} {bl.logloss:.4f}")
        ok = []
        for m in pooled["mask"]:
            d = piv["rho"].loc[m] - piv["rho"].loc[1]
            if d.mean() >= 0.02 and (d > 0).sum() >= 4:
                ok.append((label(m), round(d.mean(), 4), int((d > 0).sum())))
        print("subsets meeting rho +0.02 and >=4/5 seasons vs G1:", ok or "none")
        per_pos = df[df["mask"].isin([1, int(withg1["mask"])])].groupby("subset")[[f"rho_{p}" for p in POS]].mean()
        print(per_pos.round(3).to_string())

    print("\nraw ECR positional rank (no fit) rho by season:",
          {y: round(v, 4) for y, v in raw_ecr_rho(f).items()},
          "mean", round(np.mean(list(raw_ecr_rho(f).values())), 4))

    if perm is not None:
        best = best_mask(sub)
        print(f"\n===== permutation test, best ridge subset {label(best)}; "
              f"groups outside it tested in best+Gk; {NPERM} draws")
        piv = {m: sub.pivot(index="mask", columns="season", values=m) for m in ("rho", "mse", "logloss")}
        rr = []
        for i, g in enumerate(GROUPS):
            m = best | 1 << i
            base = m & ~(1 << i)
            real_rho = piv["rho"].loc[m] - piv["rho"].loc[base]
            real_mse = piv["mse"].loc[base] - piv["mse"].loc[m]
            real_ll = piv["logloss"].loc[base] - piv["logloss"].loc[m]
            p = perm[perm.group == g]
            pr = p.groupby("draw")[["rho", "mse", "logloss"]].mean()
            null_rho = pr.rho - piv["rho"].loc[base].mean()
            null_mse = piv["mse"].loc[base].mean() - pr.mse
            null_ll = piv["logloss"].loc[base].mean() - pr.logloss
            # per-season: fraction of draws the real model beats
            pv = lambda real, null: (1 + (null >= real).sum()) / (1 + len(null))
            rr.append(dict(group=f"{g} {NAMES[g]}", model=label(m),
                           real_rho=real_rho.mean(), null_rho_mean=null_rho.mean(), p_rho=pv(real_rho.mean(), null_rho),
                           real_mse=real_mse.mean(), p_mse=pv(real_mse.mean(), null_mse),
                           real_ll_x1000=real_ll.mean() * 1000, p_ll=pv(real_ll.mean(), null_ll),
                           seasons_help_rho=int((real_rho > 0).sum()),
                           seasons_help_mse=int((real_mse > 0).sum())))
        print(pd.DataFrame(rr).round(4).to_string(index=False))


# ---------------------------------------------------------------- leakage

def leak_check(y=2023):
    """Projection and stack for y must not change when seasons >= y are deleted."""
    import project_season as P
    import stack as S
    real = P.build

    full = S.model_board(y).proj_points
    P.build = lambda: real().pipe(lambda s: s[s.season < y])
    import board as B
    B.P.build = P.build
    cut = S.model_board(y).proj_points
    P.build = B.P.build = real
    d = (full - cut.reindex(full.index)).abs()
    print(f"projection {y}: max |diff| with seasons>={y} deleted = {d.max():.6f} "
          f"(missing {cut.reindex(full.index).isna().sum()})")
    f = pd.read_parquet(FEAT)
    g = f[f.season == y]
    for c in g.filter(regex="^G").columns:
        r = g[c].corr(g.y_pts)
        print(f"  {c:22s} corr with target {r:+.3f}  nonmissing {g[c].notna().mean():.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "build":
        build()
    elif cmd == "subsets":
        f = pd.read_parquet(FEAT)
        d = fit_subsets(f, masks())
        d.to_parquet(OUT)
        print(d.groupby("subset")[["rho", "mse", "logloss"]].mean().sort_values("rho").tail(10))
    elif cmd == "gbm":
        run_gbm(pd.read_parquet(FEAT)).to_parquet(OUT_GBM)
    elif cmd == "perm":
        f = pd.read_parquet(FEAT)
        run_perm(f, pd.read_parquet(OUT)).to_parquet(OUT_PERM)
    elif cmd == "leak":
        leak_check()
    else:
        f = pd.read_parquet(FEAT)
        report(f, pd.read_parquet(OUT),
               pd.read_parquet(OUT_GBM) if os.path.exists(OUT_GBM) else None,
               pd.read_parquet(OUT_PERM) if os.path.exists(OUT_PERM) else None)
