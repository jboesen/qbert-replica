"""Which weekly sources add information on top of expert consensus, and in what combination?

Pure data analysis; no harness is run. One row per player-week (weeks 1-17) in the
decision-relevant pool, 2020-25. Rolling origin by season: fit on seasons before y,
score y, for y = 2021-25. Everything in a row is knowable before that week's kickoff.

Source groups
  W1 consensus weekly   positional ECR rank priced on consensus.weekly_curve (fit on
                        training seasons only), log rank, expert sd / best / worst
  W2 consensus ROS      latest rest-of-season scrape informing the week
  W3 our weekly model   weekly.py projection, walk-forward (model for season s trained
                        on seasons < s), rebuilt for every candidate whether he played
  W4 usage trend        recent (last 3 games) and season baseline target, carry,
                        red-zone and snap share; vacancy ahead of him (usage.py)
  W5 game environment   implied team total, spread, game total, home
  W6 availability       injury designation and final practice status for the week
  W7 season to date     PPG so far, games played, team games missed
  W8 Kalshi (2025 only) market-implied receptions, receiving and passing yards, P(TD)

Model: ridge (lambda fixed) on standardized features with missing indicators, one fit
per position; a fixed-hyperparameter gradient-boosting check on key subsets.
Metrics: MSE on next-week PPR, within position-week Spearman, and pairwise start/sit
accuracy over every same-position pair within 5 weekly consensus ranks.

    .venv/bin/python draft/source_value_weekly.py [--rebuild]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import itertools  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "draft")
import consensus as C  # noqa: E402
import weekly as W  # noqa: E402
from build_data import norm_team  # noqa: E402

warnings.filterwarnings("ignore")

POS = ["QB", "RB", "WR", "TE"]
TOP = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
SEASONS = list(range(2020, 2026))
TEST = list(range(2021, 2026))
WEEKS = 17
RECENT = 3
LAMBDA = 10.0
NPERM = 200
SEED = 20260914
PAIR_BAND = 5
PANEL = "data/source_value_weekly_panel.parquet"
OUT = "data/source_value_weekly.parquet"
GROUPS = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
NAMES = {"W1": "consensus weekly", "W2": "consensus ROS", "W3": "our weekly model",
         "W4": "usage trend", "W5": "game environment", "W6": "availability",
         "W7": "season to date", "W8": "Kalshi"}
COLS = {
    "W1": ["w_pts", "w_logrank", "w_sd", "w_logbest", "w_logworst", "w_miss"],
    "W2": ["r_pts", "r_logrank", "r_sd", "r_miss"],
    "W3": ["proj"],
    "W4": ["tgt_rec", "tgt_base", "car_rec", "car_base", "rz_rec", "rz_base",
           "snap_rec", "snap_base", "vacancy", "use_miss"],
    "W5": ["implied", "spread", "total", "home"],
    "W6": ["inj_q", "inj_d", "prac_dnp", "prac_lim", "prac_full", "on_report"],
    "W7": ["ppg", "games", "missed", "ytd_miss"],
    "W8": ["rec", "rec_yds", "pass_yds", "td", "has_rec", "has_rec_yds", "has_pass_yds",
           "has_td"],
}
ACTIVE = {"ACT", "INA"}


# ================================================================ panel
def schedule():
    g = pd.read_csv("data/games.csv")
    g = g[(g.game_type == "REG") & (g.week <= WEEKS)]
    rows = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        rows.append(pd.DataFrame({
            "season": g.season, "week": g.week, "team": norm_team(g[f"{side}_team"]),
            "opponent": norm_team(g[f"{other}_team"]),
            "implied": g.total_line / 2 + sign * g.spread_line / 2,
            "spread": sign * g.spread_line, "total": g.total_line,
            "home": np.where(g.location.eq("Neutral"), 0.5, 1.0 if side == "home" else 0.0),
            "kick": pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00")),
            "weekday": g.weekday,
        }))
    return pd.concat(rows, ignore_index=True)


def weekly_model(s):
    """Walk-forward weekly.py for season s: everything fit on seasons < s."""
    d, dp, rfits, sch = W.history(s)
    d = d[d.season <= s].reset_index(drop=True)
    d, ks, dfn, fits, sd = W.train(d, (d.season < s).values)
    return ks, dfn.before, fits, sd, rfits, dp


def ecr_ranks(kind, s):
    x = pd.read_parquet(f"data/ecr_{kind}.parquet")
    x = x[(x.season == s) & x.pos.isin(POS)].copy()
    x["rank"] = x.groupby(["week", "pos"]).ecr.rank(method="first")
    return x


def build(s):
    print(f"building {s} ...", flush=True)
    sch = schedule()
    sch_s = sch[sch.season == s]
    plays = set(zip(sch_s.week, sch_s.team))
    ew, er = ecr_ranks("weekly", s), ecr_ranks("ros", s)
    st = pd.read_parquet(f"data/stats/w{s}.parquet")
    st = st[(st.season_type == "REG") & (st.week <= WEEKS)].copy()
    st["team"] = norm_team(st.team)
    box = st.groupby(["player_id", "week"]).agg(
        ppr=("fantasy_points_ppr", "sum"), team=("team", "last"), spos=("position", "last"),
        targets=("targets", "sum"), carries=("carries", "sum"),
        tgt_share=("target_share", "sum")).reset_index()

    # usage inputs (pbp_opps: red zone and team volume; snaps via the pfr bridge)
    import usage as U
    o = pd.read_parquet("data/pbp_opps.parquet")
    o = o[(o.season == s) & (o.week <= WEEKS)].copy()
    o["team"] = norm_team(o.team)
    tv = o.groupby(["week", "team"]).agg(team_rush=("team_rush", "max"),
                                         team_rz=("team_rz_plays", "max")).reset_index()
    rz = o.groupby(["player_id", "week"]).agg(rz=("rz_tgt", "sum"), rzc=("rz_car", "sum"))
    box = box.merge(rz.reset_index(), on=["player_id", "week"], how="left")
    box = box.merge(tv, on=["week", "team"], how="left")
    box = box.merge(U.snaps(s).drop_duplicates(["player_id", "week"]),
                    on=["player_id", "week"], how="left")
    box["car_share"] = box.carries / box.team_rush.replace(0, np.nan)
    box["rz_share"] = (box.rz.fillna(0) + box.rzc.fillna(0)) / box.team_rz.replace(0, np.nan)
    box["opp"] = box.targets.fillna(0) + box.carries.fillna(0)

    inj = pd.read_parquet(f"data/injuries_{s}.parquet")
    inj = inj[(inj.game_type == "REG") & (inj.week <= WEEKS)].copy()
    inj["team"] = norm_team(inj.team)
    late = np.nan                   # 2025 file carries no modification timestamp
    if "date_modified" in inj:
        k = inj.merge(sch_s[["week", "team", "kick"]], on=["week", "team"], how="left")
        kick_utc = k.kick.dt.tz_localize("America/New_York", ambiguous="NaT",
                                         nonexistent="NaT").dt.tz_convert("UTC")
        late = (k.date_modified >= kick_utc).mean()
    inj = inj.drop_duplicates(["gsis_id", "week"], keep="last")

    ro_path = f"data/roster_weekly_{s}.parquet"
    ro = pd.read_parquet(ro_path) if os.path.exists(ro_path) else None
    if ro is not None:
        ro = ro[(ro.game_type == "REG") & (ro.week <= WEEKS) & ro.position.isin(POS)].copy()
        ro["team"] = norm_team(ro.team)
        ro = ro.drop_duplicates(["gsis_id", "week"], keep="last")
    dep = pd.read_parquet(f"data/depth_{s}.parquet")
    if "game_type" not in dep:      # 2025 file is a single post-season snapshot: unusable
        dep = pd.DataFrame(columns=["game_type", "formation", "position", "week",
                                    "club_code", "gsis_id"])
    dep = dep[(dep.game_type == "REG") & (dep.formation == "Offense")].copy()
    dep["position"] = dep.position.replace({"FB": "RB"})
    dep = dep[dep.position.isin(POS) & dep.week.notna()]
    dep["week"] = dep.week.astype(int)
    dep["team"] = norm_team(dep.club_code)
    dep = dep.drop_duplicates(["gsis_id", "week"], keep="first")
    players = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")

    ks, dbefore, fits, sd, rfits, dp = weekly_model(s)
    dps = dp[dp.season == s].drop_duplicates("player_id").set_index("player_id")

    out = []
    ros_weeks = sorted(er.week.unique())
    for w in range(1, WEEKS + 1):
        ecw = ew[ew.week == w]
        if not len(ecw):
            continue                                   # no weekly scrape: no base layer
        rw = [x for x in ros_weeks if x <= w]
        ecr_r = er[er.week == rw[-1]] if rw else er.iloc[:0]
        cand = set(ecw.player_id) | set(ecr_r.player_id) | set(dep[dep.week == w].gsis_id)
        if ro is not None:
            cand |= set(ro[(ro.week == w) & (ro.status == "ACT")].gsis_id)
        f = pd.DataFrame({"player_id": sorted(cand)})
        f["season"], f["week"] = s, w
        e1 = ecw.set_index("player_id")
        e2 = ecr_r.drop_duplicates("player_id").set_index("player_id")
        f["pos"] = f.player_id.map(e1.pos).fillna(f.player_id.map(e2.pos))
        if ro is not None:
            rw_ = ro[ro.week == w].set_index("gsis_id")
            f["pos"] = f.pos.fillna(f.player_id.map(rw_.position))
            f["ro_team"] = f.player_id.map(rw_.team)
            f["ro_status"] = f.player_id.map(rw_.status)
        else:
            f["ro_team"], f["ro_status"] = np.nan, np.nan
        dw = dep[dep.week == w].set_index("gsis_id")
        f["pos"] = f.pos.fillna(f.player_id.map(dw.position))
        f["pos"] = f.pos.fillna(f.player_id.map(players.position))
        f = f[f.pos.isin(POS)].copy()

        past = box[box.week < w]
        later = box[box.week >= w].sort_values("week")
        last_team = past.sort_values("week").groupby("player_id").team.last()
        next_team = later.groupby("player_id").team.first()   # fallback only; see leakage notes
        f["team"] = f.ro_team.fillna(f.player_id.map(dw.team)).fillna(
            f.player_id.map(last_team)).fillna(f.player_id.map(next_team))
        f = f[f.team.notna()]

        # ---- W1 / W2 raw
        f["w_rank"] = f.player_id.map(e1["rank"])
        f["w_sd"], f["w_best"], f["w_worst"] = (f.player_id.map(e1[c]) for c in ("sd", "best", "worst"))
        f["r_rank"] = f.player_id.map(e2["rank"]) if len(e2) else np.nan
        f["r_sd"] = f.player_id.map(e2["sd"]) if len(e2) else np.nan

        # ---- W6 availability (report published before kickoff; checked below)
        iw = inj[inj.week == w].set_index("gsis_id")
        stat = f.player_id.map(iw.report_status)
        prac = f.player_id.map(iw.practice_status).fillna("")
        f["status"] = stat
        f["inj_q"] = (stat == "Questionable").astype(float)
        f["inj_d"] = (stat == "Doubtful").astype(float)
        f["prac_dnp"] = prac.str.contains("Did Not").astype(float)
        f["prac_lim"] = prac.str.contains("Limited").astype(float)
        f["prac_full"] = prac.str.contains("Full").astype(float)
        f["on_report"] = f.player_id.isin(iw.index).astype(float)
        f["ina"] = (f.ro_status == "INA").astype(float)     # gameday inactives; sensitivity only

        # ---- eligibility: team plays, not ruled Out, on the 53 (2021+)
        elig = np.array([(w, t) in plays for t in f.team]) & (stat != "Out").values
        if ro is not None:
            elig &= f.ro_status.isin(ACTIVE).values
        f = f[elig].copy()

        # ---- W5 environment
        sw = sch_s[sch_s.week == w].drop_duplicates("team").set_index("team")
        for c in ("opponent", "implied", "spread", "total", "home", "weekday"):
            f[c] = f.team.map(sw[c])

        # ---- W7 season to date (strictly before w)
        g = past.groupby("player_id")
        f["games"] = f.player_id.map(g.size()).fillna(0)
        f["ppg"] = f.player_id.map(g.ppr.mean())
        team_games = sch_s[sch_s.week < w].groupby("team").size()
        f["missed"] = (f.team.map(team_games).fillna(0) - f.games).clip(lower=0)
        f["ytd_miss"] = (f.games == 0).astype(float)

        # ---- W4 usage: last RECENT games vs season baseline, before w
        rec = past.sort_values("week").groupby("player_id").tail(RECENT).groupby("player_id")
        for c, name in (("tgt_share", "tgt"), ("car_share", "car"), ("rz_share", "rz"),
                        ("snap_pct", "snap")):
            f[f"{name}_rec"] = f.player_id.map(rec[c].mean())
            f[f"{name}_base"] = f.player_id.map(g[c].mean())
        f["use_miss"] = f.ytd_miss
        hurt = set(iw[iw.report_status.isin(["Out", "Doubtful"])].index)
        lead = past.groupby(["team", "spos", "player_id"]).opp.sum()
        vac = {}
        for (_, _), grp in lead.groupby(level=[0, 1]):
            ids = [i[2] for i in grp.sort_values(ascending=False).index]
            if ids and ids[0] in hurt:
                for p in ids[1:]:
                    vac[p] = 1.0
        f["vacancy"] = f.player_id.map(vac).fillna(0.0)

        # ---- W3 weekly model, rebuilt for every candidate
        x = pd.DataFrame({"position": f.pos.values, "n_prev": f.games.values.astype(int),
                          "ytd_ppg": f.ppg.values}, index=f.index)
        has = f.player_id.isin(dps.index) & f.player_id.map(dps.own_ppg).notna() & \
            f.player_id.map(dps.proj_ppg).notna()
        pick = f.player_id.map(dps.draft_pick).fillna(f.player_id.map(players.draft_pick))
        x["pre"] = np.where(has, f.player_id.map(dps.proj_ppg),
                            W.rookie_prior(rfits, f.pos.values, pick.values))
        x["post"] = W.pooled(x, ks)
        x["implied"] = f.implied.fillna(22.0)
        x["home"] = f.home.fillna(0.5)
        db = dbefore[dbefore.season == s]
        db = db[db.week == w].set_index(["opponent_team", "position"]).opp_def
        x["opp_def"] = [db.get((o, p), 0.0) for o, p in zip(f.opponent, f.pos)]
        f["proj"], _ = W.predict(fits, sd, x)

        # ---- target: PPR that week, 0 if he did not play
        pts = box[box.week == w].set_index("player_id").ppr
        f["ppr"] = f.player_id.map(pts).fillna(0.0)
        f["played"] = f.player_id.isin(pts.index).astype(float)

        # ---- pool
        f["proj_rank"] = f.groupby("pos").proj.rank(ascending=False, method="first")
        top = f.pos.map(TOP)
        f["in_pool"] = (f.w_rank <= top) | (f.r_rank <= top) | (f.proj_rank <= top)
        out.append(f[f.in_pool])
    p = pd.concat(out, ignore_index=True)
    p.attrs["late_injury_share"] = float(late)
    print(f"  {s}: rows {len(p)}, weeks {p.week.nunique()}, injury rows modified at/after "
          f"kickoff {late:.3%}", flush=True)
    return p.drop(columns=["ro_team", "in_pool"])


def panel(rebuild=False):
    if os.path.exists(PANEL) and not rebuild:
        return pd.read_parquet(PANEL)
    p = pd.concat([build(s) for s in SEASONS], ignore_index=True)
    p.to_parquet(PANEL)
    return p


# ================================================================ features per fold
def fold_features(p, y):
    """Feature matrix for fold y: curve fit on seasons < y, applied to all rows."""
    crv = C.weekly_curve(range(2020, y))
    X = pd.DataFrame(index=p.index)
    wr = p.w_rank.fillna(p.groupby(["season", "week", "pos"]).w_rank.transform("max") + 1)
    rr = p.r_rank
    X["w_pts"] = C.rank_points(crv, p.pos, wr.fillna(150))
    X["w_logrank"] = np.log(wr.fillna(150))
    X["w_sd"] = p.w_sd
    X["w_logbest"] = np.log(p.w_best.clip(lower=1))
    X["w_logworst"] = np.log(p.w_worst.clip(lower=1))
    X["w_miss"] = p.w_rank.isna().astype(float)
    X["r_pts"] = np.where(rr.notna(), C.rank_points(crv, p.pos, rr.fillna(150)), np.nan)
    X["r_logrank"] = np.log(rr)
    X["r_sd"] = p.r_sd
    X["r_miss"] = rr.isna().astype(float)
    for g in ("W3", "W4", "W5", "W6", "W7"):
        for c in COLS[g]:
            X[c] = p[c].astype(float)
    return X


class Fold:
    """Standardized design for one test season and position, with Gram matrices."""

    def __init__(self, X, y, tr, te, cols):
        self.cols = cols
        Xtr, Xte = X.loc[tr, cols].values.astype(float), X.loc[te, cols].values.astype(float)
        mu = np.nanmean(Xtr, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Xtr = np.where(np.isnan(Xtr), mu, Xtr)
        Xte = np.where(np.isnan(Xte), mu, Xte)
        sdv = Xtr.std(axis=0)
        sdv = np.where(sdv > 1e-9, sdv, 1.0)
        self.Xtr, self.Xte = (Xtr - mu) / sdv, (Xte - mu) / sdv
        self.ytr = y[tr]
        self.ybar = self.ytr.mean()
        self.G = self.Xtr.T @ self.Xtr
        self.b = self.Xtr.T @ (self.ytr - self.ybar)
        self.ix = {c: i for i, c in enumerate(cols)}

    def predict(self, cols):
        if not cols:
            return np.full(len(self.Xte), self.ybar)
        j = [self.ix[c] for c in cols]
        beta = np.linalg.solve(self.G[np.ix_(j, j)] + LAMBDA * np.eye(len(j)), self.b[j])
        return self.ybar + self.Xte[:, j] @ beta


# ================================================================ metrics
class Scorer:
    """Per test season: position-week groups and consensus-neighbour pairs."""

    def __init__(self, t):
        self.y = t.ppr.values
        self.gid = t.groupby(["pos", "week"]).ngroup().values
        self.ng = self.gid.max() + 1
        self.ra = t.groupby(["pos", "week"]).ppr.rank().values
        I, J = [], []
        for _, g in t.reset_index(drop=True).groupby(["pos", "week"]):
            g = g[g.w_rank.notna()]
            r, idx = g.w_rank.values, g.index.values
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    if abs(r[a] - r[b]) <= PAIR_BAND and self.y[idx[a]] != self.y[idx[b]]:
                        I.append(idx[a])
                        J.append(idx[b])
        self.I, self.J = np.array(I), np.array(J)
        self.sa = np.sign(self.y[self.I] - self.y[self.J])
        self.pos = t.pos.values

    def score(self, pred, mask=None):
        """(sse, n, sum_rho, n_groups, correct, n_pairs)"""
        pr = pd.Series(pred).groupby(self.gid).rank().values
        ra = self.ra
        n = np.bincount(self.gid, minlength=self.ng)
        ma = np.bincount(self.gid, ra, self.ng) / n
        mp = np.bincount(self.gid, pr, self.ng) / n
        da, dp = ra - ma[self.gid], pr - mp[self.gid]
        cov = np.bincount(self.gid, da * dp, self.ng)
        va = np.bincount(self.gid, da * da, self.ng)
        vp = np.bincount(self.gid, dp * dp, self.ng)
        ok = (va > 1e-12) & (n >= 3)
        rho = np.where(ok & (vp > 1e-12), cov / np.sqrt(np.maximum(va * vp, 1e-24)), 0.0)
        dpp = pred[self.I] - pred[self.J]
        corr = np.where(np.abs(dpp) < 1e-12, 0.5, (np.sign(dpp) == self.sa).astype(float))
        return (float(np.sum((self.y - pred) ** 2)), len(pred), float(rho[ok].sum()),
                int(ok.sum()), float(corr.sum()), len(corr))


def summarize(rows):
    """Aggregate score tuples -> mse, spearman, pairwise."""
    a = np.sum(np.array(rows, float), axis=0)
    return a[0] / a[1], a[2] / a[3], a[4] / a[5]


# ================================================================ main loops
def subsets():
    for k in range(0, len(GROUPS) + 1):
        for s in itertools.combinations(GROUPS, k):
            yield s


def label(s):
    return "+".join(s) if s else "null"


def run_folds(p, Xs, scorers, fn, perm_cols=None, perm_idx=None):
    """fn(fold_by_pos, season) -> dict of subset -> prediction vector over test rows."""
    res = {}
    for y in TEST:
        X = Xs[y]
        if perm_cols is not None:
            X = X.copy()
            X.loc[:, perm_cols] = X[perm_cols].values[perm_idx]
        te_all = np.where(p.season.values == y)[0]
        preds = {}
        for pos in POS:
            tr = np.where((p.season.values < y) & (p.pos.values == pos))[0]
            te = np.where((p.season.values == y) & (p.pos.values == pos))[0]
            fold = Fold(X, p.ppr.values, tr, te, list(X.columns))
            loc = np.searchsorted(te_all, te)
            for key, cols in fn.items():
                preds.setdefault(key, np.zeros(len(te_all)))[loc] = fold.predict(cols)
        for key, pr in preds.items():
            res[(key, y)] = scorers[y].score(pr)
    return res


def cols_of(s):
    return [c for g in s for c in COLS[g]]


def strata_perm(p, rng):
    idx = np.arange(len(p))
    out = idx.copy()
    for _, g in p.groupby(["season", "pos", "week"]).indices.items():
        out[g] = rng.permutation(g)
    return out


def shapley(v):
    n = len(GROUPS)
    phi = {}
    for g in GROUPS:
        others = [h for h in GROUPS if h != g]
        tot = 0.0
        for k in range(n):
            wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for s in itertools.combinations(others, k):
                with_g = tuple(h for h in GROUPS if h in s or h == g)
                tot += wgt * (v[with_g] - v[s])
        phi[g] = tot
    return phi


def gbm_check(p, Xs, scorers, keys):
    from sklearn.ensemble import HistGradientBoostingRegressor
    res = {}
    for y in TEST:
        X = Xs[y]
        te_all = np.where(p.season.values == y)[0]
        preds = {k: np.zeros(len(te_all)) for k in keys}
        for pos in POS:
            tr = np.where((p.season.values < y) & (p.pos.values == pos))[0]
            te = np.where((p.season.values == y) & (p.pos.values == pos))[0]
            loc = np.searchsorted(te_all, te)
            for k in keys:
                cols = cols_of(k)
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                  max_leaf_nodes=15, min_samples_leaf=40,
                                                  l2_regularization=1.0, random_state=0)
                m.fit(X.iloc[tr][cols].values, p.ppr.values[tr])
                preds[k][loc] = m.predict(X.iloc[te][cols].values)
        for k in keys:
            res[(k, y)] = scorers[y].score(preds[k])
    return res


# ================================================================ Kalshi block
def kalshi_block(p, best):
    import kalshi_test as K
    t = K.attach_ids(K.market_table())
    t = t.groupby(["player_id", "week"])[["rec", "rec_yds", "pass_yds", "td"]].mean().reset_index()
    d = p[p.season == 2025].reset_index(drop=True)
    X = fold_features(d, 2025)       # curve from 2020-24 only
    m = d[["player_id", "week"]].merge(t, on=["player_id", "week"], how="left")
    for k in ("rec", "rec_yds", "pass_yds", "td"):
        X[f"has_{k}"] = m[k].notna().astype(float).values
        X[k] = m[k].fillna(0.0).values
    print(f"\nKalshi: 2025 pool rows {len(d)}, with any market price "
          f"{int((X[['has_rec', 'has_rec_yds', 'has_pass_yds', 'has_td']].sum(axis=1) > 0).sum())}")
    base, full = cols_of(best), cols_of(best) + COLS["W8"]
    weeks = sorted(d.week.unique())
    sc_all = Scorer(d)

    def lowo(Xm):
        pb, pf = np.zeros(len(d)), np.zeros(len(d))
        for w in weeks:
            for pos in POS:
                tr = np.where((d.week.values != w) & (d.pos.values == pos))[0]
                te = np.where((d.week.values == w) & (d.pos.values == pos))[0]
                if not len(te):
                    continue
                fo = Fold(Xm, d.ppr.values, tr, te, full)
                pb[te], pf[te] = fo.predict(base), fo.predict(full)
        return pb, pf

    pb, pf = lowo(X)
    sb, sf = summarize([sc_all.score(pb)]), summarize([sc_all.score(pf)])
    # per-week deltas (pairwise) to count weeks helped
    wk_help = []
    for w in weeks:
        dw = d[d.week == w].reset_index(drop=True)
        s_ = Scorer(dw)
        ii = np.where(d.week.values == w)[0]
        if len(s_.I):
            wk_help.append(s_.score(pf[ii])[4] / s_.score(pf[ii])[5] - s_.score(pb[ii])[4] / s_.score(pb[ii])[5])
    rng = np.random.default_rng(SEED + 8)
    null = []
    for _ in range(NPERM):
        idx = strata_perm(d, rng)
        Xp = X.copy()
        Xp.loc[:, COLS["W8"]] = X[COLS["W8"]].values[idx]
        _, pp = lowo(Xp)
        null.append(summarize([sc_all.score(pp)]))
    null = np.array(null)
    pv = [(1 + np.sum(null[:, 0] <= sf[0])) / (NPERM + 1),
          (1 + np.sum(null[:, 1] >= sf[1])) / (NPERM + 1),
          (1 + np.sum(null[:, 2] >= sf[2])) / (NPERM + 1)]
    res = pd.DataFrame([
        {"model": "best non-market", "mse": sb[0], "spearman": sb[1], "pairwise": sb[2]},
        {"model": "best + W8 Kalshi", "mse": sf[0], "spearman": sf[1], "pairwise": sf[2],
         "p_mse": pv[0], "p_spearman": pv[1], "p_pairwise": pv[2],
         "weeks_helped": int(np.sum(np.array(wk_help) > 0)), "weeks": len(wk_help)}])
    print(res.to_string(index=False))
    res.to_parquet("data/source_value_weekly_kalshi.parquet")
    return res


# ================================================================ leakage checks
def leakage(p, scorers, Xs):
    print("\nLEAKAGE CHECKS")
    ew = pd.read_parquet("data/ecr_weekly.parquet")
    print("  weekly ECR weeks present per season:",
          ew[ew.season.isin(SEASONS)].groupby("season").week.nunique().to_dict())
    print("  weeks missing from the panel (no weekly scrape):",
          {s: sorted(set(range(1, 18)) - set(p[p.season == s].week)) for s in SEASONS})
    # Thursday games: a scrape dated Fri/Sat maps to the week, after TNF was played.
    t = p[p.season.isin(TEST)].reset_index(drop=True)
    acc_thu, acc_sun = [], []
    for y in TEST:
        s = scorers[y]
        tt = p[p.season == y].reset_index(drop=True)
        th = (tt.weekday == "Thursday").values
        pred = -tt.w_rank.fillna(999).values
        dpp = pred[s.I] - pred[s.J]
        c = np.where(np.abs(dpp) < 1e-12, .5, (np.sign(dpp) == s.sa))
        both = th[s.I] | th[s.J]
        acc_thu.append((c[both].sum(), both.sum()))
        acc_sun.append((c[~both].sum(), (~both).sum()))
    a1 = sum(x[0] for x in acc_thu) / sum(x[1] for x in acc_thu)
    a2 = sum(x[0] for x in acc_sun) / sum(x[1] for x in acc_sun)
    print(f"  raw ECR pairwise accuracy, pairs touching a Thursday-game player {a1:.3f} "
          f"(n={sum(x[1] for x in acc_thu)}) vs other pairs {a2:.3f}")
    print(f"  share of pool rows that did not play: {1 - t.played.mean():.3f}; "
          f"Doubtful rows {int(t.inj_d.sum())}, of which played {t[t.inj_d == 1].played.mean():.2f}; "
          f"INA rows {int(t.ina.sum())}, of which played {t[t.ina == 1].played.mean():.2f}")
    print("  W3 missing:", int(p.proj.isna().sum()), " (built for every candidate, not only players with a box score)")


# ================================================================ harness reference
def harness_reference():
    """Is the W6 gain beyond the discount the league harness already applies?

    The harness values a consensus player at curve points times P(plays | Questionable)
    (and treats Doubtful as unavailable). Here: that reference, W1 ridge, and W1+W6 ridge,
    per season, on the same pool and pairs; plus a Doubtful-free pool.
    """
    p = pd.read_parquet(PANEL).sort_values(["season", "pos", "week", "player_id"]).reset_index(drop=True)
    rows = []
    for drop_d in (False, True):
        q = p[p.inj_d == 0].reset_index(drop=True) if drop_d else p
        for y in TEST:
            X = fold_features(q, y)
            t = q[q.season == y].reset_index(drop=True)
            sc = Scorer(t)
            probs = W.play_probs(y)
            disc = np.where(t.inj_q == 1, probs.get("Questionable", .57),
                            np.where(t.inj_d == 1, probs.get("Doubtful", 0.0), 1.0))
            ref = X.loc[q.season == y, "w_pts"].values * disc
            te_all = np.where(q.season.values == y)[0]
            preds = {"W1": np.zeros(len(t)), "W1+W6": np.zeros(len(t)),
                     "W1+W3+W4+W5+W6+W7": np.zeros(len(t))}
            for pos in POS:
                tr = np.where((q.season.values < y) & (q.pos.values == pos))[0]
                te = np.where((q.season.values == y) & (q.pos.values == pos))[0]
                fo = Fold(X, q.ppr.values, tr, te, list(X.columns))
                loc = np.searchsorted(te_all, te)
                for k in preds:
                    preds[k][loc] = fo.predict(cols_of(tuple(k.split("+"))))
            out = {"harness_ref": sc.score(ref), **{k: sc.score(v) for k, v in preds.items()}}
            for k, v in out.items():
                m = summarize([v])
                rows.append({"pool": "no Doubtful" if drop_d else "main", "season": y,
                             "model": k, "mse": m[0], "spearman": m[1], "pairwise": m[2],
                             "correct": v[4], "n_pairs": v[5], "rho_sum": v[2], "n_groups": v[3]})
    d = pd.DataFrame(rows)
    d.to_parquet("data/source_value_weekly_harnessref.parquet")
    for pool, g in d.groupby("pool"):
        print(f"\npool: {pool}")
        piv = g.pivot(index="season", columns="model", values="pairwise")
        print((100 * piv).round(2).to_string())
        agg = g.groupby("model")[["correct", "n_pairs", "rho_sum", "n_groups"]].sum()
        print("pooled pairwise:", (100 * agg.correct / agg.n_pairs).round(2).to_dict())
        print("pooled spearman:", (agg.rho_sum / agg.n_groups).round(4).to_dict())
        for k in ("W1+W6", "W1+W3+W4+W5+W6+W7"):
            dd = 100 * (piv[k] - piv["harness_ref"])
            print(f"  {k} - harness_ref: pooled "
                  f"{100 * (agg.correct[k] / agg.n_pairs[k] - agg.correct['harness_ref'] / agg.n_pairs['harness_ref']):+.2f} pts;"
                  f" by season {' '.join(f'{x:+.2f}' for x in dd)}")


# ================================================================ main
def main():
    if "--harness-ref" in sys.argv:
        return harness_reference()
    p = panel("--rebuild" in sys.argv)
    p = p.sort_values(["season", "pos", "week", "player_id"]).reset_index(drop=True)
    print(f"panel rows {len(p)}; by season {p.groupby('season').size().to_dict()}")
    Xs = {y: fold_features(p, y) for y in TEST}
    scorers = {y: Scorer(p[p.season == y].reset_index(drop=True)) for y in TEST}
    print("pairs per season:", {y: len(scorers[y].I) for y in TEST})

    subs = list(subsets())
    fn = {s: cols_of(s) for s in subs}
    res = run_folds(p, Xs, scorers, fn)

    # raw consensus order (no model) as a reference
    raw = {}
    for y in TEST:
        tt = p[p.season == y].reset_index(drop=True)
        raw[y] = scorers[y].score(-tt.w_rank.fillna(999).values.astype(float))

    rows = []
    for (s, y), sc in res.items():
        mse, rho, acc = summarize([sc])
        rows.append({"model": "ridge", "subset": label(s), "k": len(s), "season": y,
                     **{g: g in s for g in GROUPS}, "mse": mse, "spearman": rho,
                     "pairwise": acc, "sse": sc[0], "n": sc[1], "rho_sum": sc[2],
                     "n_groups": sc[3], "correct": sc[4], "n_pairs": sc[5]})
    for y, sc in raw.items():
        mse, rho, acc = summarize([sc])
        rows.append({"model": "raw_ecr_order", "subset": "ECR rank", "k": 0, "season": y,
                     **{g: False for g in GROUPS}, "mse": np.nan, "spearman": rho,
                     "pairwise": acc, "rho_sum": sc[2], "n_groups": sc[3],
                     "correct": sc[4], "n_pairs": sc[5]})

    def pooled(s, metric):
        m = summarize([res[(s, y)] for y in TEST])
        return {"mse": -m[0], "spearman": m[1], "pairwise": m[2]}[metric]

    def season_val(s, y, metric):
        m = summarize([res[(s, y)]])
        return {"mse": -m[0], "spearman": m[1], "pairwise": m[2]}[metric]

    base = ("W1",)
    print(f"\nconsensus weekly alone (ridge W1): pairwise {pooled(base, 'pairwise'):.4f}  "
          f"spearman {pooled(base, 'spearman'):.4f}  MSE {-pooled(base, 'mse'):.2f}")
    r = summarize(list(raw.values()))
    print(f"raw ECR order: pairwise {r[2]:.4f}  spearman {r[1]:.4f}")

    # ---- Shapley
    print("\nSHAPLEY (pooled; [min, max] across seasons). MSE shown as reduction.")
    shap_rows = []
    for metric in ("spearman", "pairwise", "mse"):
        v = {s: pooled(s, metric) for s in subs}
        phi = shapley(v)
        per = {y: shapley({s: season_val(s, y, metric) for s in subs}) for y in TEST}
        for g in GROUPS:
            vals = [per[y][g] for y in TEST]
            shap_rows.append({"metric": metric, "group": g, "pooled": phi[g],
                              "min": min(vals), "max": max(vals),
                              **{f"s{y}": per[y][g] for y in TEST}})
    shap = pd.DataFrame(shap_rows)
    print(shap[["metric", "group", "pooled", "min", "max"]].to_string(index=False, float_format="%.4f"))

    # ---- adds to consensus
    print("\nADDS TO CONSENSUS: (W1 + group) - W1")
    add_rows = []
    for g in GROUPS[1:]:
        s = tuple(h for h in GROUPS if h in ("W1", g))
        d = {m: pooled(s, m) - pooled(base, m) for m in ("spearman", "pairwise", "mse")}
        pos_seasons = sum(season_val(s, y, "pairwise") > season_val(base, y, "pairwise") for y in TEST)
        add_rows.append({"group": g, "d_spearman": d["spearman"], "d_pairwise": d["pairwise"],
                         "d_mse_reduction": d["mse"], "seasons_pairwise_up": pos_seasons})
    add = pd.DataFrame(add_rows)
    print(add.to_string(index=False, float_format="%.4f"))

    # ---- top subsets
    ranked = sorted(subs, key=lambda s: pooled(s, "pairwise"), reverse=True)
    best = ranked[0]
    best_w1 = next(s for s in ranked if "W1" in s)
    print(f"\nTOP 10 SUBSETS by pooled pairwise accuracy (vs W1 alone)")
    top_rows = []
    for s in ranked[:10]:
        dy = [season_val(s, y, "pairwise") - season_val(base, y, "pairwise") for y in TEST]
        top_rows.append({"subset": label(s), "pairwise": pooled(s, "pairwise"),
                         "d_pairwise_pts": 100 * (pooled(s, "pairwise") - pooled(base, "pairwise")),
                         "d_spearman": pooled(s, "spearman") - pooled(base, "spearman"),
                         "d_mse": -(pooled(s, "mse") - pooled(base, "mse")),
                         "seasons_up": sum(x > 0 for x in dy),
                         "by_season_pts": " ".join(f"{100 * x:+.2f}" for x in dy)})
    print(pd.DataFrame(top_rows).to_string(index=False, float_format="%.4f"))
    rs = sorted(subs, key=lambda s: pooled(s, "spearman"), reverse=True)[:5]
    print("top 5 by Spearman:", [(label(s), round(pooled(s, "spearman") - pooled(base, "spearman"), 4)) for s in rs])
    print(f"best subset: {label(best)}; best containing W1: {label(best_w1)}")

    # honest selection: subset chosen on earlier test seasons, scored on the next
    print("\nFORWARD SELECTION (subset picked on earlier test seasons, scored next season):")
    for y in TEST[1:]:
        prev = [x for x in TEST if x < y]
        pick = max(subs, key=lambda s: summarize([res[(s, x)] for x in prev])[2])
        print(f"  {y}: pick {label(pick):22s} pairwise delta vs W1 "
              f"{100 * (season_val(pick, y, 'pairwise') - season_val(base, y, 'pairwise')):+.2f} pts")

    # ---- permutation test around the best subset
    print(f"\nRANDOMIZATION TEST around best subset {label(best)} "
          f"({NPERM} joint permutations within season-position-week)")
    rng = np.random.default_rng(SEED)
    perm_rows = []
    for g in GROUPS:
        with_g = tuple(h for h in GROUPS if h in best or h == g)
        without = tuple(h for h in with_g if h != g)
        obs = summarize([res[(with_g, y)] for y in TEST])
        helped = sum(season_val(with_g, y, "pairwise") > season_val(without, y, "pairwise") for y in TEST)
        helped_rho = sum(season_val(with_g, y, "spearman") > season_val(without, y, "spearman") for y in TEST)
        null = []
        for _ in range(NPERM):
            idx = strata_perm(p, rng)
            rr_ = run_folds(p, Xs, scorers, {"m": cols_of(with_g)}, COLS[g], idx)
            null.append(summarize([rr_[("m", y)] for y in TEST]))
        null = np.array(null)
        perm_rows.append({
            "group": g, "model": label(with_g),
            "obs_pairwise": obs[2], "null_mean_pairwise": null[:, 2].mean(),
            "p_pairwise": (1 + np.sum(null[:, 2] >= obs[2])) / (NPERM + 1),
            "obs_spearman": obs[1], "null_mean_spearman": null[:, 1].mean(),
            "p_spearman": (1 + np.sum(null[:, 1] >= obs[1])) / (NPERM + 1),
            "p_mse": (1 + np.sum(null[:, 0] <= obs[0])) / (NPERM + 1),
            "seasons_helped_pairwise": helped, "seasons_helped_spearman": helped_rho})
        print("  " + str(perm_rows[-1]), flush=True)
    perm = pd.DataFrame(perm_rows)
    print(perm.to_string(index=False, float_format="%.4f"))

    # ---- INA sensitivity (gameday inactives, 90 min before kickoff)
    for y in TEST:
        Xs[y]["ina"] = p.ina.values
    ina = run_folds(p, Xs, scorers, {"W1": cols_of(base), "W1+ina": cols_of(base) + ["ina"],
                                     "best": cols_of(best), "best+ina": cols_of(best) + ["ina"]})
    s_ = {k: summarize([ina[(k, y)] for y in TEST]) for k in ("W1", "W1+ina", "best", "best+ina")}
    print(f"\nINA sensitivity: pairwise W1 {s_['W1'][2]:.4f} -> +INA {s_['W1+ina'][2]:.4f}; "
          f"best {s_['best'][2]:.4f} -> +INA {s_['best+ina'][2]:.4f}")
    for y in TEST:
        Xs[y] = Xs[y].drop(columns="ina")

    # ---- GBM robustness
    keys = [base] + [tuple(h for h in GROUPS if h in ("W1", g)) for g in GROUPS[1:]] + \
        [best, tuple(GROUPS)]
    keys = list(dict.fromkeys(keys))
    gb = gbm_check(p, Xs, scorers, keys)
    print("\nGBM robustness (fixed hyperparameters), delta vs GBM W1:")
    gw = summarize([gb[(base, y)] for y in TEST])
    for k in keys:
        m = summarize([gb[(k, y)] for y in TEST])
        up = sum(summarize([gb[(k, y)]])[2] > summarize([gb[(base, y)]])[2] for y in TEST)
        print(f"  {label(k):22s} pairwise {m[2]:.4f} ({100 * (m[2] - gw[2]):+.2f} pts, up {up}/5)  "
              f"spearman {m[1]:.4f} ({m[1] - gw[1]:+.4f})  MSE {m[0]:.2f}")
        for y in TEST:
            mse, rho, acc = summarize([gb[(k, y)]])
            rows.append({"model": "gbm", "subset": label(k), "k": len(k), "season": y,
                         **{g: g in k for g in GROUPS}, "mse": mse, "spearman": rho,
                         "pairwise": acc, "sse": gb[(k, y)][0], "n": gb[(k, y)][1],
                         "rho_sum": gb[(k, y)][2], "n_groups": gb[(k, y)][3],
                         "correct": gb[(k, y)][4], "n_pairs": gb[(k, y)][5]})

    pd.DataFrame(rows).to_parquet(OUT)
    shap.to_parquet("data/source_value_weekly_shapley.parquet")
    perm.to_parquet("data/source_value_weekly_perm.parquet")

    leakage(p, scorers, Xs)
    kalshi_block(p, tuple(h for h in best if h != "W8"))


if __name__ == "__main__":
    main()
