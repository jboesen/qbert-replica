"""Is expert consensus (FantasyPros ECR) systematically biased or slow?

Pure data analysis over 2020-25, decision weeks 5-17. No harness is run.

  1. streak chasing: does last week's surprise predict next-week / next-3-week
     points beyond what the new rank implies?
  2. anchoring: does a blend of ROS ECR, preseason ECR and season-to-date PPG
     beat ROS ECR alone for rest-of-season PPG? (fit 2020-22, score 2023-25)
  3. staleness: when a starter is ruled Out, how fast does ECR re-rank the
     replacement, against his realized snap share and points?
  4. cheap extras: rookies vs veterans by week of season, QB vs skill positions,
     weekly-vs-ROS disagreement, expert spread.

Rank-implied points come from consensus.weekly_curve fit leave-one-season-out,
so no season's residuals are scored against a curve that saw that season.
Uncertainty: bootstrap over seasons (the independent unit), plus per-season values.

    .venv/bin/python draft/ecr_bias.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "draft")
from consensus import rank_points, weekly_curve  # noqa: E402

SEASONS = list(range(2020, 2026))
POS = ["QB", "RB", "WR", "TE"]
TOP = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}      # weekly start/sit-relevant ranks
TOP_ROS = {"QB": 32, "RB": 60, "WR": 80, "TE": 32}  # roster/trade-relevant ranks
W0, W1 = 5, 17
B = 1000
rng = np.random.default_rng(7)


# ---------------------------------------------------------------- loading
def load():
    st = pd.concat([pd.read_parquet(f"data/stats/w{y}.parquet") for y in SEASONS])
    st = st[st.season_type == "REG"]
    st = st[["player_id", "season", "week", "team", "position", "fantasy_points_ppr",
             "carries", "targets", "attempts"]].rename(columns={"fantasy_points_ppr": "pts"})
    st = st.astype({"season": "int64", "week": "int64"})
    g = pd.read_csv("data/games.csv")
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS)]
    tw = pd.concat([g[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
                    g[["season", "week", "away_team"]].rename(columns={"away_team": "team"})])
    lastwk = g.groupby("season").week.max().to_dict()
    inj = pd.concat([pd.read_parquet(f"data/injuries_{y}.parquet") for y in SEASONS])
    inj = inj[inj.game_type == "REG"].rename(columns={"gsis_id": "player_id"})
    inj = inj[["player_id", "season", "week", "team", "report_status"]].drop_duplicates(
        ["player_id", "season", "week"], keep="last")
    inj = inj.astype({"season": "int64", "week": "int64"})
    ids = pd.read_csv("data/playerids.csv")
    sn = pd.concat([pd.read_parquet(f"data/snaps_{y}.parquet") for y in SEASONS])
    sn = sn[sn.game_type == "REG"][["pfr_player_id", "season", "week", "offense_pct"]]
    pm = ids[ids.pfr_id.notna() & ids.gsis_id.notna()].drop_duplicates("pfr_id")
    sn = sn.merge(pm[["pfr_id", "gsis_id"]], left_on="pfr_player_id", right_on="pfr_id")
    sn = sn.rename(columns={"gsis_id": "player_id"}).drop_duplicates(
        ["player_id", "season", "week"])[["player_id", "season", "week", "offense_pct"]].astype({"season": "int64", "week": "int64"})
    rook = ids[ids.gsis_id.notna()].drop_duplicates("gsis_id")[["gsis_id", "draft_year"]]
    rook = rook.rename(columns={"gsis_id": "player_id"})

    def ecr(kind):
        x = pd.read_parquet(f"data/ecr_{kind}.parquet")
        x = x[x.season.isin(SEASONS)].copy()
        x = x.astype({c: "int64" for c in ("season", "week") if c in x})
        grp = ["season", "pos"] + (["week"] if "week" in x else [])
        x["rank"] = x.groupby(grp).ecr.rank(method="first")
        return x
    return st, tw, lastwk, inj, sn, rook, ecr("weekly"), ecr("ros"), ecr("preseason")


def fit_curve(df, ycol):
    """Rank -> mean y per position; same smoothing as consensus.weekly_curve."""
    out = {}
    for p in POS:
        by = df[df.pos == p].groupby("rank")[ycol].mean().reindex(range(1, 151))
        by = by.interpolate().ffill().bfill().rolling(5, center=True, min_periods=1).mean()
        out[p] = np.minimum.accumulate(by.fillna(0).values)
    return out


def loso(df, ycol=None, weekly=False):
    """Rank-implied value for each row, from a curve fit on the other seasons."""
    res = pd.Series(np.nan, index=df.index)
    for s in SEASONS:
        other = [y for y in SEASONS if y != s]
        crv = weekly_curve(other) if weekly else fit_curve(df[df.season.isin(other)], ycol)
        m = df.season == s
        res[m] = rank_points(crv, df.pos[m], df["rank"][m])
    return res


# ---------------------------------------------------------------- stats helpers
def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    return np.linalg.lstsq(X, y, rcond=None)[0]


def boot(df, stat):
    """Point estimate, 90% season-bootstrap interval, per-season values."""
    by = {s: g for s, g in df.groupby("season")}
    ss = sorted(by)
    point = stat(df)
    draws = []
    for _ in range(B):
        pick = rng.choice(ss, len(ss))
        draws.append(stat(pd.concat([by[s] for s in pick])))
    per = {s: stat(by[s]) for s in ss}
    lo, hi = np.nanpercentile(draws, [5, 95])
    return point, lo, hi, per


def fmt(r, nd=3):
    p, lo, hi, per = r
    signs = sum(np.sign(v) == np.sign(p) for v in per.values())
    return (f"{p:+.{nd}f} [{lo:+.{nd}f},{hi:+.{nd}f}] same sign {signs}/{len(per)} | "
            + " ".join(f"{v:+.{nd}f}" for v in per.values()))


# ---------------------------------------------------------------- panel
def build_panel(st, tw, lastwk, inj, wk, ros):
    teamweeks = set(map(tuple, tw[["season", "week", "team"]].values))
    pts = st.set_index(["player_id", "season", "week"]).pts
    team_of = st.sort_values("week").groupby(["player_id", "season"])

    p = wk[["season", "week", "player_id", "player", "pos", "ecr", "sd", "rank"]].copy()
    p["rpW"] = loso(p, weekly=True)
    key = lambda d, dw: pd.MultiIndex.from_arrays([d.player_id, d.season, d.week + dw])  # noqa: E731
    p["pts"] = pts.reindex(key(p, 0)).fillna(0).values
    p["played"] = pts.reindex(key(p, 0)).notna().values
    p["pts_prev"] = pts.reindex(key(p, -1)).values
    prev = p[["player_id", "season", "week", "rpW", "rank"]].assign(week=lambda d: d.week + 1)
    p = p.merge(prev.rename(columns={"rpW": "rpW_prev", "rank": "rank_prev"}),
                on=["player_id", "season", "week"], how="left")

    # season-to-date PPG (games played) through t-1 and t-2; team as of last game
    stc = st.sort_values(["player_id", "season", "week"]).copy()
    stc["cum"] = stc.groupby(["player_id", "season"]).pts.cumsum()
    stc["n"] = stc.groupby(["player_id", "season"]).cumcount() + 1
    def asof(lag):
        x = p[["player_id", "season", "week"]].assign(w=p.week - lag).sort_values("w")
        s = stc[["player_id", "season", "week", "cum", "n", "team"]].sort_values("week")
        m = pd.merge_asof(x, s.rename(columns={"week": "w"}), on="w",
                          by=["player_id", "season"], direction="backward")
        return m.set_index(["player_id", "season", "week"]).reindex(
            pd.MultiIndex.from_frame(p[["player_id", "season", "week"]]))
    a1, a2 = asof(1), asof(2)
    p["std1"] = (a1.cum / a1.n).values
    p["std2"] = (a2.cum / a2.n).values
    p["n1"] = a1.n.values
    p["team"] = a1.team.values

    # next-3 and rest-of-season means over the team's non-bye weeks
    def fwd(h):
        tot = np.zeros(len(p)); cnt = np.zeros(len(p))
        for d in range(h):
            w = p.week + d
            on = np.array([(s, ww, t) in teamweeks for s, ww, t in zip(p.season, w, p.team)])
            on &= (w <= p.season.map(lastwk).clip(upper=17)).values if h > 3 else \
                  (w <= p.season.map(lastwk)).values
            v = pts.reindex(pd.MultiIndex.from_arrays([p.player_id, p.season, w])).fillna(0).values
            tot += np.where(on, v, 0); cnt += on
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    p["pts3"] = fwd(3)
    p["ptsROS"] = fwd(17)

    st_inj = inj.set_index(["player_id", "season", "week"]).report_status
    p["status"] = st_inj.reindex(key(p, 0)).values

    r = ros[["season", "week", "player_id", "rank"]].rename(columns={"rank": "rankR"})
    p = p.merge(r, on=["season", "week", "player_id"], how="left")
    return p


# ---------------------------------------------------------------- Q1
def q1(p):
    print("\n=== Q1 streak chasing ===")
    d = p[p.week.between(W0, W1) & p.rpW_prev.notna() & p.pts_prev.notna()
          & ~p.status.isin(["Out", "Doubtful"]) & p.team.notna()].copy()
    d = d[d.apply(lambda r: r["rank"] <= TOP[r.pos], axis=1)]
    d["surp"] = d.pts_prev - d.rpW_prev
    d["surp_avg"] = d.pts_prev - d.std2
    d["y1"] = d.pts - d.rpW
    d["y3"] = d.pts3 - d.rpW
    # ROS version: implied next-3 PPG from the ROS rank, LOSO
    dr = d[d.rankR.notna()].copy()
    tmp = dr.rename(columns={"rank": "rw", "rankR": "rank"})
    dr["rpR"] = loso(tmp, "pts3").values
    dr["y1R"] = dr.pts - dr.rpR
    dr["y3R"] = dr.pts3 - dr.rpR
    dr["rpRR"] = loso(tmp.dropna(subset=["ptsROS"]), "ptsROS").reindex(tmp.index).values
    dr["yRR"] = dr.ptsROS - dr.rpRR
    print(f"n={len(d)} player-weeks (top ranks, weeks {W0}-{W1}, played t-1, not Out/Doubtful)")
    out = {}
    for label, df, y, ctl, x in [
            ("weekly  next-1", d, "y1", "rpW", "surp"),
            ("weekly  next-3", d, "y3", "rpW", "surp"),
            ("weekly  next-1 vs STD avg", d, "y1", "rpW", "surp_avg"),
            ("ROS     next-1", dr, "y1R", "rpR", "surp"),
            ("ROS     next-3", dr, "y3R", "rpR", "surp"),
            ("ROS     rest-of-season", dr, "yRR", "rpRR", "surp")]:
        for pos in POS + ["ALL"]:
            g = df if pos == "ALL" else df[df.pos == pos]
            g = g.dropna(subset=[y, ctl, x])
            stat = lambda h: ols(h[y].values, [h[x].values, h[ctl].values])[1]  # noqa: E731
            r = boot(g, stat)
            sdx = g[x].std()
            print(f"{label:28s} {pos:3s} n={len(g):5d} b={fmt(r)}  sd(surp)={sdx:.1f} "
                  f"-> 1sd adj {r[0]*sdx:+.2f} pts")
            out[(label, pos)] = (r, sdx)
    # shape: mean residual by surprise quintile (weekly next-1)
    d["q"] = d.groupby("pos").surp.transform(lambda s: pd.qcut(s, 5, labels=False))
    print("mean next-1 residual by last-week-surprise quintile (0=big miss .. 4=big hit):")
    print(d.groupby(["pos", "q"]).y1.mean().unstack().round(2))
    # per-season actionable counts: |adj| >= 1 pt using pooled per-position b
    for label, df in (("weekly  next-1", d), ("ROS     next-3", dr)):
        for pos in POS:
            b = out[(label, pos)][0][0]
            g = df[df.pos == pos]
            n = (np.abs(b * g.surp) >= 1).groupby(g.season).sum().mean()
            print(f"  {label} {pos}: player-weeks/season with |b*surprise|>=1pt: {n:.0f} "
                  f"(league-wide, top {TOP[pos]}); per manager ~{n/12:.1f}")
    return d


# ---------------------------------------------------------------- Q2
def q2(p, pre, st, tw, lastwk):
    print("\n=== Q2 anchoring / over-updating (ROS PPG) ===")
    d = p[p.week.between(W0, W1) & p.rankR.notna() & p.ptsROS.notna()].copy()
    d = d[d.apply(lambda r: r.rankR <= TOP_ROS[r.pos], axis=1)]
    # one row per player-week: the ROS rank is the object, weekly rank only for filtering
    train, test = d.season <= 2022, d.season >= 2023
    tr = d[train].rename(columns={"rank": "rw", "rankR": "rank"})
    crv_ros = fit_curve(tr, "ptsROS")
    d["rosimp"] = rank_points(crv_ros, d.pos, d.rankR)
    # preseason rank -> full-season PPG (non-bye team games weeks 1-17), fit on 2020-22
    seas = st[st.week <= 17].groupby(["player_id", "season"]).pts.sum().rename("tot").reset_index()
    team = st.groupby(["player_id", "season"]).team.agg(lambda s: s.mode().iloc[0]).reset_index()
    ng = tw[tw.week <= 17].groupby(["season", "team"]).size().rename("ng").reset_index()
    pr = pre.merge(seas, on=["player_id", "season"], how="left").merge(
        team, on=["player_id", "season"], how="left").merge(ng, on=["season", "team"], how="left")
    pr["ppg"] = pr.tot.fillna(0) / pr.ng.fillna(17)
    crv_pre = fit_curve(pr[pr.season <= 2022], "ppg")
    d = d.merge(pre[["player_id", "season", "rank"]].rename(columns={"rank": "rankP"}),
                on=["player_id", "season"], how="left")
    d["preimp"] = rank_points(crv_pre, d.pos, d.rankP.fillna(150))
    d["stdppg"] = d.std1.fillna(d.rosimp)
    train, test = d.season <= 2022, d.season >= 2023
    models = {"ROS raw": None, "ROS recal": ["rosimp"],
              "ROS+pre": ["rosimp", "preimp"], "ROS+STD": ["rosimp", "stdppg"],
              "blend": ["rosimp", "preimp", "stdppg"]}
    for name, cols in models.items():
        d[name] = d.rosimp
        if cols is None:
            continue
        for pos in POS:
            m = d.pos == pos
            b = ols(d.ptsROS[train & m].values, [d[c][train & m].values for c in cols])
            d.loc[m, name] = b[0] + sum(bi * d[c][m] for bi, c in zip(b[1:], cols))
            if name == "blend":
                print(f"  blend weights {pos}: const {b[0]:+.2f} ros {b[1]:+.2f} "
                      f"pre {b[2]:+.2f} std {b[3]:+.2f}")
    t = d[test].copy()

    def mae_r(g, col):
        return np.mean(np.abs(g.ptsROS - g[col])), np.corrcoef(g.ptsROS, g[col])[0, 1]
    grp = t.groupby(["season", "week", "pos"])
    sp = pd.DataFrame({name: grp.apply(lambda h, c=name: spearmanr(h.ptsROS, h[c])[0])
                       for name in models}).reset_index()
    print(f"test n={len(t)} player-weeks 2023-25")
    for name in models:
        m, r = mae_r(t, name)
        print(f"  {name:10s} MAE {m:.3f} r {r:.3f} within-pos-week spearman {sp[name].mean():.4f}")
    for name in ["ROS+pre", "ROS+STD", "blend"]:
        for k, lab in enumerate(["dMAE", "dr"]):
            stat = lambda g, k=k: mae_r(g, name)[k] - mae_r(g, "ROS recal")[k]  # noqa: E731
            print(f"  {name} - ROS recal {lab}: {fmt(boot(t, stat), 4)}")
        stat = lambda g: (g[name] - g["ROS recal"]).mean()  # noqa: E731
        print(f"  {name} - ROS recal dSpearman: {fmt(boot(sp, stat), 4)}")
    for pos in POS:
        g = t[t.pos == pos]
        h = sp[sp.pos == pos]
        print(f"  {pos}: blend MAE {mae_r(g, 'blend')[0]:.3f} vs {mae_r(g, 'ROS recal')[0]:.3f}; "
              f"spearman {h.blend.mean():.3f} vs {h['ROS recal'].mean():.3f}")
    # size: how often the blend moves a player's ROS value by >=1 PPG
    t["gap"] = t.blend - t["ROS recal"]
    print("  test player-weeks/season with |blend - ROS| >= 1 PPG:",
          (t.gap.abs() >= 1).groupby(t.season).sum().to_dict())


# ---------------------------------------------------------------- Q3
def q3(p, st, inj, sn, tw):
    print("\n=== Q3 staleness after a starter is ruled Out ===")
    s = st.merge(sn, on=["player_id", "season", "week"], how="left")
    s["opp"] = np.where(s.position == "QB", s.attempts.fillna(0) + s.carries.fillna(0),
                        s.carries.fillna(0) + s.targets.fillna(0))
    s = s[s.position.isin(["QB", "RB", "TE"])]
    tot = s.groupby(["season", "week", "team", "position"]).opp.transform("sum")
    s["share"] = s.opp / tot.replace(0, np.nan)
    played = set(map(tuple, s[["player_id", "season", "week"]].values))
    wk = p.set_index(["player_id", "season", "week"])
    out = set(map(tuple, inj[inj.report_status == "Out"][["player_id", "season", "week"]].values))
    teamweeks = set(map(tuple, tw[["season", "week", "team"]].values))
    scraped = set(map(tuple, p[["season", "week"]].drop_duplicates().values))
    rows = []
    for (season, team, pos), g in s.groupby(["season", "team", "position"]):
        for t in range(W0, W1 + 1):
            prior = g[g.week.between(t - 3, t - 1)]
            if prior.empty:
                continue
            lead = prior.groupby("player_id").opp.sum().sort_values(ascending=False)
            starter = lead.index[0]
            if prior.groupby("player_id").share.mean().get(starter, 0) < 0.5:
                continue
            if (starter, season, t) not in out or (starter, season, t - 1) in out \
                    or (starter, season, t - 1) not in played or (starter, season, t) in played:
                continue
            now = g[(g.week == t) & (g.player_id != starter)]
            if now.empty:
                continue
            repl = now.sort_values("opp", ascending=False).player_id.iloc[0]   # realized
            pre2 = lead.index[1] if len(lead) > 1 else None                      # known pregame
            for kind, pid in (("realized", repl), ("pregame#2", pre2)):
                if pid is None:
                    continue
                for k in range(-1, 4):
                    w = t + k
                    if k >= 1 and (starter, season, w) in played:
                        break
                    if (season, w, team) not in teamweeks or (season, w) not in scraped:
                        continue    # bye, or no consensus scrape that week
                    gw = g[(g.week == w) & (g.player_id == pid)]
                    e = wk.loc[(pid, season, w)] if (pid, season, w) in wk.index else None
                    rank = e["rank"] if e is not None else 151
                    rows.append(dict(season=season, team=team, pos=pos, kind=kind, k=k,
                                     t=t, pid=pid, rank=rank,
                                     rp=float(rank_points_one(p, pos, rank, season)),
                                     pts=gw.pts.sum() if len(gw) else 0.0,
                                     share=gw.share.sum() if len(gw) else 0.0,
                                     snap=gw.offense_pct.sum() if len(gw) else 0.0,
                                     team_bye=False))
    e = pd.DataFrame(rows)
    e["resid"] = e.pts - e.rp
    e["resid_ranked"] = e.resid.where(e["rank"] <= 150)
    print("'realized' = the non-starter who got the most work in week t (chosen with hindsight,"
          " so its residual is biased up);\n'pregame#2' = the #2 by opportunities over the prior"
          " 3 games (known before kickoff) -- the honest test.")
    for pos in ["QB", "RB", "TE"]:
        b = p[(p.pos == pos) & p.week.between(W0, W1) & p["rank"].between(20, 60)]
        print(f"   baseline {pos} ranks 20-60, all weeks: resid "
              f"{fmt(boot(b, lambda x: (x.pts - x.rpW).mean()), 2)}")
    for kind in ["realized", "pregame#2"]:
        for pos in ["QB", "RB", "TE"]:
            g = e[(e.kind == kind) & (e.pos == pos)]
            if g.empty:
                continue
            n_ev = g[g.k == 0].shape[0]
            print(f"-- {pos} {kind}: {n_ev} events")
            print(g.groupby("k").agg(n=("pts", "size"), rank_med=("rank", "median"),
                                     unranked=("rank", lambda r: (r > 150).mean()),
                                     snap=("snap", "mean"), share=("share", "mean"),
                                     pts=("pts", "mean"), implied=("rp", "mean"),
                                     resid=("resid", "mean"),
                                     resid_ranked=("resid_ranked", "mean")).round(2))
            for k in (0, 1):
                h = g[g.k == k]
                if h.season.nunique() >= 3:
                    print(f"   resid k={k}: {fmt(boot(h, lambda x: x.resid.mean()), 2)}")
    return e


_CURVES = {}


def rank_points_one(p, pos, rank, season):
    if season not in _CURVES:
        _CURVES[season] = weekly_curve([y for y in SEASONS if y != season])
    return rank_points(_CURVES[season], [pos], [rank])[0]


# ---------------------------------------------------------------- Q4
def q4(p, rook, d1):
    print("\n=== Q4 other cheap checks ===")
    d = p[p.week.between(W0, W1) & ~p.status.isin(["Out", "Doubtful"]) & p.team.notna()].copy()
    d = d[d.apply(lambda r: r["rank"] <= TOP[r.pos], axis=1)]
    d = d.merge(rook, on="player_id", how="left")
    d["rookie"] = d.draft_year == d.season
    d["y1"] = d.pts - d.rpW
    d["y3"] = d.pts3 - d.rpW
    d["wbin"] = pd.cut(d.week, [4, 8, 12, 17], labels=["5-8", "9-12", "13-17"])
    print("rookie minus veteran next-1 residual (pts/week), same rank-implied points:")
    for wb in ["5-8", "9-12", "13-17", None]:
        g = d if wb is None else d[d.wbin == wb]
        stat = lambda h: h[h.rookie].y1.mean() - h[~h.rookie].y1.mean()  # noqa: E731
        print(f"  weeks {wb or 'all':5s} rookies n={g.rookie.sum():4d}: {fmt(boot(g, stat), 2)}")
    for pos in ["RB", "WR", "TE", "QB"]:
        g = d[d.pos == pos]
        stat = lambda h: h[h.rookie].y1.mean() - h[~h.rookie].y1.mean()  # noqa: E731
        print(f"  {pos} rookies n={g.rookie.sum():4d}: {fmt(boot(g, stat), 2)}")
    print("rookie residual by position x weeks 5-8 / 9-17 (pooled means):")
    print(d[d.rookie].groupby(["pos", d.week >= 9]).y1.agg(["mean", "size"]).round(2))

    print("calibration slope of actual on rank-implied (1 = right spread), within top ranks:")
    for pos in POS:
        g = d[d.pos == pos]
        stat = lambda h: ols(h.pts.values, [h.rpW.values])[1]  # noqa: E731
        sp = g.groupby(["season", "week"]).apply(lambda h: spearmanr(h.pts, h.rpW)[0]).mean()
        print(f"  {pos}: slope {fmt(boot(g, stat), 2)}  within-week spearman {sp:.3f}")

    # weekly vs ROS disagreement: does the weekly matchup tilt overreact?
    dr = d[d.rankR.notna()].copy()
    tmp = dr.rename(columns={"rank": "rw", "rankR": "rank"})
    dr["rpR1"] = loso(tmp, "pts").values
    dr["tilt"] = dr.rpW - dr.rpR1
    print("weekly-vs-ROS tilt: b of next-1 residual on (weekly implied - ROS implied), ctl rpW")
    for pos in POS + ["ALL"]:
        g = dr if pos == "ALL" else dr[dr.pos == pos]
        stat = lambda h: ols(h.y1.values, [h.tilt.values, h.rpW.values])[1]  # noqa: E731
        print(f"  {pos:3s} sd(tilt)={g.tilt.std():.2f}: {fmt(boot(g, stat))}")
    # expert spread relative to the usual spread at that rank (sd grows with rank)
    d["rbin"] = (d["rank"] // 4).astype(int)
    d["sdr"] = d.sd - d.groupby(["pos", "rbin"]).sd.transform("median")
    for lab, g0 in (("all", d), ("no Questionable", d[d.status != "Questionable"])):
        print(f"expert spread ({lab}): b of next-1 residual on sd minus median sd at rank, ctl rpW")
        for pos in POS:
            g = g0[g0.pos == pos].dropna(subset=["sdr"])
            stat = lambda h: ols(h.y1.values, [h.sdr.values, h.rpW.values])[1]  # noqa: E731
            print(f"  {pos}: sd(sdr)={g.sdr.std():.2f} {fmt(boot(g, stat))}")

    print("Questionable tag: next-1 residual (pts - rank-implied), Questionable vs untagged")
    for pos in POS:
        g = d[d.pos == pos]
        stat = lambda h: h[h.status == "Questionable"].y1.mean() - h[h.status.isna()].y1.mean()  # noqa: E731
        q = g[g.status == "Questionable"]
        print(f"  {pos}: n_Q={len(q)} ({len(q)/g.season.nunique():.0f}/season), sat out "
              f"{(~q.played).mean():.0%}, Q-if-played resid {q[q.played].y1.mean():+.2f}; "
              f"Q minus untagged {fmt(boot(g, stat), 2)}")
        stat = lambda h: h[(h.status == "Questionable") & h.played].y1.mean() - h[h.status.isna()].y1.mean()  # noqa: E731
        print(f"      Q-and-active minus untagged {fmt(boot(g, stat), 2)}")
    qd = d.assign(adj=np.where(d.status == "Questionable", -1e6, 0.0))
    for lab, df in (("before inactives are known", qd),
                    ("after inactives (drop Q who sat)", qd[~((qd.status == "Questionable") & ~qd.played)])):
        for N in (2, 4, 8):
            pr = decision_pairs(df, "rank", "rpW", "pts", N)
            per = pr.groupby("season").size()
            print(f"  rule 'prefer untagged over Questionable within {N} ranks', {lab}: "
                  f"flips/season {per.mean():.0f} (~{per.mean()/12:.1f}/manager), "
                  f"gain per flip {fmt(boot(pr, lambda h: h.gain.mean()), 2)}")

    # decision tests: in near-ties (within N ranks), follow the adjusted order.
    # coefficients are fit on the other seasons, so each season is out of sample.
    d = d.merge(dr[["season", "week", "player_id", "tilt"]], how="left",
                on=["season", "week", "player_id"])
    d["surp"] = d.pts_prev - d.rpW_prev
    dros = d[d.rankR.notna()].copy()
    tmp = dros.rename(columns={"rank": "rw", "rankR": "rank"})
    dros["rpR"] = loso(tmp, "pts3").values
    dros["y3R"] = dros.pts3 - dros.rpR
    rules = [("low expert spread (weekly)", d, "rank", "rpW", "y1", "sdr", "pts"),
             ("RB weekly-vs-ROS tilt (weekly)", d, "rank", "rpW", "y1", "tilt", "pts"),
             ("last-week surprise (weekly)", d, "rank", "rpW", "y1", "surp", "pts"),
             ("last-week surprise (ROS ranks, 3 wk)", dros, "rankR", "rpR", "y3R", "surp", "pts3")]
    for name, df, rk, ctl, y, x, outcome in rules:
        df = df.dropna(subset=[x, y, ctl, outcome]).copy()
        if "RB" in name and "tilt" in name:
            df = df[df.pos == "RB"]
        df["adj"] = 0.0
        for s in SEASONS:
            for pos in POS:
                tr = (df.season != s) & (df.pos == pos)
                if tr.sum() < 50:
                    continue
                b = ols(df[y][tr].values, [df[x][tr].values, df[ctl][tr].values])[1]
                m = (df.season == s) & (df.pos == pos)
                df.loc[m, "adj"] = b * df[x][m]
        for N in (2, 4):
            base = decision_pairs(df, rk, ctl, outcome, N,
                                  swap_all=True)
            print(f"  [reference] swapping ANY pair within {N} ranks gains "
                  f"{base.gain.mean():+.2f} pts per swap") if name.startswith("low") else None
            pr = decision_pairs(df, rk, ctl, outcome, N)
            if pr.empty:
                continue
            per = pr.groupby("season").agg(flips=("gain", "size"), gain=("gain", "mean"))
            r = boot(pr, lambda h: h.gain.mean())
            print(f"  {name} N={N}: flips/season {per.flips.mean():.0f} league-wide "
                  f"(~{per.flips.mean()/12:.1f}/manager), gain per flip {fmt(r, 2)}")


def decision_pairs(df, rk, ctl, outcome, N, swap_all=False):
    a = df[["season", "week", "pos", "player_id", rk, ctl, "adj", outcome]]
    m = a.merge(a, on=["season", "week", "pos"], suffixes=("_a", "_b"))
    m = m[(m[f"{rk}_a"] < m[f"{rk}_b"]) & (m[f"{rk}_b"] - m[f"{rk}_a"] <= N)]
    if not swap_all:
        m = m[(m[f"{ctl}_b"] + m.adj_b) > (m[f"{ctl}_a"] + m.adj_a)]
    return m.assign(gain=m[f"{outcome}_b"] - m[f"{outcome}_a"])[["season", "gain"]]


def main():
    pd.set_option("display.width", 160)
    st, tw, lastwk, inj, sn, rook, wk, ros, pre = load()
    p = build_panel(st, tw, lastwk, inj, wk, ros)
    d1 = q1(p)
    q2(p, pre, st, tw, lastwk)
    q3(p, st, inj, sn, tw)
    q4(p, rook, d1)


if __name__ == "__main__":
    main()
