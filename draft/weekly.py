"""Next-game fantasy projections for QB/RB/WR/TE, refreshed every week of the season.

    .venv/bin/python draft/weekly.py 2026          # project each player's next game
    .venv/bin/python draft/weekly.py --eval        # train <= 2020, score 2021-25

A player's preseason projection is the prior, and the games he has played this season
update it - the same partial pooling as project_season.py, applied week by week, so a
hot September moves him only as far as the evidence supports. The pooled rate is then
fitted to the game in front of him: the betting market's implied team total, how the
opponent's defence has done against his position, and home field.

Projections are conditional on the player suiting up. Injury designations are applied
separately, as a probability of playing estimated from past seasons' reports.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, ".")
sys.path.insert(0, "draft")
from build_data import norm_team

POSITIONS = ["QB", "RB", "WR", "TE"]
FIRST = 2009                 # first season with a preseason projection built from history
DEF_HALFLIFE = 8             # games; a defence's recent games count most
DEF_PRIOR = 6                # games of league-average prior on a defence's rating
DEF_CARRY = 0.5              # weight last season's rating keeps into a new season
PLAY_DEFAULT = {"Out": 0.0, "Doubtful": 0.2, "Questionable": 0.8}
POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}   # startable players per position-week


# ---------------------------------------------------------------- data

def load_weeks(first=FIRST):
    files = [f for f in sorted(glob.glob("data/stats/w*.parquet"))
             if int(os.path.basename(f)[1:5]) >= first]
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[(d.season_type == "REG") & d.position.isin(POSITIONS)].copy()
    d["team"], d["opponent_team"] = norm_team(d.team), norm_team(d.opponent_team)
    d["ppr"] = d.fantasy_points_ppr
    d["touches"] = d.attempts.fillna(0) + d.carries.fillna(0) + d.targets.fillna(0)
    return d[["player_id", "player_display_name", "position", "season", "week", "team",
              "opponent_team", "ppr", "touches"]]


def load_schedule():
    """One row per team-game with the market's implied points for that team."""
    g = pd.read_csv("data/games.csv")
    g = g[g.game_type == "REG"]
    neutral = g.location.eq("Neutral")
    rows = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        rows.append(pd.DataFrame({
            "season": g.season, "week": g.week,
            "team": norm_team(g[f"{side}_team"]), "opponent": norm_team(g[f"{other}_team"]),
            # nflverse spread_line is positive when the home side is favoured
            "implied": g.total_line / 2 + sign * g.spread_line / 2,
            "home": np.where(neutral, 0.0, 1.0 if side == "home" else 0.0),
            "played": g.result.notna(),
        }))
    return pd.concat(rows, ignore_index=True)


def ytd(d):
    """This season's games so far, strictly before the current one."""
    d = d.sort_values(["player_id", "season", "week"]).copy()
    g = d.groupby(["player_id", "season"])
    d["n_prev"] = g.cumcount()
    d["ytd_ppg"] = (g.ppr.cumsum() - d.ppr) / d.n_prev.replace(0, np.nan)
    return d


# ---------------------------------------------------------------- the prior

def rookie_model(s, train):
    """Rookies have no history to pool, so their prior comes from draft slot alone."""
    fits = {}
    for pos in POSITIONS:
        t = s[train & (s.position == pos) & (s.exp == 0) & (s.games >= 1)]
        fits[pos] = sm.WLS(t.ppr_pg, sm.add_constant(-np.log(t.draft_pick)),
                           weights=t.games).fit()
    return fits


def rookie_prior(fits, position, draft_pick):
    x = -np.log(pd.Series(draft_pick).fillna(260).clip(1, 260).values)
    return np.array([fits[p].params.iloc[0] + fits[p].params.iloc[1] * v
                     for p, v in zip(position, x)])


def preseason(season, rookies):
    """Preseason points per game for everyone who could play in `season`.

    Past seasons come from the stored projections (built from strictly earlier
    seasons); an upcoming season is projected fresh and cached, since it never changes
    once the season starts.
    """
    path = f"data/preseason_{season}.parquet"
    if not os.path.exists(path):
        import board
        up = board.project_upcoming(season)
        up[["player_id", "position", "proj_ppg", "proj_games", "own_ppg"]].to_parquet(path)
    up = pd.read_parquet(path)
    return up.set_index("player_id")


def attach_prior(d, s, fits):
    """Preseason ppg for each historical player-season; rookies from draft slot."""
    p = s[["player_id", "season", "proj_ppg", "own_ppg", "draft_pick"]]
    d = d.merge(p, on=["player_id", "season"], how="left")
    rook = d.own_ppg.isna() | d.proj_ppg.isna()
    d["pre"] = np.where(rook, rookie_prior(fits, d.position, d.draft_pick), d.proj_ppg)
    d["rookie_prior"] = rook
    return d.drop(columns=["proj_ppg", "own_ppg", "draft_pick"])


# ---------------------------------------------------------------- the update

def pooled(d, ks):
    """Blend the preseason rate with this season's games: k games of prior."""
    k = d.position.map(ks)
    n = d.n_prev
    return np.where(n > 0, (k * d.pre + n * d.ytd_ppg.fillna(0)) / (k + n), d.pre)


def fit_k(d):
    ks = {}
    for pos in POSITIONS:
        t = d[d.position == pos]
        grid = np.arange(1, 31)
        err = [np.mean((t.ppr - pooled(t, {pos: k})) ** 2) for k in grid]
        ks[pos] = int(grid[int(np.argmin(err))])
    return ks


class Defence:
    """Each defence's running points allowed above expectation, per position.

    Read before each game, so no game informs its own projection. A game's value is the
    sum over every player at the position of (points scored - pooled projection), so it
    measures the defence rather than the quality of the players it happened to face.
    """

    def __init__(self, d):
        g = (d.assign(resid=d.ppr - d.post)
             .groupby(["opponent_team", "position", "season", "week"], as_index=False)
             .resid.sum()
             .sort_values(["opponent_team", "position", "season", "week"]))
        decay = 0.5 ** (1 / DEF_HALFLIFE)
        before = np.empty(len(g))
        self.state = {}
        key = None
        for i, r in enumerate(g.itertuples()):
            k = (r.opponent_team, r.position)
            if k != key:
                num = den = 0.0
                key, season = k, r.season
            if r.season != season:
                num, den, season = num * DEF_CARRY, den * DEF_CARRY, r.season
            before[i] = num / (den + DEF_PRIOR)
            num, den = num * decay + r.resid, den * decay + 1
            self.state[k] = (num, den, season)
        g["opp_def"] = before
        self.before = g[["opponent_team", "position", "season", "week", "opp_def"]]

    def now(self, team, position, season):
        num, den, last = self.state.get((team, position), (0.0, 0.0, season))
        if last < season:
            num, den = num * DEF_CARRY, den * DEF_CARRY
        return num / (den + DEF_PRIOR)


# ---------------------------------------------------------------- the game

def design(d):
    base = d.post
    return pd.DataFrame({
        "post": base,
        # A player's share of a bigger expected team score is worth more.
        "post_x_implied": base * (d.implied - 22) / 10,
        "opp_def": d.opp_def,
        "opp_x_post": d.opp_def * base / 10,
        "home": d.home - 0.5,
    }, index=d.index)


def fit_game(d):
    fits, sd = {}, {}
    for pos in POSITIONS:
        t = d[d.position == pos]
        m = sm.OLS(t.ppr, sm.add_constant(design(t))).fit()
        fits[pos] = m
        # Spread grows with the level: a 20-point player has more room to miss.
        r = np.abs(t.ppr - m.fittedvalues)
        sd[pos] = sm.OLS(r, sm.add_constant(m.fittedvalues)).fit().params.values * np.sqrt(np.pi / 2)
    return fits, sd


def predict(fits, sd, d):
    out = pd.Series(np.nan, index=d.index)
    spread = pd.Series(np.nan, index=d.index)
    for pos in POSITIONS:
        x = d[d.position == pos]
        if len(x):
            p = fits[pos].predict(sm.add_constant(design(x), has_constant="add")).clip(lower=0)
            out.loc[x.index] = p
            spread.loc[x.index] = sd[pos][0] + sd[pos][1] * p
    return out, spread


# ---------------------------------------------------------------- training table

def history(season_lt):
    """Every player-game before `season_lt`, with everything knowable before kickoff."""
    s = pd.read_parquet("data/draft_projected.parquet")
    rfits = rookie_model(s, s.season < season_lt)
    d = ytd(load_weeks())
    d = attach_prior(d, s, rfits)
    sch = load_schedule()
    d = d.merge(sch[["season", "week", "team", "implied", "home"]],
                on=["season", "week", "team"], how="left")
    d["implied"] = d.implied.fillna(22.0)
    d["home"] = d.home.fillna(0.5)
    return d, s, rfits, sch


def train(d, mask):
    ks = fit_k(d[mask])
    d["post"] = pooled(d, ks)
    dfn = Defence(d)
    d = d.drop(columns=[c for c in ("opp_def",) if c in d]).merge(
        dfn.before, on=["opponent_team", "position", "season", "week"], how="left")
    fits, sd = fit_game(d[mask])
    return d, ks, dfn, fits, sd


# ---------------------------------------------------------------- injuries

def play_probs(before):
    """P(plays) for each designation, from past seasons' reports joined to box scores."""
    files = [f for f in glob.glob("data/injuries_*.parquet")
             if int(f[-12:-8]) < before]
    if not files:
        return PLAY_DEFAULT
    inj = pd.concat([pd.read_parquet(f) for f in files])
    inj = inj[inj.position.isin(POSITIONS) & inj.report_status.isin(PLAY_DEFAULT)]
    played = load_weeks(first=inj.season.min())[["player_id", "season", "week"]]
    played["played"] = 1
    j = inj.merge(played, left_on=["gsis_id", "season", "week"],
                  right_on=["player_id", "season", "week"], how="left")
    return j.groupby("report_status").played.apply(lambda x: x.notna().mean()).to_dict()


def injury_status(season):
    path = f"data/injuries_{season}.parquet"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player_id", "week", "status"])
    inj = pd.read_parquet(path)
    inj = inj[inj.report_status.notna()].drop_duplicates(["gsis_id", "week"], keep="last")
    return inj.rename(columns={"gsis_id": "player_id", "report_status": "status"})[
        ["player_id", "week", "status"]]


# ---------------------------------------------------------------- upcoming games

def remaining_games(season, sch):
    """Every unplayed game for every team, with implied totals where the market has
    posted them and each team's average implied total otherwise."""
    g = sch[sch.season == season].copy()
    team_avg = g.groupby("team").implied.mean()
    g["implied"] = g.implied.fillna(g.team.map(team_avg)).fillna(22.0)
    return g[~g.played].sort_values(["team", "week"])


def roster(season):
    """Who could play: active skill players, on the team they last played for."""
    pl = pd.read_parquet("data/players.parquet")
    pl = pl[pl.position.isin(POSITIONS) & (pl.status == "ACT") & pl.latest_team.notna()]
    # Long-retired players can still carry an ACT status; require a recent season.
    recent = (pl.last_season >= season - 2) | (pl.rookie_season == season)
    pl = pl[recent].drop_duplicates("gsis_id")
    r = pd.DataFrame({"player_id": pl.gsis_id, "player_display_name": pl.display_name,
                      "position": pl.position, "team": norm_team(pl.latest_team),
                      "draft_pick": pl.draft_pick})
    return r


def project(season):
    """Projections for every remaining game of `season`, one row per player-game."""
    d, s, rfits, sch = history(season)
    d, ks, dfn, fits, sd = train(d, (d.season < season).values)

    r = roster(season)
    cur = d[d.season == season].sort_values("week")
    last_team = cur.groupby("player_id").team.last()          # box score beats roster
    r["team"] = r.player_id.map(last_team).fillna(r.team)
    so_far = cur.groupby("player_id").agg(n_prev=("ppr", "size"), ytd_ppg=("ppr", "mean"))
    r = r.join(so_far, on="player_id")
    r["n_prev"] = r.n_prev.fillna(0).astype(int)

    pre = preseason(season, rfits)
    r["pre"] = r.player_id.map(pre.proj_ppg)
    rook = r.pre.isna()
    r.loc[rook, "pre"] = rookie_prior(rfits, r.position[rook], r.draft_pick[rook])
    r["rookie_prior"] = rook
    r["avail"] = (r.player_id.map(pre.proj_games) / 17).fillna(0.85).clip(0.3, 1.0)
    r["post"] = pooled(r, ks)

    games = remaining_games(season, sch)
    x = r.merge(games[["team", "week", "opponent", "implied", "home"]], on="team")
    x["opp_def"] = [dfn.now(o, p, season) for o, p in zip(x.opponent, x.position)]
    x["season"] = season
    x["proj_ppr"], x["proj_sd"] = predict(fits, sd, x)

    probs = play_probs(season)
    st = injury_status(season)
    x = x.merge(st, on=["player_id", "week"], how="left")
    x["p_play"] = x.status.map(probs).fillna(1.0)
    x.attrs.update(ks=ks, probs=probs)
    return x


# ---------------------------------------------------------------- validation

def evaluate():
    d, s, rfits, sch = history(2026)
    d = d[d.season <= 2025].reset_index(drop=True)
    mask = d.season <= 2020
    d, ks, dfn, fits, sd = train(d, mask.values)
    d["proj"], d["proj_sd"] = predict(fits, sd, d)
    print("prior games in the in-season blend (k):", ks)
    for pos in POSITIONS:
        print(f"  {pos}: " + "  ".join(f"{n} {v:+.3f}" for n, v in fits[pos].params.items()))

    te = d[d.season > 2020].copy()
    te["season_avg"] = te.ytd_ppg.fillna(te.pre)
    rows = [("full model", te.proj), ("pooled rate only", te.post),
            ("season-to-date average", te.season_avg), ("preseason projection", te.pre)]
    print(f"\nholdout 2021-25, n={len(te)} player-games")
    rel = te.post >= 8                         # players anyone would think of starting
    for name, p in rows:
        r = te.ppr - p
        # Start/sit is a ranking problem: rank correlation within position and week.
        rk = te.assign(p=p).groupby(["season", "week", "position"]).apply(
            lambda g: g.ppr.corr(g.p, method="spearman"), include_groups=False).mean()
        print(f"  {name:24s} RMSE {np.sqrt(np.mean(r ** 2)):5.2f}  MAE {np.mean(np.abs(r)):5.2f}"
              f"  corr {np.corrcoef(te.ppr, p)[0, 1]:.3f}  within-week rank {rk:.3f}"
              f"  | RMSE on starters {np.sqrt(np.mean(r[rel] ** 2)):5.2f}")

    # Scored the way published projection-accuracy studies score experts: only the
    # fantasy-relevant pool (top N projected or top N actual each position-week).
    print("\nrelevant pool only (top-N projected or actual per position-week):")
    for pos, n in POOL.items():
        t = te[te.position == pos]
        top = lambda c: t.groupby(["season", "week"])[c].rank(ascending=False, method="first") <= n
        t = t[top("proj") | top("ppr")]
        for name, col in (("model", "proj"), ("season avg", "season_avg"), ("preseason", "pre")):
            slope = np.polyfit(t[col], t.ppr, 1)[0]
            print(f"  {pos} {name:10s} MAE {np.mean(np.abs(t.ppr - t[col])):5.2f}"
                  f"  r {t.ppr.corr(t[col]):.3f}  bias {np.mean(t[col] - t.ppr):+5.2f}"
                  f"  calibration slope {slope:.2f}   (n={len(t)})")

    z = (te.ppr - te.proj) / te.proj_sd
    print(f"\ncalibration: {np.mean(np.abs(z) < 1):.0%} of games within one sd (normal: 68%)")
    for wk, g in te.groupby(pd.cut(te.n_prev, [-1, 0, 3, 8, 20])):
        print(f"  after {str(wk):8s} games: RMSE model {np.sqrt(np.mean((g.ppr - g.proj) ** 2)):5.2f}"
              f"  vs season avg {np.sqrt(np.mean((g.ppr - g.season_avg) ** 2)):5.2f}  (n={len(g)})")
    print("\nP(plays) by designation:", {k: round(v, 2) for k, v in play_probs(2026).items()})


def main():
    if "--eval" in sys.argv:
        return evaluate()
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    x = project(season)
    nxt = x.sort_values("week").groupby("player_id").head(1).copy()
    nxt["exp_ppr"] = nxt.proj_ppr * nxt.p_play
    cols = ["player_id", "player_display_name", "position", "team", "week", "opponent",
            "home", "proj_ppr", "proj_sd", "status", "p_play", "exp_ppr", "pre", "ytd_ppg",
            "n_prev", "implied", "opp_def"]
    nxt = nxt[cols].sort_values("proj_ppr", ascending=False)
    nxt.to_parquet(f"data/weekly_{season}.parquet", index=False)

    print(f"=== {season} next-game projections, PPR (if he plays) ===")
    print(f"in-season blend weights (games of prior): {x.attrs['ks']}")
    for pos in POSITIONS:
        t = nxt[nxt.position == pos].head(12 if pos in ("RB", "WR") else 8)
        print(f"\n{pos}")
        for r in t.itertuples():
            flag = f"  [{r.status}]" if isinstance(r.status, str) else ""
            at = "vs" if r.home == 1 else "@ "
            print(f"  {r.player_display_name:24s} {r.team:3s} wk{r.week:<2} {at} {r.opponent:3s}"
                  f"  {r.proj_ppr:5.1f} ± {r.proj_sd:4.1f}{flag}")


if __name__ == "__main__":
    main()
