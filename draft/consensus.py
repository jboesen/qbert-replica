"""Expert consensus rankings (FantasyPros ECR), mapped to nflverse player ids.

The other managers in a league draft, start and trade off consensus. A model earns
nothing by agreeing with it; its edge is only where it disagrees and is right. So the
evaluations measure against a consensus-following manager, and this is that manager's
information, as it stood at the time:

  - preseason:  PPR positional ranks from the last snapshot before the opener
  - weekly:     PPR positional ranks published before each week's games
  - ros:        rest-of-season PPR positional ranks, same timing

Source: the dynastyprocess archive of FantasyPros scrapes (2019 on).

    .venv/bin/python draft/consensus.py
"""
import io

import numpy as np
import pandas as pd
import requests

ECR = "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.parquet"
IDS = "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv"
POS = ["QB", "RB", "WR", "TE"]
KEEP = ["season", "player_id", "player", "pos", "ecr", "sd", "best", "worst"]


def fetch():
    ecr = pd.read_parquet(io.BytesIO(requests.get(ECR, timeout=300).content))
    ids = pd.read_csv(io.BytesIO(requests.get(IDS, timeout=300).content))
    ids = ids[ids.fantasypros_id.notna() & ids.gsis_id.notna()]
    ids = ids.assign(fp=ids.fantasypros_id.astype(int).astype(str)).drop_duplicates("fp")
    ecr = ecr[ecr.pos.isin(POS)].assign(fp=ecr.id.astype(str))
    ecr = ecr.merge(ids[["fp", "gsis_id"]], on="fp", how="left").rename(
        columns={"gsis_id": "player_id"})
    ecr["date"] = pd.to_datetime(ecr.scrape_date)
    return ecr[ecr.player_id.notna()]


def week_of(dates, games):
    """The week each scrape can inform: the first week whose Sunday slate is strictly
    after the scrape date. A scrape on the Sunday itself, or on Monday after the
    Sunday games, is not pregame information for that week. (Thursday-night players
    are the residual: their latest usable ranks are the previous week's.)"""
    g = games.assign(gameday=pd.to_datetime(games.gameday))
    sun = g[g.weekday == "Sunday"].groupby(["season", "week"]).gameday.min().reset_index()
    out = []
    for d in dates:
        later = sun[sun.gameday > d]
        out.append((later.season.iloc[0], later.week.iloc[0]) if len(later) else (np.nan, np.nan))
    return pd.DataFrame(out, columns=["season", "week"], index=dates)


def positional(ecr, kind):
    """Rows from one family of positional pages: 'draft', 'weekly' or 'ros'."""
    page = ecr.fp_page.str
    if kind == "overall":           # the cross-position board a consensus drafter uses
        return ecr[(ecr.ecr_type == "ro") & page.contains("ppr-cheatsheets")]
    if kind == "draft":
        m = (ecr.ecr_type == "rp") & page.contains("cheatsheets") & ~page.contains("ros")
    elif kind == "ros":
        m = (ecr.ecr_type == "rp") & page.contains("ros-")
    else:
        m = ecr.ecr_type == "wp"
    # PPR pages for the flex positions; quarterbacks have a single page.
    m &= page.contains("ppr") | (ecr.pos == "QB")
    return ecr[m]


def weekly_curve(seasons=range(2020, 2026)):
    """Weekly consensus rank -> the mean PPR players at that rank scored that week, per
    position, smoothed and non-increasing. It puts ranks from different positions on
    one scale, so a WR2 and an RB3 can be compared for the flex."""
    w = pd.read_parquet("data/ecr_weekly.parquet")
    w = w[w.season.isin(seasons)]
    st = pd.concat([pd.read_parquet(f"data/stats/w{y}.parquet") for y in seasons])
    st = st[st.season_type == "REG"][["player_id", "season", "week", "fantasy_points_ppr"]]
    m = w.merge(st, on=["player_id", "season", "week"], how="left")
    m["fantasy_points_ppr"] = m.fantasy_points_ppr.fillna(0.0)
    m["rank"] = m.groupby(["season", "week", "pos"]).ecr.rank(method="first")
    out = {}
    for p, g in m.groupby("pos"):
        by = g.groupby("rank").fantasy_points_ppr.mean().reindex(range(1, 151))
        by = by.interpolate().ffill().bfill().rolling(5, center=True, min_periods=1).mean()
        out[p] = np.minimum.accumulate(by.values)
    return out


def rank_points(crv, pos, rank):
    r = np.clip(np.asarray(rank, float), 1, 150).astype(int) - 1
    return np.array([crv[p][i] for p, i in zip(pos, r)])


def latest(kind, season):
    """The most recent 'weekly' or 'ros' ranks for a season: (week, rows)."""
    x = pd.read_parquet(f"data/ecr_{kind}.parquet")
    x = x[x.season == season]
    if not len(x):
        raise ValueError(f"no {kind} consensus for {season}; run draft/consensus.py")
    wk = int(x.week.max())
    x = x[x.week == wk].copy()
    x["rank"] = x.groupby("pos").ecr.rank(method="first")
    return wk, x


def main():
    ecr = fetch()
    g = pd.read_csv("data/games.csv")
    g = g[g.game_type == "REG"]

    openers = pd.to_datetime(g.groupby("season").gameday.min())
    for kind, path in (("draft", "data/ecr_preseason.parquet"),
                       ("overall", "data/ecr_preseason_overall.parquet")):
        pre = positional(ecr, kind)
        rows = []
        for season, first in openers.items():
            snap = pre[(pre.date < first) & (pre.date > first - pd.Timedelta(days=60))]
            if len(snap):
                rows.append(snap[snap.date == snap.date.max()].assign(season=season))
        pre = pd.concat(rows).drop_duplicates(["season", "player_id"])
        pre[KEEP].to_parquet(path)
        print(f"preseason {kind}:", pre.groupby("season").size().to_dict())

    for kind in ("weekly", "ros"):
        x = positional(ecr, kind)
        wk = week_of(sorted(x.date.unique()), g)
        x = x.join(wk, on="date")
        x = x[x.week.notna()].sort_values("date").drop_duplicates(
            ["season", "week", "player_id"], keep="last")
        x["week"] = x.week.astype(int)
        x["season"] = x.season.astype(int)
        x[KEEP[:1] + ["week"] + KEEP[1:]].to_parquet(f"data/ecr_{kind}.parquet")
        print(f"{kind}:", x.groupby("season").week.nunique().to_dict(), "weeks per season")


if __name__ == "__main__":
    main()
