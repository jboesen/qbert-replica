"""Assemble the QBERT training table: weekly QB box-score stats joined to ESPN QBR."""
import glob
import numpy as np
import pandas as pd

TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA",
            "JAX": "JAX", "ARZ": "ARI", "HST": "HOU", "BLT": "BAL", "CLV": "CLE"}


def norm_team(s):
    return s.astype(str).str.upper().replace(TEAM_FIX)


def load_stats():
    frames = [pd.read_parquet(f) for f in sorted(glob.glob("data/stats/w*.parquet"))]
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.position == "QB") & (d.season_type.isin(["REG", "POST"]))].copy()
    d["team"] = norm_team(d.team)
    d["opponent_team"] = norm_team(d.opponent_team)
    return d


def load_qbr():
    q = pd.read_csv("data/qbr_week.csv")
    q = q[q.week_text != "Pro Bowl"].copy()
    # ESPN restarts the count in January and skips a number for the Pro Bowl;
    # nflverse just keeps counting. Map the rounds by name instead.
    rounds = {"Wild Card": 19, "Divisional Round": 20,
              "Conference Championship": 21, "Super Bowl": 22}
    post = q.season_type == "Playoffs"
    q.loc[post, "week_num"] = q.loc[post, "week_text"].map(rounds)
    q = q[q.week_num.notna()]
    q["week_num"] = q.week_num.astype(int)
    q["team_abb"] = norm_team(q.team_abb)
    return q[["season", "week_num", "team_abb", "qbr_total", "qb_plays", "pts_added",
              "name_display", "qualified"]].rename(columns={"week_num": "week",
                                                            "team_abb": "team"})


def build():
    d, q = load_stats(), load_qbr()
    # QBR carries one row per team-game (the starter), so keep each team's busiest QB.
    d["plays"] = d.attempts.fillna(0) + d.sacks_suffered.fillna(0) + d.carries.fillna(0)
    d = d.sort_values("plays", ascending=False).drop_duplicates(["season", "week", "team"])
    m = d.merge(q, on=["season", "week", "team"], how="inner", suffixes=("", "_qbr"))

    # Team rushing context: RB yards per carry, excluding the QB's own runs.
    allp = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/stats/w*.parquet"))])
    allp = allp[allp.season_type.isin(["REG", "POST"])].copy()
    allp["team"] = norm_team(allp.team)
    rb = allp[allp.position.isin(["RB", "FB"])].groupby(["season", "week", "team"]).agg(
        rb_carries=("carries", "sum"), rb_yards=("rushing_yards", "sum")).reset_index()
    m = m.merge(rb, on=["season", "week", "team"], how="left")
    m["rb_ypc"] = np.where(m.rb_carries.fillna(0) > 0, m.rb_yards / m.rb_carries, np.nan)

    g = pd.read_csv("data/games.csv")
    g = g[g.game_type != "PRO"]
    home = g[["season", "week", "home_team", "away_team", "result"]].rename(
        columns={"home_team": "team", "away_team": "opp"})
    home["home"] = 1
    away = g[["season", "week", "away_team", "home_team", "result"]].rename(
        columns={"away_team": "team", "home_team": "opp"})
    away["home"], away["result"] = 0, -away.result
    sched = pd.concat([home, away])
    sched["team"], sched["opp"] = norm_team(sched.team), norm_team(sched.opp)
    m = m.merge(sched[["season", "week", "team", "home", "result"]],
                on=["season", "week", "team"], how="left")
    return m


if __name__ == "__main__":
    m = build()
    m.to_parquet("data/qb_games.parquet")
    print(f"rows={len(m)}  seasons={m.season.min()}-{m.season.max()}  playoff rows={(m.week > 18).sum()}")
    print("qbr coverage:", m.qbr_total.notna().mean().round(3))
    print("air-yards coverage:", m.passing_yards_after_catch.notna().mean().round(3))
