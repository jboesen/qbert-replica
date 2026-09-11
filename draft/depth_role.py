"""Each player's role going into a season, from the week-1 depth chart.

A player's history says how good he is; the depth chart says whether he'll be on the
field. The season projection had only the first, which is why a backup quarterback with
a good career projected like a starter. This records, per player-season, the best depth
slot he held at his position on the last chart published before the season opened.

nflverse carries two formats: weekly charts through 2024 (depth_team 1 = starter), and
dated snapshots from 2025 on (pos_rank 1 = starter). Both reduce to the same columns.

    .venv/bin/python draft/depth_role.py          # 2012 through the current season
"""
import io
import os
import sys

import pandas as pd
import requests

URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{}.parquet"
POS = ["QB", "RB", "WR", "TE"]
OUT = "data/depth_role.parquet"


def load(y):
    path = f"data/depth_{y}.parquet"
    if not os.path.exists(path):
        r = requests.get(URL.format(y), timeout=300)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    return pd.read_parquet(path)


def opener(y):
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG")]
    return pd.Timestamp(g.gameday.min(), tz="UTC")


def season_role(y):
    d = load(y)
    if "depth_team" in d:                                  # weekly format, through 2024
        d = d[(d.week == 1) & (d.game_type == "REG") & d.position.isin(POS)]
        d = d.assign(depth=pd.to_numeric(d.depth_team, errors="coerce"),
                     team=d.club_code, pos=d.position)
    else:                                                  # dated snapshots, 2025 on
        d = d.assign(dt=pd.to_datetime(d.dt, utc=True))
        before = d[d.dt < opener(y)]
        snap = before.dt.max() if len(before) else d.dt.min()
        d = d[(d.dt == snap) & d.pos_abb.isin(POS)]
        d = d.assign(depth=d.pos_rank, pos=d.pos_abb)
    d = d[d.gsis_id.notna() & d.depth.notna()]
    role = (d.groupby(["gsis_id", "team", "pos"]).depth.min().reset_index()
            .rename(columns={"gsis_id": "player_id"}))
    role["season"] = y
    # How crowded the room is: a WR2 on a team with three good receivers is not a WR2
    # on a team with one.
    return role.sort_values("depth").drop_duplicates("player_id")


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or list(range(2012, 2027))
    out = pd.concat([season_role(y) for y in years], ignore_index=True)
    out.to_parquet(OUT)
    print(out.groupby("season").size().to_string())
    print(out.groupby(["pos", "depth"]).size().unstack().iloc[:, :4])
