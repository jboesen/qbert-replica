"""Scoring-opportunity usage per player-week from play-by-play.

The weekly stats carry targets and carries but not where on the field they happened,
and a carry at the 3 is worth several times a carry at midfield. This pulls red-zone
and goal-line opportunities, end-zone targets and each team's play volume, so usage can
be expressed as shares of what the offense actually ran.

    .venv/bin/python draft/pbp_opps.py            # 2006 through the current season
    .venv/bin/python draft/pbp_opps.py 2026       # refresh one season in place
"""
import io
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
import requests

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
OUT = "data/pbp_opps.parquet"
COLS = ["season", "week", "season_type", "posteam", "play_type", "yardline_100",
        "pass_attempt", "rush_attempt", "sack", "qb_scramble", "air_yards",
        "receiver_player_id", "rusher_player_id", "two_point_attempt"]


def season(y):
    raw = requests.get(URL.format(y), timeout=300).content
    t = pq.read_table(io.BytesIO(raw), columns=COLS).to_pandas()
    t = t[t.posteam.notna() & (t.two_point_attempt != 1)
          & t.season_type.isin(["REG", "POST"])]
    key = ["season", "week", "posteam"]

    team = t.groupby(key).agg(
        team_pass=("pass_attempt", "sum"), team_rush=("rush_attempt", "sum"),
        team_rz_plays=("yardline_100", lambda v: (v <= 20).sum())).reset_index()

    tg = t[(t.pass_attempt == 1) & t.receiver_player_id.notna() & (t.sack != 1)]
    rec = tg.assign(
        rz_tgt=tg.yardline_100 <= 20, gl_tgt=tg.yardline_100 <= 10,
        ez_tgt=tg.air_yards >= tg.yardline_100,
    ).groupby(key + ["receiver_player_id"])[["rz_tgt", "gl_tgt", "ez_tgt"]].sum()
    rec.index = rec.index.set_names(key + ["player_id"])

    ru = t[(t.rush_attempt == 1) & t.rusher_player_id.notna()]
    rush = ru.assign(
        rz_car=ru.yardline_100 <= 20, gl_car=ru.yardline_100 <= 5,
    ).groupby(key + ["rusher_player_id"])[["rz_car", "gl_car"]].sum()
    rush.index = rush.index.set_names(key + ["player_id"])

    p = rec.join(rush, how="outer").fillna(0).reset_index()
    p = p.merge(team, on=key, how="left").rename(columns={"posteam": "team"})
    return p


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or list(range(2006, 2027))
    old = pd.read_parquet(OUT) if os.path.exists(OUT) else None
    frames = []
    for y in years:
        frames.append(season(y))
        print(y, len(frames[-1]), flush=True)
    new = pd.concat(frames, ignore_index=True)
    if old is not None:
        new = pd.concat([old[~old.season.isin(years)], new], ignore_index=True)
    new.to_parquet(OUT)
    print(f"{OUT}: {len(new)} player-weeks, {new.season.min()}-{new.season.max()}")
