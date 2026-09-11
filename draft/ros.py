"""Rest-of-season value: every remaining game projected one at a time, then priced by VBD.

    .venv/bin/python draft/ros.py 2026

This is the in-season draft board - the one to use for waivers and trades. Each
remaining game carries its own opponent and implied total, bye weeks drop out
naturally, and each game is discounted by the chance he plays it: this week's injury
designation for the next game, his projected availability rate after that.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import weekly as W
from vbd import LEAGUE, add_vbd


def rest_of_season(season):
    x = W.project(season)
    nxt = x.week == x.groupby("player_id").week.transform("min")
    x["p"] = np.where(nxt, x.p_play, x.avail)
    x["exp_pts"] = x.p * x.proj_ppr
    r = x.groupby("player_id").agg(
        player_display_name=("player_display_name", "first"),
        position=("position", "first"), team=("team", "first"),
        games_left=("week", "size"), proj_ppg=("proj_ppr", "mean"),
        exp_games=("p", "sum"), ros_points=("exp_pts", "sum"))
    board, levels, _ = add_vbd(r.assign(proj_points=r.ros_points))
    r["ros_vbd"] = board.vbd
    return r.sort_values("ros_vbd", ascending=False).reset_index(), levels


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    r, levels = rest_of_season(season)
    r[["player_id", "player_display_name", "position", "team", "games_left", "proj_ppg",
       "exp_games", "ros_points", "ros_vbd"]].to_parquet(f"data/ros_{season}.parquet",
                                                         index=False)
    print(f"=== {season} rest of season, {LEAGUE['teams']}-team PPR ===")
    print("replacement level:", {k: round(v) for k, v in levels.items()}, "\n")
    show = r.head(40)[["player_display_name", "position", "team", "games_left",
                       "proj_ppg", "exp_games", "ros_points", "ros_vbd"]]
    show.index += 1
    print(show.to_string(float_format=lambda v: f"{v:.1f}"))
