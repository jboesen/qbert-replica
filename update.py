"""Refresh the in-season data from nflverse, then rebuild everything downstream of it.

    .venv/bin/python update.py            # current season, then weekly + rest-of-season
    .venv/bin/python update.py --no-build # download only

Safe to run as often as you like: each file is downloaded to a temp path, checked, and
only then swapped in, so a failed download never leaves a half-written file behind.
"""
import datetime as dt
import os
import subprocess
import sys
import time

import pandas as pd
import requests

REL = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PY = sys.executable


def current_season(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1   # Jan-Feb is last season


def fetch(url, dest, tries=4):
    for i in range(tries):             # GitHub throws the odd 5xx; back off and retry
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            break
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            if i == tries - 1:
                raise
            time.sleep(5 * 2 ** i)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(r.content)
    return tmp


def swap_parquet(url, dest, check=None):
    tmp = fetch(url, dest)
    d = pd.read_parquet(tmp)
    if not len(d) or (check and not check(d)):
        os.remove(tmp)
        raise SystemExit(f"{url} came back empty or malformed; kept the old {dest}")
    os.replace(tmp, dest)
    return d


def update_players():
    """Merge rather than replace: the live table drops some long-retired players, and
    older seasons still need their birth dates and draft slots."""
    dest = "data/players.parquet"
    tmp = fetch(f"{REL}/players/players.parquet", dest)
    new = pd.read_parquet(tmp)
    if os.path.exists(dest):
        old = pd.read_parquet(dest)
        new = pd.concat([new, old[~old.gsis_id.isin(new.gsis_id)]], ignore_index=True)
    new.to_parquet(dest)
    os.remove(tmp)
    return new


def main():
    season = current_season()
    stats = swap_parquet(f"{REL}/stats_player/stats_player_week_{season}.parquet",
                         f"data/stats/w{season}.parquet",
                         check=lambda d: {"fantasy_points_ppr", "week"} <= set(d.columns))
    print(f"stats {season}: {len(stats)} player-games through week {stats.week.max()}")

    players = update_players()
    print(f"players: {len(players)}")

    tmp = fetch(GAMES, "data/games.csv")
    g = pd.read_csv(tmp)
    if not (g.season == season).any():
        raise SystemExit(f"schedule has no {season} games; kept the old data/games.csv")
    os.replace(tmp, "data/games.csv")
    played = g[(g.season == season) & g.result.notna()]
    print(f"schedule: {len(played)} of {(g.season == season).sum()} {season} games played")

    try:
        inj = swap_parquet(f"{REL}/injuries/injuries_{season}.parquet",
                           f"data/injuries_{season}.parquet")
        print(f"injury reports: {len(inj)} rows through week {inj.week.max()}")
    except (requests.HTTPError, SystemExit) as e:
        print(f"injury reports unavailable ({e}); projections run without them")
    # Past seasons' reports calibrate how often a Questionable player actually plays.
    for yr in range(2016, season):
        if not os.path.exists(f"data/injuries_{yr}.parquet"):
            swap_parquet(f"{REL}/injuries/injuries_{yr}.parquet", f"data/injuries_{yr}.parquet")

    if "--no-build" in sys.argv:
        return
    # draft_seasons.parquet is deliberately not rebuilt: the preseason model and its
    # holdout are defined on complete seasons, and a partial one would leak into both.
    for step in (["draft/weekly.py", str(season)], ["draft/ros.py", str(season)]):
        print(f"\n$ python {' '.join(step)}", flush=True)
        subprocess.run([PY, *step], check=True)


if __name__ == "__main__":
    main()
