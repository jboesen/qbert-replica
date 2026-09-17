"""Start/sit for this week, set by expert consensus.

In the league backtest, weekly consensus (FantasyPros ECR) set better lineups than our
weekly model in every season, under both opponent designs (draft/README.md, "Does it
win leagues?"). So this sets the lineup from consensus. Each player's weekly positional
rank becomes expected points on a curve fit to 2020-25, which puts every position on one
scale for the flex. Out and doubtful players sit. Questionable players are discounted by
how often questionable players have played, from settings.p_questionable. Players on bye sit.

Refresh first: `update.py` for injury reports, then `draft/consensus.py` for the
week's ranks.

    python draft/lineup.py --mine "Ja'Marr Chase, Bijan Robinson, ..."
    python draft/lineup.py --league league.json          # uses "me" from the file
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import availability as A
import consensus as C
import settings as CFG
from vbd import LEAGUE

SET = CFG.get()
SLOTS, FLEX = LEAGUE["starters"], list(LEAGUE["flex"])


def resolve(names):
    """Free-text names -> gsis ids, via the players table (exact, then substring)."""
    p = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id")
    p = p[p.position.isin(list(SLOTS))]
    low = p.display_name.str.lower()
    ids = []
    for n in [x.strip() for x in names if x.strip()]:
        hit = p[low == n.lower()]
        if not len(hit):
            hit = p[low.str.contains(n.lower(), regex=False)]
        if len(hit):
            ids.append(hit.sort_values("last_season", ascending=False).gsis_id.iloc[0])
        else:
            print(f"  ! no player matches {n!r}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--mine", default="")
    ap.add_argument("--league")
    ap.add_argument("--settings", help="JSON of league settings; see draft/settings.py")
    a = ap.parse_args()
    names = (json.load(open(a.league))["teams"][json.load(open(a.league))["me"]]
             if a.league else a.mine.split(","))
    ids = resolve(names)

    week, ranks = C.latest("weekly", a.season)
    crv = C.weekly_curve()
    players = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == a.season) & (g.week == week)]
    playing = set(g.home_team) | set(g.away_team)
    try:
        inj = pd.read_parquet(f"data/injuries_{a.season}.parquet")
        inj = inj[inj.week == week].drop_duplicates("gsis_id", keep="last")
        status = inj.set_index("gsis_id").report_status
    except FileNotFoundError:
        status = pd.Series(dtype=object)

    r = ranks.set_index("player_id")
    rows = []
    for pid in ids:
        pos = players.position.get(pid)
        team = players.latest_team.get(pid)
        worst = ranks[ranks.pos == pos]["rank"].max()
        rank = r["rank"].get(pid, worst + 1 if pd.notna(worst) else 150)
        pts = C.rank_points(crv, [pos], [rank])[0]
        st = status.get(pid)
        note = ""
        if team not in playing:
            pts, note = -1.0, "bye"
        elif A.sits(st):
            pts, note = -1.0, st.lower()
        elif st == "Questionable":
            # One definition of what a designation costs, shared with every other tool.
            pts, note = pts * A.play_discount(st), "questionable"
        rows.append((pid, players.display_name.get(pid), pos, int(rank), pts, note))
    d = pd.DataFrame(rows, columns=["id", "name", "pos", "rank", "pts", "note"])

    start, used = [], set()
    for p, k in SLOTS.items():
        for x in d[(d.pos == p) & (d.pts >= 0)].nlargest(k, "pts").itertuples():
            start.append((p, x)); used.add(x.id)
    fl = d[d.pos.isin(FLEX) & ~d.id.isin(used) & (d.pts >= 0)].nlargest(SET.flex_slots, "pts")
    for x in fl.itertuples():
        start.append(("FLEX", x)); used.add(x.id)

    print(f"week {week} lineup, by consensus rank (expected PPR from the rank)\n")
    for slot, x in start:
        tag = f"  [{x.note}]" if x.note else ""
        print(f"  {slot:4s} {x.name:24s} {x.pos}{x.rank:<3d}  {x.pts:5.1f}{tag}")
    bench = d[~d.id.isin(used)].sort_values("pts", ascending=False)
    if len(bench):
        print("\nbench")
        for x in bench.itertuples():
            tag = f"  [{x.note}]" if x.note else ""
            print(f"       {x.name:24s} {x.pos}{x.rank:<3d}  {max(x.pts, 0):5.1f}{tag}")


if __name__ == "__main__":
    main()
