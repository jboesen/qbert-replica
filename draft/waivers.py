"""This week's waiver move, by the one decision that beat consensus in the backtest.

In the league backtest (draft/prereg_streaming.md) a team that fills next week's
starting holes on the wire gained 1-2 points of all-play and about 1-1.5 points of title
odds over a team running the plain consensus wire, positive in every season under both
opponent designs. Same information as consensus, better decision: the consensus wire
adds the best rest-of-season free agent at any position, even when your only tight end
is on bye.

So this reproduces that policy for the live season, as close to the tested code as a
live tool can be:

  - value: consensus rest-of-season rank, priced as points on the weekly rank curve and
    then over replacement at the position, as the harness's `over_replacement` does;
  - holes for the upcoming week, from what's known when waivers run: byes, anyone not
    on an active roster in the latest weekly roster file, and anyone Out or Doubtful on
    the latest injury report, assumed still out;
  - if there's a hole: add the best free agent at a position that fills it (not on bye,
    not known out) and cut the player whose loss leaves the fewest holes, preferring a
    bench player, then the lowest rest-of-season value, even when the add ranks lower;
  - if there's no hole, or no move closes one: the consensus move (best free agent in,
    worst cuttable player out, only if the add is worth more).

Refresh first: `update.py` (injury reports) and `draft/consensus.py` (rest-of-season
ranks). The weekly roster file is fetched here if it's missing or older than half a day.

    .venv/bin/python draft/waivers.py --league league.json [--week N]

league.json: {"me": "John", "teams": {"John": ["name", ...], "Alex": [...], ...}}
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import availability as A
import consensus as C
import roster as R
import settings as CFG
from lineup import resolve
from vbd import LEAGUE, add_vbd

SET = CFG.get()
POS = list(CFG.POSITIONS)
SLOTS, FLEX = LEAGUE["starters"], list(LEAGUE["flex"])
ROSTERS = ("https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/"
           "roster_weekly_{y}.parquet")


def needs(c):
    """Starting slots still empty, flex included, given position counts c."""
    short = {p: max(0, k - c.get(p, 0)) for p, k in SLOTS.items()}
    extra = sum(max(0, c.get(p, 0) - SLOTS[p]) for p in FLEX)
    short["FLEX"] = max(0, SET.flex_slots - extra)
    return short


def fresh_rosters(y):
    path = f"data/roster_weekly_{y}.parquet"
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > 12 * 3600:
        req = urllib.request.Request(ROSTERS.format(y=y), headers={"User-Agent": "research"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return pd.read_parquet(path)


def upcoming_week(y):
    """The first week with a game not yet played."""
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG")]
    open_ = g[g.result.isna()]
    return int(open_.week.min()) if len(open_) else int(g.week.max())


def values(y, w, pos):
    """Rest-of-season value over replacement for every player consensus ranks, plus
    anyone on `pos` it doesn't (one past its last-ranked player at the position)."""
    ros = pd.read_parquet("data/ecr_ros.parquet")
    ros = ros[(ros.season == y) & (ros.week < w)]
    if len(ros):
        src = ros[ros.week == ros.week.max()]
    else:                                         # before the first weekly scrape
        pre = pd.read_parquet("data/ecr_preseason.parquet")
        src = pre[pre.season == y]
    src = src.drop_duplicates("player_id").assign(
        rank=lambda x: x.groupby("pos").ecr.rank(method="first"))
    worst = src.groupby("pos")["rank"].max().to_dict()
    t = src.set_index("player_id")[["pos", "rank"]]
    extra = pos[~pos.index.isin(t.index)]
    t = pd.concat([t, pd.DataFrame({"pos": extra.values,
                                    "rank": [worst.get(p, 100) + 1 for p in extra.values]},
                                   index=extra.index)])
    t = t[t.pos.isin(POS)]
    crv = C.weekly_curve(range(2020, y))
    pts = C.rank_points(crv, t.pos, np.clip(t["rank"], 1, 150))
    b = pd.DataFrame({"position": t.pos.values, "proj_points": pts}, index=t.index)
    b, _, _ = add_vbd(b)
    return b.vbd, t["rank"], t.pos


def unavailable(y, w, ids):
    """Known out for week w at waiver time, by the shared definition in availability.py:
    bye, off the active roster, or Out/Doubtful on the latest report."""
    return A.live_reasons(y, w, ids, fresh_rosters(y))


def holes(roster, pos, gone):
    c = {}
    for p in roster:
        if p not in gone:
            c[pos[p]] = c.get(pos[p], 0) + 1
    return needs(c)


def cuttable(roster, pos):
    """The same rule the harness and the trade tool apply, from roster.py."""
    return R.cuttable(pos, roster)


def starters(roster, pos, val, gone):
    avail = [p for p in roster if p not in gone]
    used = set()
    for p, k in SLOTS.items():
        used |= set(sorted((i for i in avail if pos[i] == p), key=lambda i: -val.get(i, -1e9))[:k])
    flex = sorted((i for i in avail if pos[i] in FLEX and i not in used),
                  key=lambda i: -val.get(i, -1e9))
    used.update(flex[:SET.flex_slots])
    return used


def consensus_move(roster, free, pos, val):
    if not free:
        return None
    add = max(free, key=lambda p: val.get(p, -1e9))
    cut = cuttable(roster, pos)
    if not cut:
        return None
    drop = min(cut, key=lambda p: val.get(p, -1e9))
    return (add, drop) if val.get(add, -1e9) > val.get(drop, -1e9) else None


def streaming_move(roster, free, pos, val, gone):
    before = holes(roster, pos, gone)
    if not sum(before.values()):
        return None, before
    fills = [p for p in POS if before[p] > 0 or (p in FLEX and before["FLEX"] > 0)]
    cand = [p for p in free if pos[p] in fills and p not in gone]
    if not cand:
        return None, before
    add = max(cand, key=lambda p: val.get(p, -1e9))
    after = roster + [add]
    lineup = starters(after, pos, val, gone)
    best = None
    for d in cuttable(roster, pos):
        left = sum(holes([p for p in after if p != d], pos, gone).values())
        if left >= sum(before.values()):
            continue
        key = (left, d in lineup, val.get(d, -1e9))
        if best is None or key < best[0]:
            best = (key, d)
    return ((add, best[1]) if best else None), before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int)
    ap.add_argument("--settings", help="JSON of league settings; see draft/settings.py")
    a = ap.parse_args()
    y = a.season
    w = a.week or upcoming_week(y)

    lg = json.load(open(a.league))
    teams = {t: resolve(names) for t, names in lg["teams"].items()}
    mine = teams[lg["me"]]
    rostered = {p for r in teams.values() for p in r}

    players = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
    pos_all = players.position
    val, rank, pos = values(y, w, pos_all.reindex(list(rostered)).dropna())
    pos = pos.to_dict()
    val = val.to_dict()
    name = players.display_name.to_dict()
    free = [p for p in val if p not in rostered and pos.get(p) in POS]
    gone = unavailable(y, w, set(mine) | set(free))
    free_ok = [p for p in free if p not in gone]

    print(f"week {w} waivers for {lg['me']}  (rest-of-season consensus value over replacement)\n")
    out = [p for p in mine if p in gone]
    if out:
        print("known unavailable on your roster:")
        for p in out:
            print(f"  {name.get(p, p):24s} {pos.get(p, '?'):2s}  {gone[p]}")
    stream, before = streaming_move(mine, free_ok, pos, val, gone)
    cons = consensus_move(mine, free_ok, pos, val)
    hole_txt = ", ".join(f"{k} x{v}" for k, v in before.items() if v) or "none"
    print(f"\nstarting holes for week {w}: {hole_txt}")

    show = lambda m: (f"add {name.get(m[0], m[0])} ({pos[m[0]]}, rank {int(rank[m[0]])}), "
                      f"drop {name.get(m[1], m[1])} ({pos[m[1]]}, rank {int(rank[m[1]])})")
    if stream:
        print(f"\n>>> fill the hole: {show(stream)}")
        if cons and cons != stream:
            print(f"    (consensus alone would: {show(cons)})")
        elif not cons:
            print("    (consensus alone would stand pat)")
        if val.get(stream[0], -1e9) < val.get(stream[1], -1e9):
            print("    The add is worth less than the drop over the rest of the season; a "
                  "zero in a starting slot costs more this week than that difference.")
    elif cons:
        print(f"\n>>> no hole to fill; consensus move: {show(cons)}")
    else:
        print("\n>>> no move: no hole, and no free agent outranks your weakest cuttable player")

    top = sorted(free_ok, key=lambda p: -val[p])[:8]
    print("\nbest available (not known out):")
    for p in top:
        print(f"  {name.get(p, p):24s} {pos[p]:2s}  rank {int(rank[p]):3d}  value {val[p]:6.1f}")


if __name__ == "__main__":
    main()
