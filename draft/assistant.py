"""Draft-day assistant. Tell it who is gone and what you have; it tells you who to take.

    python draft/assistant.py --pick 18 --taken "Ja'Marr Chase, Bijan Robinson" \
                             --mine "Puka Nacua"

Recommends by the Fry-Lundberg-Ohlmann plan (which position to spend this pick on,
given who will still be there at your later picks) and shows the best available at each
position so the call is visible rather than a black box.
"""
import argparse
import sys

import pandas as pd

sys.path.insert(0, "draft")
from vbd import LEAGUE
from draft_dp import plan, snake_picks

STARTERS = LEAGUE["starters"]


def match(board, names):
    """Resolve free-text names against the board, reporting anything unmatched."""
    out, missing = [], []
    lower = board.player_display_name.str.lower()
    for n in [x.strip() for x in names.split(",") if x.strip()]:
        hit = board.index[lower == n.lower()]
        if not len(hit):
            hit = board.index[lower.str.contains(n.lower(), regex=False)]
        if len(hit):
            out.append(hit[0])
        else:
            missing.append(n)
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--seat", type=int, default=1, help="1-indexed draft slot")
    ap.add_argument("--pick", type=int, help="overall pick number now on the clock")
    ap.add_argument("--taken", default="", help="comma-separated names already drafted")
    ap.add_argument("--mine", default="", help="comma-separated names on my roster")
    a = ap.parse_args()

    board = pd.read_parquet(f"data/board_{a.season}.parquet")
    taken, miss1 = match(board, a.taken)
    mine, miss2 = match(board, a.mine)
    for n in miss1 + miss2:
        print(f"  ! no match for {n!r} - ignored (rookie, or spelled differently)")

    needs = {**STARTERS, "FLEX": LEAGUE["flex_slots"]}
    for pid in mine:
        p = board.at[pid, "position"]
        if needs.get(p, 0) > 0:
            needs[p] -= 1
        elif needs["FLEX"] > 0 and p in set(LEAGUE["flex"]):
            needs["FLEX"] -= 1

    avail = board[~board.index.isin(set(taken) | set(mine))]
    pick = a.pick or len(taken) + len(mine) + 1
    future = [p for p in snake_picks(a.seat - 1, LEAGUE["teams"], LEAGUE["rounds"])
              if p >= pick]
    if not future:
        future = [pick]
    pos, value, seq = plan(avail, future, needs)

    print(f"\npick {pick}, seat {a.seat}   roster: "
          f"{', '.join(board.loc[mine].player_display_name) or 'empty'}")
    print(f"still to fill: {', '.join(k for k, v in needs.items() for _ in range(v)) or 'nothing'}")
    print(f"\n>>> take a {pos}: {avail[avail.position == pos].player_display_name.iloc[0]}")
    print(f"    plan for your remaining picks: {' -> '.join(seq[:6])}"
          f"{' ...' if len(seq) > 6 else ''}")

    print("\nbest available by position")
    for p in STARTERS:
        top = avail[avail.position == p].head(3)
        line = ",  ".join(f"{r.player_display_name} ({r.vbd:.0f})"
                          for r in top.itertuples())
        print(f"  {p:3s} {line}")


if __name__ == "__main__":
    main()
