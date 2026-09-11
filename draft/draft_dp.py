"""Draft-pick selection by dynamic programming.

Replicates the heuristic of Fry, Lundberg & Ohlmann, "A Player Selection Heuristic for a
Sports League Draft" (JQAS 2007). Their point is that picking the highest-value player
available is wrong: what matters is the value of your finished roster, which depends on
who will still be there at your later picks. They set the problem up as a stochastic
dynamic program, note it is intractable, and reduce it to a deterministic one by
assuming the rest of the league consumes the board in a known order.

That reduction is what is implemented here. The state is (which of my picks is next,
which starting slots are still empty); the transition is choosing a position now and
inheriting the best player left at each position later.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from vbd import LEAGUE

# What a bench player is worth relative to a starter. Depth pays off at the positions
# you start several of and can flex; a second quarterback in a one-QB league mostly sits.
BENCH_VALUE = {"QB": 0.08, "TE": 0.12, "RB": 0.32, "WR": 0.30}


def snake_picks(seat, teams, rounds):
    """Pick numbers for one seat in a snake draft, 1-indexed."""
    out = []
    for r in range(rounds):
        out.append(r * teams + (seat + 1 if r % 2 == 0 else teams - seat))
    return out


def plan(board, my_picks, needs, league=LEAGUE):
    """Value-maximising assignment of positions to my remaining picks.

    Returns the position to take now, plus the projected roster value of the whole plan.
    """
    positions = list(league["starters"])
    flex_ok = set(league["flex"])

    # For each of my picks, the best remaining player at each position, assuming the
    # rest of the league takes the top of the board and I take one player per pick.
    horizon = []
    for i, pick in enumerate(my_picks):
        # Players still on `board` are already the available ones. What thins them
        # further is only the picks other teams make between now and that pick.
        consumed = (pick - my_picks[0]) - i
        row = {}
        for p in positions:
            others = board[board.position == p]
            gone = int((board.head(consumed).position == p).sum())
            row[p] = others.vbd.values[gone] if gone < len(others) else 0.0
        horizon.append(row)

    @lru_cache(maxsize=None)
    def f(i, state):
        if i == len(horizon):
            return 0.0, ()
        need = dict(zip(positions + ["FLEX"], state))
        best = (-1e9, ())
        for p in positions:
            v = horizon[i][p]
            # A pick only counts toward roster value if it fills a starting slot,
            # either at its own position or at flex.
            if need[p] > 0:
                nxt = dict(need); nxt[p] -= 1
            elif need["FLEX"] > 0 and p in flex_ok:
                nxt = dict(need); nxt["FLEX"] -= 1
            else:
                nxt, v = dict(need), v * BENCH_VALUE[p]   # bench: depth only
            val, tail = f(i + 1, tuple(nxt[k] for k in positions + ["FLEX"]))
            if v + val > best[0]:
                best = (v + val, (p,) + tail)
        return best

    state = tuple(needs[k] for k in positions + ["FLEX"])
    total, seq = f(0, state)
    f.cache_clear()
    return seq[0], total, seq
