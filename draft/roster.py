"""Roster legality and the cut rule, defined once for the draft, the wire and trades.

The three places a roster changes used to disagree about what a legal roster is. The
draft enforced a cap of two quarterbacks and seven running backs; the in-season wire
enforced only the starting minimums and never looked at the cap, so a team could stream
its way to four quarterbacks; the trade evaluator priced replacement off a third set of
per-team counts again. All three now read the same caps and the same minimums.

Enforcing the cap in season is a change to published behaviour, not a bug fix in the
sense that it would leave results alone, so it is a switch that defaults off. The
minimums, which every published arm did enforce, are always on.
"""
import numpy as np

import settings as CFG


def counts(pos, players):
    """Players held at each position.

    `pos` maps a player to its position, either as a numpy array indexed by the harness's
    integer player ids or as a dict keyed by the live tools' gsis ids. The array path is
    kept vectorised because the harness calls this inside its weekly loop.
    """
    players = list(players)
    if isinstance(pos, np.ndarray):
        sel = pos[players]
        return {p: int((sel == p).sum()) for p in CFG.POSITIONS}
    out = dict.fromkeys(CFG.POSITIONS, 0)
    for i in players:
        if pos[i] in out:
            out[pos[i]] += 1
    return out


def legal(c, s=None):
    """Whether position counts leave a legal roster: never below the starting minimums,
    and never above the caps when the cap is being enforced."""
    s = s or CFG.get()
    if any(c.get(p, 0) < s.roster_min.get(p, 0) for p in CFG.POSITIONS):
        return False
    if s.enforce_caps_in_season:
        return all(c.get(p, 0) <= s.caps.get(p, 99) for p in CFG.POSITIONS)
    return True


def cuttable(pos, held, s=None):
    """Rostered players a team may cut without dropping a position below its starters."""
    s = s or CFG.get()
    c = counts(pos, held)
    return [i for i in held if c.get(pos[i], 0) > s.roster_min.get(pos[i], 0)]


def add_allowed(pos, held, who, s=None):
    """Which of `who` a team may add without breaking its cap.

    With the cap off this is every one of them, which is what the published wire did.
    """
    s = s or CFG.get()
    if not s.enforce_caps_in_season:
        return who
    c = counts(pos, held)
    ok = [i for i in who if c.get(pos[i], 0) < s.caps.get(pos[i], 99)]
    return np.array(ok, dtype=int) if isinstance(who, np.ndarray) else ok


def forced_cut(pos, held, value, holes_after=None, s=None):
    """Who to cut when a roster is over its size, by the rule in settings.

    "lowest_value" is the consensus wire's rule: the cheapest cuttable player goes.
    "fewest_holes" is the streaming policy's: prefer the cut that leaves the fewest
    unfillable starting slots, then a bench player, then the cheapest. A roster can be
    over its size after an add, or when a player comes back from reserve.
    """
    s = s or CFG.get()
    can = cuttable(pos, held, s)
    if not can:
        return None
    if s.cut_rule == "lowest_value" or holes_after is None:
        return min(can, key=lambda i: value.get(i, -np.inf))
    return min(can, key=lambda i: (holes_after(i), value.get(i, -np.inf)))


def fits(pos, held, s=None):
    """Whether a roster is within its size as well as legal by position."""
    s = s or CFG.get()
    return len(held) <= s.roster_size and legal(counts(pos, held), s)
