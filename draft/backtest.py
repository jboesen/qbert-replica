"""Backtest the draft board: simulate real drafts, score the rosters that come out.

Twelve teams, snake draft, PPR, holdout seasons only. Each strategy occupies three
seats so no strategy gets a systematically better slot. Rosters are scored on what they
actually did that season, using the best legal lineup each week - which measures roster
quality without confounding it with in-season lineup decisions.
"""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
from vbd import LEAGUE, add_vbd
from draft_dp import plan, snake_picks

STARTERS, FLEX_OK = LEAGUE["starters"], set(LEAGUE["flex"])
TEAMS, ROUNDS = LEAGUE["teams"], LEAGUE["rounds"]


def weekly_points(season):
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/stats/w*.parquet"))])
    d = d[(d.season == season) & (d.season_type == "REG")]
    return d.pivot_table(index="player_id", columns="week",
                         values="fantasy_points_ppr", aggfunc="sum").fillna(0.0)


def score_roster(roster, wk, pos_of):
    """Best legal starting lineup each week, summed over the season."""
    total = 0.0
    have = [p for p in roster if p in wk.index]
    for week in wk.columns:
        pts = {p: wk.at[p, week] for p in have}
        used, wk_total = set(), 0.0
        for pos, n in STARTERS.items():
            cands = sorted([p for p in have if pos_of[p] == pos and p not in used],
                           key=lambda p: -pts[p])[:n]
            used.update(cands); wk_total += sum(pts[p] for p in cands)
        flex = sorted([p for p in have if pos_of[p] in FLEX_OK and p not in used],
                      key=lambda p: -pts[p])[:LEAGUE["flex_slots"]]
        total += wk_total + sum(pts[p] for p in flex)
    return total


def run_draft(board, strategies):
    """Snake draft. `board` is the shared market order; each seat picks by its strategy."""
    order = []
    for r in range(ROUNDS):
        seats = range(TEAMS) if r % 2 == 0 else reversed(range(TEAMS))
        order.extend(seats)

    rosters = {s: [] for s in range(TEAMS)}
    needs = {s: {**STARTERS, "FLEX": LEAGUE["flex_slots"]} for s in range(TEAMS)}
    taken = set()

    for pick_no, seat in enumerate(order, start=1):
        avail = board[~board.index.isin(taken)]
        if not len(avail):
            break
        strat = strategies[seat]

        if strat == "dp":
            future = [p for p in snake_picks(seat, TEAMS, ROUNDS) if p >= pick_no]
            pos, _, _ = plan(avail, future, needs[seat])
            cand = avail[avail.position == pos]
            choice = cand.index[0] if len(cand) else avail.index[0]
        elif strat == "vbd":
            choice = avail.index[0]                       # board is already VBD-sorted
        elif strat == "points":
            choice = avail.proj_points.idxmax()
        else:                                             # market: prior-year finish
            choice = avail.market_rank.idxmin()

        # Fill a starting slot if this pick can.
        p = avail.at[choice, "position"]
        if needs[seat].get(p, 0) > 0:
            needs[seat][p] -= 1
        elif needs[seat]["FLEX"] > 0 and p in FLEX_OK:
            needs[seat]["FLEX"] -= 1
        rosters[seat].append(choice)
        taken.add(choice)
    return rosters


def main():
    s = pd.read_parquet("data/draft_projected.parquet")
    results, head2head = {}, {}
    for season in range(2021, 2026):
        y = s[(s.season == season) & s.proj_points.notna()].copy()
        prev = s[s.season == season - 1].set_index("player_id").ppr
        y["prev_ppr"] = y.player_id.map(prev).fillna(0)
        y = y.set_index("player_id")
        board, _, _ = add_vbd(y)
        board["market_rank"] = board.prev_ppr.rank(ascending=False)

        wk = weekly_points(season)
        pos_of = board.position.to_dict()

        # Rotate which seats each strategy holds, so draft position cannot decide it.
        for shift in range(4):
            base = ["dp", "vbd", "points", "market"] * 3
            strategies = base[shift:] + base[:shift]
            rosters = run_draft(board, strategies)
            scores = {seat: score_roster(r, wk, pos_of) for seat, r in rosters.items()}
            for seat, sc in scores.items():
                results.setdefault(strategies[seat], []).append(sc)
            # Paired comparison: same league, same board, different seats.
            for a in ("vbd", "market", "points"):
                dp = np.mean([sc for st, sc in zip(strategies, scores.values()) if st == "dp"])
                other = np.mean([sc for st, sc in zip(strategies, scores.values()) if st == a])
                head2head.setdefault(a, []).append(dp - other)
        print(f"{season} drafted", flush=True)

    print(f"\n=== Roster quality, {ROUNDS}-round snake, 12 teams, PPR, 2021-25 ===")
    print(f"(season points from the best legal lineup each week; "
          f"{len(results['dp'])} drafts per strategy)")
    rows = [(k, np.mean(v), np.std(v), min(v), max(v)) for k, v in results.items()]
    out = pd.DataFrame(rows, columns=["strategy", "mean", "sd", "worst", "best"])
    names = {"dp": "Fry-Lundberg-Ohlmann DP", "vbd": "value-based drafting",
             "points": "best projected points", "market": "prior-year finish (market)"}
    out["strategy"] = out.strategy.map(names)
    print(out.sort_values("mean", ascending=False).to_string(
        index=False, float_format=lambda v: f"{v:.0f}"))
    print(f"\n=== DP vs each alternative, paired within the same league (n={len(head2head['vbd'])}) ===")
    for k, v in head2head.items():
        v = np.array(v)
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"  vs {names[k]:28s} {v.mean():+7.1f} points/season  "
              f"(se {se:.1f}, DP ahead in {100 * (v > 0).mean():.0f}% of leagues)")


if __name__ == "__main__":
    main()
