"""Draft structure on top of consensus values (prereg_draftstruct.md).

Every earlier draft test changed which players the test seat valued and lost to drafting
by exact consensus. These arms keep the exact consensus order and change only which
positions the seat may take when: wait on QB and TE, follow a best-ball roster template
through round 6, or lean toward the positions the room has left on the board. In-season
play is the best known policy in every arm (consensus lineups, streaming wire), so the
draft is the only thing that differs.

    .venv/bin/python draft/sim_draftstruct.py --noise 1
    .venv/bin/python draft/sim_draftstruct.py --noise 0
    .venv/bin/python draft/sim_draftstruct.py --noise 1 --leagues 2 --check   # mechanics
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
from league_backtest import (CAP, FLEX, POS, REG, ROUNDS, SECOND_AFTER, SEASONS, TEAMS,
                             WEEKS, allowed, needs)

ARMS = ("exact", "late_qb_te", "template", "reactive")

# late_qb_te: first round (1-indexed) a QB or TE may be taken, and how far past his
# consensus overall rank a player must have fallen to be taken earlier anyway.
QB_ROUND, TE_ROUND, FALL = 9, 8, 24

# template: through round TEMPLATE_ROUNDS the seat ends with 2-3 RB, 3-4 WR and at most one
# QB or TE between them, so the sixth pick is the only flexible one.
TEMPLATE_ROUNDS = 6
TEMPLATE_MIN = {"RB": 2, "WR": 3}
TEMPLATE_MAX = {"RB": 3, "WR": 4}
TEMPLATE_QBTE = 1

# reactive: the top-N consensus pool per position whose supply is tracked, and how much of
# that pool must be left beyond expectation before the seat leans toward the position.
TOP_N = {"QB": 12, "RB": 36, "WR": 36, "TE": 12}
LEAN = 0.10


def first_allowed(S, taken, c, rnd, ok=lambda i: True):
    """The first player in exact consensus order that is free, legal and passes `ok`."""
    for i in S.exact_order:
        if not taken[i] and allowed(S.pos[i], c, rnd) and ok(i):
            return int(i)
    return None


def consensus_rank(S):
    """1-based overall consensus rank per player index; players off the list rank last."""
    r = np.full(len(S.ids), len(S.exact_order) + 1)
    r[S.exact_order] = np.arange(1, len(S.exact_order) + 1)
    return r


def with_fallback(S, rule, log):
    """Wrap a structural rule so the seat still drafts when the rule leaves no legal pick,
    and record where the rule changed the pick exact consensus would have made."""
    def pick(taken, c, rnd, pick_no):
        exact = first_allowed(S, taken, c, rnd)
        mine = rule(taken, c, rnd, pick_no)
        fell_back = mine is None
        if fell_back:
            mine = exact
        log.append((rnd, mine != exact, fell_back))
        return mine
    return pick


def late_qb_te(S, log):
    rank = consensus_rank(S)
    first = {"QB": QB_ROUND, "TE": TE_ROUND}

    def rule(taken, c, rnd, pick_no):
        def ok(i):
            p = S.pos[i]
            if p not in first or rnd + 1 >= first[p]:
                return True
            return pick_no - rank[i] >= FALL          # a faller is taken anyway
        return first_allowed(S, taken, c, rnd, ok)
    return with_fallback(S, rule, log)


def template(S, log):
    def rule(taken, c, rnd, pick_no):
        if rnd >= TEMPLATE_ROUNDS:
            return first_allowed(S, taken, c, rnd)
        left = TEMPLATE_ROUNDS - rnd - 1               # my picks through the template after this

        def ok(i):
            nc = dict(c)
            nc[S.pos[i]] += 1
            if any(nc[p] > TEMPLATE_MAX[p] for p in TEMPLATE_MAX):
                return False
            if nc["QB"] + nc["TE"] > TEMPLATE_QBTE:
                return False
            # Keep the minimums reachable with the picks the template has left.
            return sum(max(0, k - nc[p]) for p, k in TEMPLATE_MIN.items()) <= left
        return first_allowed(S, taken, c, rnd, ok)
    return with_fallback(S, rule, log)


def reactive(S, log):
    """Lean toward positions the room has under-drafted against consensus."""
    order = np.array(S.exact_order)
    pool = np.zeros(len(S.ids), bool)
    for p in POS:
        at = order[S.pos[order] == p][:TOP_N[p]]
        pool[at] = True
    # How many of a position's pool exact consensus would have taken by each pick.
    gone_by = {p: np.concatenate([[0], np.cumsum(pool[order] & (S.pos[order] == p))])
               for p in POS}

    def rule(taken, c, rnd, pick_no):
        before = pick_no - 1
        lean = set()
        for p in POS:
            n = TOP_N[p]
            expected = n - gone_by[p][min(before, len(order))]
            left = int((pool & (S.pos == p) & ~taken).sum())
            if (left - expected) / n > LEAN:
                lean.add(p)
        if lean:
            x = first_allowed(S, taken, c, rnd, lambda i: S.pos[i] in lean)
            if x is not None:
                return x
        return first_allowed(S, taken, c, rnd)
    return with_fallback(S, rule, log)


POLICIES = {"late_qb_te": late_qb_te, "template": template, "reactive": reactive}


def roster_legal(S, roster):
    """The harness's roster rules, checked after the fact on the pick sequence."""
    c = dict.fromkeys(POS, 0)
    for rnd, i in enumerate(roster):
        if not allowed(S.pos[i], c, rnd):
            return False
        c[S.pos[i]] += 1
    return (len(roster) == ROUNDS and all(c[p] <= CAP[p] for p in POS)
            and not sum(needs(c).values()))


def run(check=False):
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, problems = [], []
    for y in SEASONS:
        S = LB.Season(y, wk_proj, curves)
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        every = set(range(TEAMS))
        base = [cons_val] * TEAMS
        value = np.nan_to_num(S.cons_vbd)
        for lg in range(LB.LEAGUES):
            # The same draws in the same order as run_streaming, so the control matches.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for k in range(TEAMS + 1):
                score = S.ecr_mean + LB.NOISE * S.ecr_sd * noise[k]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = LB.schedules(rng, LB.SCHEDULES)
            for seat in range(TEAMS):
                for arm in ARMS:
                    log = []
                    orders = list(ecr_orders[:TEAMS])
                    orders[seat] = (S.exact_order if arm == "exact"
                                    else POLICIES[arm](S, log))
                    drafted = LB.draft(S, orders)
                    if check:
                        problems += [(y, lg, seat, arm, t) for t in range(TEAMS)
                                     if not roster_legal(S, drafted[t])]
                    movers = {seat: LB.streaming_policy(S, gone, [])}
                    sc, moves, _ = LB.simulate(S, drafted, base, every, movers)
                    res = [LB.season_outcome(sc, s) for s in scheds]
                    mine = drafted[seat]
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": LB.all_play(sc, seat),
                        "pts_reg": sc[seat, :REG].sum(),
                        "pts_playoff": sc[seat, REG:].sum(),
                        "moves": moves[seat],
                        "draft_pos": [str(S.pos[i]) for i in mine],
                        "cons_value": float(value[mine].sum()),
                        "weekly": sc[seat].tolist(),
                        "deviations": sum(d for _, d, _ in log),
                        "fallbacks": sum(f for _, _, f in log),
                    })
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return pd.DataFrame(rows), problems


# ---------------------------------------------------------------- reports

def position_mix(d):
    """Share of the test seat's picks at each position, by round, per arm."""
    print("\nround-by-round position mix of the test seat (% of picks: QB/RB/WR/TE)")
    print("  " + "round".ljust(11) + "".join(f"{r:>13d}" for r in range(1, ROUNDS + 1)))
    for arm in ARMS:
        picks = np.array(d[d.arm == arm].draft_pos.tolist())
        cells = []
        for r in range(ROUNDS):
            col = picks[:, r]
            cells.append("/".join(f"{100 * (col == p).mean():.0f}" for p in POS))
        print(f"  {arm:11s}" + "".join(f"{x:>13s}" for x in cells))
    print("  through round 6, mean count QB RB WR TE:")
    for arm in ARMS:
        picks = np.array(d[d.arm == arm].draft_pos.tolist())[:, :TEMPLATE_ROUNDS]
        print(f"    {arm:11s} " + " ".join(f"{p} {(picks == p).sum(1).mean():.2f}" for p in POS))


def mechanics(d, problems):
    """Checks that print no outcome metric."""
    print(f"\nillegal rosters (any seat, any arm): {len(problems)}")
    for p in problems[:10]:
        print("  ", p)
    ref = pd.read_parquet(f"data/league_streaming_noise{LB.NOISE:g}.parquet")
    ref = ref[(ref.arm == "stream") & (ref.league < LB.LEAGUES)]
    key = ["season", "league", "seat"]
    cols = ["title", "playoff", "wins", "all_play", "pts_reg", "pts_playoff", "moves"]
    a = d[d.arm == "exact"].set_index(key)[cols].sort_index()
    b = ref.set_index(key)[cols].sort_index()
    same = a.shape == b.shape and np.allclose(a.values, b.values.astype(float))
    print(f"control reproduces run_streaming's stream arm exactly: {same} "
          f"({len(a)} vs {len(b)} rows)")
    position_mix(d)
    print("\npicks per test seat that differ from exact consensus in the same state, "
          "and fallbacks:")
    for arm in ARMS[1:]:
        g = d[d.arm == arm]
        print(f"  {arm:11s} differ {g.deviations.mean():.2f}   fallbacks {g.fallbacks.mean():.2f}")


def report(d):
    print("\n=== draft structure on consensus values, streaming wire in every arm ===")
    print(f"{'arm':11s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s} {'adds':>6s}")
    for arm in ARMS:
        g = d[d.arm == arm]
        print(f"{arm:11s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f} {g.moves.mean():6.2f}")
    print("\npreregistered test (prereg_draftstruct.md), minus exact, "
          "season-cluster bootstrap 90% interval:")
    for arm in ARMS[1:]:
        keep = LB.paired(d, arm, "exact", f"{arm} minus exact")
        print(f"  -> {arm}: {'passes this design' if keep else 'fails this design'}")
        by = (d[d.arm == arm].groupby("season").title.mean()
              - d[d.arm == "exact"].groupby("season").title.mean())
        print("     title by season " + " ".join(f"{100 * v:+.1f}" for v in by))

    print("\nreported, not gated")
    position_mix(d)
    print("\n  consensus value drafted (sum of cons_vbd over the 14 picks) and picks that "
          "differ from exact")
    for arm in ARMS:
        g = d[d.arm == arm]
        by = g.groupby("season").cons_value.mean()
        print(f"    {arm:11s} {g.cons_value.mean():7.0f}  by season "
              + " ".join(f"{v:6.0f}" for v in by)
              + f"   differ {g.deviations.mean():.2f}  fallbacks {g.fallbacks.mean():.2f}")
    print("\n  weekly starting-lineup points of the test seat, weeks 1-17 "
          "(arm mean; arms below as change vs exact)")
    key = ["season", "league", "seat"]
    wk = {arm: np.array(d[d.arm == arm].sort_values(key).weekly.tolist()) for arm in ARMS}
    print("    " + "week".ljust(11) + "".join(f"{w:>6d}" for w in range(1, WEEKS + 1)))
    print(f"    {'exact':11s}" + "".join(f"{v:6.1f}" for v in wk["exact"].mean(0)))
    for arm in ARMS[1:]:
        diff = (wk[arm] - wk["exact"]).mean(0)
        print(f"    {arm:11s}" + "".join(f"{v:+6.1f}" for v in diff))


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = "--check" in sys.argv
    tag = "" if LB.LEAGUES == 20 and not check else f"_check{LB.LEAGUES}"
    d, problems = run(check)
    d.to_parquet(f"data/league_draftstruct{tag}_noise{LB.NOISE:g}.parquet")
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    if check:
        mechanics(d, problems)     # no outcome metrics before the spec is frozen
    else:
        report(d)
