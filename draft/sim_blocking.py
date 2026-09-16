"""Rival-aware in-season management on top of capped mutual-benefit trades
(prereg_blocking.md).

Every policy tested so far reads only the seat's own roster. A real manager sees all
twelve rosters, the standings, the waiver order and this week's wire, and the harness
exposes exactly those. Two decisions change when the seat looks across the table:

- Blocking. In a week with no hole of its own the streaming control makes the plain
  consensus add, which is usually worth a point or two of rest-of-season value. Instead
  the seat can spend that move denying the free agent who would most raise the starting
  lineup of the rival directly above it in the standings. That rival sits behind the seat
  in waiver order (order is reverse standings), so the seat can always get there first.
- Rival-need-aware trading. Among the offers the mutual-benefit rule already accepts,
  offers within a tenth of the best gain are near-equivalent for the seat, so it can
  choose the one that helps the receiving side least, and avoid feeding the rival it is
  racing.

Blocking mostly transfers value rather than creating it, so the place to look for it is
title odds, not all-play.

    .venv/bin/python draft/sim_blocking.py --noise 1 [--leagues 2 --mechanics]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
import sim_trades as ST
import sim_tradecap as TC
from league_backtest import POS, REG, TEAMS, WEEKS

EPS = ST.EPS
GAIN_MIN = TC.GAIN_MIN     # a capped trade must project this many lineup points
BLOCK_K = 40               # free agents considered for a block, best first by wire value
NEAR_BEST = ST.NEAR_BEST   # "near-equivalent" offers, sim_trades' own 0.9
BLOCK_MIN = 8.0            # rest-of-season lineup points a barred block must deny
BLOCK_COST = 4.0           # rest-of-season lineup points a barred block may cost me
CHECK = False              # mechanics check: the control paths must match sim_tradecap

# (arm, block spec (harm floor, own-cost ceiling) or None, trade pick rule)
ARMS = [
    ("mutual_cap2", None, "mine"),
    ("block", (EPS, np.inf), "mine"),
    ("block_bar", (BLOCK_MIN, BLOCK_COST), "mine"),
    ("trade_rival", None, "rival"),
    ("block_bar_trade", (BLOCK_MIN, BLOCK_COST), "rival"),
]
VERDICT = "block_bar"


# ---------------------------------------------------------------- inputs

def future_avail_all(S, y, gone, teams):
    """ST.future_avail over every waiver week, not just the trade window.

    Blocking happens on the wire in weeks 2-17, so the rest-of-season lineup value of a
    rival's roster has to be computable in all of them. Same rule as ST.future_avail:
    future weeks know only the schedule, week v also the known-unavailable flags.
    """
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG")]
    plays = {(w, t) for w, a, h in zip(g.week, g.away_team, g.home_team) for t in (a, h)}
    out = {}
    for v in range(LB.FIRST_WAIVER, WEEKS + 1):
        tm = teams[v - 1]
        a = np.array([[(j, t) in plays for j in range(v, WEEKS + 1)] for t in tm])
        a[:, 0] &= ~gone[:, v - 1]
        out[v] = a
    return out


# ---------------------------------------------------------------- blocking on the wire

def blocking_policy(S, gone, state, spec, log):
    """Hole-aware streaming, except that a week with no hole may be spent on a block.

    The seat reads, at its own turn in the waiver order: every roster as it stands right
    now, the standings through the weeks already played, this week's priority order, and
    the free pool left after the teams ahead of it have moved. All of it is on a real
    league's home page at that moment; none of it is week-w information.
    """
    stream = LB.streaming_policy(S, gone, [])
    if spec is None:
        return stream
    harm_min, cost_max = spec

    def move(S_, held, val, free, w):
        g = gone[:, w - 1]
        if sum(LB.holes(S, held, g).values()):
            return stream(S_, held, val, free, w)       # my own lineup comes first
        t = state["target"]
        if t is None:                                   # leading the league: nobody to block
            return stream(S_, held, val, free, w)
        pts, avail = state["pts"][:, w - 1], state["avail"][w]
        theirs = np.array(state["rosters"][t])
        cut = LB.cuttable(S, theirs)
        if not len(cut) or not len(free):
            return stream(S_, held, val, free, w)

        # What the wire is worth to the target: its rest-of-season starting lineup after
        # it adds the free agent and cuts its own worst, the drop its consensus policy
        # would make. Only the best BLOCK_K free agents by wire value are worth testing.
        cand = free[np.argsort(-val[free], kind="stable")[:BLOCK_K]]
        base_t = ST.lineup_value(pts, avail, theirs[None, :])[0]
        vd = np.where(np.isfinite(val[cut]), val[cut], -np.inf)
        dt = int(cut[int(np.argmin(vd))])
        keep = np.array([i for i in theirs if i != dt])
        harm = ST.lineup_value(
            pts, avail, np.column_stack([np.tile(keep, (len(cand), 1)), cand])) - base_t
        j = int(np.argmax(harm))
        if harm[j] <= EPS or harm[j] < harm_min:
            return stream(S_, held, val, free, w)
        add = int(cand[j])

        # My own drop: the cuttable player who costs me least, and never one that opens a
        # hole in my own week-w lineup, since a block that costs me a starter is not free.
        mine = np.array(held)
        base_m = ST.lineup_value(pts, avail, mine[None, :])[0]
        rows, drops = [], []
        for d in LB.cuttable(S, mine):
            after = [i for i in held if i != d] + [add]
            if sum(LB.holes(S, after, g).values()):
                continue
            rows.append(after)
            drops.append(int(d))
        if not rows:
            return stream(S_, held, val, free, w)
        cost = ST.lineup_value(pts, avail, np.array(rows)) - base_m
        b = int(np.argmax(cost))
        if cost[b] < -cost_max:
            return stream(S_, held, val, free, w)

        cons = LB.consensus_move(S, held, val, free, w)
        mv = (add, drops[b])
        log.append({
            "week": w, "target": t, "add": S.ids[add], "add_pos": S.pos[add],
            "drop": S.ids[drops[b]], "drop_pos": S.pos[drops[b]],
            "harm": float(harm[j]), "cost": float(cost[b]),
            "is_top_fa": bool(add == int(free[int(np.argmax(val[free]))])),
            "would_start": bool(add in LB.starters(S, list(keep) + [add], val, g)),
            "same_as_cons": mv == cons, "cons_moves": cons is not None,
            "add_val": float(val[add]), "drop_val": float(val[drops[b]]),
        })
        return mv

    return move


# ---------------------------------------------------------------- the trade pick

def search_pick(S, rosters, seat, v, val, pts, avail, k, rule, target):
    """TC.search_mutual's enumeration, with a choice of which accepted offer to take.

    rule "mine" takes the largest own gain, exactly as TC.search_mutual does; the
    mechanics check asserts the two agree. Rule "rival" treats offers within NEAR_BEST of
    the best gain as equivalent to the seat and breaks that tie on what the trade does to
    the other side: avoid the standings-closest rival first, then give the receiving side
    the smallest lineup gain, then the seat's own gain.
    """
    mine = np.array(rosters[seat])
    base = ST.lineup_value(pts, avail, mine[None, :])[0]
    on = np.zeros(len(S.ids), bool)
    for r in rosters:
        on[r] = True
    free = np.where(~on & np.isfinite(val))[0]
    fa = int(free[np.argmax(val[free])]) if len(free) else -1
    cm = ST.counts_of(S, mine)
    pairs = [(a, b) for a in range(len(mine)) for b in range(a + 1, len(mine))]
    cand, meta = [], []
    for o in range(TEAMS):
        if o == seat:
            continue
        theirs = np.array(rosters[o])
        co = ST.counts_of(S, theirs)
        up1 = ST.lineup_value(pts, avail, np.column_stack([np.tile(mine, (len(theirs), 1)), theirs]))
        up2 = ST.lineup_value(pts, avail, np.column_stack(
            [np.tile(mine, (len(theirs), 1)), theirs, np.full(len(theirs), fa)]))
        for j, r in enumerate(theirs):
            pr = S.pos[r]
            if not np.isfinite(val[r]):
                continue
            if up1[j] > base + EPS:
                for a, gv in enumerate(mine):
                    if not val[gv] / (1 + k) >= val[r]:
                        continue
                    pg = S.pos[gv]
                    c1 = dict(cm); c1[pg] -= 1; c1[pr] += 1
                    c2 = dict(co); c2[pr] -= 1; c2[pg] += 1
                    if not (ST.legal(c1) and ST.legal(c2)):
                        continue
                    new = mine.copy(); new[a] = r
                    cand.append(new)
                    meta.append((o, (int(gv),), int(r), -1))
            if up2[j] > base + EPS and fa >= 0:
                for a, b in pairs:
                    g1, g2 = mine[a], mine[b]
                    if not (val[g1] + val[g2]) / (1 + k) >= val[r]:
                        continue
                    c1 = dict(cm); c1[S.pos[g1]] -= 1; c1[S.pos[g2]] -= 1
                    c1[pr] += 1; c1[S.pos[fa]] += 1
                    c2 = dict(co); c2[pr] -= 1; c2[S.pos[g1]] += 1; c2[S.pos[g2]] += 1
                    if not (ST.legal(c1) and ST.legal(c2)):
                        continue
                    new = np.concatenate([np.delete(mine, [a, b]), [r, fa]])
                    cand.append(new)
                    meta.append((o, (int(g1), int(g2)), int(r), fa))
    diag = {"n_near": 0}
    if not cand:
        return None, diag
    gain = ST.lineup_value(pts, avail, np.array(cand)) - base
    ok = np.where(gain > EPS)[0]
    if not len(ok):
        return None, diag
    opp_gain = np.full(len(cand), np.nan)
    by_opp = {}
    for i in ok:
        by_opp.setdefault(meta[i][0], []).append(i)
    for o, idx in by_opp.items():
        theirs = np.array(rosters[o])
        before = ST.lineup_value(pts, avail, theirs[None, :])[0]
        gives = np.array([list(meta[i][1]) + [-1] * (2 - len(meta[i][1])) for i in idx])
        after = TC.opponent_after(S, theirs, np.array([meta[i][2] for i in idx]), gives, val)
        opp_gain[idx] = ST.lineup_value(pts, avail, after) - before
    pick = ok[opp_gain[ok] >= -EPS]
    if not len(pick):
        return None, diag
    if rule == "mine":
        best = pick[np.argmax(gain[pick])]
    else:
        near = pick[gain[pick] >= NEAR_BEST * gain[pick].max()]
        diag["n_near"] = len(near)
        key = np.lexsort((-gain[near], opp_gain[near],
                          np.array([meta[i][0] == target for i in near])))
        best = near[key[0]]
    o, give, get, add = meta[best]
    return {"opp": o, "give": give, "get": get, "fa": add, "gain": float(gain[best]),
            "base": float(base), "opp_gain": float(opp_gain[best])}, diag


# ---------------------------------------------------------------- the season

def simulate(S, drafted, values, seat, movers, state, rule, log, dec):
    """TC.simulate at (mutual, cap 2, GAIN_MIN), with the standings published to the
    waiver policy each week and a choice of trade pick rule."""
    rosters = [list(r) for r in drafted]
    state["rosters"] = rosters
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    every = set(range(TEAMS))
    done = 0
    for w in range(WEEKS):
        for t in range(TEAMS):
            scores[t, w] = LB.week_points(S, rosters[t], S.ecr_val, w)
        if w + 1 < WEEKS:
            order = [t for t in LB.priority(scores, w) if t in every]
            # Waiver order is worst record first, so the standings are its reverse and the
            # rival immediately above me always moves after me: he is blockable, and the
            # one below me never is. Leading the league leaves nobody to block.
            standings = list(reversed(order))
            r = standings.index(seat)
            state["target"] = standings[r - 1] if r > 0 else None
            if CHECK and state["target"] is not None:
                assert order.index(state["target"]) > order.index(seat)
            moves += LB.waiver_week(S, rosters, values, order, w + 2, movers)
            v = w + 2
            if v not in ST.TRADE_WEEKS or done >= 2:
                continue
            val = values[seat][:, v - 1]
            pts, avail = state["pts"][:, v - 1], state["avail"][v]
            if rule == "mine" and not CHECK:
                found, diag = TC.search_mutual(S, rosters, seat, v, val, pts, avail, 0.0)
            else:
                found, diag = search_pick(S, rosters, seat, v, val, pts, avail, 0.0,
                                          rule, state["target"])
                if CHECK:
                    ref, _ = TC.search_mutual(S, rosters, seat, v, val, pts, avail, 0.0)
                    if rule == "mine":
                        assert (ref is None and found is None) or all(
                            ref[x] == found[x] for x in ("opp", "give", "get", "fa", "gain"))
                    elif found is not None:
                        # a rival-aware pick is still an accepted offer, so it cannot beat
                        # the best own gain, and it must stay within the tolerance
                        assert found["gain"] <= ref["gain"] + EPS
                        assert found["gain"] >= NEAR_BEST * ref["gain"] - EPS
                        assert found["opp_gain"] >= -EPS
            take = found is not None and found["gain"] >= GAIN_MIN
            dec.append({"week": v, "offer": found is not None,
                        "gain": found["gain"] if found else np.nan, "executed": take,
                        "target": state["target"], **diag})
            if not take:
                continue
            o = found["opp"]
            opp_before = ST.lineup_value(pts, avail, np.array(rosters[o])[None, :])[0]
            before = list(rosters[seat])
            drop = ST.execute(S, rosters, seat, found, val)
            done += 1
            opp_after = ST.lineup_value(pts, avail, np.array(rosters[o])[None, :])[0]
            assert abs((opp_after - opp_before) - found["opp_gain"]) < 1e-6
            assert opp_after >= opp_before - EPS
            cf = sum(LB.week_points(S, rosters[seat], S.ecr_val, j) -
                     LB.week_points(S, before, S.ecr_val, j) for j in range(v - 1, WEEKS))
            log.append({
                "week": v, "opp": o, "target": state["target"],
                "hit_target": o == state["target"],
                "structure": "1-for-1" if found["fa"] < 0 else "2-for-1",
                "give_pos": "/".join(S.pos[i] for i in found["give"]),
                "get_pos": S.pos[found["get"]],
                "proj_gain": found["gain"], "opp_proj_gain": float(opp_after - opp_before),
                "cf_pts": cf,
            })
    return scores, moves, rosters


def run():
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, block_log, trade_log, decisions = [], [], [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        ST.POS_CODE_ARR = np.array([ST.POS_CODE.get(p, -1) for p in S.pos])
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        teams = ST.team_at(S, y)
        pts = ST.ros_points(S, y, crv)
        avail = future_avail_all(S, y, gone, teams)
        for v in ST.TRADE_WEEKS:      # the wider table must agree with the trade one
            assert (avail[v] == ST.future_avail(S, y, gone, teams)[v]).all()
        for lg in range(LB.LEAGUES):
            # The same draws in the same order as sim_tradecap, so the drafts match.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for kk in range(TEAMS + 1):
                score = S.ecr_mean + LB.NOISE * S.ecr_sd * noise[kk]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = LB.schedules(rng, LB.SCHEDULES)
            base = [cons_val] * TEAMS
            for seat in range(TEAMS):
                orders = list(ecr_orders[:TEAMS])
                orders[seat] = S.exact_order
                drafted = LB.draft(S, orders)
                for arm, spec, rule in ARMS:
                    state = {"pts": pts, "avail": avail, "target": None}
                    blog, log, dec = [], [], []
                    movers = {seat: blocking_policy(S, gone, state, spec, blog)}
                    sc, moves, _ = simulate(S, drafted, base, seat, movers, state, rule,
                                            log, dec)
                    res = [LB.season_outcome(sc, s) for s in scheds]
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": LB.all_play(sc, seat),
                        "pts_reg": sc[seat, :REG].sum(),
                        "pts_playoff": sc[seat, REG:].sum(),
                        "moves": moves[seat], "trades": len(log), "blocks": len(blog),
                    })
                    for e in blog + log + dec:
                        e.update(season=y, league=lg, seat=seat, arm=arm)
                    block_log += blog
                    trade_log += log
                    decisions += dec
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(block_log), pd.DataFrame(trade_log),
            pd.DataFrame(decisions))


def report(d, bl, tl, dec):
    print("\n=== blocking: rival-aware waivers and trades on top of mutual_cap2 ===")
    print(f"{'arm':17s} {'title':>7s} {'all-play':>9s} {'reg pts':>8s} {'adds':>6s} "
          f"{'trades':>7s} {'blocks':>7s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:17s} {100 * g.title.mean():6.1f}% {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.moves.mean():6.2f} {g.trades.mean():7.2f} "
              f"{g.blocks.mean():7.2f}")
    print("\npaired against the mutual_cap2 control (season-cluster bootstrap 90% interval):")
    verdict = {}
    ctrl = d[d.arm == "mutual_cap2"]
    for arm, *_ in ARMS[1:]:
        verdict[arm] = LB.paired(d, arm, "mutual_cap2", f"{arm} - mutual_cap2")
        pos = d[d.arm == arm].merge(ctrl, on=["season", "league", "seat"])
        by = pos.assign(x=pos.all_play_x - pos.all_play_y).groupby("season").x.mean()
        tag = "  <- preregistered verdict" if arm == VERDICT else ""
        print(f"    rule {'passes' if verdict[arm] else 'fails'}; seasons positive "
              f"{(by > 0).sum()}/5" + tag)
        print("    title by season " + " ".join(
            f"{100 * x:+.1f}" for x in pos.assign(x=pos.title_x - pos.title_y)
            .groupby("season").x.mean()))

    n = d.groupby("arm").size()
    print("\nreported, not gated: blocking moves")
    if len(bl):
        print(f"  {'arm':17s} {'blocks':>7s} {'harm':>7s} {'cost':>7s} {'top FA':>7s} "
              f"{'starts':>7s} {'differs':>8s} {'wk':>5s}")
        for arm, *_ in ARMS[1:]:
            x = bl[bl.arm == arm]
            if not len(x):
                continue
            print(f"  {arm:17s} {len(x) / n[arm]:7.2f} {x.harm.mean():+7.1f} "
                  f"{x.cost.mean():+7.1f} {x.is_top_fa.mean():7.2f} "
                  f"{x.would_start.mean():7.2f} {(~x.same_as_cons).mean():8.2f} "
                  f"{x.week.mean():5.1f}")
        print("  (harm: the target's projected rest-of-season starting-lineup gain that the "
              "block denies;\n   cost: my own projected change; top FA: the blocked player "
              "was also the best free agent\n   by wire value; starts: he would have been in "
              "the target's week-w starting lineup;\n   differs: the block is not the move "
              "consensus would have made)")
        for arm, *_ in ARMS[1:]:
            x = bl[bl.arm == arm]
            if len(x):
                print(f"  {arm:17s} blocks by season " + " ".join(
                    f"{len(x[x.season == y]) / (n[arm] / 5):.2f}" for y in LB.SEASONS)
                    + "   positions " + " ".join(
                        f"{p} {c / n[arm]:.2f}" for p, c in x.add_pos.value_counts().items()))
    else:
        print("  none")

    print("\nreported, not gated: trades")
    print(f"  {'arm':17s} {'trades':>7s} {'proj gain':>9s} {'opp gain':>9s} "
          f"{'to target':>10s} {'near set':>9s} {'cf pts':>7s}")
    for arm, *_ in ARMS[1:] + [ARMS[0]]:
        t, x = tl[tl.arm == arm], dec[dec.arm == arm]
        if not len(t):
            continue
        near = x.n_near[x.n_near > 0].mean() if "n_near" in x else np.nan
        print(f"  {arm:17s} {len(t) / n[arm]:7.2f} {t.proj_gain.median():9.1f} "
              f"{t.opp_proj_gain.median():9.1f} {t.hit_target.mean():10.2f} "
              f"{near:9.1f} {t.cf_pts.mean():+7.1f}")
    print("  (to target: the share of executed trades made with the standings-closest "
          "rival;\n   near set: offers within NEAR_BEST of the best gain, when the rival "
          "rule is used)")
    return verdict


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = "" if LB.LEAGUES == 20 else f"_check{LB.LEAGUES}"
    CHECK = bool(check) and "--mechanics" in sys.argv
    d, bl, tl, dec = run()
    d.to_parquet(f"data/league_blocking{check}_noise{LB.NOISE:g}.parquet")
    bl.to_parquet(f"data/league_blocking_blocks{check}_noise{LB.NOISE:g}.parquet")
    tl.to_parquet(f"data/league_blocking_log{check}_noise{LB.NOISE:g}.parquet")
    dec.to_parquet(f"data/league_blocking_decisions{check}_noise{LB.NOISE:g}.parquet")
    if check and "--mechanics" in sys.argv:
        sys.exit()                   # the mechanics check reads the files, not outcomes
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    report(d, bl, tl, dec)
