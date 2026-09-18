"""Does acting on the QB / own-receiver correlation win games? (prereg_stack.md)

prereg_variance.md killed the last independent-player variance lever: weekly spread rises
with consensus value at every position, so the highest-projected lineup is already close
to the widest one, and 12 projected points bought at most +0.33 of extra spread against
-1.63 of floor. The one lever an independent-player model cannot see is correlation. A
quarterback and the receiver he throws to share drives, so starting both widens the weekly
total without moving either player's mean.

The correlation is measured from prior seasons in prereg_stack.md rather than taken from
correlate.py's assumed default, and what it buys is small and known before the run: at
rho ~= 0.25 a stacked lineup gains roughly half a point of standard deviation, so a stack
is worth about one projected point, not three. The arms spend that, spend more only when
the seat is the underdog, and take a stack partner off the wire when he costs nothing.

Arms: mutual_cap2 (control, sim_tradecap's verdict arm), stack1, stack_dog, stack_wire.

    .venv/bin/python draft/sim_stack.py --noise 1
    .venv/bin/python draft/sim_stack.py --noise 1 --leagues 2 --schedules 20 --mechanics
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import correlate as CO
import league_backtest as LB
import sim_trades as ST
import sim_tradecap as TC
import sim_variance as SV
import winprob as WP
import settings as CFG
from league_backtest import FLEX, POS, REG, SLOTS, TEAMS, WEEKS

SET = CFG.get()
C_PLAIN = 1.0        # projected points a stack is worth at the measured correlation
C_DOG = 3.0          # what an underdog will pay, where width is worth more than mean
WIRE_EPS = 0.5       # "near-equal" consensus value on the wire, in points over replacement
CONTROL = "mutual_cap2"
CONTROL_SPEC = SV.CONTROL_SPEC            # ("mutual", 2, TC.GAIN_MIN)
LINEUP_ARMS = ["stack1", "stack_dog"]
ARMS = LINEUP_ARMS + ["stack_wire"]
VERDICT = "stack1"
RECEIVERS = CO.RECEIVERS
ACC0 = {"weeks": 0, "avail": 0, "free": 0, "paid": 0, "dmu": 0.0, "und": 0, "und_paid": 0,
        "won": 0, "won_cons": 0, "res_st": 0.0, "res_st_n": 0, "res_un": 0.0, "res_un_n": 0}


# ---------------------------------------------------------------- lineups

def best_lineup(S, roster, w, force=()):
    """The most-projected legal lineup in week w that starts everyone in `force`.

    Slots are filled the way LB.week_points fills them (top of each position, then the
    best flex left over), which is the maximum, so with nothing forced this is the
    consensus lineup. A forced pass catcher can sit in his own slot or in the flex, so
    both are tried and the better kept. Returns (projected points, lineup) or None.
    """
    r = [i for i in roster if S.elig[i, w]]
    if any(i not in r for i in force):
        return None
    v = {i: S.ecr_val[i, w] for i in r}
    by = {p: sorted([i for i in r if S.pos[i] == p], key=lambda i: -v[i]) for p in POS}
    have = {p: len(by[p]) for p in POS}
    need = {p: min(k, have[p]) for p, k in SLOTS.items()}
    nflex = SET.flex_slots if sum(have[p] - need[p] for p in FLEX) > 0 else 0
    forced_flex = [i for i in force if S.pos[i] in FLEX]
    best = None
    for fx in [None] + (forced_flex[:1] if nflex else []):
        used, ok = [], True
        for p in SLOTS:
            must = [i for i in force if S.pos[i] == p and i != fx]
            if len(must) > need[p]:
                ok = False
                break
            rest = [i for i in by[p] if i not in must and i != fx]
            used += must + rest[:need[p] - len(must)]
        if not ok:
            continue
        if nflex:
            if fx is not None:
                used.append(fx)
            else:
                left = [i for i in r if S.pos[i] in FLEX and i not in used]
                if left:
                    used.append(max(left, key=lambda i: v[i]))
        mu = sum(v[i] for i in used)
        if best is None or mu > best[0]:
            best = (mu, tuple(sorted(used)))
    return best


def stack_pairs(S, roster, team_w, w):
    """(quarterback, own pass catcher) pairs on a roster, both eligible this week."""
    live = [i for i in roster if S.elig[i, w]]
    return [(q, m) for q, mates in CO.stacks(S.pos, team_w, live) for m in mates]


def week_table(S, roster, team_w, w, opp_mu, real0):
    """One seat-week, priced once and shared by both lineup arms and all twenty schedules.

    Holds the consensus lineup, the cheapest stacked lineup at each budget, and every
    possible head-to-head opponent's projected total, which is what tells the seat whether
    it is the underdog. `real0` is the control's own realised points for the week, so an
    arm that declines to stack scores exactly what the control scored.
    """
    mu0, lu0 = best_lineup(S, roster, w) or (0.0, ())
    pairs = stack_pairs(S, roster, team_w, w)
    started = sum(1 for q, m in pairs if q in lu0 and m in lu0)
    out = {"mu0": mu0, "lu0": lu0, "real0": float(real0), "avail": bool(pairs),
           "free": started > 0, "npair": len(pairs), "opp_mu": opp_mu}
    for name, c in (("plain", C_PLAIN), ("dog", C_DOG)):
        best = None
        for q, m in pairs:
            got = best_lineup(S, roster, w, force=(q, m))
            if got is None or got[0] < mu0 - c - 1e-9:
                continue
            # Cheapest stack first; among equals, the one that starts more stacked pairs.
            key = (got[0], sum(1 for a, b in pairs if a in got[1] and b in got[1]))
            if best is None or key > best[0]:
                best = (key, got[1])
        changed = best is not None and best[1] != lu0
        lu = best[1] if changed else lu0
        out[name] = {"changed": changed, "lu": lu,
                     "mu": float(S.ecr_val[list(lu), w].sum()) if lu else 0.0,
                     "real": float(S.actual[list(lu), w].sum()) if changed else float(real0)}
    # C = 0: the consensus lineup, what stack_dog plays in a week it is not the underdog.
    out["none"] = {"changed": False, "lu": lu0, "mu": mu0, "real": float(real0)}
    return out


def opponent_mus(S, snap_w, seat, w):
    """Each other team's consensus-lineup projection: the number the seat compares itself
    with to know whether it is the underdog. Every opponent starts its consensus lineup on
    the control's path, so this is what the seat is actually scored against."""
    mu = np.full(TEAMS, np.nan)
    for t in range(TEAMS):
        if t == seat:
            continue
        lu = WP.consensus_lineup(S, snap_w[t], w)
        mu[t] = S.ecr_val[list(lu), w].sum() if lu else 0.0
    return mu


# ---------------------------------------------------------------- the wire arm

def stack_wire_policy(S, gone, teams, log):
    """The streaming wire, with the add broken toward a pass catcher on a rostered QB's
    NFL team.

    The swap is only ever to a player at the same position and within WIRE_EPS of the
    intended add's consensus value, so the move still closes the same hole, cuts the same
    player and stays legal. Paying real value for a stack partner is not what this arm
    asks; it asks whether a free preference is worth having.
    """
    base = LB.streaming_policy(S, gone, [])

    def move(S_, held, val, free, w):
        got = base(S_, held, val, free, w)
        if got is None:
            return None
        add, drop = got
        if S.pos[add] not in RECEIVERS:
            return got
        tm = teams[w - 1]
        mine = {tm[i] for i in held if S.pos[i] == "QB" and tm[i]}
        if not mine:
            return got
        alt = [int(i) for i in free if S.pos[i] == S.pos[add] and tm[i] in mine
               and np.isfinite(val[i]) and val[i] >= val[add] - WIRE_EPS]
        if not alt or int(add) in alt:
            return got
        pick = max(alt, key=lambda i: val[i])
        log.append({"week": w, "gave_up": float(val[add] - val[pick]),
                    "add_pos": S.pos[pick]})
        return (pick, drop)

    return move


# ---------------------------------------------------------------- the season

def play(tab, ctrl, scheds, seat, arm):
    """One lineup arm's regular season under every schedule, on the control's roster path.

    stack1 ignores the opponent so its score repeats across schedules; stack_dog reads the
    week's opponent, so both run the loop per schedule and are scored the same way.
    """
    pts = np.zeros((len(scheds), REG))
    acc = dict(ACC0)
    opps = [[next(b if a == seat else a for a, b in rd if seat in (a, b)) for rd in s[:REG]]
            for s in scheds]
    for si in range(len(scheds)):
        for w in range(REG):
            t = tab[w]
            opp = opps[si][w]
            under = bool(t["mu0"] < t["opp_mu"][opp])
            if arm == "stack_dog":
                use = t["dog"] if under else t["none"]
            else:
                use = t["plain"]
            pts[si, w] = use["real"] if use["changed"] else t["real0"]
            acc["weeks"] += 1
            acc["avail"] += t["avail"]
            acc["free"] += t["free"]
            acc["und"] += under
            # Realised spread: the week's points minus its own projection, split by
            # whether the lineup actually started a quarterback with his own receiver.
            r = pts[si, w] - (use["mu"] if use["changed"] else t["mu0"])
            k = "st" if (t["free"] or use["changed"]) else "un"
            acc[f"res_{k}"] += r * r
            acc[f"res_{k}_n"] += 1
            if use["changed"]:
                acc["paid"] += 1
                acc["und_paid"] += under
                acc["dmu"] += t["mu0"] - use["mu"]
                acc["won"] += pts[si, w] > ctrl[opp, w]
                acc["won_cons"] += t["real0"] > ctrl[opp, w]
    return pts, acc


def control_acc(tab):
    """The control's own stack diagnostics: how often a stack existed at all, how often
    its consensus lineup started one for free, and its realised spread either way."""
    acc = dict(ACC0)
    for w in range(REG):
        t = tab[w]
        acc["weeks"] += 1
        acc["avail"] += t["avail"]
        acc["free"] += t["free"]
        r = t["real0"] - t["mu0"]
        k = "st" if t["free"] else "un"
        acc[f"res_{k}"] += r * r
        acc[f"res_{k}_n"] += 1
    return acc


def row(y, lg, seat, arm, ctrl, mine, scheds, res, moves, acc):
    """One arm's season summary. `mine` is the seat's weekly points per schedule; `res`
    is set instead for an arm scored on its own single score matrix."""
    if res is not None:
        ap = LB.all_play(ctrl, seat)
        title = float(np.mean([r[2][seat] for r in res]))
        playoff = float(np.mean([r[1][seat] for r in res]))
        wins = float(np.mean([r[0][seat] for r in res]))
        reg = float(ctrl[seat, :REG].sum())
    else:
        aps, titles, playoffs, winss = [], [], [], []
        for si, s in enumerate(scheds):
            sc = ctrl.copy()
            sc[seat, :REG] = mine[si]
            w_, p_, t_ = LB.season_outcome(sc, s)
            aps.append(LB.all_play(sc, seat))
            titles.append(t_[seat])
            playoffs.append(p_[seat])
            winss.append(w_[seat])
        ap, title = float(np.mean(aps)), float(np.mean(titles))
        playoff, wins = float(np.mean(playoffs)), float(np.mean(winss))
        reg = float(mine.mean(0).sum())
    return {"season": y, "league": lg, "seat": seat, "arm": arm, "title": title,
            "playoff": playoff, "wins": wins, "all_play": ap, "pts_reg": reg,
            "pts_playoff": float(ctrl[seat, REG:].sum()), "moves": int(moves[seat]), **acc}


def run(check=False):
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, mech, wire_log = [], [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        ST.POS_CODE_ARR = np.array([ST.POS_CODE.get(p, -1) for p in S.pos])
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        teams = ST.team_at(S, y)                        # NFL team as of the wire decision
        wk_teams = CO.weekly_teams(S.ids, y, WEEKS)     # NFL team in the week itself
        inp = {"pts": ST.ros_points(S, y, crv),
               "avail": ST.future_avail(S, y, gone, teams)}
        base_vals = [cons_val] * TEAMS
        for lg in range(LB.LEAGUES):
            # The same draws in the same order as sim_tradecap, so the control's rosters
            # and scores are identical to its mutual_cap2 arm.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for kk in range(TEAMS + 1):
                score = S.ecr_mean + LB.NOISE * S.ecr_sd * noise[kk]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = LB.schedules(rng, LB.SCHEDULES)
            for seat in range(TEAMS):
                orders = list(ecr_orders[:TEAMS])
                orders[seat] = S.exact_order
                drafted = LB.draft(S, orders)
                t0 = time.perf_counter()
                movers = {seat: LB.streaming_policy(S, gone, [])}
                ctrl, moves, final, snap = SV.path(S, drafted, base_vals, seat, movers, inp)
                if check:
                    mv2 = {seat: LB.streaming_policy(S, gone, [])}
                    sc2, mo2, fin2 = TC.simulate(S, drafted, base_vals, seat, mv2,
                                                 CONTROL_SPEC, inp, [], [])
                    mech.append({"season": y, "league": lg, "seat": seat,
                                 "d_scores": float(np.abs(ctrl - sc2).max()),
                                 "same_moves": bool(np.array_equal(moves, mo2)),
                                 "same_rosters": all(sorted(a) == sorted(b)
                                                     for a, b in zip(final, fin2))})
                tab = []
                for w in range(REG):
                    tb = week_table(S, snap[w][seat], wk_teams[w], w,
                                    opponent_mus(S, snap[w], seat, w), ctrl[seat, w])
                    tab.append(tb)
                    if check:
                        mech[-1].setdefault("cons_same", []).append(
                            tb["lu0"] == WP.consensus_lineup(S, snap[w][seat], w))
                        mech[-1].setdefault("legal", []).append(all(
                            WP.legality(S, snap[w][seat], tb[k]["lu"], w)["legal"]
                            for k in ("plain", "dog", "none")))
                        mech[-1].setdefault("dmu", []).append(
                            max(tb["mu0"] - tb[k]["mu"] for k in ("plain", "dog")))
                res = [LB.season_outcome(ctrl, s) for s in scheds]
                rows.append(row(y, lg, seat, CONTROL, ctrl, None, scheds, res, moves,
                                control_acc(tab)))
                for arm in LINEUP_ARMS:
                    mine, acc = play(tab, ctrl, scheds, seat, arm)
                    rows.append(row(y, lg, seat, arm, ctrl, mine, scheds, None, moves, acc))

                # The wire arm changes who is on the roster, so it needs its own season.
                wlog = []
                mv3 = {seat: stack_wire_policy(S, gone, teams, wlog)}
                sc3, mo3, fin3 = TC.simulate(S, drafted, base_vals, seat, mv3,
                                             CONTROL_SPEC, inp, [], [])
                res3 = [LB.season_outcome(sc3, s) for s in scheds]
                wacc = dict(ACC0, weeks=REG, paid=len(wlog),
                            dmu=float(sum(e["gave_up"] for e in wlog)))
                rows.append(row(y, lg, seat, "stack_wire", sc3, None, scheds, res3, mo3,
                                wacc))
                for e in wlog:
                    e.update(season=y, league=lg, seat=seat)
                wire_log += wlog
                if check:
                    mech[-1]["sec"] = time.perf_counter() - t0
                    mech[-1]["wire_swaps"] = len(wlog)
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(mech), pd.DataFrame(wire_log)


# ---------------------------------------------------------------- reporting

def report(d):
    print("\n=== stacking: acting on the quarterback / own-receiver correlation ===")
    print("12-team PPR leagues on 2021-25. Every arm plays mutual_cap2 (hole-aware "
          "streaming plus at\nmost two win-win trades). stack1 and stack_dog change only "
          "the weeks 1-14 lineup; stack_wire\nchanges only which free agent the wire takes "
          "when two are worth nearly the same.\n")
    print(f"{'arm':12s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:12s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% {g.pts_reg.mean():8.0f}")

    print(f"\npaired against the {CONTROL} control (season-cluster bootstrap 90% interval):")
    verdict = {}
    key = ["season", "league", "seat"]
    b = d[d.arm == CONTROL].set_index(key)
    for arm in ARMS:
        verdict[arm] = LB.paired(d, arm, CONTROL, f"{arm} - {CONTROL}")
        a = d[d.arm == arm].set_index(key)
        ap = (a.all_play - b.loc[a.index].all_play).groupby("season").mean()
        ti = (a.title - b.loc[a.index].title).groupby("season").mean()
        tag = "  <- preregistered verdict" if arm == VERDICT else ""
        print(f"    rule {'passes' if verdict[arm] else 'fails'}; seasons positive "
              f"{(ap > 0).sum()}/5{tag}")
        print("    title by season " + " ".join(f"{100 * v:+.1f}" for v in ti))

    print("\ndiagnostics (not gated). All-play is schedule-free, while a stack's benefit "
          "is head-to-head,\nso both are reported and neither alone settles it.")
    print(f"  {'arm':12s} {'stack avail':>12s} {'free stack':>11s} {'paid':>8s} "
          f"{'pts paid':>9s} {'h2h won':>9s} {'vs consensus':>13s}")
    for arm in [CONTROL] + ARMS:
        g = d[d.arm == arm]
        n, k = g.weeks.sum(), g.paid.sum()
        av = f"{100 * g.avail.sum() / n:11.1f}%" if g.avail.sum() else f"{'n/a':>12s}"
        fr = f"{100 * g.free.sum() / n:10.1f}%" if g.free.sum() else f"{'n/a':>11s}"
        print(f"  {arm:12s} {av} {fr} {100 * k / n:7.2f}% {g.dmu.sum() / max(k, 1):9.2f} "
              f"{g.won.sum() / max(k, 1):9.3f} "
              f"{(g.won.sum() - g.won_cons.sum()) / max(k, 1):+13.4f}")
    print("\n  realised weekly spread, lineups that started a stack against those that did "
          "not\n  (root mean square of the week's points minus its own projection):")
    for arm in [CONTROL] + ARMS:
        g = d[d.arm == arm]
        ns, nu = g.res_st_n.sum(), g.res_un_n.sum()
        if not (ns or nu):
            continue
        rs = np.sqrt(g.res_st.sum() / ns) if ns else np.nan
        ru = np.sqrt(g.res_un.sum() / nu) if nu else np.nan
        print(f"  {arm:12s} stacked {rs:5.2f} (n={ns:,})   unstacked {ru:5.2f} (n={nu:,})")
    print("\n  underdog weeks, and how often each arm paid for a stack in one:")
    for arm in LINEUP_ARMS:
        g = d[d.arm == arm]
        print(f"  {arm:12s} underdog in {100 * g.und.sum() / g.weeks.sum():.1f}% of weeks, "
              f"paid in {100 * g.und_paid.sum() / max(g.und.sum(), 1):.2f}% of those")
    print("  by season, share of weeks each arm paid for a stack:")
    for arm in ARMS:
        g = d[d.arm == arm].groupby("season")
        print(f"    {arm:12s} " + " ".join(f"{100 * x.paid.sum() / x.weeks.sum():.2f}"
                                           for _, x in g))
    return verdict


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    if "--schedules" in sys.argv:
        LB.SCHEDULES = int(sys.argv[sys.argv.index("--schedules") + 1])
    CFG.announce_run(noise=LB.NOISE, leagues=LB.LEAGUES)
    mechanics = "--mechanics" in sys.argv
    check = "" if (LB.LEAGUES == 20 and LB.SCHEDULES == 20) \
        else f"_check{LB.LEAGUES}x{LB.SCHEDULES}"
    d, mech, wl = run(check=bool(check) and mechanics)
    d.to_parquet(f"data/league_stack{check}_noise{LB.NOISE:g}.parquet")
    if len(wl):
        wl.to_parquet(f"data/league_stack_wire{check}_noise{LB.NOISE:g}.parquet")
    if check and mechanics:
        print(f"\nmechanics, {LB.LEAGUES} leagues x {LB.SCHEDULES} schedules, "
              f"noise {LB.NOISE:g}:")
        print(f"  control reproduces {CONTROL}: max abs score difference "
              f"{mech.d_scores.max():.6g}, moves {mech.same_moves.all()}, rosters "
              f"{mech.same_rosters.all()}")
        same = np.concatenate(mech.cons_same.values)
        legal = np.concatenate(mech.legal.values)
        dmu = np.concatenate(mech.dmu.values)
        print(f"  best_lineup with nothing forced equals winprob's consensus lineup in "
              f"{same.mean():.3f} of {len(same)} seat-weeks")
        print(f"  every stacked lineup legal in {legal.mean():.3f} of them; largest "
              f"projected points given up {dmu.max():.2f} (bars {C_PLAIN}/{C_DOG})")
        print(f"  seconds per seat-season: mean {mech.sec.mean():.2f}, "
              f"max {mech.sec.max():.2f}")
        print(f"  wire swaps per seat-season: mean {mech.wire_swaps.mean():.2f}, "
              f"max {mech.wire_swaps.max()}")
        for arm in [CONTROL] + ARMS:
            g = d[d.arm == arm]
            print(f"    {arm:12s} stack available {100 * g.avail.sum() / g.weeks.sum():5.1f}%"
                  f" of weeks, started free {100 * g.free.sum() / g.weeks.sum():5.1f}%, "
                  f"paid for {100 * g.paid.sum() / g.weeks.sum():5.2f}%")
        sys.exit()                   # the mechanics check prints no outcome metric
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    report(d)
