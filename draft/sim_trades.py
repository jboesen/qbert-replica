"""Roster-fit trades on top of hole-aware streaming (prereg_trades.md, part B).

Streaming passed because it acts on need rather than on value: a zero in a starting slot
costs more than the rest-of-season value a move gives up. A trade is the same idea one
step further. Two teams can price players identically and both still gain, because a
fourth receiver is worth little to a team that starts two, and a lot to a team whose
second receiver is hurt. So the test seat looks for trades its consensus opponents
accept on value alone while it gains on fit: its own starting lineup over the rest of
the season, byes included.

Opponents never propose. They accept an offer when the consensus rest-of-season value
they receive, divided by (1 + k), is at least the value they give; k is an endowment
premium. The test seat makes at most one trade a week, in weeks 3-11, after the week's
waivers have cleared. Both sides keep 14 players: in a 2-for-1 the side one player short
adds the best free agent by consensus value and the side one player long cuts its
lowest-valued cuttable player, both by the harness's own wire rules.

    .venv/bin/python draft/sim_trades.py --noise 1 [--leagues 2]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
import roster as R
import weekly as W
import settings as CFG
from league_backtest import FLEX, POS, REG, SLOTS, TEAMS, WEEKS

SET = CFG.get()
TRADE_WEEKS = SET.trade_weeks       # decisions before weeks 3..11 by default
K_SWEEP = (0.0, 0.1, 0.2)
NEAR_BEST = SET.trade_near_best     # test B: "equal value" means within this of the best gain
EPS = 1e-6
POS_CODE = {p: i for i, p in enumerate(POS)}


# ---------------------------------------------------------------- inputs per season

def ros_points(S, y, crv):
    """Raw consensus rest-of-season points per game at each waiver decision, the same
    ranks league_backtest.waiver_values prices, before pricing over replacement.

    A starting lineup scores raw points, so roster fit is judged on these; opponents
    price trades over replacement, as they price the wire.
    """
    n = len(S.ids)
    ix = {p: i for i, p in enumerate(S.ids)}
    ros = pd.read_parquet("data/ecr_ros.parquet")
    ros = ros[ros.season == y]
    pre = pd.read_parquet("data/ecr_preseason.parquet")
    pre = pre[pre.season == y]
    out = np.full((n, WEEKS), np.nan)
    for w in range(LB.FIRST_WAIVER, WEEKS + 1):
        earlier = ros[ros.week < w]
        src = earlier[earlier.week == earlier.week.max()] if len(earlier) else pre
        src = src.assign(rank=src.groupby("pos").ecr.rank(method="first"))
        worst = src.groupby("pos")["rank"].max().to_dict()
        c_rank = np.full(n, np.nan)
        for p, r in zip(src.player_id, src["rank"]):
            if p in ix:
                c_rank[ix[p]] = r
        for p in POS:
            at = S.pos == p
            c_rank[at & np.isnan(c_rank)] = worst.get(p, 100) + 1
            out[at, w - 1] = C.rank_points(crv, S.pos[at], np.clip(c_rank[at], 1, 150))
    return out


def team_at(S, y):
    """Each player's NFL team as of each decision: the latest weekly roster row at or
    before week w-1, the same row known_unavailable reads. Row w-1 = decision before w."""
    ix = {p: i for i, p in enumerate(S.ids)}
    ro = pd.read_parquet(f"data/roster_weekly_{y}.parquet")
    ro = ro[(ro.game_type == "REG") & (ro.week <= WEEKS) & ro.gsis_id.isin(ix)]
    ro = ro.assign(team=W.norm_team(ro.team)).sort_values("week")
    rows = {w: list(zip(x.gsis_id, x.team)) for w, x in ro.groupby("week")}
    cur = np.array([""] * len(S.ids), dtype=object)
    out = np.empty((WEEKS, len(S.ids)), dtype=object)
    out[0] = cur
    for w in range(2, WEEKS + 1):
        for p, t in rows.get(w - 1, []):
            cur[ix[p]] = t
        out[w - 1] = cur.copy()
    return out


def future_avail(S, y, gone, teams):
    """For the decision before week v: player x (weeks v..17), whether he is expected to
    be available. Future weeks know only the schedule (a bye); week v also uses the
    streaming policy's known-unavailable flags. No team on file means unavailable."""
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG")]
    plays = {(w, t) for w, a, h in zip(g.week, g.away_team, g.home_team) for t in (a, h)}
    out = {}
    for v in TRADE_WEEKS:
        tm = teams[v - 1]
        a = np.array([[(j, t) in plays for j in range(v, WEEKS + 1)] for t in tm])
        a[:, 0] &= ~gone[:, v - 1]
        out[v] = a
    return out


def playoff_schedule(S, y, teams):
    """prereg_trades.md test B: how soft each player's weeks 15-17 opponents have been so
    far, per decision week (the Subvertadown strength-of-schedule idea).

    For every game already played, each offense's points at a position are compared with
    that offense's own average at the position so far; a defense's adjusted points
    allowed is its mean of those residuals. A player's score is the mean over his team's
    weeks 15-17 opponents. Only games before week v are read.
    """
    st = pd.read_parquet(f"data/stats/w{y}.parquet")
    st = st[(st.season_type == "REG") & st.position.isin(POS) & (st.week <= WEEKS)]
    st = st.assign(team=W.norm_team(st.team), opp=W.norm_team(st.opponent_team))
    gm = st.groupby(["week", "team", "opp", "position"]).fantasy_points_ppr.sum().reset_index()
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG") & g.week.between(15, 17)]
    opps = {}
    for w, a, h in zip(g.week, g.away_team, g.home_team):
        opps.setdefault(a, []).append(h)
        opps.setdefault(h, []).append(a)
    out = {}
    for v in TRADE_WEEKS:
        x = gm[gm.week < v]
        x = x.assign(res=x.fantasy_points_ppr
                     - x.groupby(["team", "position"]).fantasy_points_ppr.transform("mean"))
        allowed = x.groupby(["opp", "position"]).res.mean().to_dict()
        tm = teams[v - 1]
        sc = np.zeros(len(S.ids))
        for i in range(len(S.ids)):
            o = opps.get(tm[i], [])
            if o and S.pos[i] in POS:
                sc[i] = np.mean([allowed.get((d, S.pos[i]), 0.0) for d in o])
        out[v] = sc
    return out


# ---------------------------------------------------------------- lineup value

def lineup_value(pts, avail, rosters):
    """Rest-of-season starting-lineup points for a batch of rosters.

    pts: player values (n,), avail: player x weeks bool, rosters: (C, m) player indices,
    -1 for an empty spot. With a fixed weekly value per player the best lineup takes the
    top k at each position and, for the flex, the best leftover, which is the (k+1)-th
    best at RB, WR or TE. Unavailable players count as zero in that week.
    """
    if len(rosters) > 4000:           # bound memory: C x m x W floats, several copies
        return np.concatenate([lineup_value(pts, avail, rosters[i:i + 4000])
                               for i in range(0, len(rosters), 4000)])
    idx = np.where(rosters < 0, 0, rosters)
    empty = rosters < 0
    x = np.where(empty[:, :, None], 0.0, pts[idx][:, :, None] * avail[idx])   # C x m x W
    code = np.where(empty, -1, POS_CODE_ARR[idx])
    total = np.zeros(x.shape[0])
    flex = np.zeros((x.shape[0], x.shape[2]))
    F = SET.flex_slots
    spare = []
    for p, c in POS_CODE.items():
        xp = np.where((code == c)[:, :, None], x, 0.0)
        k = SLOTS[p]
        top = -np.sort(-xp, axis=1)[:, :k + F, :]
        total += top[:, :k, :].sum(axis=(1, 2))
        if p in FLEX and top.shape[1] > k:
            spare.append(top[:, k:, :])
    if spare:                              # the flex slots take the best leftovers
        pool = -np.sort(-np.concatenate(spare, axis=1), axis=1)
        flex = pool[:, :F, :].sum(axis=1)
    return total + flex.sum(axis=1)


POS_CODE_ARR = None       # player -> position code, set per season


# ---------------------------------------------------------------- the trade search

def counts_of(S, players):
    return {p: int((S.pos[players] == p).sum()) for p in POS}


def legal(c):
    """The same legality test the wire and the draft use, from roster.py."""
    return R.legal(c)


def search(S, rosters, seat, v, val, pts, avail, k, sos=None):
    """The test seat's best accepted offer before week v, or None.

    Candidates are every 1-for-1 and every 2-for-1 (the seat gives two, receives one)
    with each opponent that the opponent accepts on value and that leaves both rosters
    legal. The seat takes the one that raises its own rest-of-season lineup the most;
    with `sos`, among offers within NEAR_BEST of that gain, the one whose incoming player
    has the softest weeks 15-17 schedule.
    """
    mine = np.array(rosters[seat])
    base = lineup_value(pts, avail, mine[None, :])[0]
    on = np.zeros(len(S.ids), bool)
    for r in rosters:
        on[r] = True
    free = np.where(~on & np.isfinite(val))[0]
    fa = int(free[np.argmax(val[free])]) if len(free) else -1
    cm = counts_of(S, mine)
    pairs = [(a, b) for a in range(len(mine)) for b in range(a + 1, len(mine))]
    cand, meta = [], []
    for o in range(TEAMS):
        if o == seat:
            continue
        theirs = np.array(rosters[o])
        co = counts_of(S, theirs)
        # A player worth receiving must lift the lineup even before anything is given up,
        # since giving players away can only lower it. That bound prunes most of them.
        up1 = lineup_value(pts, avail, np.column_stack([np.tile(mine, (len(theirs), 1)), theirs]))
        up2 = lineup_value(pts, avail, np.column_stack(
            [np.tile(mine, (len(theirs), 1)), theirs, np.full(len(theirs), fa)]))
        for j, r in enumerate(theirs):
            pr = S.pos[r]
            if not np.isfinite(val[r]):
                continue
            if up1[j] > base + EPS:
                for a, g in enumerate(mine):
                    if not val[g] / (1 + k) >= val[r]:
                        continue
                    pg = S.pos[g]
                    c1 = dict(cm); c1[pg] -= 1; c1[pr] += 1
                    c2 = dict(co); c2[pr] -= 1; c2[pg] += 1
                    if not (legal(c1) and legal(c2)):
                        continue
                    new = mine.copy(); new[a] = r
                    cand.append(new)
                    meta.append((o, (int(g),), int(r), -1))
            if up2[j] > base + EPS and fa >= 0:
                for a, b in pairs:
                    g1, g2 = mine[a], mine[b]
                    if not (val[g1] + val[g2]) / (1 + k) >= val[r]:
                        continue
                    c1 = dict(cm); c1[S.pos[g1]] -= 1; c1[S.pos[g2]] -= 1
                    c1[pr] += 1; c1[S.pos[fa]] += 1
                    c2 = dict(co); c2[pr] -= 1; c2[S.pos[g1]] += 1; c2[S.pos[g2]] += 1
                    # The opponent then cuts one; cuttable() never breaks its minimums,
                    # so its roster is legal if it was legal at 15.
                    if not (legal(c1) and legal(c2)):
                        continue
                    new = np.concatenate([np.delete(mine, [a, b]), [r, fa]])
                    cand.append(new)
                    meta.append((o, (int(g1), int(g2)), int(r), fa))
    if not cand:
        return None
    gain = lineup_value(pts, avail, np.array(cand)) - base
    ok = np.where(gain > EPS)[0]
    if not len(ok):
        return None
    best = ok[np.argmax(gain[ok])]
    if sos is not None:
        near = ok[gain[ok] >= NEAR_BEST * gain[best]]
        key = np.lexsort((-gain[near], -sos[[meta[i][2] for i in near]]))
        best = near[key[0]]
    o, give, get, add = meta[best]
    return {"opp": o, "give": give, "get": get, "fa": add, "gain": float(gain[best]),
            "base": float(base)}


def execute(S, rosters, seat, t, val):
    """Swap the players; in a 2-for-1 the seat adds the free agent and the opponent cuts
    its lowest-valued cuttable player, as the consensus wire would."""
    o = t["opp"]
    sizes = (len(rosters[seat]), len(rosters[o]))
    mine = [i for i in rosters[seat] if i not in t["give"]] + [t["get"]]
    theirs = [i for i in rosters[o] if i != t["get"]] + list(t["give"])
    drop = -1
    if t["fa"] >= 0:
        mine.append(t["fa"])
        cut = LB.cuttable(S, np.array(theirs))
        vd = np.where(np.isfinite(val[cut]), val[cut], -np.inf)
        drop = int(cut[int(np.argmin(vd))])
        theirs = [i for i in theirs if i != drop]
    rosters[seat], rosters[o] = mine, theirs
    # Cheap invariants, checked on every trade: 14 players each, legal minimums, and no
    # player on two rosters.
    for r, size in zip((mine, theirs), sizes):
        assert len(r) == len(set(r)) == size and legal(counts_of(S, np.array(r)))
    allp = [i for r in rosters for i in r]
    assert len(allp) == len(set(allp))
    return drop


# ---------------------------------------------------------------- the season

def simulate(S, drafted, values, seat, movers, trade=None):
    """league_backtest.simulate with a trade window after each week's waivers.

    With trade=None this is exactly league_backtest.simulate, which the control arm must
    reproduce. trade = (k, use_sos, inputs, log).
    """
    rosters = [list(r) for r in drafted]
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    every = set(range(TEAMS))
    for w in range(WEEKS):
        for t in range(TEAMS):
            scores[t, w] = LB.week_points(S, rosters[t], S.ecr_val, w)
        if w + 1 < WEEKS:
            order = [t for t in LB.priority(scores, w) if t in every]
            moves += LB.waiver_week(S, rosters, values, order, w + 2, movers)
            v = w + 2
            if trade is not None and v in TRADE_WEEKS:
                k, use_sos, inp, log = trade
                val = values[seat][:, v - 1]
                found = search(S, rosters, seat, v, val, inp["pts"][:, v - 1],
                               inp["avail"][v], k, inp["sos"][v] if use_sos else None)
                if found:
                    before = list(rosters[seat])
                    drop = execute(S, rosters, seat, found, val)
                    after = list(rosters[seat])
                    # Realised payoff with both rosters frozen from here: what the trade
                    # itself did to the seat's starting lineup, before later moves.
                    cf = sum(LB.week_points(S, after, S.ecr_val, j) -
                             LB.week_points(S, before, S.ecr_val, j) for j in range(v - 1, WEEKS))
                    log.append({
                        "week": v, "opp": found["opp"],
                        "structure": "1-for-1" if found["fa"] < 0 else "2-for-1",
                        "give": [S.ids[i] for i in found["give"]], "get": S.ids[found["get"]],
                        "give_pos": "/".join(S.pos[i] for i in found["give"]),
                        "get_pos": S.pos[found["get"]],
                        "val_given": float(sum(val[i] for i in found["give"])),
                        "val_received": float(val[found["get"]]),
                        "proj_gain": found["gain"], "cf_pts": cf,
                        "fa": S.ids[found["fa"]] if found["fa"] >= 0 else None,
                        "opp_drop": S.ids[drop] if drop >= 0 else None,
                        "sos_get": float(inp["sos"][v][found["get"]]),
                    })
    return scores, moves, rosters


def arms():
    out = [("stream", None)]
    for k in K_SWEEP:
        out += [(f"tradeA_k{k:g}", (k, False)), (f"tradeB_k{k:g}", (k, True))]
    return out


def run():
    global POS_CODE_ARR
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, trade_log = [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        POS_CODE_ARR = np.array([POS_CODE.get(p, -1) for p in S.pos])
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        teams = team_at(S, y)
        pts = ros_points(S, y, crv)
        # The lineup values must be the same ranks the wire prices, before replacement.
        for w in range(LB.FIRST_WAIVER, WEEKS + 1):
            assert np.allclose(LB.over_replacement(S, pts[:, w - 1]), cons_val[:, w - 1],
                               equal_nan=True)
        inp = {"pts": pts, "avail": future_avail(S, y, gone, teams),
               "sos": playoff_schedule(S, y, teams)}
        for lg in range(LB.LEAGUES):
            # The same draws in the same order as run_streaming, so drafts and control match.
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
                week_sc = {}
                for arm, spec in arms():
                    movers = {seat: LB.streaming_policy(S, gone, [])}
                    log = []
                    trade = None if spec is None else (spec[0], spec[1], inp, log)
                    sc, moves, _ = simulate(S, drafted, base, seat, movers, trade)
                    week_sc[arm] = sc[seat]
                    res = [LB.season_outcome(sc, s) for s in scheds]
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": LB.all_play(sc, seat),
                        "pts_reg": sc[seat, :REG].sum(),
                        "pts_playoff": sc[seat, REG:].sum(),
                        "moves": moves[seat], "trades": len(log),
                    })
                    for e in log:
                        e.update(season=y, league=lg, seat=seat, arm=arm)
                    trade_log += log
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trade_log)


def report(d, tl):
    print("\n=== trades: roster-fit trades on top of hole-aware streaming ===")
    print("12-team PPR leagues on 2021-25, exact-consensus draft and consensus lineups, "
          "every other\nseat on the consensus wire; the test seat streams in every arm\n")
    print(f"{'arm':13s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s} {'adds':>6s} {'trades':>7s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:13s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f} {g.moves.mean():6.2f} "
              f"{g.trades.mean():7.2f}")
    print("\npaired against the streaming control (season-cluster bootstrap 90% interval):")
    verdict = {}
    for arm, _ in arms()[1:]:
        verdict[arm] = LB.paired(d, arm, "stream", f"{arm} minus stream")
        print(f"  -> {'passes this design' if verdict[arm] else 'fails this design'}")

    print("\nreported, not gated: executed trades per league-seat season")
    if not len(tl):
        print("  none")
        return verdict
    n = d.groupby("arm").size()
    print(f"  {'arm':13s} {'trades':>7s} {'1-for-1':>8s} {'2-for-1':>8s} {'val given':>10s} "
          f"{'val recv':>9s} {'proj gain':>10s} {'cf pts':>7s} {'paired pts':>11s}")
    for arm, _ in arms()[1:]:
        g = tl[tl.arm == arm]
        a = d[d.arm == arm].set_index(["season", "league", "seat"])
        b = d[d.arm == "stream"].set_index(["season", "league", "seat"])
        paired_pts = ((a.pts_reg + a.pts_playoff) - (b.pts_reg + b.pts_playoff)).mean()
        s = g.structure.value_counts()
        print(f"  {arm:13s} {len(g) / n[arm]:7.2f} {s.get('1-for-1', 0) / n[arm]:8.2f} "
              f"{s.get('2-for-1', 0) / n[arm]:8.2f} {g.val_given.mean():10.1f} "
              f"{g.val_received.mean():9.1f} {g.proj_gain.mean():10.1f} {g.cf_pts.mean():+7.1f} "
              f"{paired_pts:+11.1f}")
    print("  (val: consensus rest-of-season points per game over replacement, summed per side;"
          "\n   proj gain: projected rest-of-season lineup points; cf pts: realised lineup "
          "points\n   weeks v-17 with the post-trade roster minus the pre-trade roster, both "
          "frozen;\n   paired pts: season points, arm minus control)")
    print("\n  by season, trades per league-seat: " + "  ".join(
        f"{arm} " + " ".join(f"{len(tl[(tl.arm == arm) & (tl.season == y)]) / (n[arm] / 5):.1f}"
                             for y in LB.SEASONS) for arm, _ in arms()[1:3]))
    print("  positions received (all arms): "
          + " ".join(f"{p} {c}" for p, c in tl.get_pos.value_counts().items()))
    return verdict


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    CFG.announce_run(noise=LB.NOISE, leagues=LB.LEAGUES)
    check = "" if LB.LEAGUES == 20 else f"_check{LB.LEAGUES}"
    d, tl = run()
    d.to_parquet(f"data/league_trades{check}_noise{LB.NOISE:g}.parquet")
    tl.to_parquet(f"data/league_trades_log{check}_noise{LB.NOISE:g}.parquet")
    if check and "--mechanics" in sys.argv:
        sys.exit()                   # the mechanics check reads the files, not outcomes
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    report(d, tl)
