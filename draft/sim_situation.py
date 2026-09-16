"""Situation-aware trades and waivers on top of the capped mutual-benefit trader
(prereg_situation.md).

Every policy this harness has tested is stationary: it does the same thing in week 3 at
0-2 as in week 11 at 8-2. Real managers don't. Two things about a seat's own situation
are knowable at decision time and should change what it does.

- Weeks remaining. `sim_tradecap`'s 10-point bar is a rest-of-season projected gain, so
  the same bar is a much harder test in week 11 (7 weeks left) than in week 3 (15). Held
  flat it quietly turns into "trade early or not at all". Scaled by weeks left it is a
  constant points-per-week bar.
- Playoff odds. A seat that is close to the cut should spend its scarce trades and its
  bench on winning now; one that is nearly out should hold the bar high and buy upside
  instead of a slightly better floor. Odds are estimated from games already played plus a
  Monte Carlo of the weeks left, with the weekly spread fit on earlier seasons only.

The control is `sim_tradecap`'s `mutual_cap2` exactly: fill holes on waivers, at most two
trades a season, each worth at least 10 projected rest-of-season lineup points and never
lowering the opponent's own projected lineup.

    .venv/bin/python draft/sim_situation.py --noise 1 [--leagues 2 --mechanics]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
import sim_trades as ST
import sim_tradecap as TC
import winprob as WP
from league_backtest import FLEX, POS, REG, SLOTS, TEAMS, WEEKS

EPS = ST.EPS
GAIN_MIN = TC.GAIN_MIN          # 10 rest-of-season lineup points, tradecap's frozen bar
BASE_WEEK = 3                   # the week the flat bar was calibrated on
WEEKS_AT_BASE = WEEKS + 1 - BASE_WEEK     # 15 weeks left at the first trade decision
CAP = 2                         # tradecap's mutual_cap2: at most two trades a season
ODDS_CUT = 0.25                 # below this a seat is a seller
BUY_MULT, SELL_MULT = 0.5, 2.0  # what the odds regime does to the trade bar
NEAR_ADD = 1.0                  # ppg of rest-of-season value a tilted waiver add may give up
DRAWS = 1000                    # Monte Carlo scenarios per playoff-odds estimate

# (arm, scale the bar by weeks left, tilt the bar by odds, tilt waiver adds by odds)
ARMS = [
    ("control", False, False, False),
    ("scaled", True, False, False),
    ("odds", False, True, False),
    ("situation", True, True, True),
]
VERDICT = "situation"


# ---------------------------------------------------------------- inputs per season

def avail_all(S, y, gone, teams):
    """ST.future_avail over every waiver decision week 2..17, not just the trade window.

    Same rule and the same sources; the mechanics check asserts it equals ST.future_avail
    on weeks 3-11.
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


def spread_shape(S, pts, model):
    """Per player and decision week, the spread of a weekly score around its consensus
    value: the residual pool's sd and its 10th and 90th percentiles.

    The pools are winprob's, fit on seasons before y and centred to mean zero, so these
    describe shape only and never move a player's expected points. A player is placed in
    a pool by his own consensus points per game at that decision, which is the same
    quantity winprob buckets on.
    """
    n = len(S.ids)
    sd = np.zeros((n, WEEKS))
    lo = np.zeros((n, WEEKS))
    hi = np.zeros((n, WEEKS))
    stats = {}
    for p, (edges, pools) in model.items():
        stats[p] = [(r.std(), np.quantile(r, .1), np.quantile(r, .9)) for r in pools]
    for p in POS:
        at = np.where(S.pos == p)[0]
        if not len(at):
            continue
        edges = model[p][0]
        for w in range(LB.FIRST_WAIVER - 1, WEEKS):
            v = np.nan_to_num(pts[at, w], nan=0.0)
            b = np.digitize(v, edges)
            s = np.array([stats[p][k] for k in b])
            sd[at, w], lo[at, w], hi[at, w] = s[:, 0], s[:, 1], s[:, 2]
    return sd, lo, hi


# ---------------------------------------------------------------- playoff odds

def lineup_by_week(pts, avail, roster):
    """ST.lineup_value for one roster, broken out per remaining week.

    Same slot rule (top k at each position, then the best leftover flex, a player counted
    as zero in a week he is expected unavailable); the mechanics check asserts it sums to
    ST.lineup_value.
    """
    r = np.array([i for i in roster if i >= 0])
    x = pts[r][:, None] * avail[r]
    x = np.nan_to_num(x, nan=0.0)
    code = ST.POS_CODE_ARR[r]
    total = np.zeros(x.shape[1])
    flex = np.zeros(x.shape[1])
    for p, c in ST.POS_CODE.items():
        xp = np.where((code == c)[:, None], x, 0.0)
        k = SLOTS[p]
        top = -np.sort(-xp, axis=0)[:k + 1]
        total += top[:k].sum(0)
        if p in FLEX and top.shape[0] > k:
            flex = np.maximum(flex, top[k])
    return total + flex


def week_starters(pos, pts, av0, roster):
    """The players the same rule starts in the first remaining week."""
    r = [i for i in roster if i >= 0]
    v = {i: (pts[i] if (np.isfinite(pts[i]) and av0[i]) else 0.0) for i in r}
    used = []
    for p, k in SLOTS.items():
        at = sorted((i for i in r if pos[i] == p), key=lambda i: -v[i])
        used += at[:k]
    left = [i for i in r if pos[i] in FLEX and i not in used]
    if left:
        used.append(max(left, key=lambda i: v[i]))
    return used


def team_sd(pos, pts, av0, roster, sd):
    """A team's weekly score sd: its projected starters' spreads added in quadrature.

    Starters are drawn independently here, which is the same assumption winprob makes
    within a lineup; with seven of them the total is close enough to normal to sample as
    one number instead of seven.
    """
    s = sd[week_starters(pos, pts, av0, roster)]
    return float(np.sqrt((s ** 2).sum()))


def playoff_odds(pos, rosters, seat, v, pts, avail, scores, sd, rng):
    """P(the seat finishes in the top six) at the decision before week v, from games
    already played and a Monte Carlo of the weeks left.

    The harness scores twenty schedule draws over one set of weekly scores, so at decision
    time there is no single schedule to run standings on. It already solves that for
    waiver priority by using all-play instead of head-to-head (LB.priority), and this uses
    the same convention: the seat is "in" if its all-play wins over weeks 1-14 finish top
    six, ties broken by points for. Future weekly scores are each team's own projected
    starting lineup for that week plus a draw from the fitted spread.
    """
    played = min(v - 1, REG)
    s = scores[:, :played]
    aw = np.array([(s[t] > np.delete(s, t, 0)).sum() for t in range(TEAMS)], float)
    pf = s.sum(1)
    left = REG - played
    if left <= 0:                     # the regular season is over; the field is known
        better = (aw > aw[seat]) | ((aw == aw[seat]) & (pf > pf[seat]))
        better[seat] = False
        return float(better.sum() < 6)
    mu = np.array([lineup_by_week(pts, avail, r)[:left] for r in rosters])
    sds = np.array([team_sd(pos, pts, avail[:, 0], r, sd) for r in rosters])
    sim = mu[None] + sds[None, :, None] * rng.standard_normal((DRAWS, TEAMS, left))
    wins = aw[None] + (sim[:, :, None, :] > sim[:, None, :, :]).sum(axis=(2, 3))
    tot = pf[None] + sim.sum(2)
    better = ((wins > wins[:, [seat]])
              | ((wins == wins[:, [seat]]) & (tot > tot[:, [seat]])))
    better[:, seat] = False
    return float((better.sum(1) < 6).mean())


# ---------------------------------------------------------------- the situational rules

def gain_bar(v, odds, scale, tilt):
    """The projected rest-of-season gain a capped trade must clear before week v."""
    bar = GAIN_MIN
    if scale:
        bar *= (WEEKS + 1 - v) / WEEKS_AT_BASE
    if tilt:
        bar *= BUY_MULT if odds >= ODDS_CUT else SELL_MULT
    return bar


def tilted_add(pool, val, lo_w, hi_w, get_odds):
    """Among free agents within NEAR_ADD of the best rest-of-season value, the one whose
    weekly score distribution suits the seat's situation: the highest 90th percentile when
    the seat needs an unlikely run, the highest 10th percentile when it is protecting a
    berth. With one candidate in the band this is the streaming policy's own argmax, and
    the odds are not even estimated.

    Returns (add, odds), odds NaN when nothing was close enough for the tilt to bite.
    """
    best = pool[int(np.argmax(val[pool]))]
    near = pool[val[pool] >= val[best] - NEAR_ADD]
    if len(near) < 2:
        return int(best), np.nan
    odds = get_odds()
    shape = hi_w[near] if odds < ODDS_CUT else lo_w[near]
    key = np.lexsort((-val[near], -shape))
    return int(near[key[0]]), odds


def situation_policy(S, gone, get_odds, lo, hi, log):
    """LB.streaming_policy with the add chosen by tilted_add instead of by value alone.

    The hole rule, the fallback and the drop rule are unchanged; only which of several
    near-equally valued free agents is taken depends on the seat's playoff odds.
    """
    def move(S_, held, val, free, w):
        lo_w, hi_w = lo[:, w - 1], hi[:, w - 1]
        cons = LB.consensus_move(S, held, val, free, w)
        g = gone[:, w - 1]
        before = LB.holes(S, held, g)
        if not sum(before.values()):
            # No hole: consensus's own move, but the add may be a near-equal alternative.
            add, odds = tilted_add(free, val, lo_w, hi_w, get_odds)
            can_cut = LB.cuttable(S, held)
            if not len(can_cut):
                return None
            vd = np.where(np.isfinite(val[can_cut]), val[can_cut], -np.inf)
            drop = int(can_cut[int(np.argmin(vd))])
            mine = (add, drop) if val[add] > vd.min() else None
            log.append({"week": w, "hole": False, "odds": odds,
                        "same_as_stream": mine == cons,
                        "add_pos": S.pos[add] if mine else None})
            return mine
        fills = [p for p in POS if before[p] > 0 or (p in FLEX and before["FLEX"] > 0)]
        cand = free[np.isin(S.pos[free], fills) & ~g[free]]
        if not len(cand):
            return cons
        add, odds = tilted_add(cand, val, lo_w, hi_w, get_odds)
        after_add = list(held) + [add]
        lineup = LB.starters(S, after_add, val, g)
        best = None
        for d in LB.cuttable(S, held):
            leftover = sum(LB.holes(S, [i for i in after_add if i != d], g).values())
            if leftover >= sum(before.values()):
                continue
            key = (leftover, d in lineup, val[d] if np.isfinite(val[d]) else -np.inf)
            if best is None or key < best[0]:
                best = (key, int(d))
        if best is None:
            return cons
        mine = (add, best[1])
        # What plain streaming would have added in the same state, for the diagnostics.
        stream_add = int(cand[int(np.argmax(val[cand]))])
        log.append({"week": w, "hole": True, "odds": odds,
                    "same_as_stream": add == stream_add, "add_pos": S.pos[add]})
        return mine
    return move


# ---------------------------------------------------------------- the season

def simulate(S, drafted, values, seat, spec, inp, log, dec, wlog):
    """sim_tradecap.simulate with a situational trade bar and, optionally, situational
    waivers. spec = (scale, tilt, waiver_tilt); the control is (False, False, False) and
    must reproduce tradecap's mutual_cap2 exactly.
    """
    scale, tilt, wtilt = spec
    rosters = [list(r) for r in drafted]
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    every = set(range(TEAMS))
    done = 0
    cell = {}          # this week's odds, estimated at most once and only if asked for

    def get_odds():
        if "v" not in cell:
            raise RuntimeError("odds asked for outside a decision week")
        if "p" not in cell:
            v = cell["v"]
            cell["p"] = playoff_odds(S.pos, rosters, seat, v, inp["pts"][:, v - 1],
                                     inp["avail"][v], scores, inp["sd"][:, v - 1],
                                     inp["rng"](v))
        return cell["p"]

    movers = {seat: situation_policy(S, inp["gone"], get_odds, inp["lo"], inp["hi"], wlog)
              if wtilt else LB.streaming_policy(S, inp["gone"], [])}
    for w in range(WEEKS):
        for t in range(TEAMS):
            scores[t, w] = LB.week_points(S, rosters[t], S.ecr_val, w)
        if w + 1 >= WEEKS:
            continue
        v = w + 2
        # At most one odds estimate a week, made the first time a decision asks for it and
        # reused by the rest of that week, so waivers and the trade window agree.
        cell.clear()
        cell["v"] = v
        order = [t for t in LB.priority(scores, w) if t in every]
        moves += LB.waiver_week(S, rosters, values, order, v, movers)
        if v not in ST.TRADE_WEEKS or done >= CAP:
            continue
        val = values[seat][:, v - 1]
        pts, avail = inp["pts"][:, v - 1], inp["avail"][v]
        found, diag = TC.search_mutual(S, rosters, seat, v, val, pts, avail, 0.0, mutual=True)
        # Nothing to judge when the week offers no mutual trade, so no odds are estimated.
        odds = get_odds() if (tilt and found is not None) else cell.get("p", np.nan)
        bar = gain_bar(v, odds, scale, tilt) if found is not None else np.nan
        take = found is not None and found["gain"] >= bar
        flat = found is not None and found["gain"] >= GAIN_MIN
        dec.append({"week": v, "offer": found is not None, "odds": odds,
                    "bar": bar, "gain": found["gain"] if found else np.nan,
                    "executed": take, "flat_would": flat, "changed": take != flat, **diag})
        if not take:
            continue
        o = found["opp"]
        opp_before = ST.lineup_value(pts, avail, np.array(rosters[o])[None, :])[0]
        before = list(rosters[seat])
        drop = ST.execute(S, rosters, seat, found, val)
        done += 1
        opp_after = ST.lineup_value(pts, avail, np.array(rosters[o])[None, :])[0]
        assert opp_after >= opp_before - EPS
        cf = sum(LB.week_points(S, rosters[seat], S.ecr_val, j) -
                 LB.week_points(S, before, S.ecr_val, j) for j in range(v - 1, WEEKS))
        log.append({
            "week": v, "opp": o, "odds": odds, "bar": bar,
            "structure": "1-for-1" if found["fa"] < 0 else "2-for-1",
            "give_pos": "/".join(S.pos[i] for i in found["give"]),
            "get_pos": S.pos[found["get"]],
            "proj_gain": found["gain"], "opp_proj_gain": float(opp_after - opp_before),
            "cf_pts": cf, "opp_drop": S.ids[drop] if drop >= 0 else None,
        })
    return scores, moves, rosters


def run(mechanics=False):
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, trade_log, decisions, waiver_log = [], [], [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        ST.POS_CODE_ARR = np.array([ST.POS_CODE.get(p, -1) for p in S.pos])
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        teams = ST.team_at(S, y)
        pts = ST.ros_points(S, y, crv)
        avail = avail_all(S, y, gone, teams)
        sd, lo, hi = spread_shape(S, pts, WP.fit_spread(y, curves))
        if mechanics:
            ref = ST.future_avail(S, y, gone, teams)
            for v in ST.TRADE_WEEKS:
                assert (ref[v] == avail[v]).all()
        inp = {"pts": pts, "avail": avail, "sd": sd, "lo": lo, "hi": hi, "gone": gone}
        for lg in range(LB.LEAGUES):
            # The same draws in the same order as sim_trades and run_streaming.
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
                # The odds stream depends only on the state being judged, so every arm
                # that asks the same question at the same week gets the same answer.
                inp["rng"] = lambda v, y=y, lg=lg, seat=seat: np.random.default_rng(
                    [y, lg, seat, v, 11])
                for arm, scale, tilt, wtilt in ARMS:
                    log, dec, wl = [], [], []
                    sc, moves, _ = simulate(S, drafted, base, seat,
                                            (scale, tilt, wtilt), inp, log, dec, wl)
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
                    for e in log + dec + wl:
                        e.update(season=y, league=lg, seat=seat, arm=arm)
                    trade_log += log
                    decisions += dec
                    waiver_log += wl
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(trade_log), pd.DataFrame(decisions),
            pd.DataFrame(waiver_log))


def report(d, tl, dec, wl):
    print("\n=== situation: weeks-remaining and playoff-odds-aware trades and waivers ===")
    print(f"{'arm':12s} {'title':>7s} {'all-play':>9s} {'reg pts':>8s} {'adds':>6s} "
          f"{'trades':>7s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:12s} {100 * g.title.mean():6.1f}% {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.moves.mean():6.2f} {g.trades.mean():7.2f}")
    print("\npaired against the control (tradecap mutual_cap2), season-cluster bootstrap:")
    verdict = {}
    for arm, *_ in ARMS[1:]:
        verdict[arm] = LB.paired(d, arm, "control", f"{arm} - control")
        g = d[d.arm == arm]
        pos = g.merge(d[d.arm == "control"], on=["season", "league", "seat"])
        by = pos.assign(x=pos.all_play_x - pos.all_play_y).groupby("season").x.mean()
        tag = "  <- preregistered verdict" if arm == VERDICT else ""
        print(f"    rule {'passes' if verdict[arm] else 'fails'}; seasons positive "
              f"{(by > 0).sum()}/5; trades/season {g.trades.mean():.2f}" + tag)
        print("    title by season " + " ".join(
            f"{100 * x:+.1f}" for x in pos.assign(x=pos.title_x - pos.title_y)
            .groupby("season").x.mean()))

    print("\ndiagnostics: trade decisions (searched weeks only)")
    print(f"  {'arm':12s} {'searched':>8s} {'bar':>6s} {'executed':>8s} {'changed':>8s} "
          f"{'odds':>6s} {'odds|chg':>8s} {'wk':>5s} {'gain':>7s}")
    for arm, *_ in ARMS:
        x = dec[dec.arm == arm]
        t = tl[tl.arm == arm]
        ch = x[x.changed]
        print(f"  {arm:12s} {len(x):8d} {x.bar.mean():6.2f} {x.executed.mean():8.3f} "
              f"{x.changed.mean():8.3f} {x.odds.mean():6.2f} "
              f"{(ch.odds.mean() if len(ch) else np.nan):8.2f} {t.week.mean():5.1f} "
              f"{t.proj_gain.median():7.1f}")
    print("  odds distribution at trade decisions (control arm has no odds tilt):")
    for arm, *_ in ARMS[2:]:
        x = dec[(dec.arm == arm) & dec.odds.notna()]
        if len(x):
            q = np.quantile(x.odds, [.1, .25, .5, .75, .9])
            print(f"  {arm:12s} deciles " + " ".join(f"{v:.2f}" for v in q)
                  + f"   seller share {(x.odds < ODDS_CUT).mean():.3f}")
    print("  trades per season by season, by arm:")
    for arm, *_ in ARMS:
        g = d[d.arm == arm]
        print(f"  {arm:12s} " + " ".join(f"{x:.2f}" for x in g.groupby("season").trades.mean()))
    print("  trade week distribution (share by week):")
    for arm, *_ in ARMS:
        t = tl[tl.arm == arm]
        if len(t):
            s = t.week.value_counts(normalize=True).sort_index()
            print(f"  {arm:12s} " + " ".join(f"w{w} {x:.2f}" for w, x in s.items()))

    if len(wl):
        print("\ndiagnostics: situational waiver adds")
        for arm, g in wl.groupby("arm"):
            tilt = g[g.odds.notna()]            # weeks with two or more near-equal adds
            sell = tilt[tilt.odds < ODDS_CUT]
            print(f"  {arm:12s} decisions {len(g)}, a choice to make in "
                  f"{len(tilt) / len(g):.3f} of them, differ from streaming "
                  f"{(~g.same_as_stream).mean():.3f}")
            print(f"    of the weeks with a choice, seller {len(sell) / max(len(tilt), 1):.3f}; "
                  f"differ as seller {(~sell.same_as_stream).mean() if len(sell) else np.nan:.3f}, "
                  f"as buyer "
                  f"{(~tilt[tilt.odds >= ODDS_CUT].same_as_stream).mean():.3f}")
            if len(tilt):
                q = np.quantile(tilt.odds, [.1, .25, .5, .75, .9])
                print("    odds deciles at those weeks " + " ".join(f"{v:.2f}" for v in q))
            x = g[~g.same_as_stream & g.add_pos.notna()]
            if len(x):
                print("    positions added when it differs: "
                      + " ".join(f"{p} {c}" for p, c in x.add_pos.value_counts().items()))
                print("    by hole / no hole: "
                      + " ".join(f"{k} {v}" for k, v in x.hole.value_counts().items()))
    return verdict


def mechanics_check(d, tl):
    """Does the control reproduce sim_tradecap's mutual_cap2 on the leagues run here?"""
    ref = pd.read_parquet(f"data/league_tradecap_noise{LB.NOISE:g}.parquet")
    ref = ref[(ref.arm == "mutual_cap2") & (ref.league < LB.LEAGUES)]
    key = ["season", "league", "seat"]
    cols = ["title", "playoff", "wins", "all_play", "pts_reg", "pts_playoff", "moves",
            "trades"]
    a = d[d.arm == "control"].set_index(key)[cols].sort_index()
    b = ref.set_index(key)[cols].sort_index()
    print(f"\nmechanics, {LB.LEAGUES} leagues per season, noise {LB.NOISE:g}:")
    print(f"  control rows {len(a)}, tradecap mutual_cap2 rows {len(b)}")
    print(f"  max absolute difference {np.abs(a.values - b.loc[a.index].values).max()}")
    print(f"  identical: {a.equals(b.loc[a.index])}")
    t = tl[tl.arm == "control"]
    print(f"  control trades: every one clears {GAIN_MIN}: "
          f"{bool((t.proj_gain >= GAIN_MIN).all())}; opponent never worse: "
          f"{bool((t.opp_proj_gain >= -EPS).all())}; weeks "
          f"{sorted(t.week.unique())}; at most 2 a seat: "
          f"{int(d[d.arm == 'control'].trades.max())}")
    for arm, *_ in ARMS[1:]:
        g = d[d.arm == arm]
        print(f"  {arm:12s} trades/seat max {int(g.trades.max())}, mean {g.trades.mean():.2f}")


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = "" if LB.LEAGUES == 20 else f"_check{LB.LEAGUES}"
    mech = bool(check) and "--mechanics" in sys.argv
    d, tl, dec, wl = run(mech)
    d.to_parquet(f"data/league_situation{check}_noise{LB.NOISE:g}.parquet")
    tl.to_parquet(f"data/league_situation_log{check}_noise{LB.NOISE:g}.parquet")
    dec.to_parquet(f"data/league_situation_decisions{check}_noise{LB.NOISE:g}.parquet")
    wl.to_parquet(f"data/league_situation_waivers{check}_noise{LB.NOISE:g}.parquet")
    if mech:
        mechanics_check(d, tl)
        sys.exit()                   # the mechanics check reads the files, not outcomes
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    report(d, tl, dec, wl)
