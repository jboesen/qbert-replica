"""Roster-fit trades at a realistic rate and with opponents who look at their own roster
(prereg_tradecap.md).

sim_trades.py found that trading for the seat's own starting lineup beats streaming alone,
but on two generous assumptions: the seat traded about 8 times a season (real Sleeper
teams: 0.4), and opponents took any offer that was fair by consensus value, even one that
gutted their own lineup. This module keeps sim_trades' search and execution unchanged and
tightens both.

- Caps: at most 1, 2 or 4 executed trades a season. A scarce trade should go to a big gain,
  so a capped seat takes its week's best offer only when the projected rest-of-season
  lineup gain is at least GAIN_MIN points; otherwise it waits for a later week.
- Mutual benefit: the opponent also refuses any trade that lowers its own rest-of-season
  starting-lineup value, by the same lineup function the seat uses on the same consensus
  numbers. Both teams gain on fit, neither loses on value.

    .venv/bin/python draft/sim_tradecap.py --noise 1 [--leagues 2 --mechanics]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
import sim_trades as ST
from league_backtest import POS, REG, TEAMS, WEEKS

EPS = ST.EPS
GAIN_MIN = 10.0        # rest-of-season lineup points a capped trade must project
CHECK_SEARCH = False   # mechanics check: the mutual search, unfiltered, must equal ST.search

# (arm, acceptance, cap, minimum gain). stream and tradeA_k0 are sim_trades' arms exactly.
ARMS = [
    ("stream", None, None, 0.0),
    ("tradeA_k0", "value", None, 0.0),
    ("value_cap1", "value", 1, GAIN_MIN),
    ("value_cap2", "value", 2, GAIN_MIN),
    ("value_cap4", "value", 4, GAIN_MIN),
    ("mutual", "mutual", None, 0.0),
    ("mutual_cap1", "mutual", 1, GAIN_MIN),
    ("mutual_cap2", "mutual", 2, GAIN_MIN),
    ("mutual_cap4", "mutual", 4, GAIN_MIN),
    ("mutual_cap2_nomin", "mutual", 2, 0.0),
]
VERDICT = "mutual_cap2"


# ---------------------------------------------------------------- the opponent's side

def opponent_after(S, theirs, get, gives, val):
    """The opponent's roster after each candidate, the way ST.execute builds it: the
    incoming players appended after its own, and in a 2-for-1 its lowest-valued cuttable
    player cut (first such player in that order, as np.argmin picks). Vectorised over
    candidates because the Python cuttable() is far too slow for thousands of offers.

    theirs: (m,) roster; get: (C,) player each candidate takes; gives: (C, 2) players it
    receives, -1 in the second column for a 1-for-1. Returns (C, m+1), -1 for empty.
    """
    n = len(get)
    keep = np.tile(theirs, (n, 1))
    keep = keep[keep != get[:, None]].reshape(n, len(theirs) - 1)
    new = np.column_stack([keep, gives])                     # (C, m+1)
    two = gives[:, 1] >= 0
    if two.any():
        r = new[two]
        code = ST.POS_CODE_ARR[r]
        counts = np.stack([(code == c).sum(1) for c in range(len(POS))], 1)
        mins = np.array([LB.ROSTER_MIN.get(p, 0) for p in POS])
        can = counts[np.arange(len(r))[:, None], code] > mins[code]
        v = val[r]
        vd = np.where(np.isfinite(v), v, -np.inf)
        vd = np.where(can, vd, np.inf)
        cut = np.argmin(vd, axis=1)
        r[np.arange(len(r)), cut] = -1
        new[two] = r
    return new


def search_mutual(S, rosters, seat, v, val, pts, avail, k, mutual=True):
    """ST.search's candidates and choice, with the opponent's own-lineup test on top.

    The enumeration below copies ST.search line for line (same order, so the same ties
    break the same way); the mechanics check asserts that without the filter it returns
    ST.search's trade. Also returns what the week offered: the best positive offer the
    opponent accepts on value alone, and the best that also passes its lineup test.
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
                for a, g in enumerate(mine):
                    if not val[g] / (1 + k) >= val[r]:
                        continue
                    pg = S.pos[g]
                    c1 = dict(cm); c1[pg] -= 1; c1[pr] += 1
                    c2 = dict(co); c2[pr] -= 1; c2[pg] += 1
                    if not (ST.legal(c1) and ST.legal(c2)):
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
                    if not (ST.legal(c1) and ST.legal(c2)):
                        continue
                    new = np.concatenate([np.delete(mine, [a, b]), [r, fa]])
                    cand.append(new)
                    meta.append((o, (int(g1), int(g2)), int(r), fa))
    diag = {"value_best": np.nan, "mutual_best": np.nan, "n_value": 0, "n_mutual": 0}
    if not cand:
        return None, diag
    gain = ST.lineup_value(pts, avail, np.array(cand)) - base
    ok = np.where(gain > EPS)[0]
    diag["n_value"] = len(ok)
    if not len(ok):
        return None, diag
    diag["value_best"] = float(gain[ok].max())
    # The opponent's lineup before and after, only for offers the seat would want at all.
    opp_gain = np.full(len(cand), np.nan)
    by_opp = {}
    for i in ok:
        by_opp.setdefault(meta[i][0], []).append(i)
    for o, idx in by_opp.items():
        theirs = np.array(rosters[o])
        before = ST.lineup_value(pts, avail, theirs[None, :])[0]
        gives = np.array([list(meta[i][1]) + [-1] * (2 - len(meta[i][1])) for i in idx])
        after = opponent_after(S, theirs, np.array([meta[i][2] for i in idx]), gives, val)
        opp_gain[idx] = ST.lineup_value(pts, avail, after) - before
    mut = ok[opp_gain[ok] >= -EPS]
    diag["n_mutual"] = len(mut)
    if len(mut):
        diag["mutual_best"] = float(gain[mut].max())
    pick = mut if mutual else ok
    if not len(pick):
        return None, diag
    best = pick[np.argmax(gain[pick])]
    o, give, get, add = meta[best]
    return {"opp": o, "give": give, "get": get, "fa": add, "gain": float(gain[best]),
            "base": float(base), "opp_gain": float(opp_gain[best])}, diag


# ---------------------------------------------------------------- the season

def simulate(S, drafted, values, seat, movers, spec=None, inp=None, log=None, dec=None):
    """ST.simulate with a trade cap, a minimum gain and a choice of acceptance rule.

    spec None is the stream control (league_backtest.simulate). spec = (accept, cap,
    gain_min). A capped seat stops searching once the cap is spent.
    """
    rosters = [list(r) for r in drafted]
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    every = set(range(TEAMS))
    done = 0
    for w in range(WEEKS):
        for t in range(TEAMS):
            scores[t, w] = LB.week_points(S, rosters[t], S.ecr_val, w)
        if w + 1 < WEEKS:
            order = [t for t in LB.priority(scores, w) if t in every]
            moves += LB.waiver_week(S, rosters, values, order, w + 2, movers)
            v = w + 2
            if spec is None or v not in ST.TRADE_WEEKS:
                continue
            accept, cap, gmin = spec
            if cap is not None and done >= cap:
                continue
            val = values[seat][:, v - 1]
            pts, avail = inp["pts"][:, v - 1], inp["avail"][v]
            if accept == "value" and not CHECK_SEARCH:
                found, diag = ST.search(S, rosters, seat, v, val, pts, avail, 0.0), {}
            else:
                found, diag = search_mutual(S, rosters, seat, v, val, pts, avail, 0.0,
                                            mutual=accept == "mutual")
                if CHECK_SEARCH:
                    ref = ST.search(S, rosters, seat, v, val, pts, avail, 0.0)
                    if accept == "value":
                        assert (ref is None and found is None) or all(
                            ref[x] == found[x] for x in ("opp", "give", "get", "fa", "gain"))
                    elif found is not None:
                        # a mutual offer is also a value offer, so it can't beat the best one
                        assert found["gain"] <= ref["gain"] + EPS
            take = found is not None and found["gain"] >= gmin
            dec.append({"week": v, "offer": found is not None,
                        "gain": found["gain"] if found else np.nan, "executed": take, **diag})
            if not take:
                continue
            o = found["opp"]
            opp_before = ST.lineup_value(pts, avail, np.array(rosters[o])[None, :])[0]
            before = list(rosters[seat])
            drop = ST.execute(S, rosters, seat, found, val)
            done += 1
            opp_after = ST.lineup_value(pts, avail, np.array(rosters[o])[None, :])[0]
            if accept == "mutual":
                # The executed trade must be the one the opponent evaluated.
                assert abs((opp_after - opp_before) - found["opp_gain"]) < 1e-6
                assert opp_after >= opp_before - EPS
            cf = sum(LB.week_points(S, rosters[seat], S.ecr_val, j) -
                     LB.week_points(S, before, S.ecr_val, j) for j in range(v - 1, WEEKS))
            log.append({
                "week": v, "opp": o,
                "structure": "1-for-1" if found["fa"] < 0 else "2-for-1",
                "give_pos": "/".join(S.pos[i] for i in found["give"]),
                "get_pos": S.pos[found["get"]],
                "val_given": float(sum(val[i] for i in found["give"])),
                "val_received": float(val[found["get"]]),
                "proj_gain": found["gain"], "opp_proj_gain": float(opp_after - opp_before),
                "cf_pts": cf, "opp_drop": S.ids[drop] if drop >= 0 else None,
            })
    return scores, moves, rosters


def run():
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, trade_log, decisions = [], [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        ST.POS_CODE_ARR = np.array([ST.POS_CODE.get(p, -1) for p in S.pos])
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        teams = ST.team_at(S, y)
        pts = ST.ros_points(S, y, crv)
        for w in range(LB.FIRST_WAIVER, WEEKS + 1):
            assert np.allclose(LB.over_replacement(S, pts[:, w - 1]), cons_val[:, w - 1],
                               equal_nan=True)
        inp = {"pts": pts, "avail": ST.future_avail(S, y, gone, teams)}
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
                for arm, accept, cap, gmin in ARMS:
                    movers = {seat: LB.streaming_policy(S, gone, [])}
                    log, dec = [], []
                    spec = None if accept is None else (accept, cap, gmin)
                    sc, moves, _ = simulate(S, drafted, base, seat, movers, spec, inp, log, dec)
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
                    for e in log + dec:
                        e.update(season=y, league=lg, seat=seat, arm=arm)
                    trade_log += log
                    decisions += dec
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trade_log), pd.DataFrame(decisions)


def report(d, tl, dec):
    print("\n=== tradecap: capped and mutual-benefit trades on top of streaming ===")
    print(f"{'arm':18s} {'title':>7s} {'all-play':>9s} {'reg pts':>8s} {'adds':>6s} "
          f"{'trades':>7s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:18s} {100 * g.title.mean():6.1f}% {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.moves.mean():6.2f} {g.trades.mean():7.2f}")
    print("\npaired against the streaming control (season-cluster bootstrap 90% interval):")
    verdict = {}
    for arm, *_ in ARMS[1:]:
        verdict[arm] = LB.paired(d, arm, "stream", f"{arm} - stream")
        g = d[d.arm == arm]
        pos = g.merge(d[d.arm == "stream"], on=["season", "league", "seat"])
        by = pos.assign(x=pos.all_play_x - pos.all_play_y).groupby("season").x.mean()
        tag = "  <- preregistered verdict" if arm == VERDICT else ""
        print(f"    rule {'passes' if verdict[arm] else 'fails'}; seasons positive "
              f"{(by > 0).sum()}/5; trades/season {g.trades.mean():.2f} by season "
              + " ".join(f"{x:.2f}" for x in g.groupby("season").trades.mean()) + tag)
        print("    title by season " + " ".join(
            f"{100 * x:+.1f}" for x in pos.assign(x=pos.title_x - pos.title_y)
            .groupby("season").x.mean()))
    print("\ncontext, paired against uncapped tradeA_k0:")
    for arm, *_ in ARMS[2:]:
        LB.paired(d, arm, "tradeA_k0", f"{arm} - tradeA_k0")

    print("\nreported, not gated: decision weeks (searched weeks only)")
    print(f"  {'arm':18s} {'searched':>8s} {'value>0':>8s} {'mutual>0':>9s} {'mut>=min':>9s} "
          f"{'executed':>8s} {'proj gain':>9s} {'opp gain':>9s} {'cf pts':>7s} {'wk':>5s}")
    for arm, *_ in ARMS[1:]:
        x = dec[dec.arm == arm]
        t = tl[tl.arm == arm]
        mutual_cols = "n_mutual" in x and x.n_mutual.notna().any()
        vpos = (x.n_value > 0).mean() if mutual_cols else x.offer.mean()
        mpos = (x.n_mutual > 0).mean() if mutual_cols else np.nan
        mmin = (x.mutual_best >= GAIN_MIN).mean() if mutual_cols else np.nan
        print(f"  {arm:18s} {len(x):8d} {vpos:8.2f} {mpos:9.2f} {mmin:9.2f} "
              f"{x.executed.mean():8.2f} {t.proj_gain.median():9.1f} "
              f"{t.opp_proj_gain.median():9.1f} {t.cf_pts.mean():+7.1f} {t.week.mean():5.1f}")
    m = dec[(dec.arm == "mutual")]
    if len(m):
        print("\n  uncapped mutual arm, share of weeks with a positive offer, by week "
              "(value-only / mutual):")
        g = m.groupby("week")
        print("  " + "  ".join(f"w{w} {(a.n_value > 0).mean():.2f}/{(a.n_mutual > 0).mean():.2f}"
                               for w, a in g))
        w3 = m[m.week == 3]
        print(f"  week 3 (untraded rosters): value-only offers per seat {w3.n_value.mean():.0f},"
              f" mutual {w3.n_mutual.mean():.0f}; best gain median value-only "
              f"{w3.value_best.median():.1f}, mutual {w3.mutual_best.median():.1f}")
    t = tl[tl.arm == "tradeA_k0"]
    if len(t):
        print(f"\n  uncapped tradeA_k0: share of executed trades that don't lower the "
              f"opponent's lineup {(t.opp_proj_gain >= -EPS).mean():.2f}, median opponent "
              f"change {t.opp_proj_gain.median():+.1f}")
    n = d.groupby("arm").size()
    print("\n  trade structure (1-for-1 share) and positions received by arm:")
    for arm, *_ in ARMS[1:]:
        t = tl[tl.arm == arm]
        if len(t):
            print(f"  {arm:18s} 1-for-1 {(t.structure == '1-for-1').mean():.2f}  get "
                  + " ".join(f"{p} {c / n[arm]:.2f}" for p, c in t.get_pos.value_counts().items()))
    return verdict


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = "" if LB.LEAGUES == 20 else f"_check{LB.LEAGUES}"
    CHECK_SEARCH = bool(check) and "--mechanics" in sys.argv
    d, tl, dec = run()
    d.to_parquet(f"data/league_tradecap{check}_noise{LB.NOISE:g}.parquet")
    tl.to_parquet(f"data/league_tradecap_log{check}_noise{LB.NOISE:g}.parquet")
    dec.to_parquet(f"data/league_tradecap_decisions{check}_noise{LB.NOISE:g}.parquet")
    if check and "--mechanics" in sys.argv:
        sys.exit()                   # the mechanics check reads the files, not outcomes
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    report(d, tl, dec)
