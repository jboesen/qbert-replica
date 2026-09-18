"""Are the more correct settings also better, or at least harmless? (prereg_settings.md)

settings.py consolidated every league rule and left three switches defaulting to the
published behaviour because turning them on moves numbers. Two of them are testable here:
the in-season roster cap, which the draft enforces and the wire does not, and replacement
pricing, which counts down an assumed number of rostered players per team rather than
looking at who is genuinely unrostered. Both are properties of the simulated world rather
than of one seat, so each arm runs the WHOLE league under the changed setting and is
paired against the same policy in the published world. The comparison is therefore between
two leagues, not between two policies; see the preregistration for what that number means.

The policy under test is fixed at mutual_cap2, sim_tradecap.py's verdict arm, and each
world is also run with the stream control so the policy's own edge can be read inside each
world. sim_tradecap.py and league_backtest.py are imported unchanged.

    .venv/bin/python draft/sim_settings.py --noise 1 [--leagues 2 --mechanics]
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
import roster as R
import sim_trades as ST
import sim_tradecap as TC
import settings as CFG
from league_backtest import POS, REG, TEAMS, WEEKS

BASE = CFG.get()
EPS = ST.EPS
GAIN_MIN = BASE.trade_gain_min
MUTUAL2 = ("mutual", 2, GAIN_MIN)       # sim_tradecap's verdict arm, unchanged

# A real platform's position limits rather than the draft simulator's bot-sanity values.
# ESPN's defaults for a 12-team league; Sleeper has no limits at all, which would make an
# arm identical to the control. Chosen from the platform, not fitted to anything here.
ESPN_CAPS = {"QB": 4, "RB": 8, "WR": 8, "TE": 3}

# (world, settings). The draft is never touched: league_backtest binds its own CAP at
# import, so every arm drafts the same rosters and the pairing survives.
WORLDS = {
    "default": BASE,
    "caps": BASE.replace(enforce_caps_in_season=True),
    "pool": BASE.replace(replacement="pool"),
    "both": BASE.replace(enforce_caps_in_season=True, replacement="pool"),
    "espn": BASE.replace(enforce_caps_in_season=True, caps=ESPN_CAPS),
}

# (arm, world, trade spec). The two default arms must reproduce sim_tradecap.py exactly.
ARMS = [
    ("stream", "default", None),
    ("ctrl", "default", MUTUAL2),
    ("caps_on_stream", "caps", None),
    ("caps_on", "caps", MUTUAL2),
    ("pool_stream", "pool", None),
    ("pool", "pool", MUTUAL2),
    ("both_stream", "both", None),
    ("both", "both", MUTUAL2),
    ("caps_espn_stream", "espn", None),
    ("caps_espn", "espn", MUTUAL2),
]
# Each switch answers its own question, so each carries its own verdict against ctrl.
VERDICT = ("caps_on", "pool", "both")
HARMLESS = 0.005          # pooled |all-play| change inside this band counts as harmless


# ---------------------------------------------------------------- pool replacement

def rostered_mask(n, rosters):
    on = np.zeros(n, bool)
    for r in rosters:
        on[np.asarray(r, dtype=int)] = True
    return on


def pool_levels(S, pts_w, rosters):
    """Replacement at each position: the best genuinely unrostered player this week.

    The published harness instead counts down to league-wide starter demand over the whole
    ranked pool (vbd.replacement_levels), which does not know who is actually held.
    """
    on = rostered_mask(len(S.pos), rosters)
    ok = np.isfinite(pts_w)
    out = {}
    for p in POS:
        at = (S.pos == p) & ~on & ok
        out[p] = float(pts_w[at].max()) if at.any() else 0.0
    return out


def mult_levels(S, pts_w, cons_w):
    """The published multiplier levels, read back out of the values the harness computed,
    so the diagnostic cannot disagree with the run it describes."""
    out = {}
    for p in POS:
        at = (S.pos == p) & np.isfinite(pts_w) & np.isfinite(cons_w)
        out[p] = float(np.median(pts_w[at] - cons_w[at])) if at.any() else np.nan
    return out


def priced(S, pts_w, levels):
    """A week's value column priced against the given per-position replacement."""
    out = np.full(len(S.pos), np.nan)
    ok = np.isfinite(pts_w)
    for p in POS:
        at = (S.pos == p) & ok
        out[at] = pts_w[at] - levels[p]
    return out


def pool_wire(pts, orig):
    """Reprice the shared value array from the real free-agent pool before each round.

    Every seat reads the same array, so one write per distinct array serves the whole
    league. Pricing happens once per decision week, before the wire runs, so the trade
    search later in that week sees the same numbers the wire did, exactly as the fixed
    multiplier column does in the published arms.
    """
    def wrapped(S, rosters, values, order, w, movers=None):
        col = priced(S, pts[:, w - 1], pool_levels(S, pts[:, w - 1], rosters))
        seen = set()
        for v in values:
            if id(v) not in seen:
                seen.add(id(v))
                v[:, w - 1] = col
        return orig(S, rosters, values, order, w, movers)
    return wrapped


# ---------------------------------------------------------------- binding diagnostics

def diag_wire(pts, orig, counts, rows, tag, test_seat):
    """Record, in the published control only, how often the unenforced cap would bind and
    how far pool replacement sits from the multiplier it replaces.

    Roster composition is read before each week's wire round, so week w is the roster that
    played week w-1. Nothing here changes a decision.
    """
    def wrapped(S, rosters, values, order, w, movers=None):
        for t in range(TEAMS):
            c = R.counts(S.pos, rosters[t])
            for p in POS:
                counts[(w, p, t == test_seat, c[p])] += 1
        pw, cw = pts[:, w - 1], values[test_seat][:, w - 1]
        ml, pl = mult_levels(S, pw, cw), pool_levels(S, pw, rosters)
        on = rostered_mask(len(S.pos), rosters)
        pool_col = priced(S, pw, pl)
        free = np.where(~on & np.isfinite(cw))[0]
        same, n_free = np.nan, len(free)
        if n_free:
            same = float(free[np.argmax(cw[free])] == free[np.argmax(pool_col[free])])
        rows.append({**tag, "week": w, "n_free": n_free, "top_add_same": same,
                     **{f"mult_{p}": ml[p] for p in POS},
                     **{f"pool_{p}": pl[p] for p in POS}})
        return orig(S, rosters, values, order, w, movers)
    return wrapped


def check_wire(orig, s):
    """Mechanics only: assert every roster in the league is legal under the active world."""
    def wrapped(S, rosters, values, order, w, movers=None):
        for t in range(TEAMS):
            c = R.counts(S.pos, rosters[t])
            assert R.legal(c, s), f"illegal roster week {w} team {t}: {c}"
            assert len(rosters[t]) == s.roster_size
        return orig(S, rosters, values, order, w, movers)
    return wrapped


# ---------------------------------------------------------------- the run

def run(mechanics=False):
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, trade_log, decisions, diag = [], [], [], []
    counts = Counter()
    orig_wire = LB.waiver_week
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
            # The same draws in the same order as sim_tradecap, so the drafts match.
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
                for arm, world, spec in ARMS:
                    s = WORLDS[world]
                    CFG.activate(s, announce=False)
                    pool = s.replacement == "pool"
                    base = [cons_val.copy()] * TEAMS if pool else [cons_val] * TEAMS
                    wire = orig_wire
                    if pool:
                        wire = pool_wire(pts, wire)
                    if arm == "ctrl":
                        tag = {"season": y, "league": lg, "seat": seat}
                        wire = diag_wire(pts, wire, counts, diag, tag, seat)
                    if mechanics:
                        wire = check_wire(wire, s)
                    LB.waiver_week = wire
                    try:
                        movers = {seat: LB.streaming_policy(S, gone, [])}
                        log, dec = [], []
                        sc, moves, _ = TC.simulate(S, drafted, base, seat, movers, spec,
                                                   inp, log, dec)
                    finally:
                        LB.waiver_week = orig_wire
                        CFG.activate(BASE, announce=False)
                    res = [LB.season_outcome(sc, sch) for sch in scheds]
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
    cnt = pd.DataFrame([{"week": w, "pos": p, "test_seat": t, "held": k, "n": n}
                        for (w, p, t, k), n in counts.items()])
    return (pd.DataFrame(rows), pd.DataFrame(trade_log), pd.DataFrame(decisions),
            pd.DataFrame(diag), cnt)


# ---------------------------------------------------------------- reporting

def reproduces(d):
    """The two default-world arms against sim_tradecap.py's committed full-size rows.

    A reduced run holds leagues 0..N-1, which are the same leagues under the same seeds,
    so it is checked against the same committed file rather than a second reduced one.
    """
    ref = pd.read_parquet(f"data/league_tradecap_noise{LB.NOISE:g}.parquet")
    key = ["season", "league", "seat"]
    cols = ["title", "playoff", "wins", "all_play", "pts_reg", "pts_playoff", "moves",
            "trades"]
    ok = True
    for mine, theirs in (("stream", "stream"), ("ctrl", "mutual_cap2")):
        a = d[d.arm == mine].set_index(key)[cols].sort_index()
        b = ref[ref.arm == theirs].set_index(key)[cols].sort_index()
        b = b.loc[a.index]
        md = float((a - b).abs().to_numpy().max())
        print(f"  {mine:8s} vs sim_tradecap {theirs:12s}: {len(a)} rows, "
              f"max abs difference {md:.6g}")
        ok &= md == 0.0
    return ok


def report_binding(cnt, diag):
    """How often the unenforced cap actually binds, and how far pool sits from multiplier.

    Separate from the pass/fail: if the cap never binds the switch is cosmetic.
    """
    print("\n=== does the unenforced cap ever bind? (published control, all 12 seats) ===")
    print("rosters read before each week's wire round, weeks 2-17\n")
    tot = cnt.n.sum() / len(POS)
    print(f"{'pos':4s} {'cap':>4s} {'max held':>9s} {'mean held':>10s} "
          f"{'team-weeks over cap':>20s} {'espn cap':>9s} {'over espn':>10s}")
    for p in POS:
        g = cnt[cnt.pos == p]
        cap, ecap = BASE.caps[p], ESPN_CAPS[p]
        over = g[g.held > cap].n.sum()
        oesp = g[g.held > ecap].n.sum()
        mean = (g.held * g.n).sum() / g.n.sum()
        print(f"{p:4s} {cap:4d} {int(g.held.max()):9d} {mean:10.2f} "
              f"{over:11d} ({100 * over / tot:5.2f}%) {ecap:9d} "
              f"{oesp:6d} ({100 * oesp / tot:4.2f}%)")
    any_over = cnt[cnt.apply(lambda r: r["held"] > BASE.caps[r["pos"]], axis=1)]
    if len(any_over):
        print("\n  team-weeks over the 2/2/7/7 cap, by position and week:")
        for p in POS:
            g = any_over[any_over.pos == p]
            if len(g):
                by = g.groupby("week").n.sum()
                print(f"    {p}: " + " ".join(f"w{w} {n}" for w, n in by.items()))
        print("\n  over the cap, test seat vs the eleven consensus seats (per team-week):")
        for p in POS:
            g = cnt[(cnt.pos == p)]
            for lab, flag in (("test seat", True), ("consensus", False)):
                h = g[g.test_seat == flag]
                o = h[h.held > BASE.caps[p]].n.sum()
                if h.n.sum():
                    print(f"    {p} {lab:10s} {100 * o / h.n.sum():6.3f}%")
    else:
        print("\n  no roster in any published-control league-week exceeds any cap.")

    print("\n=== pool vs multiplier replacement (published control) ===")
    print("per-position replacement level, consensus rest-of-season points per game\n")
    print(f"{'pos':4s} {'multiplier':>11s} {'pool':>8s} {'pool - mult':>12s} "
          f"{'|diff| p90':>11s}")
    for p in POS:
        m, q = diag[f"mult_{p}"], diag[f"pool_{p}"]
        print(f"{p:4s} {m.mean():11.2f} {q.mean():8.2f} {(q - m).mean():12.2f} "
              f"{(q - m).abs().quantile(.9):11.2f}")
    print(f"\n  free agents with a value at a decision: {diag.n_free.mean():.0f} per week")
    print(f"  the wire's best available add is the same player under both pricings in "
          f"{100 * diag.top_add_same.mean():.1f}% of team-weeks")
    by = diag.groupby("week").top_add_same.mean()
    print("  by week: " + " ".join(f"w{w} {100 * v:.0f}%" for w, v in by.items()))


def report(d, tl, dec, diag, cnt):
    print("\n=== settings: do the more correct switches help, hurt, or do nothing? ===")
    print("mutual_cap2 in every policy arm; each world applies to all twelve seats\n")
    print(f"{'arm':18s} {'world':8s} {'title':>7s} {'all-play':>9s} {'reg pts':>8s} "
          f"{'adds':>6s} {'trades':>7s}")
    worlds = dict((a, w) for a, w, _ in ARMS)
    for arm, *_ in ARMS:
        g = d[d.arm == arm]
        print(f"{arm:18s} {worlds[arm]:8s} {100 * g.title.mean():6.1f}% "
              f"{100 * g.all_play.mean():8.1f}% {g.pts_reg.mean():8.0f} "
              f"{g.moves.mean():6.2f} {g.trades.mean():7.2f}")

    print("\nPRIMARY: the same policy in a changed world, against the published world")
    print("(a two-league comparison: the test seat's all-play against its own eleven "
          "opponents,\n under the changed rule, minus the same seat's all-play under the "
          "published rule)\n")
    verdict = {}
    for arm, world, spec in ARMS:
        if arm in ("stream", "ctrl") or spec is None:
            continue
        verdict[arm] = LB.paired(d, arm, "ctrl", f"{arm} - ctrl")
        pos = d[d.arm == arm].merge(d[d.arm == "ctrl"], on=["season", "league", "seat"])
        by = pos.assign(x=pos.all_play_x - pos.all_play_y).groupby("season").x.mean()
        pooled = float(pos.all_play_x.mean() - pos.all_play_y.mean())
        dt = float(pos.title_x.mean() - pos.title_y.mean())
        harmless = abs(pooled) <= HARMLESS and dt >= LB.TITLE_GUARD
        tag = "  <- preregistered verdict arm" if arm in VERDICT else "  (sensitivity)"
        print(f"    rule {'passes' if verdict[arm] else 'fails'}; harmless band "
              f"{'met' if harmless else 'missed'}; seasons positive {(by > 0).sum()}/5"
              + tag)

    print("\nSECONDARY: the policy's own edge inside each world (mutual_cap2 - stream)")
    for arm, world, spec in ARMS:
        if spec is None:
            continue
        ref = "stream" if arm == "ctrl" else f"{arm}_stream"
        LB.paired(d, arm, ref, f"{arm} - {ref}")

    print("\nCONTEXT: the world's effect on the stream control alone")
    for arm, world, spec in ARMS:
        if spec is not None or arm == "stream":
            continue
        LB.paired(d, arm, "stream", f"{arm} - stream")

    print("\nreported, not gated: trades and wire activity by arm")
    print(f"  {'arm':18s} {'searched':>8s} {'mutual>0':>9s} {'executed':>8s} "
          f"{'proj gain':>9s} {'trades/season':>13s}")
    for arm, world, spec in ARMS:
        if spec is None:
            continue
        x = dec[dec.arm == arm]
        t = tl[tl.arm == arm]
        mpos = (x.n_mutual > 0).mean() if len(x) and "n_mutual" in x else np.nan
        print(f"  {arm:18s} {len(x):8d} {mpos:9.2f} {x.executed.mean():8.2f} "
              f"{t.proj_gain.median():9.1f} {d[d.arm == arm].trades.mean():13.2f}")

    report_binding(cnt, diag)
    return verdict


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    CFG.announce_run(noise=LB.NOISE, leagues=LB.LEAGUES)
    check = "" if LB.LEAGUES == 20 else f"_check{LB.LEAGUES}"
    mech = bool(check) and "--mechanics" in sys.argv
    d, tl, dec, diag, cnt = run(mechanics=mech)
    d.to_parquet(f"data/league_settings{check}_noise{LB.NOISE:g}.parquet")
    tl.to_parquet(f"data/league_settings_log{check}_noise{LB.NOISE:g}.parquet")
    dec.to_parquet(f"data/league_settings_decisions{check}_noise{LB.NOISE:g}.parquet")
    diag.to_parquet(f"data/league_settings_diag{check}_noise{LB.NOISE:g}.parquet")
    cnt.to_parquet(f"data/league_settings_counts{check}_noise{LB.NOISE:g}.parquet")
    print("\nreproduction against sim_tradecap.py:")
    same = reproduces(d)
    print(f"  -> {'exact' if same else 'DIFFERS'}")
    if mech:
        print("mechanics check: rosters legal in every world (asserted), "
              "reproduction above; no outcome metrics printed")
        sys.exit(0 if same else 1)
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    report(d, tl, dec, diag, cnt)
