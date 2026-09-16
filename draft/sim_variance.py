"""Opponent-aware and situation-aware variance in weekly lineups (prereg_variance.md).

winprob.py asked which lineup beats the most of the eleven other teams. That objective is
nearly linear in score, so it almost never left consensus and changed nothing. The real
variance argument is situational and head-to-head: the right lineup depends on the one
opponent you actually play this week, and on what a win is worth to you. A big underdog
should buy variance with projected points; a big favourite should not. A seat whose
playoff life turns on this week should pay more for variance than one already locked in
or already out.

So this module keeps consensus's numbers and changes only the weekly lineup decision, on
top of the best realistic policy so far (mutual_cap2 from sim_tradecap.py: hole-aware
streaming plus at most two win-win trades a season).

  - Objective: P(my lineup outscores this week's head-to-head opponent's lineup), under
    winprob.py's empirical spread model fit on prior seasons only. The opponent is drawn,
    not treated as a fixed number.
  - Price: a budget C of projected points the lineup may give up against the consensus
    lineup. C = 0 is consensus; a large C buys a lot of variance.
  - Situation sets C: the swing in playoff odds between winning and losing this week,
    computed from games played so far only.

    .venv/bin/python draft/sim_variance.py --noise 1
    .venv/bin/python draft/sim_variance.py --noise 1 --leagues 2 --mechanics
"""
import sys
import time
from functools import lru_cache
from math import erf, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
import sim_trades as ST
import sim_tradecap as TC
import weekly as W
import winprob as WP
from league_backtest import POS, REG, TEAMS, WEEKS

DRAWS = 2000                 # Monte Carlo scenarios per seat-week, shared by every arm
MARGIN_SE = 2.0              # leave consensus only past this many Monte Carlo std errors
C_MAX = 12.0                 # the largest projected-point budget any arm will spend
C_STEP = 3.0
LEVELS = np.arange(0.0, C_MAX + C_STEP / 2, C_STEP)      # 0, 3, 6, 9, 12
LOCK = 0.90                  # playoff odds at or above this count as locked in
DEAD = 0.05                  # at or below this, out of the race
MUSTWIN = 0.20               # leverage above this is a must-win week (reporting only)
CONTROL = "mutual_cap2"      # sim_tradecap's verdict arm, and this test's control
CONTROL_SPEC = ("mutual", 2, TC.GAIN_MIN)
ARMS = ["h2h3", "h2h12", "situ", "live"]
VERDICT = "situ"


# ---------------------------------------------------------------- playoff odds

# P(I finish ahead of one team) depends only on the win gap and the games left, so the
# whole seat-week calculation collapses to a lookup plus an 11-team Poisson binomial.
# Cached because the same standings shape recurs across schedules, seats and leagues.

@lru_cache(maxsize=None)
def _odds(dwins2, left, pfwin):
    """P(top 6 of 12) given doubled win gaps to the other eleven and `left` games each.

    Each remaining game is a coin flip, so a team's final wins are current wins plus
    Binomial(left, 1/2); the gap to one team is normal with variance left/2. The eleven
    comparisons are treated as independent, which they are not (opponents play each
    other), but this only has to sort weeks into a few budget levels.
    """
    ps = []
    sd = sqrt(left / 2) if left > 0 else 0.0
    for d2, pw in zip(dwins2, pfwin):
        d = d2 / 2.0
        if sd == 0.0:
            ps.append(1.0 if d > 0 else (float(pw) if d == 0 else 0.0))
        else:
            ps.append(0.5 * (1 + erf(d / (sd * sqrt(2)))))
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[6:].sum())


def _key(v, pf, me, left):
    others = [t for t in range(TEAMS) if t != me]
    d2 = tuple(int(round(2 * (v[me] - v[t]))) for t in others)
    pfw = tuple(bool(pf[me] > pf[t]) for t in others)
    return _odds(d2, left, pfw)


def odds_now(wins, pf, me, w):
    """Playoff odds before week w is played, from weeks 0..w-1 only."""
    return _key(wins, pf, me, REG - w)


def leverage(wins, pf, me, opp, w):
    """How much this week's head-to-head result moves my playoff odds.

    The other ten teams also play this week; their games are unresolved, so they carry
    half a win. Only results already played enter `wins` and `pf`.
    """
    left = REG - w - 1
    out = []
    for iwin in (True, False):
        v = wins.astype(float).copy()
        v += 0.5
        v[me] -= 0.5
        v[opp] -= 0.5
        v[me] += 1.0 if iwin else 0.0
        v[opp] += 0.0 if iwin else 1.0
        out.append(_key(v, pf, me, left))
    return out[0] - out[1]


def budget(arm, wins, pf, me, opp, w):
    """The projected-point budget this arm will spend on variance in week w."""
    if arm == "h2h3":
        return 3.0, np.nan
    if arm == "h2h12":
        return C_MAX, np.nan
    if arm == "situ":
        lev = leverage(wins, pf, me, opp, w)
        c = C_STEP * round((C_MAX / C_STEP) * lev)
        return float(min(max(c, 0.0), C_MAX)), lev
    # live: spend everything while the seat is alive but unsettled, nothing otherwise
    o = odds_now(wins, pf, me, w)
    return (C_MAX if DEAD < o < LOCK else 0.0), o


# ---------------------------------------------------------------- the roster path

def path(S, drafted, values, seat, movers, inp):
    """TC.simulate for the control arm, with every team's roster recorded each week.

    Copied from sim_tradecap.simulate so the control's scores are identical; the only
    addition is the weekly snapshot, which the lineup policy needs. The mechanics check
    asserts the scores, moves and final rosters match TC.simulate exactly.
    """
    rosters = [list(r) for r in drafted]
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    every = set(range(TEAMS))
    done = 0
    accept, cap, gmin = CONTROL_SPEC
    snap = []
    for w in range(WEEKS):
        snap.append([list(r) for r in rosters])
        for t in range(TEAMS):
            scores[t, w] = LB.week_points(S, rosters[t], S.ecr_val, w)
        if w + 1 < WEEKS:
            order = [t for t in LB.priority(scores, w) if t in every]
            moves += LB.waiver_week(S, rosters, values, order, w + 2, movers)
            v = w + 2
            if v not in ST.TRADE_WEEKS or done >= cap:
                continue
            val = values[seat][:, v - 1]
            pts, avail = inp["pts"][:, v - 1], inp["avail"][v]
            found, _ = TC.search_mutual(S, rosters, seat, v, val, pts, avail, 0.0,
                                        mutual=True)
            if found is None or found["gain"] < gmin:
                continue
            ST.execute(S, rosters, seat, found, val)
            done += 1
    return scores, moves, rosters, snap


# ---------------------------------------------------------------- the weekly decision

def week_table(S, model, quest, snap_w, seat, w, rng):
    """Everything the arms need for one seat-week, precomputed once and shared.

    Returns the candidate lineups, their realised points, their projected points, and
    for every possible head-to-head opponent and every budget level the lineup the policy
    would start and the win probability it and the consensus lineup carry.
    """
    cache = {}

    def scen(i):
        if i not in cache:
            cache[i] = WP.draw(model, S.pos[i], S.ecr_val[i, w], quest[i, w], S.p_q, rng)
        return cache[i]

    cons = WP.consensus_lineup(S, snap_w[seat], w)
    cands = [cons] + sorted(WP.candidates(S, model, snap_w[seat], w, quest) - {cons})
    players = sorted({i for c in cands for i in c})
    col = {i: k for k, i in enumerate(players)}
    D = (np.array([scen(i) for i in players], np.float32) if players
         else np.zeros((0, DRAWS), np.float32))
    A = np.zeros((len(cands), len(players)), np.float32)
    for k, c in enumerate(cands):
        A[k, [col[i] for i in c]] = 1.0
    mine = A @ D                                            # (ncand, DRAWS)
    mu = np.array([S.ecr_val[list(c), w].sum() if c else 0.0 for c in cands])
    real = np.array([S.actual[list(c), w].sum() if c else 0.0 for c in cands])
    sd = mine.std(1) if len(cands) else np.zeros(0)
    nq = np.array([int(quest[list(c), w].sum()) if c else 0 for c in cands])
    # How much shape a budget can actually buy: the widest and narrowest lineup the seat
    # could legally field for each budget level, against the consensus lineup's own sd.
    band = np.array([[sd[mu >= mu[0] - c - 1e-9].min(), sd[mu >= mu[0] - c - 1e-9].max()]
                     for c in LEVELS]) if len(cands) else np.zeros((len(LEVELS), 2))

    nlev = len(LEVELS)
    pick = np.full((TEAMS, nlev), -1, int)
    pw_pick = np.full((TEAMS, nlev), np.nan)
    pw_cons = np.full(TEAMS, np.nan)
    opp_mu = np.full(TEAMS, np.nan)
    for t in range(TEAMS):
        if t == seat:
            continue
        lu = WP.consensus_lineup(S, snap_w[t], w)
        tot = (np.sum([scen(i) for i in lu], axis=0).astype(np.float32) if lu
               else np.zeros(DRAWS, np.float32))
        opp_mu[t] = S.ecr_val[list(lu), w].sum() if lu else 0.0
        srt = np.sort(tot)
        below = np.searchsorted(srt, mine.ravel(), side="left").reshape(mine.shape)
        pw = below.mean(1) / DRAWS
        pw_cons[t] = pw[0]
        for j, c in enumerate(LEVELS):
            ok = np.where(mu >= mu[0] - c - 1e-9)[0]
            b = int(ok[np.argmax(pw[ok])])                  # first maximum keeps consensus
            if b:
                # Paired standard error of the gain over consensus, on my own draws; the
                # opponent's simulation noise is shared by both lineups and drops out.
                g = (np.searchsorted(srt, mine[b], side="left")
                     - np.searchsorted(srt, mine[0], side="left")) / DRAWS
                if pw[b] - pw[0] <= MARGIN_SE * g.std() / sqrt(DRAWS):
                    b = 0
            pick[t, j] = b
            pw_pick[t, j] = pw[b]
    return {"cands": cands, "mu": mu, "real": real, "pick": pick, "sd": sd, "nq": nq,
            "pw_pick": pw_pick, "pw_cons": pw_cons, "opp_mu": opp_mu, "band": band}


def legality(S, roster, lu, w):
    """Mechanics: winprob's own check, so a started lineup is one the harness would allow."""
    return WP.legality(S, roster, lu, w)


# ---------------------------------------------------------------- the season

def play(tab, ctrl, scheds, seat, arm):
    """One arm's regular season under every schedule, on the control's roster path.

    Returns the seat's per-schedule weekly points and a diagnostics accumulator. The
    lineup depends on the week's opponent and on the standings so far, both of which are
    schedule-specific, so this loop runs once per schedule draw.
    """
    lev_idx = {float(c): j for j, c in enumerate(LEVELS)}
    pts = np.zeros((len(scheds), REG))
    acc = {"weeks": 0, "differ": 0, "pw_gain": 0.0, "won": 0, "won_cons": 0,
           "budget": 0.0, "mustwin": 0, "mw_differ": 0, "mw_pw_gain": 0.0,
           "mw_won": 0, "mw_won_cons": 0, "dmu": 0.0, "und_differ": 0, "und": 0,
           "dsd": 0.0, "dq": 0, "dsd_up": 0}
    prep = [[(np.array([p[0] for p in rd]), np.array([p[1] for p in rd]),
              next(b if a == seat else a for a, b in rd if seat in (a, b)))
             for rd in s[:REG]] for s in scheds]
    for si in range(len(scheds)):
        wins = np.zeros(TEAMS)
        pf = np.zeros(TEAMS)
        for w in range(REG):
            ha, hb, opp = prep[si][w]
            t = tab[w]
            c, sit = budget(arm, wins, pf, seat, opp, w)
            k = t["pick"][opp, lev_idx[c]]
            pts[si, w] = t["real"][k]
            # Diagnostics: what the model expected against what the week actually did.
            acc["weeks"] += 1
            acc["budget"] += c
            oppp = ctrl[opp, w]
            mw = arm == "situ" and not np.isnan(sit) and sit >= MUSTWIN
            if k:
                gain = t["pw_pick"][opp, lev_idx[c]] - t["pw_cons"][opp]
                acc["differ"] += 1
                acc["dmu"] += t["mu"][0] - t["mu"][k]
                acc["dsd"] += t["sd"][k] - t["sd"][0]
                acc["dsd_up"] += t["sd"][k] > t["sd"][0]
                acc["dq"] += int(t["nq"][k] - t["nq"][0])
                acc["pw_gain"] += gain
                acc["won"] += t["real"][k] > oppp
                acc["won_cons"] += t["real"][0] > oppp
                if mw:
                    acc["mw_differ"] += 1
                    acc["mw_pw_gain"] += gain
                    acc["mw_won"] += t["real"][k] > oppp
                    acc["mw_won_cons"] += t["real"][0] > oppp
            if t["mu"][0] < t["opp_mu"][opp]:
                acc["und"] += 1
                acc["und_differ"] += k != 0
            acc["mustwin"] += mw
            # Standings after week w, for the next week's situation. Ties go to the
            # second team, as season_outcome resolves them.
            sc = ctrl[:, w].copy()
            sc[seat] = pts[si, w]
            np.add.at(wins, np.where(sc[ha] > sc[hb], ha, hb), 1)
            pf += sc
    return pts, acc


def run(check=False):
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, mech = [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        S.p_q = W.play_probs(y).get("Questionable", 0.57)
        quest = WP.questionable(S, y)
        model = WP.fit_spread(y, curves)
        ST.POS_CODE_ARR = np.array([ST.POS_CODE.get(p, -1) for p in S.pos])
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        teams = ST.team_at(S, y)
        pts_ros = ST.ros_points(S, y, crv)
        inp = {"pts": pts_ros, "avail": ST.future_avail(S, y, gone, teams)}
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
                movers = {seat: LB.streaming_policy(S, gone, [])}
                ctrl, moves, final, snap = path(S, drafted, base_vals, seat, movers, inp)
                if check:
                    mv2 = {seat: LB.streaming_policy(S, gone, [])}
                    sc2, mo2, fin2 = TC.simulate(S, drafted, base_vals, seat, mv2,
                                                 CONTROL_SPEC, inp, [], [])
                    mech.append({"season": y, "same_scores": np.array_equal(ctrl, sc2),
                                 "same_moves": np.array_equal(moves, mo2),
                                 "same_rosters": all(sorted(a) == sorted(b)
                                                     for a, b in zip(final, fin2))})
                tab = []
                for w in range(REG):
                    mc = np.random.default_rng([y, lg, seat, w, 11])
                    t0 = time.perf_counter()
                    tab.append(week_table(S, model, quest, snap[w], seat, w, mc))
                    if check:
                        legal = all(legality(S, snap[w][seat], c, w)["legal"]
                                    for c in tab[-1]["cands"])
                        mech[-1].setdefault("legal", []).append(legal)
                        mech[-1].setdefault("sec", []).append(time.perf_counter() - t0)
                        mech[-1].setdefault("ncand", []).append(len(tab[-1]["cands"]))
                        mech[-1].setdefault("band", []).append(tab[-1]["band"])
                res = [LB.season_outcome(ctrl, s) for s in scheds]
                rows.append(row(y, lg, seat, CONTROL, ctrl, ctrl, scheds, res, moves, {}))
                for arm in ARMS:
                    mine, acc = play(tab, ctrl, scheds, seat, arm)
                    rows.append(row(y, lg, seat, arm, ctrl, mine, scheds, None, moves, acc))
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(mech)


def row(y, lg, seat, arm, ctrl, mine, scheds, res, moves, acc):
    """One arm's season summary. `mine` is the seat's weekly points, per schedule for the
    variance arms and the control's single row for the control."""
    others = np.delete(ctrl[:, :REG], seat, 0)
    if res is not None:                                   # control: schedule-free scores
        ap = LB.all_play(ctrl, seat)
        title = np.mean([r[2][seat] for r in res])
        playoff = np.mean([r[1][seat] for r in res])
        wins = np.mean([r[0][seat] for r in res])
        reg = ctrl[seat, :REG].sum()
    else:
        aps, titles, playoffs, winss = [], [], [], []
        for si, s in enumerate(scheds):
            sc = ctrl.copy()
            sc[seat, :REG] = mine[si]
            w_, p_, t_ = LB.season_outcome(sc, s)
            aps.append(LB.all_play(sc, seat))
            titles.append(t_[seat]); playoffs.append(p_[seat]); winss.append(w_[seat])
        ap, title = float(np.mean(aps)), float(np.mean(titles))
        playoff, wins = float(np.mean(playoffs)), float(np.mean(winss))
        reg = float(mine.mean(0).sum())
    return {"season": y, "league": lg, "seat": seat, "arm": arm, "title": title,
            "playoff": playoff, "wins": wins, "all_play": ap, "pts_reg": reg,
            "pts_playoff": ctrl[seat, REG:].sum(), "moves": moves[seat], **acc}


# ---------------------------------------------------------------- reporting

def report(d):
    print("\n=== situational, opponent-aware variance in weekly lineups ===")
    print("12-team PPR leagues on 2021-25. Every arm plays mutual_cap2 (streaming plus at "
          "most two\nwin-win trades); only the seat's weeks 1-14 lineups differ, and they "
          "are chosen against the\nweek's actual head-to-head opponent, per schedule draw.\n")
    print(f"{'arm':10s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:10s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
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

    print("\ndiagnostics (seat-week-schedule decisions, weeks 1-14; not gated):")
    print(f"  {'arm':10s} {'differ':>7s} {'budget':>7s} {'pts given':>10s} "
          f"{'E[dP(win)]':>11s} {'realised':>9s} {'underdog differ':>16s}")
    for arm in ARMS:
        g = d[d.arm == arm]
        n, nd = g.weeks.sum(), g.differ.sum()
        if not nd:
            print(f"  {arm:10s} {0.0:6.1f}%")
            continue
        print(f"  {arm:10s} {100 * nd / n:6.1f}% {g.budget.sum() / n:7.2f} "
              f"{g.dmu.sum() / nd:10.2f} {g.pw_gain.sum() / nd:+11.4f} "
              f"{(g.won.sum() - g.won_cons.sum()) / nd:+9.4f} "
              f"{100 * g.und_differ.sum() / max(g.und.sum(), 1):15.1f}%")
        print(f"             where it differs: lineup sd {g.dsd.sum() / nd:+.2f} "
              f"(wider in {100 * g.dsd_up.sum() / nd:.0f}% of them), questionable "
              f"starters {g.dq.sum() / nd:+.2f}")
    print("  by season, share of weeks the lineup differs from consensus:")
    for arm in ARMS:
        g = d[d.arm == arm].groupby("season")
        print(f"    {arm:10s} " + " ".join(f"{100 * x.differ.sum() / x.weeks.sum():.1f}"
                                           for _, x in g))
    g = d[d.arm == VERDICT]
    mw, mwd = g.mustwin.sum(), g.mw_differ.sum()
    print(f"\n  {VERDICT} in must-win weeks (playoff-odds swing >= {MUSTWIN}): "
          f"{100 * mw / g.weeks.sum():.1f}% of weeks, lineup differs in "
          f"{100 * mwd / max(mw, 1):.1f}% of them")
    if mwd:
        print(f"    expected win-probability gain {g.mw_pw_gain.sum() / mwd:+.4f}, realised "
              f"{(g.mw_won.sum() - g.mw_won_cons.sum()) / mwd:+.4f}")
    return verdict


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    if "--schedules" in sys.argv:
        LB.SCHEDULES = int(sys.argv[sys.argv.index("--schedules") + 1])
    if "--draws" in sys.argv:
        DRAWS = int(sys.argv[sys.argv.index("--draws") + 1])
    WP.DRAWS = DRAWS
    chk = LB.LEAGUES != 20 or LB.SCHEDULES != 20
    d, mech = run(chk)
    tag = "" if not chk else f"_check{LB.LEAGUES}x{LB.SCHEDULES}"
    d.to_parquet(f"data/league_variance{tag}_noise{LB.NOISE:g}.parquet")
    if chk:
        print(f"\nmechanics, {LB.LEAGUES} leagues x {LB.SCHEDULES} schedules, "
              f"noise {LB.NOISE:g}:")
        print(f"  control reproduces {CONTROL} exactly: scores "
              f"{mech.same_scores.all()}, moves {mech.same_moves.all()}, "
              f"rosters {mech.same_rosters.all()}")
        legal = np.concatenate(mech.legal.values)
        sec = np.concatenate(mech.sec.values)
        nc = np.concatenate(mech.ncand.values)
        print(f"  candidate lineups legal in {legal.mean():.3f} of {len(legal)} seat-weeks")
        print(f"  candidates per seat-week: mean {nc.mean():.1f}, max {nc.max()}")
        print(f"  seconds per seat-week table: mean {sec.mean():.3f}, max {sec.max():.3f}")
        print("  how often each arm leaves the consensus lineup, and what it spends:")
        for arm in ARMS:
            g = d[d.arm == arm]
            print(f"    {arm:10s} differs {100 * g.differ.sum() / g.weeks.sum():5.2f}% of "
                  f"decisions, mean budget {g.budget.sum() / g.weeks.sum():5.2f} pts, "
                  f"must-win weeks {100 * g.mustwin.sum() / g.weeks.sum():5.1f}%")
        band = np.concatenate([np.stack(b) for b in mech.band.values])   # (weeks, lev, 2)
        print("  lineup sd a budget can buy (mean over seat-weeks), by budget level:")
        for j, c in enumerate(LEVELS):
            print(f"    C={c:4.0f}  narrowest {band[:, j, 0].mean():5.2f}  "
                  f"widest {band[:, j, 1].mean():5.2f}")
        sys.exit()
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    report(d)
