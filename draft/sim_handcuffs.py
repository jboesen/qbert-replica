"""Streaming plus a contingent running-back stash, against streaming alone
(prereg_handcuffs.md).

Consensus-following managers value players by current rest-of-season rank, so they
only pick up a backup running back after the starter ahead of him is hurt. A team that
already holds him wins that race. Holding another team's backup adds a new route to a
starter-level player; holding the backup behind your own starter only insures a slot you
already fill. So the test seat keeps hole-aware streaming exactly as it passed, and in
weeks it has no hole-filling move to make it may spend the week's move holding up to K
backups behind other teams' lead backs. A diagnostic arm stashes behind its own backs.
Every opponent keeps the consensus wire, as in run_streaming.

    .venv/bin/python draft/sim_handcuffs.py --noise 1
    .venv/bin/python draft/sim_handcuffs.py --noise 0 --leagues 2   # mechanics check
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import league_backtest as LB
import handcuff as HC
from build_data import norm_team

K = 2                    # stash slots
FULL_USAGE = 0.8         # a returning lead is back at 80% of his share when stashed
RETURN_GAMES = 2         # ... in each of his team's last two games


# ---------------------------------------------------------------- inputs per season

def rb_replacement(S, y, crv):
    """Consensus rest-of-season values, as the control computes them, plus the running
    back replacement level behind them at each decision.

    Contingent value has to be on the same scale as the values it competes with, points
    per week over replacement, and waiver_values does not return the level it subtracts.
    Wrapping over_replacement for one call records it without touching the control's
    numbers: the wrapper returns exactly what the original does.
    """
    levels = []
    orig = LB.over_replacement

    def spy(S_, pts):
        out = orig(S_, pts)
        at = np.where((S_.pos == "RB") & np.isfinite(out) & np.isfinite(pts))[0]
        levels.append(float(pts[at[0]] - out[at[0]]) if len(at) else np.nan)
        return out
    LB.over_replacement = spy
    try:
        cons_val, _ = LB.waiver_values(S, y, crv, None)
    finally:
        LB.over_replacement = orig
    # waiver_values prices two arrays per week, consensus first, identical without fits.
    repl = {w: levels[2 * (w - LB.FIRST_WAIVER)] for w in range(LB.FIRST_WAIVER, LB.WEEKS + 1)}
    return cons_val, repl


def roster_info(S, y):
    """At the decision before week w: each player's team, and whether he is known out
    for a reason other than a bye (off the 53, or Out or Doubtful last week), from the
    same latest-roster-row reading as known_unavailable. Row w-1 is the decision before
    week w."""
    n = len(S.ids)
    ix = {p: i for i, p in enumerate(S.ids)}
    inj = pd.read_parquet(f"data/injuries_{y}.parquet")
    out = {(p, w) for p, w, s in zip(inj.gsis_id, inj.week, inj.report_status) if s in LB.OUT}
    ro = pd.read_parquet(f"data/roster_weekly_{y}.parquet")
    ro = ro[(ro.game_type == "REG") & (ro.week <= LB.WEEKS) & ro.gsis_id.isin(ix)]
    ro = ro.assign(team=norm_team(ro.team)).sort_values("week")
    rows = {w: list(zip(x.gsis_id, x.status, x.team)) for w, x in ro.groupby("week")}
    last = {}
    team = np.full((n, LB.WEEKS), None, object)
    hurt = np.ones((n, LB.WEEKS), bool)
    for w in range(LB.FIRST_WAIVER, LB.WEEKS + 1):
        for p, s, t in rows.get(w - 1, []):
            last[p] = (w - 1, s, t)
        for p, (wk, s, t) in last.items():
            team[ix[p], w - 1] = t
            hurt[ix[p], w - 1] = s not in LB.ACTIVE or (p, wk) in out
    return team, hurt


def candidates(S, y, repl):
    """Stash candidates per decision week, best contingent value first.

    A candidate is a team's second back, in the harness's player pool, whose lead played
    the team's last game and whose fitted chance of inheriting the role is at least
    MIN_P_TRANSFER. Contingent value, in points per week over replacement:
    f_miss(w) x (expected fill-in PPR - RB replacement level at w).
    """
    fits = HC.fit_before(y)
    u, d = HC.season_states(y, fits)
    ix = {p: i for i, p in enumerate(S.ids)}
    d = d[(d.p >= HC.MIN_P_TRANSFER) & d.back.isin(ix)].copy()
    d["cv"] = d.f_miss * (d.fill - d.week.map(repl))
    out = {}
    for w, g in d.sort_values(["week", "cv"], ascending=[True, False]).groupby("week"):
        out[int(w)] = [dict(back=ix[r.back], lead_id=r.lead, lead=ix.get(r.lead, -1),
                            team=r.team, cv=r.cv, p=r.p, s1=r.s1)
                       for r in g.itertuples()]
    return u, out


# ---------------------------------------------------------------- policies

def protected_consensus(S, held, val, free, w, keep):
    """consensus_move, except stashed players are never the cut. With nothing kept it is
    consensus_move line for line."""
    add = free[int(np.argmax(val[free]))]
    can_cut = LB.cuttable(S, held)
    if keep and len(can_cut):
        can_cut = can_cut[~np.isin(can_cut, list(keep))]
    if not len(can_cut):
        return None
    vd = np.where(np.isfinite(val[can_cut]), val[can_cut], -np.inf)
    drop = int(can_cut[int(np.argmin(vd))])
    return (int(add), drop) if val[add] > vd.min() else None


def protected_streaming(S, gone, log, keep):
    """LB.streaming_policy with two changes that only bite when something is stashed:
    its consensus fallback never cuts a stash, and among drops that close as many holes
    and are equally (non-)starters, a stash is cut last. So a hole-filling move takes a
    stash's slot only when it needs it. With `keep` empty it is streaming_policy exactly,
    which the mechanics check verifies."""
    def move(S_, held, val, free, w):
        cons = protected_consensus(S, held, val, free, w, keep)
        g = gone[:, w - 1]
        before = LB.holes(S, held, g)
        if not sum(before.values()):
            return cons
        fills = [p for p in LB.POS if before[p] > 0 or (p in LB.FLEX and before["FLEX"] > 0)]
        cand = free[np.isin(S.pos[free], fills) & ~g[free]]
        if not len(cand):
            return cons
        add = int(cand[int(np.argmax(val[cand]))])
        after_add = list(held) + [add]
        lineup = LB.starters(S, after_add, val, g)
        best = None
        for d in LB.cuttable(S, held):
            left = sum(LB.holes(S, [i for i in after_add if i != d], g).values())
            if left >= sum(before.values()):
                continue
            key = (left, d in lineup, d in keep, val[d] if np.isfinite(val[d]) else -np.inf)
            if best is None or key < best[0]:
                best = (key, int(d))
        if best is None:
            return cons
        mine = (add, best[1])
        log.append({"week": w, "add": S.ids[add], "drop": S.ids[best[1]],
                    "add_pos": S.pos[add], "drop_pos": S.pos[best[1]],
                    "drop_stash": best[1] in keep})
        return mine
    return move


def stash_policy(S, gone, team_at, hurt, u, cands, mode, log, episodes, roster_by_week):
    """Streaming, plus up to K stashed backups (mode "cross": behind a lead back on an NFL
    team none of my running backs plays for; "own": behind a lead back I hold).

    The stash competes for the week's one move only when streaming has no hole-filling
    move to make, and takes it only if its contingent value beats the bench player it
    cuts and its gain beats the gain of the consensus move it displaces.
    """
    stash = {}                       # player index -> what he was stashed behind
    stream = protected_streaming(S, gone, log, stash)
    no_one_out = np.zeros(len(S.ids), bool)

    def release(w):
        for b, s in list(stash.items()):
            st = u.state(s["team"], w)
            if st is not None and S.ids[b] not in (st["lead"], st["back"]):
                why = "demoted"
            else:
                games = [x for x in u.team_weeks.get(s["team"], []) if s["since"] <= x < w]
                gone_at = [k for k, x in enumerate(games) if u.work(s["team"], x, s["lead_id"]) == 0]
                back = games[-RETURN_GAMES:]
                if not (gone_at and len(games) - 1 - gone_at[-1] >= RETURN_GAMES and all(
                        u.share(s["team"], x, s["lead_id"]) >= FULL_USAGE * s["s1"] for x in back)):
                    continue
                why = "lead_back"
            del stash[b]
            episodes[s["episode"]].update(released_week=w, reason=why)

    def stash_move(held, val, free, w, cons):
        g = gone[:, w - 1]
        rb_teams = {team_at[i, w - 1] for i in held if S.pos[i] == "RB"}
        held_ids = {S.ids[i] for i in held}
        free_set = set(int(i) for i in free)
        pick = None
        for c in cands.get(w, []):
            if c["back"] not in free_set or (c["lead"] >= 0 and hurt[c["lead"], w - 1]):
                continue
            if mode == "cross" and c["team"] in rb_teams:
                continue
            if mode == "own" and c["lead_id"] not in held_ids:
                continue
            pick = c
            break
        if pick is None:
            return None
        # The cut comes off the bench: not a starter this week from players known to be
        # available, nor a starter on value alone (a starter on bye is still a starter).
        lineup = LB.starters(S, held, val, g) | LB.starters(S, held, val, no_one_out)
        can = [int(i) for i in LB.cuttable(S, held) if i not in stash and i not in lineup]
        if not can:
            return None
        dv = np.array([val[i] if np.isfinite(val[i]) else -np.inf for i in can])
        drop = can[int(np.argmin(dv))]
        # A stash worth less than replacement even as the fill-in is no route to a
        # starter, however little the bench player he replaces is worth.
        if not pick["cv"] > max(dv.min(), 0.0):
            return None
        if cons is not None:
            cut = val[cons[1]] if np.isfinite(val[cons[1]]) else -np.inf
            if val[cons[0]] - cut >= pick["cv"] - dv.min():
                return None
        stash[pick["back"]] = dict(pick, since=w, episode=len(episodes))
        episodes.append({"week": w, "player": S.ids[pick["back"]], "lead": pick["lead_id"],
                         "team": pick["team"], "cv": pick["cv"], "p": pick["p"],
                         "drop": S.ids[drop], "drop_val": dv.min(),
                         "displaced_cons": cons is not None, "released_week": np.nan,
                         "reason": "", "end_week": np.nan, "_ix": pick["back"]})
        return (pick["back"], drop)

    def move(S_, held, val, free, w):
        release(w)
        n = len(log)
        m = stream(S, held, val, free, w)
        if len(log) == n and len(stash) < K:          # no hole-filling move this week
            m = stash_move(held, val, free, w, m) or m
        if m is not None:
            stash.pop(m[1], None)
            for e in episodes:
                if e["_ix"] == m[1] and np.isnan(e["end_week"]):
                    e["end_week"] = w
                    if not e["reason"]:
                        e.update(released_week=w, reason="cut_for_hole")
        roster_by_week[w] = [i for i in held if m is None or i != m[1]] + (
            [m[0]] if m is not None else [])
        return m
    return move


def lineup_players(S, roster, value, w):
    """The players week_points starts in week index w, same rule, same tie order."""
    r = np.array(roster)
    pos = S.pos[r]
    v = np.where(S.elig[r, w], value[r, w], -np.inf)
    used = np.zeros(len(r), bool)
    for p, k in LB.SLOTS.items():
        cand = np.where((pos == p) & ~used & np.isfinite(v))[0]
        used[cand[np.argsort(-v[cand])][:k]] = True
    cand = np.where(np.isin(pos, LB.FLEX) & ~used & np.isfinite(v))[0]
    if len(cand):
        used[cand[np.argmax(v[cand])]] = True
    return set(r[used].tolist())


# ---------------------------------------------------------------- the experiment

def run_handcuffs(check=False):
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, stashes, audit = [], [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        crv = LB.C.weekly_curve(range(2020, y))
        cons_val, repl = rb_replacement(S, y, crv)
        gone = LB.known_unavailable(S, y)
        team_at, hurt = roster_info(S, y)
        u, cands = candidates(S, y, repl)
        every = set(range(LB.TEAMS))
        for lg in range(LB.LEAGUES):
            # The same draws in the same order as run_streaming, so drafts and control match.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((LB.TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for k in range(LB.TEAMS + 1):
                score = S.ecr_mean + LB.NOISE * S.ecr_sd * noise[k]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = LB.schedules(rng, LB.SCHEDULES)
            base = [cons_val] * LB.TEAMS
            for seat in range(LB.TEAMS):
                orders = list(ecr_orders[:LB.TEAMS])
                orders[seat] = S.exact_order
                drafted = LB.draft(S, orders)
                arms = [("stream", LB.streaming_policy(S, gone, []), None)]
                if check:        # the protected copy with nothing stashed must match it
                    arms.append(("stream_copy", protected_streaming(S, gone, [], {}), None))
                for mode in ("cross", "own"):
                    eps, rbw = [], {}
                    pol = stash_policy(S, gone, team_at, hurt, u, cands, mode, [], eps, rbw)
                    arms.append((mode, pol, (eps, rbw)))
                for arm, pol, track in arms:
                    sc, moves, final = LB.simulate(S, drafted, base, every, {seat: pol})
                    res = [LB.season_outcome(sc, s) for s in scheds]
                    n_stash = 0
                    if track is not None:
                        eps, rbw = track
                        n_stash = len(eps)
                        rost = list(drafted[seat])
                        by_week = []
                        for w in range(1, LB.WEEKS + 1):
                            rost = rbw.get(w, rost)
                            by_week.append(rost)
                        for e in eps:
                            end = LB.WEEKS + 1 if np.isnan(e["end_week"]) else int(e["end_week"])
                            starts = [x for x in range(int(e["week"]), end)
                                      if e["_ix"] in lineup_players(S, by_week[x - 1], S.ecr_val, x - 1)]
                            e.update(season=y, league=lg, seat=seat, arm=arm,
                                     weeks_held=end - int(e["week"]), starts=len(starts),
                                     starts_reg=sum(x <= LB.REG for x in starts),
                                     start_pts=float(sum(S.actual[e["_ix"], x - 1] for x in starts)),
                                     lead_missed=int(any(u.work(e["team"], x, e["lead"]) == 0
                                                         for x in u.team_weeks.get(e["team"], [])
                                                         if int(e["week"]) <= x < end)))
                            stashes.append({k: v for k, v in e.items() if k != "_ix"})
                        if check:
                            audit.append(audit_rosters(S, y, lg, seat, arm, sc, final, by_week))
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": LB.all_play(sc, seat),
                        "pts_reg": sc[seat, :LB.REG].sum(),
                        "pts_playoff": sc[seat, LB.REG:].sum(),
                        "moves": moves[seat],
                        "stash_moves": n_stash,
                    })
            print(f"{y} league {lg + 1}/{LB.LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(stashes), pd.DataFrame(audit)


def audit_rosters(S, y, lg, seat, arm, sc, final, by_week):
    """Mechanics only: roster legality all season for the test seat, final legality for
    everyone, and the lineup reconstruction agreeing with the harness's own scores."""
    bad_week = 0
    for w, r in enumerate(by_week):
        c = {p: int((S.pos[r] == p).sum()) for p in LB.POS}
        if len(r) != LB.ROUNDS or len(set(r)) != len(r) or any(
                c[p] < k for p, k in LB.SLOTS.items()):
            bad_week += 1
    all_ids = [i for r in final for i in r]
    pts_ok = all(abs(sum(S.actual[i, w] for i in lineup_players(S, by_week[w], S.ecr_val, w))
                     - sc[seat, w]) < 1e-6 for w in range(LB.WEEKS))
    return {"season": y, "league": lg, "seat": seat, "arm": arm, "bad_weeks": bad_week,
            "final_sizes_ok": all(len(r) == LB.ROUNDS for r in final),
            "no_shared_players": len(all_ids) == len(set(all_ids)), "lineup_matches": pts_ok}


def check_handcuffs(d, st, audit):
    """Mechanics check at reduced leagues: prints no outcome metric."""
    print("\n=== mechanics check (no outcome metrics) ===")
    ref = pd.read_parquet(f"data/league_streaming_noise{LB.NOISE:g}.parquet")
    key = ["season", "league", "seat"]
    cols = ["title", "playoff", "wins", "all_play", "pts_reg", "pts_playoff", "moves"]
    ref = ref[(ref.arm == "stream") & (ref.league < LB.LEAGUES)].set_index(key)[cols].sort_index()
    for arm in ("stream", "stream_copy"):
        mine = d[d.arm == arm].set_index(key)[cols].sort_index()
        same = mine.index.equals(ref.index) and np.allclose(mine.values, ref.values, atol=1e-9)
        print(f"  {arm} reproduces run_streaming's stream arm exactly: {same}")
    print(f"  test-seat weeks with an illegal roster: {int(audit.bad_weeks.sum())}; "
          f"final rosters 14 each: {bool(audit.final_sizes_ok.all())}; "
          f"no player on two rosters: {bool(audit.no_shared_players.all())}; "
          f"lineup reconstruction matches scores: {bool(audit.lineup_matches.all())}")
    if not len(st):
        print("  no stash moves")
        return
    seats = d[d.arm == "cross"].groupby("season").size()
    for arm, g in st.groupby("arm"):
        print(f"\n  {arm}: stash moves per league-seat season by season "
              + " ".join(f"{len(g[g.season == y]) / n:.2f}" for y, n in seats.items()))
        print(f"    week added: {g.week.describe()[['min', '50%', 'max']].to_dict()}")
        print(f"    cv: {g.cv.describe()[['min', '50%', 'max']].round(2).to_dict()}   "
              f"drop value: {g.drop_val.describe()[['min', '50%', 'max']].round(2).to_dict()}")
        print(f"    p transfer: {g.p.describe()[['min', '50%', 'max']].round(2).to_dict()}   "
              f"displaced a consensus move: {g.displaced_cons.mean():.2f}")
        print(f"    release reasons: {g.reason.replace('', 'still held').value_counts().to_dict()}")
        print(f"    weeks held: median {g.weeks_held.median():.0f}; lead missed a game while "
              f"held: {g.lead_missed.mean():.2f}")
        print("    examples:")
        print(g[["season", "week", "player", "lead", "team", "cv", "p", "drop",
                 "released_week", "reason"]].head(6).to_string(index=False))


def report_handcuffs(d, st):
    print("\n=== waivers: streaming plus a contingent-RB stash vs streaming alone ===")
    print("12-team PPR leagues on 2021-25, exact-consensus draft and consensus lineups, "
          "every other\nseat on the consensus wire; one add/drop per team per week; "
          f"K = {K} stash slots\n")
    print(f"{'arm':6s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s} {'adds':>6s} {'stashes':>8s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:6s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f} {g.moves.mean():6.2f} "
              f"{g.stash_moves.mean():8.2f}")
    print("\npreregistered test (prereg_handcuffs.md), season-cluster bootstrap 90% interval:")
    keep = LB.paired(d, "cross", "stream", "cross stash minus stream")
    print(f"  -> {'passes this design' if keep else 'fails this design'}")
    print("\ndiagnostic, not gated:")
    LB.paired(d, "own", "stream", "own stash minus stream")
    LB.paired(d, "cross", "own", "cross minus own stash")
    for arm in ("cross", "own"):
        a = d[d.arm == arm].set_index(["season", "league", "seat"])
        b = d[d.arm == "stream"].set_index(["season", "league", "seat"]).loc[a.index]
        by = (a.title - b.title).groupby("season").mean()
        print(f"  {arm} title change by season: " + " ".join(f"{100 * v:+.1f}" for v in by))

    print("\nreported, not gated: stash episodes (per league-seat season)")
    seats = d[d.arm == "cross"].groupby("season").size()
    print(f"  {'arm':5s} {'season':6s} {'stashes':>8s} {'started':>8s} {'starts':>7s} "
          f"{'pts/start':>10s} {'start pts':>10s} {'lead missed':>12s}")
    for arm in ("cross", "own"):
        g = st[st.arm == arm] if len(st) else st
        for y, n in list(seats.items()) + [("all", seats.sum())]:
            x = g if y == "all" else g[g.season == y]
            if not len(x):
                print(f"  {arm:5s} {str(y):6s} {0:8.2f}")
                continue
            print(f"  {arm:5s} {str(y):6s} {len(x) / n:8.2f} {(x.starts > 0).mean():8.2f} "
                  f"{x.starts.sum() / n:7.2f} {x.start_pts.sum() / max(x.starts.sum(), 1):10.1f} "
                  f"{x.start_pts.sum() / n:10.1f} {x.lead_missed.mean():12.2f}")
    print("  (started: share of stashes that ever started for the test seat while on its "
          "roster;\n   starts and start pts: per league-seat season; lead missed: share of "
          "stashes whose\n   lead back had a game with no work while the stash was held)")
    if len(st):
        print("  release reasons: " + "; ".join(
            f"{arm} " + ", ".join(f"{k} {v}" for k, v in
                                  g.reason.replace("", "held to end").value_counts().items())
            for arm, g in st.groupby("arm")))


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = LB.LEAGUES != 20
    tag = f"_check{LB.LEAGUES}" if check else ""     # never overwrite a real result
    d, st, audit = run_handcuffs(check)
    d.to_parquet(f"data/league_handcuffs{tag}_noise{LB.NOISE:g}.parquet")
    st.to_parquet(f"data/league_handcuffs_stashes{tag}_noise{LB.NOISE:g}.parquet")
    print(f"\nopponent noise: {LB.NOISE:g} x ECR sd")
    if check:
        check_handcuffs(d, st, audit)
    else:
        report_handcuffs(d, st)
