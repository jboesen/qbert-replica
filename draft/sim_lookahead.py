"""Look-ahead streaming: fill a starting hole a week or two before it arrives.

Hole-aware streaming (prereg_streaming.md) beat the consensus wire by filling next
week's holes. It waits until the hole is one week out, though, and by then the rest of
the league has had a waiver round to take the player who fills it. Streaming practice
(Subvertadown, 2-4 week holds on r/fantasyfootball) says to grab him earlier. This
tests that against streaming itself, the best known policy, with the same consensus
values and the same knowable availability.

Spec frozen in draft/prereg_lookahead.md before the real run:

  - Availability for week w+k at the decision before week w: the bye schedule for week
    w+k, plus known_unavailable's roster and injury reading as of week w-1, assumed to
    persist. For k = 0 this is known_unavailable exactly.
  - If week w has a hole: the streaming move. Otherwise, for the earliest week in the
    horizon with a hole and a legal filling move: add the best free agent (by the same
    rest-of-season consensus value) who fills it and is available that week, and cut
    the least useful cuttable player without creating a hole in any horizon week.
    Otherwise: the consensus move.
  - Arms: stream (control, run_streaming's stream arm), look2 (H=2), look3 (H=3).

    .venv/bin/python draft/sim_lookahead.py --noise 1
    .venv/bin/python draft/sim_lookahead.py --noise 0
    .venv/bin/python draft/sim_lookahead.py --noise 1 --leagues 2    # mechanics check only
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB
import weekly as W

POS, FLEX = LB.POS, LB.FLEX
TEAMS, REG, WEEKS = LB.TEAMS, LB.REG, LB.WEEKS
SEASONS = LB.SEASONS
LEAGUES, SCHEDULES = LB.LEAGUES, LB.SCHEDULES
NOISE = 1.0
HORIZONS = (2, 3)            # weeks looked at, the upcoming one included
ARMS = ("stream",) + tuple(f"look{h}" for h in HORIZONS)


# ---------------------------------------------------------------- availability ahead

def horizon_unavailable(S, y, h):
    """G[i, w-1, k]: at the decision before week w, player i is known unavailable in week
    w+k. Same reading as LB.known_unavailable, which is the k = 0 slice: the latest roster
    row at or before week w-1 decides status, Out/Doubtful and team, and only the bye
    check moves forward to week w+k. Nothing published after week w-1 is read, so an
    injury is assumed to last and a trade is not foreseen. Weeks past the season stay
    True and are never looked at."""
    n = len(S.ids)
    ix = {p: i for i, p in enumerate(S.ids)}
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG")]
    plays = {(w, t) for w, a, hm in zip(g.week, g.away_team, g.home_team) for t in (a, hm)}
    inj = pd.read_parquet(f"data/injuries_{y}.parquet")
    out = {(p, w) for p, w, s in zip(inj.gsis_id, inj.week, inj.report_status) if s in LB.OUT}
    ro = pd.read_parquet(f"data/roster_weekly_{y}.parquet")
    ro = ro[(ro.game_type == "REG") & (ro.week <= WEEKS) & ro.gsis_id.isin(ix)]
    ro = ro.assign(team=W.norm_team(ro.team)).sort_values("week")
    last = {}
    rows = {w: list(zip(x.gsis_id, x.status, x.team)) for w, x in ro.groupby("week")}
    G = np.ones((n, WEEKS, h), bool)
    for w in range(LB.FIRST_WAIVER, WEEKS + 1):
        for p, s, t in rows.get(w - 1, []):
            last[p] = (w - 1, s, t)
        for p, (wk, s, t) in last.items():
            hurt = s not in LB.ACTIVE or (p, wk) in out
            for k in range(h):
                if w + k <= WEEKS:
                    G[ix[p], w - 1, k] = hurt or (w + k, t) not in plays
    return G


def n_holes(S, players, gone_w):
    return sum(LB.holes(S, players, gone_w).values())


# ---------------------------------------------------------------- the policy

def lookahead_policy(S, G, h, log):
    """Streaming with a horizon of h weeks.

    A hole next week is the streaming decision, untouched, including its fallback. With
    none, the seat looks at weeks w+1..w+h-1 in order. Since the injury reading is the
    same for every week in the horizon, a future hole with no hole in week w can only be
    a bye, so this is in effect bye planning. The pre-acquisition ignores whether the add
    outranks the drop, as streaming does, but it may not open a hole in any week of the
    horizon: trading next week's lineup for a later one would just move the zero.
    """
    stream = LB.streaming_policy(S, G[:, :, 0], [])

    def move(S, held, val, free, w):
        g = G[:, w - 1, :]
        if n_holes(S, held, g[:, 0]):
            return stream(S, held, val, free, w)
        span = [k for k in range(h) if w + k <= WEEKS]
        before = {k: n_holes(S, held, g[:, k]) for k in span}
        for k in span[1:]:
            if not before[k]:
                continue
            short = LB.holes(S, held, g[:, k])
            fills = [p for p in POS if short[p] > 0 or (p in FLEX and short["FLEX"] > 0)]
            cand = free[np.isin(S.pos[free], fills) & ~g[free, k]]
            if not len(cand):
                continue
            add = int(cand[int(np.argmax(val[cand]))])
            after_add = list(held) + [add]
            lineup = LB.starters(S, after_add, val, g[:, 0])
            best = None
            for d in LB.cuttable(S, held):
                rest = [i for i in after_add if i != d]
                left = {j: n_holes(S, rest, g[:, j]) for j in span}
                if left[k] >= before[k] or any(left[j] > before[j] for j in span):
                    continue
                key = (left[k], sum(left.values()), d in lineup,
                       val[d] if np.isfinite(val[d]) else -np.inf)
                if best is None or key < best[0]:
                    best = (key, int(d))
            if best is None:
                continue
            mine = (add, best[1])
            cons = LB.consensus_move(S, held, val, free, w)
            log.append({"week": w, "target": w + k, "add": add, "drop": best[1],
                        "add_pos": S.pos[add], "drop_pos": S.pos[best[1]],
                        "add_val": val[add], "drop_val": best[0][3],
                        "same_as_cons": mine == cons})
            return mine
        return LB.consensus_move(S, held, val, free, w)
    return move


def recording(policy, seen):
    """Wrap a seat's policy to keep, per decision week, its roster and the free agents at
    its turn. Used on the control to ask whether streaming would still have found the
    look-ahead's player free when his week came."""
    def move(S, held, val, free, w):
        seen[w] = (set(held.tolist()), set(free.tolist()))
        return policy(S, held, val, free, w)
    return move


def simulate(S, drafted, values, movers, seat, check):
    """LB.simulate, plus the test seat's roster for each scored week and, in a check, a
    legality audit of every roster after every waiver round."""
    rosters = [list(r) for r in drafted]
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    hist, bad = [], 0
    every = set(range(TEAMS))
    for w in range(WEEKS):
        for t in range(TEAMS):
            scores[t, w] = LB.week_points(S, rosters[t], S.ecr_val, w)
        hist.append(list(rosters[seat]))
        if w + 1 < WEEKS:
            order = [t for t in LB.priority(scores, w) if t in every]
            moves += LB.waiver_week(S, rosters, values, order, w + 2, movers)
            if check:
                flat = [i for r in rosters for i in r]
                bad += len(flat) != len(set(flat))
                for t in range(TEAMS):
                    c = {p: int((S.pos[rosters[t]] == p).sum()) for p in POS}
                    bad += len(rosters[t]) != len(drafted[t])
                    bad += any(c[p] < k for p, k in LB.ROSTER_MIN.items())
    return scores, moves, hist, bad


def lineup_ids(S, roster, w):
    """Who starts in week w under consensus lineups with real eligibility, the same
    choice LB.week_points scores."""
    r = np.array(roster)
    pos = S.pos[r]
    v = np.where(S.elig[r, w], S.ecr_val[r, w], -np.inf)
    used = np.zeros(len(r), bool)
    for p, k in LB.SLOTS.items():
        cand = np.where((pos == p) & ~used & np.isfinite(v))[0]
        used[cand[np.argsort(-v[cand])][:k]] = True
    cand = np.where(np.isin(pos, FLEX) & ~used & np.isfinite(v))[0]
    if len(cand):
        used[cand[np.argmax(v[cand])]] = True
    return set(r[used].tolist())


def follow_up(S, e, hist, sc_arm, sc_stream, seen, gone, val, seat):
    """What became of one pre-acquisition when its target week t arrived."""
    t, add = e["target"], e["add"]
    roster = hist[t - 1]
    e["rostered"] = add in roster
    e["started"] = add in lineup_ids(S, roster, t - 1)
    # Needed: take him off the roster that played week t and a known hole reopens.
    e["needed"] = e["rostered"] and (
        n_holes(S, [i for i in roster if i != add], gone[:, t - 1])
        > n_holes(S, roster, gone[:, t - 1]))
    e["paired_pts"] = sc_arm[seat, t - 1] - sc_stream[seat, t - 1]
    # The control's view at its own turn: at the decision he was added, and before week t.
    held_w, free_w = seen[e["week"]]
    held_t, free_t = seen[t]
    e["stream_free_at_add"] = add in free_w
    if add in held_t:
        e["stream_at_target"] = "own"
    elif add in free_t:
        e["stream_at_target"] = "free"
    elif not np.isfinite(val[add, t - 1]):
        e["stream_at_target"] = "unvalued"     # off the wire's list, owner unknown
    else:
        e["stream_at_target"] = "taken"
    short = LB.holes(S, list(held_t), gone[:, t - 1])
    p = S.pos[add]
    e["stream_hole_fits"] = bool(short.get(p, 0) > 0 or (p in FLEX and short["FLEX"] > 0))
    return e


# ---------------------------------------------------------------- the run

def run(check=False):
    curves = LB.rank_curve()
    # Season() needs weekly projections only for model-lineup arms, not run here; with
    # none the model values fall back to the preseason rate, unused by waivers.
    no_proj = pd.DataFrame(columns=["player_id", "season", "week", "proj"])
    rows, moves_log, audit = [], [], []
    for y in SEASONS:
        S = LB.Season(y, no_proj, curves)
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        G = horizon_unavailable(S, y, max(HORIZONS))
        gone = G[:, :, 0]
        if check:
            audit.append({"season": y, "gone_matches": bool(
                (gone == LB.known_unavailable(S, y)).all())})
        base = [cons_val] * TEAMS
        for lg in range(LEAGUES):
            # The same draws in the same order as run_streaming, so drafts and control match.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for k in range(TEAMS + 1):
                score = S.ecr_mean + NOISE * S.ecr_sd * noise[k]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = LB.schedules(rng, SCHEDULES)
            for seat in range(TEAMS):
                orders = list(ecr_orders[:TEAMS])
                orders[seat] = S.exact_order
                drafted = LB.draft(S, orders)
                seen, logs, out = {}, {}, {}
                for arm in ARMS:
                    if arm == "stream":
                        pol = recording(LB.streaming_policy(S, gone, []), seen)
                    else:
                        logs[arm] = []
                        pol = lookahead_policy(S, G, int(arm[4:]), logs[arm])
                    sc, moves, hist, bad = simulate(S, drafted, base, {seat: pol}, seat, check)
                    out[arm] = (sc, hist)
                    res = [LB.season_outcome(sc, s) for s in scheds]
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": LB.all_play(sc, seat),
                        "pts_reg": sc[seat, :REG].sum(),
                        "pts_playoff": sc[seat, REG:].sum(),
                        "moves": moves[seat],
                        "illegal": bad,
                    })
                for arm, log in logs.items():
                    for e in log:
                        follow_up(S, e, out[arm][1], out[arm][0], out["stream"][0], seen,
                                  gone, cons_val, seat)
                        e.update(season=y, league=lg, seat=seat, arm=arm,
                                 add=S.ids[e["add"]], drop=S.ids[e["drop"]],
                                 add_name=S.name[e["add"]])
                        moves_log.append(e)
            print(f"{y} league {lg + 1}/{LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(moves_log), pd.DataFrame(audit)


# ---------------------------------------------------------------- reporting

def report(d, mv):
    print("\n=== waivers: look-ahead streaming vs streaming ===")
    print("12-team PPR leagues on 2021-25, exact-consensus draft and consensus lineups, "
          "every other\nseat on the consensus wire; one add/drop per team per week\n")
    print(f"{'arm':7s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s} {'adds':>6s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:7s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f} {g.moves.mean():6.2f}")
    print("\npreregistered test (prereg_lookahead.md), season-cluster bootstrap 90% interval:")
    for arm in ARMS[1:]:
        keep = LB.paired(d, arm, "stream", f"{arm} minus stream")
        print(f"  -> {arm}: {'passes this design' if keep else 'fails this design'}")
    key = ["season", "league", "seat"]
    for arm in ARMS[1:]:
        a = d[d.arm == arm].set_index(key)
        b = d[d.arm == "stream"].set_index(key).loc[a.index]
        print(f"  {arm} title by season: " + " ".join(
            f"{100 * v:+.1f}" for v in (a.title - b.title).groupby("season").mean()))

    seats = d[d.arm == "stream"].groupby("season").size()
    print("\nreported, not gated: pre-acquisitions by the test seat")
    if not len(mv):
        print("  none")
        return
    for arm, m in mv.groupby("arm"):
        n = len(m)
        print(f"  {arm}: {n / seats.sum():.2f} per seat-season "
              f"(by season " + " ".join(f"{len(m[m.season == y]) / k:.2f}"
                                         for y, k in seats.items()) + ")")
        print(f"    same move consensus would make: {100 * m.same_as_cons.mean():.0f}%;  "
              f"add ranks below drop: {100 * (m.add_val < m.drop_val).mean():.0f}%;  "
              f"lead time: " + " ".join(f"{k} wk {100 * v:.0f}%" for k, v in
                                        (m.target - m.week).value_counts(normalize=True)
                                        .sort_index().items()))
        print(f"    when his week came: still rostered {100 * m.rostered.mean():.0f}%, "
              f"still needed {100 * m.needed.mean():.0f}%, started {100 * m.started.mean():.0f}%;"
              f"  that week the arm outscored stream by {m.paired_pts.mean():+.1f}")
        at = m.stream_at_target.value_counts(normalize=True)
        fits = m[m.stream_hole_fits]
        print("    in the stream arm before his week: " + ", ".join(
            f"{k} {100 * v:.0f}%" for k, v in at.items())
            + f";  free when look-ahead took him {100 * m.stream_free_at_add.mean():.0f}%")
        if len(fits):
            print(f"    where streaming had a hole he fills ({100 * len(fits) / n:.0f}% of "
                  f"cases): taken by another team {100 * (fits.stream_at_target == 'taken').mean():.0f}%")
        print("    by position added: " + " ".join(
            f"{p} {k}" for p, k in m.add_pos.value_counts().items()))


if __name__ == "__main__":
    if "--noise" in sys.argv:
        NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = LEAGUES != LB.LEAGUES
    d, mv, audit = run(check)
    if check:
        # A reduced run writes its own files and prints mechanics only, never outcomes.
        d.to_parquet(f"data/league_lookahead_check{LEAGUES}_noise{NOISE:g}.parquet")
        mv.to_parquet(f"data/league_lookahead_moves_check{LEAGUES}_noise{NOISE:g}.parquet")
        print(f"\nmechanics, {LEAGUES} leagues per season, noise {NOISE:g}:")
        print(f"  horizon k=0 equals known_unavailable: {audit.gone_matches.tolist()}")
        print(f"  illegal roster events: {int(d.illegal.sum())}")
        ref = pd.read_parquet(f"data/league_streaming_noise{NOISE:g}.parquet")
        ref = ref[ref.arm == "stream"].set_index(["season", "league", "seat"])
        mine = d[d.arm == "stream"].set_index(["season", "league", "seat"])
        cols = ["all_play", "pts_reg", "pts_playoff", "title", "playoff", "wins", "moves"]
        same = (mine[cols] == ref.loc[mine.index, cols]).all().all()
        print(f"  control reproduces run_streaming's stream arm exactly: {same}")
        seats = len(mine)
        for arm, m in mv.groupby("arm"):
            print(f"  {arm}: {len(m)} pre-acquisitions over {seats} seat-seasons; "
                  f"lead " + str((m.target - m.week).value_counts().to_dict())
                  + "; add " + str(m.add_pos.value_counts().to_dict())
                  + "; drop " + str(m.drop_pos.value_counts().to_dict()))
            for e in m.head(6).itertuples():
                print(f"    {e.season} lg{e.league} seat{e.seat} wk{e.week}->{e.target}: "
                      f"add {e.add_name} {e.add_pos} ({e.add_val:.1f}), drop {e.drop_pos} "
                      f"({e.drop_val:.1f})")
        sys.exit()
    d.to_parquet(f"data/league_lookahead_noise{NOISE:g}.parquet")
    mv.to_parquet(f"data/league_lookahead_moves_noise{NOISE:g}.parquet")
    print(f"\nopponent noise: {NOISE:g} x ECR sd")
    report(d, mv)
