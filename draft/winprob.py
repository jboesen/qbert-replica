"""Win-probability lineups: start the lineup that beats the most teams, not the one
that scores the most.

Every attempt to out-forecast consensus has failed, so this one keeps consensus's own
values and changes only the decision. A consensus manager starts the highest expected
points. But all-play counts how many of the eleven other teams you outscore each week,
and that count is not linear in your score: a team projected to lose gains from
variance and a team projected to win loses from it. So the lineup with the most
expected wins can differ from the lineup with the most expected points, with no better
forecast of any player.

Spec frozen in draft/prereg_winprob.md before the real run:

  - Spread model: for season y, the empirical distribution of actual weekly PPR around
    the harness's consensus value (weekly positional rank -> points), per position and
    value quintile, fit on seasons 2020..y-1 and centred to mean zero in each bucket,
    so a player's expected points are exactly consensus's value.
  - Policy: in weeks 1-14 the test seat enumerates its legal lineups and starts the one
    with the most expected opponents outscored, given that the eleven opponents start by
    consensus. Weeks 15-17 keep consensus lineups.
  - Arms: exact-consensus draft plus consensus lineups (cons, the control) or plus
    win-probability lineups (winprob). Opponents are unchanged from the harness.

    .venv/bin/python draft/winprob.py --noise 1
    .venv/bin/python draft/winprob.py --noise 0
    .venv/bin/python draft/winprob.py --noise 1 --leagues 2    # mechanics check only
"""
import sys
import time
from itertools import combinations, product

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import league_backtest as LB
import weekly as W

POS, SLOTS, FLEX = LB.POS, LB.SLOTS, LB.FLEX
TEAMS, REG, WEEKS = LB.TEAMS, LB.REG, LB.WEEKS
SEASONS = LB.SEASONS
LEAGUES, SCHEDULES = LB.LEAGUES, LB.SCHEDULES
NOISE = 1.0
FIRST_FIT = 2020            # the first season with weekly consensus ranks for a full year
BUCKETS = 5                 # value quintiles per position
MIN_BUCKET = 30             # a thinner bucket borrows its neighbour's residuals
DRAWS = 10000               # Monte Carlo scenarios per week, shared by every lineup
# Leave consensus only when the estimated gain clears this many Monte Carlo standard
# errors. The true gains are hundredths of an opponent, so without a margin most
# departures chase simulation noise; a check at 2,000 draws with no margin picked a
# different lineup under a second seed in 18% of weeks.
MARGIN_SE = 2.0
# The most starters one position can take: its slots plus the flex. A player beaten on
# value by this many others in his own spread bucket is never worth starting, because
# swapping him for an unused one shifts the whole score distribution up.
MAX_START = {p: SLOTS[p] + (p in FLEX) for p in POS}


# ---------------------------------------------------------------- the spread model

def fit_spread(y, curves):
    """Residual pools per position and value bucket, from seasons FIRST_FIT..y-1.

    The value is the harness's own: the player's weekly positional consensus rank read
    through LB.rank_curve. The pool is player-weeks the harness would let a manager
    start and that the mixture below doesn't already handle: ranked that week, team not
    on bye, not Out, Doubtful or Questionable. Roster status (ACT/INA) is not used here
    because the weekly roster files start in 2021, and 2021's fit has only 2020.
    """
    g = pd.read_csv("data/games.csv")
    ecr = pd.read_parquet("data/ecr_weekly.parquet")
    rows = []
    for s in range(FIRST_FIT, y):
        w = ecr[(ecr.season == s) & (ecr.week <= WEEKS)]
        st = pd.read_parquet(f"data/stats/w{s}.parquet")
        st = st[st.season_type == "REG"]
        pts = st.groupby(["player_id", "week"]).fantasy_points_ppr.sum()
        # A player's team for the bye check is his most common team in that season's
        # box scores; a player with none never played and scores zero either way.
        team = st.groupby("player_id").team.agg(lambda t: t.mode().iloc[0])
        gs = g[(g.season == s) & (g.game_type == "REG")]
        plays = {(wk, t) for wk, a, h in zip(gs.week, gs.away_team, gs.home_team)
                 for t in (a, h)}
        inj = pd.read_parquet(f"data/injuries_{s}.parquet")
        flagged = {(p, wk) for p, wk, r in zip(inj.gsis_id, inj.week, inj.report_status)
                   if r in LB.OUT or r == "Questionable"}
        w = w.assign(rank=w.groupby(["week", "pos"]).ecr.rank(method="first"))
        w = w.assign(team=w.player_id.map(team))
        keep = [(p, wk) not in flagged and (pd.isna(t) or (wk, t) in plays)
                for p, wk, t in zip(w.player_id, w.week, w.team)]
        w = w[keep]
        actual = pts.reindex(list(zip(w.player_id, w.week))).fillna(0.0).values
        value = np.zeros(len(w))
        for p in POS:
            m = (w.pos == p).values
            value[m] = LB.curve_points(curves, p, w["rank"].values[m])
        rows.append(pd.DataFrame({"pos": w.pos.values, "value": value, "actual": actual}))
    d = pd.concat(rows)

    model = {}
    for p in POS:
        x = d[d.pos == p]
        edges = np.unique(np.quantile(x.value, np.arange(1, BUCKETS) / BUCKETS))
        b = np.digitize(x.value, edges)
        pools = [(x.actual - x.value).values[b == k] for k in range(len(edges) + 1)]
        # Borrow from the nearest bucket with enough cases, so no pool is a handful.
        filled = []
        for k in range(len(pools)):
            near = sorted(range(len(pools)), key=lambda j: abs(j - k))
            filled.append(next(pools[j] for j in near if len(pools[j]) >= MIN_BUCKET))
        # Centre each pool: the test is about the shape of the distribution, so the
        # expected points must stay exactly consensus's value.
        model[p] = (edges, [r - r.mean() for r in filled])
    return model


def spread_summary(model):
    """Size, spread and skew per bucket; the preregistration quotes it."""
    out = []
    for p, (edges, pools) in model.items():
        for k, r in enumerate(pools):
            sd = r.std()
            out.append({"pos": p, "bucket": k, "n": len(r), "sd": sd,
                        "skew": ((r - r.mean()) ** 3).mean() / sd ** 3,
                        "p10": np.quantile(r, .1), "p50": np.quantile(r, .5),
                        "p90": np.quantile(r, .9)})
    return pd.DataFrame(out)


def draw(model, pos, value, quest, p_q, rng):
    """DRAWS scenario scores for one player. A questionable player's harness value is
    his value times the chance he plays, so he is drawn as that mixture: he plays with
    probability p_q and then scores like a player of the undiscounted value."""
    v = value / p_q if quest else value
    edges, pools = model[pos]
    r = pools[int(np.digitize(v, edges))]
    s = v + r[rng.integers(0, len(r), DRAWS)]
    if quest:
        s = s * (rng.random(DRAWS) < p_q)
    return s


# ---------------------------------------------------------------- lineups

def consensus_lineup(S, roster, w):
    """The players week_points starts on consensus values, in the same order of choice."""
    r = np.array(roster)
    pos = S.pos[r]
    v = np.where(S.elig[r, w], S.ecr_val[r, w], -np.inf)
    used = np.zeros(len(r), bool)
    for p, k in SLOTS.items():
        cand = np.where((pos == p) & ~used & np.isfinite(v))[0]
        used[cand[np.argsort(-v[cand])][:k]] = True
    cand = np.where(np.isin(pos, FLEX) & ~used & np.isfinite(v))[0]
    if len(cand):
        used[cand[np.argmax(v[cand])]] = True
    return tuple(sorted(r[used]))


def candidates(S, model, roster, w, quest):
    """Every legal lineup of eligible players, after dropping players who are dominated
    within their own spread bucket. Slots a roster can't fill stay empty, as in the
    harness; the flex takes any eligible RB, WR or TE left over."""
    by = {p: [] for p in POS}
    groups = {}
    for i in sorted(roster, key=lambda i: -S.ecr_val[i, w]):
        if not S.elig[i, w] or S.pos[i] not in by:
            continue
        p = S.pos[i]
        q = bool(quest[i, w])
        v = S.ecr_val[i, w] / S.p_q if q else S.ecr_val[i, w]
        key = (p, int(np.digitize(v, model[p][0])), q)
        groups[key] = groups.get(key, 0) + 1
        if groups[key] <= MAX_START[p]:
            by[p].append(i)
    out = set()
    for picks in product(*(combinations(by[p], min(k, len(by[p])))
                           for p, k in SLOTS.items())):
        chosen = set(i for c in picks for i in c)
        rest = [i for p in FLEX for i in by[p] if i not in chosen]
        if not rest:
            out.add(tuple(sorted(chosen)))
        for f in rest:
            out.add(tuple(sorted(chosen | {f})))
    return out


def winprob_week(S, model, rosters, seat, w, quest, rng):
    """The lineup with the most expected opponents outscored in week w.

    Returns (lineup, expected count for the consensus lineup, for the chosen lineup).
    Consensus stands unless the best lineup beats it by more than MARGIN_SE standard
    errors.
    Players are drawn independently; every lineup is scored on the same draws, so the
    comparison between two lineups carries far less noise than either estimate.
    """
    cache = {}

    def scen(i):
        if i not in cache:
            cache[i] = draw(model, S.pos[i], S.ecr_val[i, w], quest[i, w], S.p_q, rng)
        return cache[i]

    opp = []
    for t in range(TEAMS):
        if t == seat:
            continue
        lu = consensus_lineup(S, rosters[t], w)
        opp.append(np.sum([scen(i) for i in lu], axis=0) if lu else np.zeros(DRAWS))
    # sum_j P(mine > opponent j) at a score s is the share of all opponents' pooled
    # draws below s, times the number of draws; strict, as all_play is.
    pooled = np.sort(np.concatenate(opp))

    cons = consensus_lineup(S, rosters[seat], w)
    cands = [cons] + sorted(candidates(S, model, rosters[seat], w, quest) - {cons})
    players = sorted({i for c in cands for i in c})
    col = {i: k for k, i in enumerate(players)}
    D = np.array([scen(i) for i in players], dtype=np.float32) if players else \
        np.zeros((0, DRAWS), np.float32)
    A = np.zeros((len(cands), len(players)), np.float32)
    for k, c in enumerate(cands):
        A[k, [col[i] for i in c]] = 1.0
    mine = A @ D
    below = np.searchsorted(pooled.astype(np.float32), mine.ravel(), side="left")
    # Per my draw, the expected number of opponents that score beats.
    per = below.reshape(mine.shape) / DRAWS
    exp = per.mean(1)
    best = int(np.argmax(exp))             # the first maximum, so a tie keeps consensus
    # The paired standard error of the gain over consensus, from my draws. It leaves out
    # the opponents' simulation noise, which both lineups share.
    se = (per[best] - per[0]).std() / np.sqrt(DRAWS)
    if best and exp[best] - exp[0] <= MARGIN_SE * se:
        best = 0
    return cands[best], exp[0], exp[best]


def questionable(S, y):
    """The harness's pregame questionable flags, player x week."""
    ix = {p: i for i, p in enumerate(S.ids)}
    inj = pd.read_parquet(f"data/injuries_{y}.parquet")
    q = np.zeros((len(S.ids), WEEKS), bool)
    for p, wk, s in zip(inj.gsis_id, inj.week, inj.report_status):
        if s == "Questionable" and p in ix and wk <= WEEKS:
            q[ix[p], wk - 1] = True
    return q


# ---------------------------------------------------------------- the experiment

def run(check=False):
    curves = LB.rank_curve()
    # Season() needs weekly projections only for the model-lineup arms, which aren't
    # run here; with none it values those players at their preseason rate, unused.
    no_proj = pd.DataFrame(columns=["player_id", "season", "week", "proj"])
    rows, mech = [], []
    for y in SEASONS:
        S = LB.Season(y, no_proj, curves)
        S.p_q = W.play_probs(y).get("Questionable", 0.57)      # the harness's discount
        quest = questionable(S, y)
        model = fit_spread(y, curves)
        for lg in range(LEAGUES):
            # The same draws in the same order as the main harness, so the drafts and
            # the control's scores match its exact-consensus arm.
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
                rosters = LB.draft(S, orders)
                base = np.array([LB.lineup_points(S, r, S.ecr_val) for r in rosters])
                test = base.copy()
                proj_cons, proj_best, differ = [], [], []
                for w in range(REG):
                    # Its own stream per decision, apart from the harness's draws, and
                    # the same in both opponent designs.
                    mc = np.random.default_rng([y, lg, seat, w, 7])
                    t0 = time.perf_counter()
                    lu, e_cons, e_best = winprob_week(S, model, rosters, seat, w, quest, mc)
                    cons = consensus_lineup(S, rosters[seat], w)
                    if lu != cons:
                        test[seat, w] = S.actual[list(lu), w].sum()
                    proj_cons.append(e_cons)
                    proj_best.append(e_best)
                    differ.append(lu != cons)
                    if check:
                        mech.append(legality(S, rosters[seat], lu, w)
                                    | {"sec": time.perf_counter() - t0, "differ": lu != cons,
                                       "season": y})
                for arm, sc in (("cons", base), ("winprob", test)):
                    res = [LB.season_outcome(sc, s) for s in scheds]
                    others = np.delete(sc[:, :REG], seat, 0)
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": LB.all_play(sc, seat),
                        "pts_reg": sc[seat, :REG].sum(),
                        "pts_playoff": sc[seat, REG:].sum(),
                        # per week 1-14: opponents outscored, and what the model expected
                        "beat": (sc[seat, :REG] > others).sum(0).tolist(),
                        "proj_cons": proj_cons, "proj_best": proj_best,
                        "differ": differ,
                    })
            print(f"{y} league {lg + 1}/{LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(mech)


def legality(S, roster, lu, w):
    """Mechanics: a started lineup is on the roster, eligible, unique, and fills the
    slots the way the harness would."""
    lu = list(lu)
    pos = list(S.pos[lu])
    n = {p: pos.count(p) for p in POS}
    elig = [i for i in roster if S.elig[i, w]]
    have = {p: sum(S.pos[i] == p for i in elig) for p in POS}
    need = {p: min(k, have[p]) for p, k in SLOTS.items()}
    flex_left = sum(have[p] - need[p] for p in FLEX)
    size = sum(need.values()) + (1 if flex_left > 0 else 0)
    ok = (set(lu) <= set(roster) and len(set(lu)) == len(lu)
          and all(S.elig[i, w] for i in lu) and len(lu) == size
          and all(n[p] >= need[p] for p in POS)
          and n["QB"] == need["QB"]
          and sum(n[p] - need[p] for p in FLEX) == size - sum(need.values()))
    return {"legal": ok}


def report(d):
    print("\n=== win-probability lineups vs consensus lineups ===")
    print("12-team PPR leagues on 2021-25, exact-consensus draft in both arms; weeks 1-14 "
          "the test seat starts\nthe lineup with the most expected opponents outscored, "
          "weeks 15-17 consensus lineups\n")
    print(f"{'arm':8s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:8s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f}")

    print("\npreregistered test (prereg_winprob.md), season-cluster bootstrap 90% interval:")
    keep = LB.paired(d, "winprob", "cons", "winprob minus consensus")
    key = ["season", "league", "seat"]
    a = d[d.arm == "winprob"].set_index(key)
    b = d[d.arm == "cons"].set_index(key).loc[a.index]
    by = (a.title - b.title).groupby("season").mean()
    print("  title by season " + " ".join(f"{100 * v:+.1f}" for v in by))
    print(f"  -> {'passes this design' if keep else 'fails this design'}")

    # Diagnostics, not gated: how often it acts, and where the change comes from.
    wk = pd.DataFrame({
        "season": np.repeat(a.index.get_level_values("season"), REG),
        "change": (np.concatenate(a.beat.values) - np.concatenate(b.beat.values)) / (TEAMS - 1),
        "differ": np.concatenate(a.differ.values),
        "proj_cons": np.concatenate(a.proj_cons.values),
        "gain": np.concatenate(a.proj_best.values) - np.concatenate(a.proj_cons.values),
    })
    wk["underdog"] = wk.proj_cons < (TEAMS - 1) / 2
    print("\ndiagnostics (seat-weeks 1-14):")
    print("  lineup differs from consensus: "
          f"{100 * wk.differ.mean():.1f}% of weeks; by season "
          + " ".join(f"{100 * v:.1f}" for v in wk.groupby("season").differ.mean()))
    dw = wk[wk.differ]
    print(f"  in those weeks: model-expected gain {dw.gain.mean():+.3f} opponents, "
          f"realized {dw.change.mean() * (TEAMS - 1):+.3f}")
    for lab, g in (("underdog", wk[wk.underdog]), ("favourite", wk[~wk.underdog])):
        print(f"  {lab:9s} ({100 * len(g) / len(wk):.0f}% of weeks, differs "
              f"{100 * g.differ.mean():.1f}%): paired all-play change "
              f"{100 * g.change.mean():+.2f} pp; by season "
              + " ".join(f"{100 * v:+.2f}" for v in g.groupby("season").change.mean()))
    return keep


if __name__ == "__main__":
    if "--noise" in sys.argv:
        NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = LEAGUES != LB.LEAGUES
    d, mech = run(check)
    if check:
        # A reduced run writes its own file and prints mechanics only, never outcomes.
        d.to_parquet(f"data/league_winprob_check{LEAGUES}_noise{NOISE:g}.parquet")
        print(f"\nmechanics, {LEAGUES} leagues per season, noise {NOISE:g}:")
        print(f"  lineups legal: {mech.legal.mean():.3f} of {len(mech)} seat-weeks")
        print(f"  seconds per decision: mean {mech.sec.mean():.3f}, max {mech.sec.max():.3f}")
        print("  lineup differs from consensus: "
              + " ".join(f"{y} {100 * g.differ.mean():.1f}%" for y, g in mech.groupby("season")))
        ref = pd.read_parquet("data/league_backtest.parquet" if NOISE == 1 else
                              f"data/league_backtest_noise{NOISE:g}.parquet")
        ref = ref[(ref.draft == "exact") & (ref.lineup == "ecr")].set_index(
            ["season", "league", "seat"])
        mine = d[d.arm == "cons"].set_index(["season", "league", "seat"])
        cols = ["all_play", "pts_reg", "pts_playoff", "title"]
        same = (mine[cols] == ref.loc[mine.index, cols]).all().all()
        print(f"  control reproduces the harness's exact-consensus arm exactly: {same}")
        sys.exit()
    d.to_parquet(f"data/league_winprob_noise{NOISE:g}.parquet")
    print(f"\nopponent noise: {NOISE:g} x ECR sd")
    report(d)
