"""FAAB (blind-bid) waivers: does bidding just enough beat bidding like everyone else?

Hole-aware streaming is the one waiver policy that beat consensus in this harness, but
the harness runs waivers in worst-record-first priority, and many leagues run a budget
instead. Under FAAB the decision is not only whom to claim but what to pay, and real
bids are skewed: a few managers pay a large share of their budget for players nobody
else wanted (see data/faab_bids.parquet, from real Sleeper leagues). A manager who
knows who else is likely to want a player can bid the least that probably wins and keep
the rest for later weeks. This tests that bid, holding the claim itself fixed.

Spec frozen in draft/prereg_faab.md before the real run:

  - Mechanism: every team has $100 for the season. Each waiver round every team claims
    the player its own policy would add now; the highest bid wins (ties to waiver
    priority, worst all-play first), pays its bid, and is done for the week. Losers
    re-claim on what is left. With every bid equal this is exactly the harness's
    priority wire, which the mechanics check confirms.
  - Opponents: the consensus wire's target, and a bid drawn from the real Sleeper claims
    for that week group and that player's rest-of-season positional rank bucket, capped
    at the budget left. A team's bid on a player in a week is drawn once.
  - Test seat, both arms: the streaming policy's target. Control (ctrl) draws its bid
    as the opponents do. Test (faab) bids the least whole dollar that wins with
    probability P_STAR against the opponents currently claiming the same player, under
    a pacing cap.

    .venv/bin/python draft/sim_faab.py --noise 1
    .venv/bin/python draft/sim_faab.py --noise 0
    .venv/bin/python draft/sim_faab.py --noise 1 --leagues 2    # mechanics check only
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB

POS = LB.POS
TEAMS, REG, WEEKS = LB.TEAMS, LB.REG, LB.WEEKS
SEASONS = LB.SEASONS
LEAGUES, SCHEDULES = LB.LEAGUES, LB.SCHEDULES
NOISE = 1.0
FIRST = LB.FIRST_WAIVER
BUDGET = 100
BID_SEASONS = (2023, 2024, 2025)
P_STAR = 0.75
# Pacing: never bid more than PACE / (decisions left, this one included) of what is left,
# so a $100 budget can't be spent in the first month. Decisions run before weeks 2-17.
PACE = 3.0
# Bid cells: week group x rank bucket. Upper edges, inclusive.
WEEK_GROUPS = (4, 8, 12, 17)
RANK_EDGES = (12, 24, 36, 60)          # positional rest-of-season rank; beyond is the tail
MIN_CELL = 200                         # a thinner cell pools with its whole week group


# ---------------------------------------------------------------- the bid model

def cell(w, rank):
    return (int(np.searchsorted(WEEK_GROUPS, w)), int(np.searchsorted(RANK_EDGES, rank)))


def fit_bids(seasons):
    """Distribution of a single manager's bid, in whole dollars of a $100 budget, per cell.

    Every submitted claim counts, winning or failed, because a simulated opponent is one
    bidder: drawing from winning bids alone would hand each of them the top of several
    real bidders' bids. Bids are rescaled from each league's own budget.
    """
    d = pd.read_parquet("data/faab_bids.parquet")
    d = d[d.season.isin(seasons) & d.week.between(FIRST, WEEKS)]
    dollars = np.clip(np.floor(d.bid_pct.values + 0.5), 0, BUDGET).astype(int)
    keys = [cell(w, r) for w, r in zip(d.week, d["rank"])]
    d = d.assign(dollars=dollars, cell=keys, wg=[k[0] for k in keys])
    out = {}
    for wg in range(len(WEEK_GROUPS)):
        group = d[d.wg == wg].dollars.values
        for rb in range(len(RANK_EDGES) + 1):
            x = d[d.cell == (wg, rb)].dollars.values
            if len(x) < MIN_CELL:
                x = group
            pmf = np.bincount(x, minlength=BUDGET + 1) / len(x)
            out[(wg, rb)] = (pmf, np.cumsum(pmf))
    return out


def ros_rank(S, y):
    """Positional rest-of-season rank per player at each decision, by the same rule
    waiver_values prices with: the latest scrape strictly before week w, preseason ranks
    before the first, one past the last ranked player for anyone unranked."""
    n = len(S.ids)
    ix = {p: i for i, p in enumerate(S.ids)}
    ros = pd.read_parquet("data/ecr_ros.parquet")
    ros = ros[ros.season == y]
    pre = pd.read_parquet("data/ecr_preseason.parquet")
    pre = pre[pre.season == y]
    rank = np.full((n, WEEKS), np.nan)
    for w in range(FIRST, WEEKS + 1):
        earlier = ros[ros.week < w]
        src = earlier[earlier.week == earlier.week.max()] if len(earlier) else pre
        src = src.assign(rank=src.groupby("pos").ecr.rank(method="first"))
        worst = src.groupby("pos")["rank"].max().to_dict()
        c = np.full(n, np.nan)
        for p, r in zip(src.player_id, src["rank"]):
            if p in ix:
                c[ix[p]] = r
        for p in POS:
            at = S.pos == p
            c[at & np.isnan(c)] = worst.get(p, 100) + 1
        rank[:, w - 1] = c
    return rank


class Auction:
    """Budgets and bids for one simulated season.

    `style` is the test seat's bidding: "draw" (as the opponents), "pstar" (the least
    bid that wins with probability P_STAR) or "zero" (every team bids $0, the mechanics
    check that must reproduce the priority wire).
    """

    def __init__(self, key, seat, style, rank, env, est):
        self.key, self.seat, self.style = key, seat, style
        self.rank, self.env, self.est = rank, env, est
        self.budget = np.full(TEAMS, BUDGET)
        self.draws = {}
        self.seat_bids = {}

    def draw(self, t, p, w):
        """A team's bid on a player in a week: one uniform per (league, team, week,
        player), so both arms of a seat see the same opponent bids for the same claim."""
        k = (t, p, w)
        if k not in self.draws:
            u = np.random.default_rng([*self.key, t, w, p]).random()
            _, cdf = self.env[cell(w, self.rank[p, w - 1])]
            self.draws[k] = int(min(np.searchsorted(cdf, u, side="right"), BUDGET))
        return min(self.draws[k], int(self.budget[t]))

    def cap(self, w):
        left = WEEKS + 1 - w
        return int(np.floor(self.budget[self.seat] * min(1.0, PACE / left)))

    def pstar(self, p, w, rivals, prio):
        """Least whole-dollar bid with win probability >= P_STAR against the rivals, each
        bidding from the estimated distribution capped at its known remaining budget.
        A tie goes to whoever is ahead in priority. If no bid under the cap reaches
        P_STAR, bid the cap."""
        cap = self.cap(w)
        if not rivals:
            return 0
        pmf, _ = self.est[cell(w, self.rank[p, w - 1])]
        b = np.arange(BUDGET + 1)
        win = np.ones(BUDGET + 1)
        for r in rivals:
            B = int(self.budget[r])
            q = pmf[:B + 1].copy()
            q[B] += pmf[B + 1:].sum()                   # bids above his budget become B
            below = np.concatenate([[0.0], np.cumsum(q)])[np.minimum(b, B + 1)]
            tie = np.where(b <= B, q[np.minimum(b, B)], 0.0)
            win *= below + (tie if prio[self.seat] < prio[r] else 0.0)
        ok = np.where((win >= P_STAR) & (b <= cap))[0]
        return int(ok[0]) if len(ok) else cap

    def bid(self, t, p, w, claims, prio):
        if self.style == "zero":
            return 0
        if t != self.seat or self.style == "draw":
            return self.draw(t, p, w)
        k = (p, w)
        if k not in self.seat_bids:
            # Fixed at the seat's first claim on him this week, like a sealed bid.
            rivals = [r for r, m in claims.items() if r != t and m and m[0] == p]
            self.seat_bids[k] = self.pstar(p, w, rivals, prio)
        return min(self.seat_bids[k], int(self.budget[t]))


# ---------------------------------------------------------------- the wire

def faab_week(S, rosters, values, order, w, movers, auction, diag):
    """One blind-bid waiver run before week w.

    Each round every team still waiting claims what its policy adds from the current
    pool. The best (bid, priority) claim wins and that team is done. A team with nothing
    to claim enters at $0, so once no positive claims remain the rest go in priority
    order, each on the pool as it stands at its turn, which is the priority wire.
    """
    on_roster = np.zeros(len(S.ids), bool)
    for r in rosters:
        on_roster[r] = True
    prio = {t: i for i, t in enumerate(order)}
    pending = list(order)
    moves = np.zeros(TEAMS, int)
    seat = auction.seat
    first = True
    while pending:
        claims = {}
        for t in pending:
            val = values[t][:, w - 1]
            free = np.where(~on_roster & np.isfinite(val))[0]
            claims[t] = (movers.get(t, LB.consensus_move)(S, np.array(rosters[t]), val, free, w)
                         if len(free) else None)
        bids = {t: auction.bid(t, claims[t][0], w, claims, prio) if claims[t] else 0
                for t in pending}
        if first and seat in claims:
            diag["intended"] = claims[seat][0] if claims[seat] else None
            first = False
        win = max(pending, key=lambda t: (bids[t], -prio[t]))
        pending.remove(win)
        move = claims[win]
        if move is None:
            continue
        add, drop = move
        rivals = [r for r in pending if claims[r] and claims[r][0] == add]
        if win == seat:
            price = max([bids[r] + (prio[r] < prio[seat]) for r in rivals], default=0)
            diag.update(added=add, paid=bids[win], overpay=bids[win] - price)
        diag["wins"].append((w, auction.rank[add, w - 1], bids[win], len(rivals) + 1))
        auction.budget[win] -= bids[win]
        rosters[win] = [i for i in rosters[win] if i != drop] + [int(add)]
        on_roster[add], on_roster[drop] = True, False
        moves[win] += 1
    return moves


def simulate(S, drafted, values, movers, auction):
    """LB.simulate with the blind-bid wire in place of the priority wire."""
    rosters = [list(r) for r in drafted]
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    log = {"spend": np.zeros(WEEKS), "intended": 0, "won_intended": 0, "claims_won": 0,
           "overpay": 0.0, "wins": []}
    for w in range(WEEKS):
        for t in range(TEAMS):
            scores[t, w] = LB.week_points(S, rosters[t], S.ecr_val, w)
        if w + 1 < WEEKS:
            order = LB.priority(scores, w)
            diag = {"intended": None, "added": None, "paid": 0, "overpay": 0, "wins": log["wins"]}
            moves += faab_week(S, rosters, values, order, w + 2, movers, auction, diag)
            log["spend"][w + 1] = diag["paid"]
            if diag["intended"] is not None:
                log["intended"] += 1
                log["won_intended"] += diag["added"] == diag["intended"]
            if diag["added"] is not None:
                log["claims_won"] += 1
                log["overpay"] += diag["overpay"]
    return scores, moves, rosters, log


def legal(S, rosters, auction):
    """Fourteen players each, nobody on two rosters, starters never cut, budgets >= 0."""
    flat = [i for r in rosters for i in r]
    ok = len(flat) == len(set(flat)) and all(len(r) == LB.ROUNDS for r in rosters)
    for r in rosters:
        c = {p: int((S.pos[r] == p).sum()) for p in POS}
        ok &= all(c[p] >= k for p, k in LB.SLOTS.items())
    return bool(ok and (auction.budget >= 0).all())


# ---------------------------------------------------------------- the experiment

def run(check=False):
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    env = fit_bids(BID_SEASONS)
    rows, mech = [], []
    for y in SEASONS:
        S = LB.Season(y, wk_proj, curves)
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        rank = ros_rank(S, y)
        # The test seat's estimate leaves out the season being scored.
        est = fit_bids(tuple(s for s in BID_SEASONS if s != y))
        every = set(range(TEAMS))
        for lg in range(LEAGUES):
            # The same draws in the same order as run_streaming, so drafts match.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for k in range(TEAMS + 1):
                score = S.ecr_mean + NOISE * S.ecr_sd * noise[k]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = LB.schedules(rng, SCHEDULES)
            base = [cons_val] * TEAMS
            for seat in range(TEAMS):
                orders = list(ecr_orders[:TEAMS])
                orders[seat] = S.exact_order
                drafted = LB.draft(S, orders)
                if check:
                    # With every bid $0 the blind-bid wire must be the priority wire.
                    scratch = []
                    ref, _, _ = LB.simulate(S, drafted, base, every,
                                            {seat: LB.streaming_policy(S, gone, scratch)})
                    a = Auction((y, lg), seat, "zero", rank, env, est)
                    zero, _, _, _ = simulate(S, drafted, base,
                                             {seat: LB.streaming_policy(S, gone, [])}, a)
                    mech.append({"season": y, "league": lg, "seat": seat, "arm": "zero",
                                 "same_as_priority": bool((ref == zero).all())})
                for arm, style in (("ctrl", "draw"), ("faab", "pstar")):
                    t0 = time.time()
                    a = Auction((y, lg), seat, style, rank, env, est)
                    # The streaming log is not wanted here; each call gets a scratch list.
                    movers = {seat: LB.streaming_policy(S, gone, [])}
                    sc, moves, rosters, log = simulate(S, drafted, base, movers, a)
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
                        "spend": BUDGET - a.budget[seat],
                        "spend_by_week": log["spend"].tolist(),
                        "intended": log["intended"], "won_intended": log["won_intended"],
                        "claims_won": log["claims_won"], "overpay": log["overpay"],
                        "opp_spend": float(np.mean(np.delete(BUDGET - a.budget, seat))),
                    })
                    if check:
                        mech.append({"season": y, "league": lg, "seat": seat, "arm": arm,
                                     "legal": legal(S, rosters, a), "sec": time.time() - t0,
                                     "moves": int(moves.sum()), "wins": log["wins"]})
            print(f"{y} league {lg + 1}/{LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(mech)


def report(d):
    print("\n=== FAAB waivers: least bid that wins with p* vs bidding like the league ===")
    print(f"12-team PPR leagues on 2021-25, exact-consensus draft, consensus lineups, "
          f"${BUDGET} FAAB;\nthe test seat claims by streaming in both arms, "
          f"p* = {P_STAR}, pacing {PACE:g} / decisions left\n")
    print(f"{'arm':6s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s} {'adds':>6s} {'spend':>6s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:6s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f} {g.moves.mean():6.2f} "
              f"{g.spend.mean():6.1f}")
    print("\npreregistered test (prereg_faab.md), season-cluster bootstrap 90% interval:")
    keep = LB.paired(d, "faab", "ctrl", "faab minus ctrl")
    print(f"  -> {'passes this design' if keep else 'fails this design'}")
    diagnostics(d)
    return keep


def diagnostics(d):
    print("\nreported, not gated (test seat, per league-seat season):")
    for arm, g in d.groupby("arm", sort=False):
        sp = np.array(g.spend_by_week.tolist())
        cum = sp.cumsum(1).mean(0)
        print(f"  {arm}: cumulative $ spent before weeks 2..17: "
              + " ".join(f"{v:.0f}" for v in cum[1:]))
        print(f"  {arm}: intended claims {g.intended.mean():.1f}, won "
              f"{100 * g.won_intended.sum() / max(g.intended.sum(), 1):.0f}%, claims won "
              f"{g.claims_won.mean():.1f}, average overpay "
              f"${g.overpay.sum() / max(g.claims_won.sum(), 1):.1f} per claim won, "
              f"opponents' mean spend ${g.opp_spend.mean():.0f}")
        by = g.groupby("season").apply(
            lambda x: 100 * x.won_intended.sum() / max(x.intended.sum(), 1))
        print(f"  {arm}: intended-claim win rate by season "
              + " ".join(f"{v:.0f}%" for v in by))


if __name__ == "__main__":
    if "--noise" in sys.argv:
        NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = LEAGUES != LB.LEAGUES
    d, mech = run(check)
    if check:
        # A reduced run writes its own file and prints mechanics only, never outcomes.
        d.to_parquet(f"data/league_faab_check{LEAGUES}_noise{NOISE:g}.parquet")
        mech.to_parquet(f"data/league_faab_mech_check{LEAGUES}_noise{NOISE:g}.parquet")
        z = mech[mech.arm == "zero"]
        m = mech[mech.arm != "zero"]
        print(f"\nmechanics, {LEAGUES} leagues per season, noise {NOISE:g}:")
        print(f"  $0 bids reproduce the priority streaming wire exactly: "
              f"{z.same_as_priority.all()} ({len(z)} seat-seasons)")
        print(f"  rosters and budgets legal: {m.legal.all()}; seconds per season sim "
              f"{m.sec.mean():.2f}; league adds per season {m.moves.mean():.0f}")
        w = pd.DataFrame([x for ws in m.wins for x in ws],
                         columns=["week", "rank", "bid", "bidders"])
        w["rb"] = np.searchsorted(RANK_EDGES, w["rank"])
        print("  simulated winning bids by rank bucket (median / mean $, bidders):")
        for rb, g in w.groupby("rb"):
            print(f"    bucket {rb}: n {len(g)}  {g.bid.median():.0f} / {g.bid.mean():.1f}  "
                  f"bidders {g.bidders.mean():.2f}")
        diagnostics(d)
        sys.exit()
    d.to_parquet(f"data/league_faab_noise{NOISE:g}.parquet")
    print(f"\nopponent noise: {NOISE:g} x ECR sd")
    report(d)
