"""League backtest: do these tools win fantasy leagues against managers who follow
consensus?

Projection accuracy isn't the goal; beating eleven other managers is. They draft and
set lineups off expert consensus (FantasyPros ECR), so the tools earn something only
where they disagree with consensus and are right. This measures exactly that. Spec
agreed with Codex (gpt-5.6-luna) after two rounds of debate:

  - 12-team leagues on each holdout season 2021-25, 14 rounds, PPR, 1QB/2RB/2WR/1TE/1FLEX.
  - Eleven consensus seats draft from the preseason overall PPR ECR, each perturbed by
    the experts' own spread (ECR sd), and start lineups by weekly positional ECR.
  - One test seat; 2x2 ablation of its decision policies: draft by our board (VBD on
    the walk-forward projection, fit only on earlier seasons) or by consensus, and set
    lineups by our weekly projections or by consensus. Everything else is identical.
    A hindsight-lineup arm is reported as a diagnostic upper bound only.
  - Every input is as of the decision: preseason boards from earlier seasons, weekly
    projections from earlier weeks, consensus ranks scraped strictly before the week's
    Sunday slate. Eligibility is known pregame: on the active roster (ACT/INA), team
    not on bye, not Out/Doubtful on the official report. A started player who then
    doesn't play scores zero, for every policy.
  - Scored on what players actually did. Weeks 1-14 head-to-head on random schedules,
    top six to a weeks 15-17 playoff (seeds 1-2 bye).

Objective and headline: the paired change in championship probability, test seat vs a
consensus seat in the same league. Decision statistic, declared before running: a
policy is kept if its pooled paired change in all-play win rate (schedule-free, far
less noisy) is positive and positive in at least 4 of 5 seasons, and its pooled change
in title probability is no worse than -1.0 percentage point. Seasons are the
independent unit (the player outcomes are fixed); schedule draws only average out
schedule luck.

Label: draft + start/sit edge against synthetic consensus - no waivers, no trades.

    .venv/bin/python draft/league_backtest.py
"""
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import availability as A
import roster as R
import settings as CFG
from vbd import LEAGUE, add_vbd
from draft_dp import BENCH_VALUE, snake_picks
import board as B
import consensus as C
import stack as K
import usage as U
import weekly as W

# Every league rule and modelling assumption comes from settings.py, which defaults to
# exactly the league these results were published on. Nothing below is a second copy.
SET = CFG.get()
POS = list(CFG.POSITIONS)
SLOTS = LEAGUE["starters"]
FLEX = list(LEAGUE["flex"])
TEAMS, ROUNDS = LEAGUE["teams"], LEAGUE["rounds"]
SEASONS = range(SET.seasons[0], SET.seasons[-1] + 1)
REG, WEEKS = SET.reg_weeks, SET.weeks
LEAGUES, SCHEDULES = SET.leagues, SET.schedules
# Roster rules for every seat alike, the way draft simulators keep bots sane: at most
# two QBs and two TEs (no second of either before round 9), at most seven RBs or WRs,
# and once the picks left only just cover the empty starting slots, draft for them.
CAP, SECOND_AFTER = SET.caps, SET.second_after
ONE_EACH = SET.one_each                      # positions where a second is held back
ACTIVE = set(SET.active_statuses)            # on the 53; INA is the gameday inactive list
OUT = set(SET.out_statuses)
TITLE_GUARD = -0.010
# Opponents' deviation from consensus, in units of the experts' own spread. 1 is the
# main design; 0 is the sensitivity check where every opponent follows consensus exactly.
NOISE = SET.noise
# prereg_adp.md: the draft-market arms, scored with consensus lineups only, and the most
# consensus-implied season value adp_gap will give up to take a player the market won't
# leave it. Declared before the run, not tuned.
ADP_ARMS = ("adp", "adp_gap")
ADP_DELTA = 10.0


# ---------------------------------------------------------------- inputs per season

def rank_curve():
    """Consensus rank -> expected points, per position, from 2020 (before the test
    seasons): the mean actual score at each weekly rank, smoothed and non-increasing.
    It's how a consensus manager compares a WR2 with an RB3 for the flex."""
    w = pd.read_parquet("data/ecr_weekly.parquet")
    w = w[w.season == 2020]
    st = pd.read_parquet("data/stats/w2020.parquet")
    st = st[st.season_type == "REG"][["player_id", "week", "fantasy_points_ppr"]]
    m = w.merge(st, on=["player_id", "week"], how="left").fillna({"fantasy_points_ppr": 0})
    m["rank"] = m.groupby(["week", "pos"]).ecr.rank(method="first")
    curves = {}
    for p, g in m.groupby("pos"):
        by = g.groupby("rank").fantasy_points_ppr.mean().reindex(range(1, 121))
        sm = by.rolling(5, center=True, min_periods=1).mean().ffill()
        curves[p] = np.minimum.accumulate(sm.values)        # rank 1 -> index 0
    return curves


def curve_points(curves, pos, rank):
    c = curves[pos]
    return c[np.clip(np.asarray(rank, int) - 1, 0, len(c) - 1)]


def weekly_projections():
    """Our weekly model's pregame projections for 2021-25 (trained on <= 2020)."""
    d, s, rfits, sch = W.history(2026)
    d = d[d.season <= 2025].reset_index(drop=True)
    d, *_, fits, sd = W.train(d, (d.season <= 2020).values)
    d["proj"], _ = W.predict(fits, sd, d)
    return d[d.season > 2020][["player_id", "season", "week", "proj"]]


class Season:
    """Everything a season needs, as player x week arrays (index 0 = week 1)."""

    def __init__(self, y, wk_proj, curves):
        self.y = y
        board = B.project_upcoming(y, rookies=True)
        board = board[(board.own_w > 4) | ((board.exp == 0) & board.role.isin(["d1", "d2"]))]
        raw = board.drop_duplicates("player_id").set_index("player_id")
        board, _, _ = add_vbd(raw)
        ecr = pd.read_parquet("data/ecr_preseason_overall.parquet")
        ecr = ecr[ecr.season == y].set_index("player_id")

        ids = sorted(set(board.index) | set(ecr.index))
        self.ids = ids
        ix = {p: i for i, p in enumerate(ids)}
        players = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
        pos = pd.Series(board.position).combine_first(ecr.pos).reindex(ids)
        pos = pos.fillna(players.position.reindex(ids))
        self.pos = pos.values
        self.name = players.display_name.reindex(ids).fillna(pd.Series(ecr.player)).values
        n = len(ids)

        # Draft preferences: our board by VBD; consensus by ECR and its spread.
        self.model_order = [ix[p] for p in board.index]
        self.ecr_mean = ecr.ecr.reindex(ids).values
        spread = ecr.sd.reindex(ids)
        self.ecr_sd = spread.fillna(spread.median()).values

        # Actual points; 0 where he didn't play.
        st = pd.read_parquet(f"data/stats/w{y}.parquet")
        st = st[(st.season_type == "REG") & (st.week <= WEEKS) & st.player_id.isin(ix)]
        self.actual = np.zeros((n, WEEKS))
        for r in st.itertuples():
            self.actual[ix[r.player_id], r.week - 1] += r.fantasy_points_ppr

        # Pregame eligibility.
        ro = pd.read_parquet(f"data/roster_weekly_{y}.parquet")
        ro = ro[(ro.game_type == "REG") & (ro.week <= WEEKS) & ro.gsis_id.isin(ix)]
        g = pd.read_csv("data/games.csv")
        g = g[(g.season == y) & (g.game_type == "REG")]
        plays = {(w, t) for w, a, h in zip(g.week, g.away_team, g.home_team) for t in (a, h)}
        inj = pd.read_parquet(f"data/injuries_{y}.parquet")
        out = {(p, w) for p, w, s in zip(inj.gsis_id, inj.week, inj.report_status) if s in OUT}
        self.elig = np.zeros((n, WEEKS), bool)
        ro = ro.assign(team=W.norm_team(ro.team))
        for p, w, t, s in zip(ro.gsis_id, ro.week, ro.team, ro.status):
            if s in ACTIVE and (w, t) in plays and (p, w) not in out:
                self.elig[ix[p], w - 1] = True
        q = {(p, w) for p, w, s in zip(inj.gsis_id, inj.week, inj.report_status)
             if s == "Questionable"}
        p_q = W.play_probs(y).get("Questionable", 0.57)

        # Our lineup values: the weekly projection, carried forward through weeks he has
        # no box score, the preseason rate before his first; discounted if questionable.
        wp = wk_proj[wk_proj.season == y]
        self.model_val = np.full((n, WEEKS), np.nan)
        for r in wp.itertuples():
            if r.player_id in ix and r.week <= WEEKS:
                self.model_val[ix[r.player_id], r.week - 1] = r.proj
        pre = board.proj_ppg.reindex(ids).fillna(0).values
        mv = pd.DataFrame(self.model_val).ffill(axis=1)
        self.model_val = mv.fillna(pd.Series(pre)).T.fillna(pd.Series(pre)).T.values
        self.model_val = np.where(np.isnan(self.model_val), pre[:, None], self.model_val)

        # Consensus lineup values: weekly positional rank, carried forward to weeks with
        # no scrape (week 1 from the preseason positional ranks); unranked players rank
        # one past the last ranked player at the position. Then rank -> points.
        wk = pd.read_parquet("data/ecr_weekly.parquet")
        wk = wk[(wk.season == y) & wk.player_id.isin(ix)]
        pre_pos = pd.read_parquet("data/ecr_preseason.parquet")
        pre_pos = pre_pos[(pre_pos.season == y) & pre_pos.player_id.isin(ix)]
        rank = np.full((n, WEEKS), np.nan)
        pr = pre_pos.assign(r=pre_pos.groupby("pos").ecr.rank(method="first"))
        last = {ix[p]: r for p, r in zip(pr.player_id, pr.r)}
        worst = pr.groupby("pos").r.max().to_dict()
        for w in range(1, WEEKS + 1):
            this = wk[wk.week == w]
            if len(this):
                this = this.assign(r=this.groupby("pos").ecr.rank(method="first"))
                last = {ix[p]: r for p, r in zip(this.player_id, this.r)}
                worst = this.groupby("pos").r.max().to_dict()
            for i in range(n):
                rank[i, w - 1] = last.get(i, worst.get(self.pos[i], 100) + 1)
        self.ecr_val = np.zeros((n, WEEKS))
        for p in POS:
            m = self.pos == p
            self.ecr_val[m] = curve_points(curves, p, rank[m])

        # The same pregame availability discount for every lineup policy: a questionable
        # player is worth his value times the chance he plays.
        for p, w in q:
            if p in ix and w <= WEEKS:
                self.model_val[ix[p], w - 1] *= p_q
                self.ecr_val[ix[p], w - 1] *= p_q

        # Blends, declared before any blend result was seen: an independent model and
        # the consensus average out each other's errors (preseason ADP and early-season
        # results each correlate ~.6 with the rest of the season; their mean ~.68).
        self.blend_val = (self.model_val + self.ecr_val) / 2
        ours = np.full(n, len(self.model_order) + 1.0)
        ours[self.model_order] = np.arange(1, len(self.model_order) + 1)
        theirs = pd.Series(self.ecr_mean).rank().fillna(np.isfinite(self.ecr_mean).sum() + 1).values
        both = (ours + theirs) / 2
        self.blend_order = list(np.argsort(both, kind="stable"))
        ok = np.where(np.isfinite(self.ecr_mean))[0]
        self.exact_order = list(ok[np.argsort(self.ecr_mean[ok], kind="stable")])

        # Values for the market-aware draft (prereg.md): consensus-implied season points,
        # and the stacked projection, with the curve and weights fit only on seasons
        # before y; both priced over replacement.
        crv, coefs = K.fit_before(y)
        t = K.table(y, raw)
        cons = pd.Series(K.ecr_points(crv, t.pos, t["rank"]), index=t.index)
        self.cons_vbd = self._vbd(cons, ids)
        self.stack_vbd = self._vbd(K.apply(t, crv, coefs), ids)
        # Who the opponents are predicted to take next: the consensus overall order.
        self.market_key = np.where(np.isfinite(self.ecr_mean), self.ecr_mean, 1e6)

        # Draft-market prices (prereg_adp.md): the last snapshot of real drafts before
        # the opener, on a 12-team board, so an ADP is comparable to a pick number here.
        mkt = pd.read_parquet("data/adp_ffc.parquet")
        mkt = mkt[(mkt.season == y) & mkt.player_id.notna()]
        mkt = mkt.drop_duplicates("player_id").set_index("player_id")
        self.adp = mkt.adp.reindex(ids).values
        # ADP order first, then everyone else by consensus. The tail is load-bearing: the
        # skill-position ADP list runs 146-206 deep by season and a 14-round draft takes
        # 168 picks, so in 2022 it is exhausted before the draft ends.
        adp_key = np.where(np.isfinite(self.adp), self.adp, 1e6)
        order = np.lexsort((self.market_key, adp_key))
        self.adp_order = [i for i in order
                          if np.isfinite(self.adp[i]) or np.isfinite(self.ecr_mean[i])]

    def _vbd(self, pts, ids):
        b = pd.DataFrame({"position": self.pos, "proj_points": pts.reindex(ids).values},
                         index=ids)
        b = b[b.position.isin(POS) & b.proj_points.notna()]
        b, _, _ = add_vbd(b)
        return b.vbd.reindex(ids).values


# ---------------------------------------------------------------- draft and lineups

def needs(c):
    """Starting slots still empty, flex included, given position counts c."""
    short = {p: max(0, k - c[p]) for p, k in SLOTS.items()}
    extra = sum(max(0, c[p] - SLOTS[p]) for p in FLEX)
    short["FLEX"] = max(0, SET.flex_slots - extra)
    return short


def allowed(pos, c, rnd):
    if pos not in c or c[pos] >= CAP.get(pos, 99):
        return False
    if pos in ONE_EACH and c[pos] >= 1 and rnd + 1 < SECOND_AFTER:
        return False
    short = needs(c)
    if ROUNDS - rnd <= sum(short.values()):          # must fill the lineup now
        return short.get(pos, 0) > 0 or (pos in FLEX and short["FLEX"] > 0)
    return True


def plan_position(horizon, need):
    """Fry-Lundberg-Ohlmann: the position to take now so the finished lineup is worth
    the most, given the best player expected at each position at each of my picks.
    Same recursion and bench weights as draft_dp.plan."""
    keys = POS + ["FLEX"]

    @lru_cache(maxsize=None)
    def f(i, state):
        if i == len(horizon):
            return 0.0, None
        nd = dict(zip(keys, state))
        best = (-1e18, None)
        for p in POS:
            v = horizon[i][p]
            if nd[p] > 0:
                nx = dict(nd); nx[p] -= 1
            elif nd["FLEX"] > 0 and p in FLEX:
                nx = dict(nd); nx["FLEX"] -= 1
            else:
                nx, v = dict(nd), v * BENCH_VALUE[p]
            val, _ = f(i + 1, tuple(nx[k] for k in keys))
            if v + val > best[0]:
                best = (v + val, p)
        return best

    return f(0, tuple(need[k] for k in keys))[1]


def market_policy(S, seat, value):
    """prereg.md: value players by `value`, predict that opponents take the consensus
    order until my next pick, and plan the position with plan_position."""
    mine = snake_picks(seat, TEAMS, ROUNDS)

    def pick(taken, c, rnd, pick_no):
        avail = np.where(~taken & np.isfinite(value))[0]
        if not len(avail):
            return None
        order = avail[np.argsort(S.market_key[avail], kind="stable")]
        pos_o, val_o = S.pos[order], value[order]
        future = [p for p in mine if p >= pick_no]
        horizon = []
        for i, pk in enumerate(future):
            head = pos_o[:(pk - future[0]) - i]            # taken by others before pk
            row = {}
            for p in POS:
                vals = val_o[pos_o == p]
                gone = int((head == p).sum())
                row[p] = vals[gone:].max() if len(vals) > gone else 0.0
            horizon.append(row)
        choice = plan_position(horizon, needs(c))
        if allowed(choice, c, rnd):
            at = avail[S.pos[avail] == choice]
            return at[np.argmax(value[at])]
        for x in avail[np.argsort(-value[avail])]:            # the plan's slot is closed
            if allowed(S.pos[x], c, rnd):
                return x
        return None

    return pick


def adp_gap_policy(S, seat, value, delta=ADP_DELTA):
    """prereg_adp.md: value players by consensus, but pay market prices. Never spend a
    pick on a player the market says will still be there next time, when someone the
    market will take is worth nearly as much. A player with no ADP counts as safe:
    undrafted in a 12-team market means he lasts."""
    mine = snake_picks(seat, TEAMS, ROUNDS)

    def pick(taken, c, rnd, pick_no):
        avail = np.where(~taken & np.isfinite(value))[0]
        ok = avail[[allowed(S.pos[x], c, rnd) for x in avail]] if len(avail) else avail
        if not len(ok):
            return None
        best = ok[np.argmax(value[ok])]
        later = [p for p in mine if p > pick_no]
        if not later:                                   # last pick: nothing to wait for
            return best
        unsafe = ok[np.nan_to_num(S.adp[ok], nan=1e6) < later[0]]
        if not len(unsafe) or best in unsafe:
            return best
        cand = unsafe[np.argmax(value[unsafe])]
        return cand if value[cand] >= value[best] - delta else best

    return pick


def draft(S, orders):
    """Snake draft; each seat takes the first player on its own list it may take, or
    asks its policy, when the seat's entry is a function instead of a list."""
    taken = np.zeros(len(S.ids), bool)
    rosters = [[] for _ in range(TEAMS)]
    counts = [dict.fromkeys(POS, 0) for _ in range(TEAMS)]
    ptr = [0] * TEAMS
    pick_no = 0
    for rnd in range(ROUNDS):
        seats = range(TEAMS) if rnd % 2 == 0 else reversed(range(TEAMS))
        for seat in seats:
            pick_no += 1
            order = orders[seat]
            if callable(order):
                p = order(taken, counts[seat], rnd, pick_no)
                if p is not None:
                    taken[p] = True
                    rosters[seat].append(p)
                    counts[seat][S.pos[p]] += 1
                continue
            i = ptr[seat]
            while i < len(order):
                p = order[i]
                if not taken[p] and allowed(S.pos[p], counts[seat], rnd):
                    break
                i += 1
            if i == len(order):
                continue
            p = order[i]
            taken[p] = True
            rosters[seat].append(p)
            counts[seat][S.pos[p]] += 1
            # A skipped player may become allowed later, so only advance past taken ones.
            while ptr[seat] < len(order) and taken[order[ptr[seat]]]:
                ptr[seat] += 1
    return rosters


def week_points(S, roster, value, w):
    """Actual points in week w from the lineup a policy sets on its values."""
    r = np.array(roster)
    pos = S.pos[r]
    v = np.where(S.elig[r, w], value[r, w], -np.inf)
    used = np.zeros(len(r), bool)
    total = 0.0
    for p, k in SLOTS.items():
        cand = np.where((pos == p) & ~used & np.isfinite(v))[0]
        pick = cand[np.argsort(-v[cand])][:k]
        used[pick] = True
        total += S.actual[r[pick], w].sum()
    cand = np.where(np.isin(pos, FLEX) & ~used & np.isfinite(v))[0]
    if len(cand):
        pick = cand[np.argsort(-v[cand], kind="stable")][:SET.flex_slots]
        total += S.actual[r[pick], w].sum()
    return total


def lineup_points(S, roster, value):
    """Actual points each week from the lineup a policy sets on its values."""
    return np.array([week_points(S, roster, value, w) for w in range(WEEKS)])


def schedules(rng, n):
    """Random 14-week head-to-head schedules: a shuffled round robin plus three repeats."""
    base = []
    teams = list(range(TEAMS))
    for rnd in range(TEAMS - 1):                       # circle method
        pairs = [(teams[i], teams[TEAMS - 1 - i]) for i in range(TEAMS // 2)]
        base.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    out = []
    for _ in range(n):
        lab = rng.permutation(TEAMS)
        rounds = [base[i] for i in rng.permutation(TEAMS - 1)]
        rounds += [base[i] for i in rng.choice(TEAMS - 1, REG - (TEAMS - 1), replace=False)]
        out.append([[(lab[a], lab[b]) for a, b in rd] for rd in rounds])
    return out


def bracket(seed, scores, s=SET):
    """The champion of a single-elimination playoff, given the seeding.

    Configurable where it used to be a hand-written six-team bracket. Byes are given to
    the top seeds in the first round; every later round pairs the best remaining seed
    with the worst, which is what the published bracket did when it put the lower of the
    two quarter-final winners against the #1 seed.

    A tie on the week's score goes to the better seed. The one exception is the final,
    where it goes to the winner of the half of the bracket containing the top seed,
    because with two teams left there is no matchup to assign and the order is only a
    tiebreak. That is the published rule and it is kept so results do not move.
    """
    alive = list(range(s.playoff_teams))              # positions in `seed`, best first
    week = s.reg_weeks                                # 0-indexed first playoff week
    while len(alive) > 1:
        if week >= s.weeks:
            raise ValueError("the playoff bracket does not fit in the weeks available")
        byes = s.playoff_byes if week == s.reg_weeks else 0
        if s.playoff_reseed and len(alive) > 2:
            alive = sorted(alive)
        rest = alive[byes:]
        nxt = alive[:byes]
        for i in range(len(rest) // 2):
            a, b = rest[i], rest[-1 - i]
            nxt.append(a if scores[seed[a], week] >= scores[seed[b], week] else b)
        if len(rest) % 2:                             # an odd man out advances
            nxt.append(rest[len(rest) // 2])
        alive, week = nxt, week + 1
    return seed[alive[0]]


def season_outcome(scores, sched):
    """Wins, playoff berth and title per team for one schedule. scores: teams x weeks."""
    wins = np.zeros(TEAMS)
    for w, rd in enumerate(sched):
        for a, b in rd:
            wins[a if scores[a, w] > scores[b, w] else b] += 1
    pf = scores[:, :REG].sum(1)
    seed = sorted(range(TEAMS), key=lambda t: (-wins[t], -pf[t]))
    playoff = np.zeros(TEAMS, bool)
    playoff[seed[:SET.playoff_teams]] = True
    title = np.zeros(TEAMS, bool)
    title[bracket(seed, scores)] = True
    return wins, playoff, title


def all_play(scores, t):
    wk = scores[:, :REG]
    return ((wk[t] > np.delete(wk, t, 0)).sum() / ((TEAMS - 1) * REG))


# ---------------------------------------------------------------- waivers

# Waiver rules, applied identically to every seat (prereg_waivers.md): one add/drop a
# week, roster size fixed at 14, no FAAB, no trades, no IR slot. A team may never cut
# below a legal starting lineup, so the wire cannot leave it unable to field one.
ROSTER_MIN = SET.roster_min
FIRST_WAIVER = SET.first_waiver_week   # the first decision follows week 1


def waiver_values(S, y, crv, fits):
    """Rest-of-season value per player at each decision point, for both waiver policies.

    Both policies price a player through the same consensus rank curve, so the only
    thing that differs between them is the order. The usage signal enters as an implied
    rank: recent role puts a player at some percentile of his position, that percentile
    is read on the consensus rank scale, and he is valued at the better of that and his
    consensus rank. Pricing it any other way would be a level shift rather than a
    disagreement, since the signal is fit on players who are playing and consensus
    ranks everyone. Consensus being slow on role is the whole hypothesis, so the signal
    may promote a player but never demote one.
    """
    n = len(S.ids)
    ix = {p: i for i, p in enumerate(S.ids)}
    ros = pd.read_parquet("data/ecr_ros.parquet")
    ros = ros[ros.season == y]
    pre = pd.read_parquet("data/ecr_preseason.parquet")
    pre = pre[pre.season == y]
    # No fits means only the consensus values are wanted; the usage arm stays at consensus.
    uv = U.season_values(y, fits) if fits is not None else pd.DataFrame(
        columns=["player_id", "week", "value"])
    cons = np.full((n, WEEKS), np.nan)
    test = np.full((n, WEEKS), np.nan)
    for w in range(FIRST_WAIVER, WEEKS + 1):
        # The latest rest-of-season scrape strictly before the upcoming week; before the
        # first one of the season, the preseason positional ranks stand in.
        earlier = ros[ros.week < w]
        src = earlier[earlier.week == earlier.week.max()] if len(earlier) else pre
        src = src.assign(rank=src.groupby("pos").ecr.rank(method="first"))
        worst = src.groupby("pos")["rank"].max().to_dict()
        c_rank = np.full(n, np.nan)
        for p, r in zip(src.player_id, src["rank"]):
            if p in ix:
                c_rank[ix[p]] = r
        u = np.full(n, np.nan)
        this = uv[uv.week == w]
        for p, v in zip(this.player_id, this.value):
            if p in ix:
                u[ix[p]] = v

        t_rank = np.full(n, np.nan)
        for p in POS:
            at = S.pos == p
            deep = int((at & np.isfinite(c_rank)).sum())    # how far down consensus ranks
            # A player consensus does not rank sits one past the last player it does.
            c_rank[at & np.isnan(c_rank)] = worst.get(p, 100) + 1
            t_rank[at] = c_rank[at]
            seen = at & np.isfinite(u)
            if not seen.sum() or not deep:
                continue
            order = np.argsort(-u[seen], kind="stable")
            r = np.empty(int(seen.sum()))
            r[order] = np.arange(1, seen.sum() + 1)
            # The signal ranks only players who are playing; consensus ranks a deeper
            # pool. Stretching by that ratio makes the two percentiles comparable.
            r *= deep / seen.sum()
            t_rank[seen] = np.minimum(c_rank[seen], r)
        for arr, rk in ((cons, c_rank), (test, t_rank)):
            pts = np.full(n, np.nan)
            for p in POS:
                at = S.pos == p
                pts[at] = C.rank_points(crv, S.pos[at], np.clip(rk[at], 1, 150))
            arr[:, w - 1] = over_replacement(S, pts)
    return cons, test


def over_replacement(S, pts):
    """Price a week's projected points against the replacement at each position.

    Raw points cannot be compared across positions on a waiver wire any more than on
    draft day: a QB15 outscores a WR40 every week, so a policy comparing points takes a
    quarterback every time and ends the season starting four of them. In a one-QB league
    the second one is worth nothing, which is exactly what value over replacement says.
    """
    b = pd.DataFrame({"position": S.pos, "proj_points": pts})
    b = b[b.position.isin(POS) & b.proj_points.notna()]
    b, _, _ = add_vbd(b)
    out = np.full(len(S.pos), np.nan)
    out[b.index.values] = b.vbd.values
    return out


def priority(scores, upto):
    """Waiver order: worst record first, ties by points for.

    Record here is all-play through the weeks already played, not head-to-head. The
    harness scores every roster against twenty schedule draws, and a schedule-dependent
    order would mean re-running the wire once per draw, with the roster a team ends up
    holding decided partly by schedule luck. All-play is the same standings idea with
    the luck taken out.
    """
    s = scores[:, :upto + 1]
    rate = [(s[t] > np.delete(s, t, 0)).sum() for t in range(TEAMS)]
    pf = s.sum(1)
    return sorted(range(TEAMS), key=lambda t: (rate[t], pf[t]))


def cuttable(S, held):
    """Rostered players a team may cut without dropping a position below its starters."""
    return np.array(R.cuttable(S.pos, held), dtype=int)


def consensus_move(S, held, val, free, w):
    """The consensus wire policy: add the best free agent, drop the worst cuttable player,
    only when the add is worth more. Returns (add, drop) or None."""
    free = R.add_allowed(S.pos, held, free)     # a no-op unless the cap is enforced
    if not len(free):
        return None
    add = free[int(np.argmax(val[free]))]
    can_cut = cuttable(S, held)
    if not len(can_cut):
        return None
    # A player the policy cannot value at all is the first one cut.
    vd = np.where(np.isfinite(val[can_cut]), val[can_cut], -np.inf)
    drop = int(can_cut[int(np.argmin(vd))])
    return (int(add), drop) if val[add] > vd.min() else None


def waiver_week(S, rosters, values, order, w, movers=None):
    """One waiver round before week w. Teams act in priority order on their own policy,
    so a team can lose the player it wanted to one picking ahead of it. `movers` swaps in
    a different decision for some seats; everyone else runs the consensus move."""
    on_roster = np.zeros(len(S.ids), bool)
    for r in rosters:
        on_roster[r] = True
    moves = np.zeros(TEAMS, int)
    # A league allowing more than one move a week runs the whole priority pass again,
    # so the worst team still picks first on its second move. The default is one.
    for _ in range(SET.waiver_moves_per_week):
        for t in order:
            val = values[t][:, w - 1]
            free = np.where(~on_roster & np.isfinite(val))[0]
            if not len(free):
                continue
            move = (movers or {}).get(t, consensus_move)(S, np.array(rosters[t]), val, free, w)
            if move is None:
                continue
            add, drop = move
            rosters[t] = [i for i in rosters[t] if i != drop] + [int(add)]
            on_roster[add], on_roster[drop] = True, False
            moves[t] += 1
    return moves


def simulate(S, drafted, values, who, movers=None):
    """Play a season week by week, running the wire between weeks.

    Waiver priority depends on the standings so far, so the order cannot be known in
    advance: scoring and transacting have to interleave.
    """
    rosters = [list(r) for r in drafted]
    scores = np.zeros((TEAMS, WEEKS))
    moves = np.zeros(TEAMS, int)
    for w in range(WEEKS):
        for t in range(TEAMS):
            scores[t, w] = week_points(S, rosters[t], S.ecr_val, w)
        if w + 1 < WEEKS and who:
            order = [t for t in priority(scores, w) if t in who]
            moves += waiver_week(S, rosters, values, order, w + 2, movers)
    return scores, moves, rosters


# ---------------------------------------------------------------- the waiver experiment

def run_waivers():
    curves = rank_curve()
    wk_proj = weekly_projections()
    rows = []
    for y in SEASONS:
        S = Season(y, wk_proj, curves)
        # Both fits see only seasons before y, so nothing is fit on the season scored.
        crv = C.weekly_curve(range(2020, y))
        fits = U.fit_before(y)
        cons_val, test_val = waiver_values(S, y, crv, fits)
        every = set(range(TEAMS))
        for lg in range(LEAGUES):
            # The same draws in the same order as the main harness, so the drafts match.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for k in range(TEAMS + 1):
                score = S.ecr_mean + NOISE * S.ecr_sd * noise[k]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = schedules(rng, SCHEDULES)
            for seat in range(TEAMS):
                orders = list(ecr_orders[:TEAMS])
                orders[seat] = S.exact_order          # exact consensus draft in every arm
                drafted = draft(S, orders)
                base = [cons_val] * TEAMS
                mine = list(base)
                mine[seat] = test_val
                for arm, v, who in (("none", base, set()), ("cons", base, every),
                                    ("usage", mine, every), ("solo", base, {seat})):
                    sc, moves, _ = simulate(S, drafted, v, who)
                    res = [season_outcome(sc, s) for s in scheds]
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": all_play(sc, seat),
                        "pts_reg": sc[seat, :REG].sum(),
                        "pts_playoff": sc[seat, REG:].sum(),
                        "moves": moves[seat],
                    })
            print(f"{y} league {lg + 1}/{LEAGUES}", flush=True)
    return pd.DataFrame(rows)


def paired(d, arm, ref, label):
    """Paired change of one arm against another, in the same league, seat and schedules."""
    key = ["season", "league", "seat"]
    m = ["title", "playoff", "wins", "all_play", "pts_reg", "pts_playoff"]
    a = d[d.arm == arm].set_index(key)[m]
    b = d[d.arm == ref].set_index(key)[m]
    diff = a - b.loc[a.index]
    by = diff.groupby("season").all_play.mean()
    rng = np.random.default_rng(1)
    seasons = diff.groupby("season").mean()
    boot = pd.DataFrame([seasons.loc[rng.choice(seasons.index, len(seasons))].mean()
                         for _ in range(2000)])
    lo, hi = boot.quantile(.05), boot.quantile(.95)
    keep = (diff.all_play.mean() > 0 and (by > 0).sum() >= 4
            and diff.title.mean() >= TITLE_GUARD)
    print(f"  {label:26s} title {100 * diff.title.mean():+.1f} pp "
          f"[{100 * lo.title:+.1f}, {100 * hi.title:+.1f}]   all-play "
          f"{100 * diff.all_play.mean():+.1f} pp [{100 * lo.all_play:+.1f}, "
          f"{100 * hi.all_play:+.1f}]  by season "
          + " ".join(f"{100 * v:+.1f}" for v in by)
          + f"   reg pts {diff.pts_reg.mean():+.0f}")
    return keep


def report_waivers(d):
    print("\n=== waivers: usage-based breakout detector vs consensus waivers ===")
    print("12-team PPR leagues on 2021-25, exact-consensus draft and consensus lineups in "
          "every arm;\none add/drop per team per week, priority worst record first\n")
    print(f"{'arm':6s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s} {'adds':>6s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:6s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f} {g.moves.mean():6.2f}")

    print("\npreregistered test (prereg_waivers.md), season-cluster bootstrap 90% interval:")
    keep = paired(d, "usage", "cons", "usage minus consensus")
    print(f"  -> {'passes this design' if keep else 'fails this design'}")
    print("\ndiagnostics: how much the wire moves anything at all")
    paired(d, "cons", "none", "everyone drafts, no wire")
    paired(d, "solo", "none", "only my seat works the wire")
    print("('cons minus none' is close to zero by construction: when every team gets the "
          "same\n lever, the relative standings need not move. 'solo minus none' is the "
          "lever's size.)")


# ---------------------------------------------------------------- streaming (prereg_streaming.md)

def known_unavailable(S, y):
    """Who is known to be unavailable for week w at the waiver decision before it.

    One definition of availability, in availability.py, shared with the live tools. Row
    w-1 of the result is the decision made before week w.
    """
    return A.harness_grid(S.ids, y, WEEKS)


def holes(S, players, gone_w):
    """Starting slots, flex included, the players not known unavailable can't fill."""
    c = dict.fromkeys(POS, 0)
    for i in players:
        if not gone_w[i] and S.pos[i] in c:
            c[S.pos[i]] += 1
    return needs(c)


def starters(S, players, val, gone_w):
    """The week's lineup filled by rest-of-season value from players known available.
    Weekly consensus for the week isn't out at waiver time, so this is the best guess."""
    players = [i for i in players if not gone_w[i]]
    v = {i: val[i] if np.isfinite(val[i]) else -np.inf for i in players}
    used = set()
    for p, k in SLOTS.items():
        used |= set(sorted((i for i in players if S.pos[i] == p), key=lambda i: -v[i])[:k])
    flex = sorted((i for i in players if S.pos[i] in FLEX and i not in used),
                  key=lambda i: -v[i])
    used.update(flex[:SET.flex_slots])
    return used


def streaming_policy(S, gone, log):
    """prereg_streaming.md: fill next week's starting hole when there is one, even with a
    player consensus ranks below the one cut; otherwise move exactly as consensus does.

    The consensus wire ranks free agents on rest-of-season value alone, so it will add a
    fourth receiver while the only tight end sits on bye. A zero in a starting slot is a
    sure loss of points that week, and the bet is that it outweighs the rest-of-season
    value given up. Each hole-driven move is logged with what consensus would have done.
    """
    def move(S, held, val, free, w):
        cons = consensus_move(S, held, val, free, w)
        g = gone[:, w - 1]
        before = holes(S, held, g)
        if not sum(before.values()):
            return cons
        fills = [p for p in POS if before[p] > 0 or (p in FLEX and before["FLEX"] > 0)]
        cand = free[np.isin(S.pos[free], fills) & ~g[free]]
        cand = R.add_allowed(S.pos, held, cand)     # a no-op unless the cap is enforced
        if not len(cand):
            return cons
        add = int(cand[int(np.argmax(val[cand]))])
        after_add = list(held) + [add]
        lineup = starters(S, after_add, val, g)
        best = None
        for d in cuttable(S, held):
            left = sum(holes(S, [i for i in after_add if i != d], g).values())
            if left >= sum(before.values()):         # the move has to close a hole
                continue
            key = (left, d in lineup, val[d] if np.isfinite(val[d]) else -np.inf)
            if best is None or key < best[0]:
                best = (key, int(d))
        if best is None:
            return cons
        mine = (add, best[1])
        log.append({"week": w, "holes": sum(before.values()), "add": S.ids[add],
                    "drop": S.ids[best[1]], "add_pos": S.pos[add], "drop_pos": S.pos[best[1]],
                    "add_val": val[add], "drop_val": best[0][2], "same_as_cons": mine == cons,
                    "cons_moves": cons is not None, "cf_pts": cf_points(S, held, mine, cons, w)})
        return mine
    return move


def cf_points(S, held, mine, cons, w):
    """Week-w lineup points with the streaming move minus with the consensus move (or no
    move), on the same roster: the one-week payoff of acting on the hole."""
    def pts(m):
        r = list(held) if m is None else [i for i in held if i != m[1]] + [m[0]]
        return week_points(S, r, S.ecr_val, w - 1)
    return pts(mine) - pts(cons)


def run_streaming():
    curves = rank_curve()
    wk_proj = weekly_projections()
    rows, moves_log = [], []
    for y in SEASONS:
        S = Season(y, wk_proj, curves)
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = waiver_values(S, y, crv, None)     # consensus values only
        gone = known_unavailable(S, y)
        every = set(range(TEAMS))
        for lg in range(LEAGUES):
            # The same draws in the same order as run_waivers, so drafts and control match.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for k in range(TEAMS + 1):
                score = S.ecr_mean + NOISE * S.ecr_sd * noise[k]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = schedules(rng, SCHEDULES)
            base = [cons_val] * TEAMS
            for seat in range(TEAMS):
                orders = list(ecr_orders[:TEAMS])
                orders[seat] = S.exact_order
                drafted = draft(S, orders)
                log = []
                arms = (("cons", None), ("stream", {seat: streaming_policy(S, gone, log)}))
                week_sc = {}
                for arm, movers in arms:
                    sc, moves, _ = simulate(S, drafted, base, every, movers)
                    week_sc[arm] = sc[seat]
                    res = [season_outcome(sc, s) for s in scheds]
                    rows.append({
                        "season": y, "league": lg, "seat": seat, "arm": arm,
                        "title": np.mean([r[2][seat] for r in res]),
                        "playoff": np.mean([r[1][seat] for r in res]),
                        "wins": np.mean([r[0][seat] for r in res]),
                        "all_play": all_play(sc, seat),
                        "pts_reg": sc[seat, :REG].sum(),
                        "pts_playoff": sc[seat, REG:].sum(),
                        "moves": moves[seat],
                    })
                for e in log:
                    e.update(season=y, league=lg, seat=seat, paired_pts=(
                        week_sc["stream"][e["week"] - 1] - week_sc["cons"][e["week"] - 1]))
                moves_log += log
            print(f"{y} league {lg + 1}/{LEAGUES}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(moves_log)


def report_streaming(d, mv):
    print("\n=== waivers: hole-aware streaming vs consensus waivers ===")
    print("12-team PPR leagues on 2021-25, exact-consensus draft and consensus lineups, "
          "every other\nseat on the consensus wire; one add/drop per team per week\n")
    print(f"{'arm':6s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s} {'adds':>6s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:6s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f} {g.moves.mean():6.2f}")
    print("\npreregistered test (prereg_streaming.md), season-cluster bootstrap 90% interval:")
    keep = paired(d, "stream", "cons", "stream minus consensus")
    print(f"  -> {'passes this design' if keep else 'fails this design'}")

    seats = d[d.arm == "stream"].groupby("season").size()
    print("\nreported, not gated: hole-driven moves by the test seat (per league-seat season)")
    if not len(mv):
        print("  none")
        return
    diff = mv[~mv.same_as_cons]
    print(f"  {'season':6s} {'hole moves':>10s} {'differ':>7s} {'cf pts':>7s} {'paired pts':>11s}")
    for y, n_seats in seats.items():
        m, x = mv[mv.season == y], diff[diff.season == y]
        print(f"  {y:<6d} {len(m) / n_seats:10.2f} {len(x) / n_seats:7.2f} "
              f"{x.cf_pts.mean():+7.1f} {x.paired_pts.mean():+11.1f}")
    print(f"  {'all':6s} {len(mv) / seats.sum():10.2f} {len(diff) / seats.sum():7.2f} "
          f"{diff.cf_pts.mean():+7.1f} {diff.paired_pts.mean():+11.1f}")
    print("  (differ: the move is not the one consensus would make in the same state; "
          "cf pts: that\n   week's lineup points with the move minus with consensus's move, "
          "same roster; paired pts:\n   the stream arm's score that week minus the control "
          "arm's)")
    print("  holes filled, by position added: "
          + " ".join(f"{p} {k}" for p, k in diff.add_pos.value_counts().items()))


# ---------------------------------------------------------------- the experiment

def run():
    curves = rank_curve()
    wk_proj = weekly_projections()
    rows = []
    for y in SEASONS:
        S = Season(y, wk_proj, curves)
        vals = {"model": S.model_val, "ecr": S.ecr_val, "blend": S.blend_val,
                "hindsight": S.actual}
        for lg in range(LEAGUES):
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((TEAMS + 1, len(S.ids)))
            ecr_orders = []
            for k in range(TEAMS + 1):                  # 12 opponents' lists + a spare
                score = S.ecr_mean + NOISE * S.ecr_sd * noise[k]
                ok = np.where(np.isfinite(score))[0]
                ecr_orders.append(list(ok[np.argsort(score[ok])]))
            scheds = schedules(rng, SCHEDULES)
            for seat in range(TEAMS):
                # "exact" drafts on the consensus mean with no noise. The opponents' noise is
                # pure error by construction, so following consensus exactly already beats
                # them; the blend has to be judged against this, not against "ecr".
                own = {"model": S.model_order, "ecr": ecr_orders[TEAMS], "blend": S.blend_order,
                       "exact": S.exact_order,
                       "market": market_policy(S, seat, S.cons_vbd),
                       "market_stack": market_policy(S, seat, S.stack_vbd),
                       "adp": S.adp_order,
                       "adp_gap": adp_gap_policy(S, seat, S.cons_vbd)}
                for d_arm in ("model", "ecr", "blend", "exact", "market", "market_stack",
                              "adp", "adp_gap"):
                    orders = list(ecr_orders[:TEAMS])
                    orders[seat] = own[d_arm]
                    rosters = draft(S, orders)
                    base = np.array([lineup_points(S, r, S.ecr_val) for r in rosters])
                    # The market arms are a draft test, so only consensus lineups.
                    lineups = ("ecr",) if d_arm in ADP_ARMS else (
                        "model", "ecr", "blend", "hindsight")
                    for l_arm in lineups:
                        sc = base.copy()
                        sc[seat] = lineup_points(S, rosters[seat], vals[l_arm])
                        res = [season_outcome(sc, s) for s in scheds]
                        rows.append({
                            "season": y, "league": lg, "seat": seat,
                            "draft": d_arm, "lineup": l_arm,
                            "title": np.mean([r[2][seat] for r in res]),
                            # how much the title rate swings with the schedule alone
                            "title_sched_sd": np.std([r[2][seat] for r in res]),
                            "titles": [int(r[2][seat]) for r in res],   # per schedule
                            "playoff": np.mean([r[1][seat] for r in res]),
                            "wins": np.mean([r[0][seat] for r in res]),
                            "all_play": all_play(sc, seat),
                            "pts_reg": sc[seat, :REG].sum(),
                            "pts_playoff": sc[seat, REG:].sum(),
                        })
            print(f"{y} league {lg + 1}/{LEAGUES}", flush=True)
    return pd.DataFrame(rows)


def report(d):
    key = ["season", "league", "seat"]
    base = d[(d.draft == "ecr") & (d.lineup == "ecr")].set_index(key)
    m = ["title", "playoff", "wins", "all_play", "pts_reg", "pts_playoff"]
    print("\n=== draft + start/sit edge against synthetic consensus, no waivers or trades ===")
    print("12-team PPR leagues on 2021-25, 20 perturbed drafts x 12 seats x 20 schedules "
          "per season; changes are paired against a consensus seat in the same league\n")
    print(f"{'draft':6s} {'lineups':10s} {'title':>7s} {'playoff':>8s} {'title|po':>9s} "
          f"{'wins':>6s} {'all-play':>9s} {'reg pts':>8s} {'po pts':>7s} | d all-play by season")
    verdicts = []
    for (da, la), g in d.groupby(["draft", "lineup"], sort=False):
        x = g.set_index(key)
        diff = (x[m] - base[m])
        by = diff.groupby("season").all_play.mean()
        cells = (f"{100 * x.title.mean():6.1f}% {100 * x.playoff.mean():7.1f}% "
                 f"{100 * x.title.sum() / max(x.playoff.sum(), 1e-9):8.1f}% "
                 f"{x.wins.mean():6.2f} {100 * x.all_play.mean():8.1f}% "
                 f"{x.pts_reg.mean():8.0f} {x.pts_playoff.mean():7.0f}")
        print(f"{da:6s} {la:10s} {cells} | " + " ".join(f"{100 * v:+.1f}" for v in by))
        if (da, la) != ("ecr", "ecr") and la != "hindsight":
            keep = (diff.all_play.mean() > 0 and (by > 0).sum() >= 4
                    and diff.title.mean() >= TITLE_GUARD)
            verdicts.append((da, la, diff, by, keep))

    print("\npaired change vs a consensus seat (season-cluster bootstrap 90% interval):")
    rng = np.random.default_rng(0)
    for da, la, diff, by, keep in verdicts:
        seasons = diff.groupby("season").mean()
        boot = [seasons.loc[rng.choice(seasons.index, len(seasons))].mean() for _ in range(2000)]
        boot = pd.DataFrame(boot)
        lo, hi = boot.quantile(.05), boot.quantile(.95)
        print(f"  {da} draft + {la} lineups: title {100 * diff.title.mean():+.1f} pp "
              f"[{100 * lo.title:+.1f}, {100 * hi.title:+.1f}]   all-play "
              f"{100 * diff.all_play.mean():+.1f} pp [{100 * lo.all_play:+.1f}, "
              f"{100 * hi.all_play:+.1f}]   playoff {100 * diff.playoff.mean():+.1f} pp   "
              f"wins {diff.wins.mean():+.2f}   -> {'KEEP' if keep else 'not shown to help'}")
    # What the model adds on top of consensus: blend vs exact consensus, same lineups.
    print("\nwhat the model adds: blend draft minus exact-consensus draft, same lineup policy")
    for la in ("ecr", "blend"):
        a = d[(d.draft == "blend") & (d.lineup == la)].set_index(key)[m]
        b = d[(d.draft == "exact") & (d.lineup == la)].set_index(key)[m]
        diff = a - b
        by = diff.groupby("season").all_play.mean()
        print(f"  {la} lineups: title {100 * diff.title.mean():+.1f} pp   all-play "
              f"{100 * diff.all_play.mean():+.1f} pp  (by season "
              + " ".join(f"{100 * v:+.1f}" for v in by) + f")   playoff "
              f"{100 * diff.playoff.mean():+.1f} pp")

    # The preregistered test (prereg.md): market-aware drafts against exact consensus.
    b = d[(d.draft == "exact") & (d.lineup == "ecr")].set_index(key)[m]
    for spec, arms in (("prereg.md", ("market", "market_stack")),
                       ("prereg_adp.md", ADP_ARMS)):
        arms = [a for a in arms if ((d.draft == a) & (d.lineup == "ecr")).any()]
        if not arms:
            continue
        print(f"\npreregistered ({spec}): draft minus exact-consensus draft, consensus "
              "lineups (season-cluster bootstrap 90% interval)")
        for arm in arms:
            rng = np.random.default_rng(1)               # per arm, as the spec declares
            a = d[(d.draft == arm) & (d.lineup == "ecr")].set_index(key)[m]
            diff = a - b.loc[a.index]
            by = diff.groupby("season").all_play.mean()
            seasons = diff.groupby("season").mean()
            boot = pd.DataFrame([seasons.loc[rng.choice(seasons.index, len(seasons))].mean()
                                 for _ in range(2000)])
            lo, hi = boot.quantile(.05), boot.quantile(.95)
            keep = (diff.all_play.mean() > 0 and (by > 0).sum() >= 4
                    and diff.title.mean() >= TITLE_GUARD)
            print(f"  {arm:13s} title {100 * diff.title.mean():+.1f} pp "
                  f"[{100 * lo.title:+.1f}, {100 * hi.title:+.1f}]   all-play "
                  f"{100 * diff.all_play.mean():+.1f} pp [{100 * lo.all_play:+.1f}, "
                  f"{100 * hi.all_play:+.1f}]  by season "
                  + " ".join(f"{100 * v:+.1f}" for v in by)
                  + f"   playoff {100 * diff.playoff.mean():+.1f} pp   -> "
                  + ("passes this design" if keep else "fails this design"))

    print(f"\nschedule sensitivity: for a fixed roster and lineups, the title rate moves by "
          f"{100 * d.title_sched_sd.mean():.1f} pp (sd) across schedule draws alone")
    # The same for a paired difference: arm and baseline share each schedule draw.
    bt = d[(d.draft == "ecr") & (d.lineup == "ecr")].set_index(key).titles
    for (da, la), g in d.groupby(["draft", "lineup"], sort=False):
        if la == "hindsight" or (da, la) == ("ecr", "ecr"):
            continue
        at = g.set_index(key).titles
        sds = [np.std(np.array(a) - np.array(b)) for a, b in zip(at, bt.loc[at.index])]
        print(f"  paired title difference, {da} draft + {la} lineups: sd {100 * np.mean(sds):.1f} pp "
              f"across schedules (per league-seat)")
    print("(hindsight lineups are a diagnostic ceiling on start/sit, not a policy)")


if __name__ == "__main__":
    if "--noise" in sys.argv:
        NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:      # mechanics checks only; a real run uses the default
        LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    CFG.announce_run(noise=NOISE, leagues=LEAGUES)
    # Never overwrite a real result with a short check.
    check = "" if LEAGUES == 20 else f"_check{LEAGUES}"
    if "--streaming" in sys.argv:
        d, mv = run_streaming()
        d.to_parquet(f"data/league_streaming{check}_noise{NOISE:g}.parquet")
        mv.to_parquet(f"data/league_streaming_moves{check}_noise{NOISE:g}.parquet")
        print(f"\nopponent noise: {NOISE:g} x ECR sd")
        report_streaming(d, mv)
        sys.exit()
    if "--waivers" in sys.argv:
        d = run_waivers()
        d.to_parquet(f"data/league_waivers{check}_noise{NOISE:g}.parquet")
        print(f"\nopponent noise: {NOISE:g} x ECR sd")
        report_waivers(d)
        sys.exit()
    d = run()
    path = ("data/league_backtest.parquet" if NOISE == 1.0 else
            f"data/league_backtest_noise{NOISE:g}.parquet")
    if LEAGUES != 20:                # never overwrite a real result with a short check
        path = f"data/league_backtest_check{LEAGUES}_noise{NOISE:g}.parquet"
    d.to_parquet(path)
    print(f"\nopponent noise: {NOISE:g} x ECR sd")
    report(d)
