"""Trade evaluator: what a trade does to each side's season, in wins.

A player is worth, in a trade, what he adds to the lineup of the team that gets him - not
his projection in isolation. A third running back is worth only the weeks he would
actually start: byes, injuries, and the gap between him and whoever else would fill the
slot, down to the waiver wire. So each side's rest of season is simulated with and
without the trade:

  - every remaining week, each player is active, on bye, or hurt. Injuries follow a
    two-state chain whose rates reproduce the player's projected availability, so a
    fragile star misses a run of weeks rather than a scattering of single games;
  - the lineup is set each week from who is available, on projections, the way a
    manager would, with the waiver wire as an always-available fallback at every slot;
  - the week's expected lineup score becomes a win probability against a league-average
    opponent, and those sum to expected wins.

Both rosters share the same simulated injuries (common random numbers), so the
difference between "with" and "without" is the trade, not simulation noise.

    # evaluate a trade someone offered you
    python draft/trade.py --league league.json --with Alex --give "Puka Nacua" \
                          --get "Bijan Robinson"

    # or without a league file
    python draft/trade.py --mine "..." --theirs "..." --give "..." --get "..."

    # search for trades that help you and that the other side should also accept
    python draft/trade.py --league league.json --suggest [--with Alex]

league.json: {"me": "John", "teams": {"John": ["name", ...], "Alex": [...], ...}}
"""
import argparse
import itertools
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "draft")
import availability as A
import correlate as CO
import roster as R
import settings as CFG
from vbd import LEAGUE
from assistant import match
import consensus as C

SET = CFG.get()
POS = list(CFG.POSITIONS)
FLEX = list(LEAGUE["flex"])
SLOTS = LEAGUE["starters"]
# Weekly PPR sd as a function of the player's mean, per position (player-seasons with 8+
# games, 2018-25). Scoring is right-skewed for RB/WR/TE, so this is a rough spread only.
SD = {"QB": (5.22, .14), "RB": (2.43, .38), "WR": (2.27, .41), "TE": (1.57, .49)}
REG_END, LAST_WEEK = SET.reg_weeks, SET.weeks


def sd_of(pos, mean):
    a, b = SD[pos]
    return a + b * mean


# ---------------------------------------------------------------- inputs

def load_values(season, source="consensus"):
    """Per player: points per game when active, and share of games he's active for."""
    ros = f"data/ros_{season}.parquet"
    if os.path.exists(ros):
        r = pd.read_parquet(ros).set_index("player_id")
        v = r[["player_display_name", "position"]].copy()
        if {"proj_ppg", "exp_games"} <= set(r.columns):
            v["ppg"] = r.proj_ppg
            v["avail"] = r.exp_games / r.games_left.clip(lower=1)
        else:          # points total only: treat as a healthy-rate player
            v["ppg"] = r.ros_points / r.games_left.clip(lower=1) / 0.9
            v["avail"] = 0.9
    else:
        b = pd.read_parquet(f"data/board_{season}.parquet")
        v = b[["player_display_name", "position"]].copy()
        v["ppg"] = b.proj_ppg
        v["avail"] = b.proj_games / SET.weeks
    v["avail"] = A.clip_rate(v.avail)

    # Points per game from consensus rest-of-season ranks where consensus has one:
    # consensus out-forecasts our model for season totals and single weeks alike (see
    # "Does it win leagues?" in the README). Our model still supplies availability, and
    # the rate for anyone consensus doesn't rank.
    if source == "consensus":
        try:
            _, r = C.latest("ros", season)
            ppg = pd.Series(C.rank_points(C.weekly_curve(), r.pos, r["rank"]),
                            index=r.player_id.values)
            ppg = ppg[~ppg.index.duplicated()]
            has = v.index.isin(ppg.index)
            v.loc[has, "ppg"] = ppg.reindex(v.index[has]).values
        except (FileNotFoundError, ValueError) as e:
            print(f"  ! consensus values unavailable ({e}); using the model's")

    players = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
    v["team"] = players.latest_team.reindex(v.index)
    v["status"] = players.status.reindex(v.index)
    return v[v.position.isin(POS)]


def schedule(season, start=None):
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == season) & (g.game_type == "REG")]
    open_ = g[g.result.isna()]
    week = start or (int(open_.week.min()) if len(open_) else LAST_WEEK + 1)
    weeks = list(range(week, LAST_WEEK + 1))
    plays = {w: set(g[g.week == w].home_team) | set(g[g.week == w].away_team) for w in weeks}
    return week, weeks, plays


def injury_report(season):
    """This week's chance of missing, by designation, from the shared definition."""
    return A.report_status(season)


def weekly_override(season, week):
    """This week's matchup projection (points if he plays) and chance he plays."""
    path = f"data/weekly_{season}.parquet"
    if not os.path.exists(path):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    w = pd.read_parquet(path)
    w = w[w.week == week].set_index("player_id")
    p_play = w.p_play if "p_play" in w else pd.Series(dtype=float)
    return w.proj_ppr, p_play


# ---------------------------------------------------------------- simulation

class Season:
    """Rest-of-season simulator. One instance fixes the random draws, so every roster
    evaluated through it sees the same injuries for the same player."""

    def __init__(self, values, season, sims=2000, seed=0, start=None, live=True,
                 rostered=None):
        """`live` reads this week's injury report and matchup projections; a backtest
        of a past season turns it off and starts wherever it likes. `rostered` is every
        player held in the league, which is what prices replacement off the real pool."""
        self.v, self.sims, self.seed = values, sims, seed
        self.rostered = set(rostered) if rostered is not None else None
        self.week, self.weeks, self.plays = schedule(season, start)
        self.W = len(self.weeks)
        self.reg = np.array([w <= REG_END for w in self.weeks])
        none = pd.Series(dtype=float)
        self.hurt0 = injury_report(season) if live else none
        self.matchup, p_play = weekly_override(season, self.week) if live else (none, none)
        if len(p_play):                   # the weekly model's read of the injury report
            self.hurt0 = 1 - p_play
        self.cache, self.memo = {}, {}
        self.waiver = self._waiver_line()
        self.opp_mu, self.opp_var = self._average_team()

    def _waiver_line(self):
        """Best player nobody rosters, per position: a weekly stream, always available.

        With real rosters in hand the honest answer is the best genuinely unrostered
        player, which is what settings.replacement == "pool" uses. The published default
        instead counts down an assumed number of players rostered per team, which is
        kept because changing it would move results that are already published.
        """
        out = {}
        for p in POS:
            exp = self.v[self.v.position == p].ppg * self.v.avail
            if SET.replacement == "pool" and self.rostered is not None:
                free = exp[~exp.index.isin(self.rostered)].sort_values(ascending=False)
                out[p] = float(free.iloc[0]) if len(free) else 0.0
                continue
            ranked = exp.sort_values(ascending=False)
            k = int(round(SET.teams * SET.rostered_mult[p]))
            out[p] = float(ranked.iloc[min(k, len(ranked) - 1)])
        return out

    def _average_team(self):
        """A league-average lineup: each starting slot filled by the average starter."""
        exp = self.v.ppg * self.v.avail
        mu = var = 0.0
        used = {}
        for p, k in SLOTS.items():
            n = SET.teams * k
            top = exp[self.v.position == p].nlargest(n)
            used[p] = n
            mu += k * top.mean()
            var += k * sd_of(p, top.mean()) ** 2
        flex_pool = pd.concat([exp[self.v.position == p].sort_values(ascending=False)
                               .iloc[used[p]:] for p in FLEX])
        top = flex_pool.nlargest(SET.teams * SET.flex_slots)
        mu += SET.flex_slots * top.mean()
        var += SET.flex_slots * sd_of("WR", top.mean()) ** 2
        return mu, var

    def player(self, pid):
        """Simulated points-if-started, sims x weeks; NaN when on bye or hurt."""
        if pid in self.cache:
            return self.cache[pid]
        r = self.v.loc[pid]
        rng = np.random.default_rng([self.seed, zlib.crc32(pid.encode())])
        out = A.injury_path(rng, self.sims, self.W, r.avail,
                            self.hurt0.get(pid, 0.0), r.status in ("RES", "PUP", "NFI"))
        bye = np.array([r.team not in self.plays[w] for w in self.weeks])
        mean = np.full(self.W, r.ppg)
        if self.W and pid in self.matchup.index:
            mean[0] = self.matchup[pid]
        pts = np.where(out | bye[None, :], np.nan, mean[None, :])
        self.cache[pid] = pts
        return pts

    def lineup(self, roster):
        """Expected lineup points and variance, sims x weeks, starting the best available."""
        S, W = self.sims, self.W
        mu = np.zeros((S, W))
        var = np.zeros((S, W))
        nxt = {}                                   # best non-starter per flex position
        for p, k in SLOTS.items():
            ids = [x for x in roster if self.v.at[x, "position"] == p]
            cols = [np.nan_to_num(self.player(x), nan=-np.inf) for x in ids]
            cols += [np.full((S, W), self.waiver[p])] * (k + 1)
            m = -np.sort(-np.stack(cols, axis=-1), axis=-1)
            mu += m[..., :k].sum(-1)
            var += (sd_of(p, m[..., :k]) ** 2).sum(-1)
            nxt[p] = m[..., k]
        for _ in range(SET.flex_slots):
            cand = np.stack([nxt[p] for p in FLEX], axis=-1)
            pick = cand.argmax(-1)
            flex = np.take_along_axis(cand, pick[..., None], -1)[..., 0]
            mu += flex
            var += np.choose(pick, [sd_of(p, flex) ** 2 for p in FLEX])
            # A second flex slot takes the next man at whichever position just supplied one.
            for j, p in enumerate(FLEX):
                nxt[p] = np.where(pick == j, -np.inf, nxt[p])
        if SET.correlations:
            var = var + self._stack_var(roster)
        return mu, var

    def _stack_var(self, roster):
        """Extra lineup variance from starting a quarterback with his own receiver.

        Priced on the expected starters rather than per simulated week, which is an
        approximation: the pair is counted whenever both would normally start, not only
        in the scenarios where they actually do. The correlation itself is an assumption
        and is off by default. See correlate.py.
        """
        if not len(roster):
            return 0.0
        v = self.v.loc[list(roster)]
        exp = v.ppg * v.avail
        pairs = []
        for qb, mates in CO.stacks(v.position, v.team, list(roster)):
            best = exp[mates].idxmax()
            pairs.append((("QB", float(exp[qb])), (v.at[best, "position"], float(exp[best]))))
        return CO.stack_variance(pairs, lambda p, m: sd_of(p, m))

    def outlook(self, roster):
        key = frozenset(roster)
        if key not in self.memo:
            mu, var = self.lineup(roster)
            win = norm.cdf((mu - self.opp_mu) / np.sqrt(var + self.opp_var))
            self.memo[key] = {"wins": (win * self.reg).sum(1),           # per sim
                              "pts": mu[:, self.reg].sum(1),
                              "playoff_pts": mu[:, ~self.reg].sum(1)}
        return self.memo[key]

    def surplus(self, pid):
        """Rough standalone worth: expected points over the waiver line."""
        r = self.v.loc[pid]
        return (r.ppg - self.waiver[r.position]) * r.avail


# ---------------------------------------------------------------- trades

def drop_for_space(sim, roster, n, exact=True):
    """Cut the n players whose loss costs the least (a full roster is assumed). The
    exact version simulates each cut; the quick one drops the lowest surplus.

    The cut is taken from the players roster.py says are cuttable, so a forced cut can
    never leave the team unable to field a legal starting lineup.
    """
    for _ in range(n):
        can = R.cuttable(sim.v.position, roster) or list(roster)
        if exact:
            base = sim.outlook(roster)["wins"].mean()
            cost = {x: base - sim.outlook([y for y in roster if y != x])["wins"].mean()
                    for x in can}
        else:
            cost = {x: sim.surplus(x) for x in can}
        roster = [y for y in roster if y != min(cost, key=cost.get)]
    return roster


def apply(sim, roster, out, inc, exact=True):
    after = [x for x in roster if x not in out] + list(inc)
    extra = max(0, len(inc) - len(out))
    kept = drop_for_space(sim, after, extra, exact) if extra else after
    return kept, [x for x in after if x not in kept]


def evaluate(sim, mine, theirs, give, get, exact=True):
    """Change in each side's outlook, with standard errors from the paired sims."""
    res = {}
    for side, roster, out, inc in (("me", mine, give, get), ("them", theirs, get, give)):
        before = sim.outlook(roster)
        after_roster, dropped = apply(sim, roster, out, inc, exact)
        after = sim.outlook(after_roster)
        d = {k: after[k] - before[k] for k in before}
        res[side] = {k: (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) for k, v in d.items()}
        res[side]["dropped"] = dropped
    return res


def names(v, ids):
    return ", ".join(v.loc[list(ids)].player_display_name) or "-"


def report(v, res, give, get):
    print(f"\nyou give: {names(v, give)}\nyou get:  {names(v, get)}\n")
    print(f"{'':6s} {'wins':>14s} {'pts, weeks to ' + str(REG_END):>20s} {'playoff pts':>14s}")
    for side in ("me", "them"):
        r = res[side]
        cells = [f"{r[k][0]:+.2f} ± {r[k][1]:.2f}" if k == "wins" else f"{r[k][0]:+.1f}"
                 for k in ("wins", "pts", "playoff_pts")]
        label = "you" if side == "me" else "them"
        print(f"{label:6s} {cells[0]:>14s} {cells[1]:>20s} {cells[2]:>14s}")
        if r["dropped"]:
            print(f"{'':6s} (must drop {names(v, r['dropped'])} to make room)")
    mine, theirs = res["me"]["wins"][0], res["them"]["wins"][0]
    verdict = ("accept - it helps you" if mine > 0.02 else
               "decline - it costs you" if mine < -0.02 else "a wash for you")
    print(f"\n>>> {verdict}" + ("; and it helps them too, so it should get done" if
                                mine > 0.02 and theirs > 0 else ""))


def relevant(sim, roster, n=10):
    """Players worth trading: above the waiver line, most valuable first."""
    gain = {x: (sim.v.at[x, "ppg"] - sim.waiver[sim.v.at[x, "position"]]) * sim.v.at[x, "avail"]
            for x in roster}
    return [x for x in sorted(gain, key=gain.get, reverse=True) if gain[x] > 0][:n]


def suggest(v, season, me, teams, partner=None, top=10):
    """Trades where both sides gain wins: screen cheaply, then re-run the best carefully."""
    held = {p for r in teams.values() for p in r}
    screen = Season(v, season, sims=300, seed=1, rostered=held)
    exact = Season(v, season, sims=3000, seed=2, rostered=held)
    if not trade_window_open(screen.week):
        return
    mine = teams[me]
    rows = []
    for t, theirs in teams.items():
        if t == me or (partner and t != partner):
            continue
        a, b = relevant(screen, mine), relevant(screen, theirs)
        shapes = [(1, 1), (2, 1), (1, 2), (2, 2)]
        for ng, nr in shapes:
            for give in itertools.combinations(a, ng):
                for get in itertools.combinations(b, nr):
                    r = evaluate(screen, mine, theirs, give, get, exact=False)
                    if r["me"]["wins"][0] > 0 and r["them"]["wins"][0] > 0:
                        rows.append((t, give, get, r["me"]["wins"][0], r["them"]["wins"][0]))
    rows.sort(key=lambda x: -min(x[3], x[4] * 2))   # good for me, and a real gain for them
    out = []
    for t, give, get, *_ in rows[:top * 3]:
        r = evaluate(exact, mine, teams[t], give, get)
        if r["me"]["wins"][0] > 0 and r["them"]["wins"][0] > 0:
            out.append((t, give, get, r))
    out.sort(key=lambda x: -x[3]["me"]["wins"][0])
    print(f"\n=== trades that help both sides (wins added, rest of season) ===")
    if not out:
        print("none found - your roster and theirs don't have complementary needs")
    for t, give, get, r in out[:top]:
        print(f"\nwith {t}:  give {names(v, give)}  |  get {names(v, get)}")
        print(f"    you {r['me']['wins'][0]:+.2f} wins ({r['me']['pts'][0]:+.0f} pts)   "
              f"them {r['them']['wins'][0]:+.2f} wins ({r['them']['pts'][0]:+.0f} pts)")


def trade_window_open(week, s=SET):
    """Whether a trade may still be made. Past the deadline, or once the playoffs have
    started, the answer is no however good the trade looks."""
    if week > s.trade_deadline_week:
        print(f"\nthe trade deadline was week {s.trade_deadline_week}; it is week {week}.")
        return False
    if week > s.reg_weeks and not s.trade_in_playoffs:
        print(f"\nthe playoffs have started; no trades after week {s.reg_weeks}.")
        return False
    return True


def resolve(v, text_or_list):
    names_ = text_or_list if isinstance(text_or_list, list) else text_or_list.split(",")
    ids, missing = match(v, ",".join(names_))
    for n in missing:
        print(f"  ! no projection for {n!r} - ignored (not on an active roster, "
              f"or spelled differently)")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--league", help="league.json with every roster")
    ap.add_argument("--with", dest="partner", help="the other team, by its name in league.json")
    ap.add_argument("--mine", default="")
    ap.add_argument("--theirs", default="")
    ap.add_argument("--give", default="")
    ap.add_argument("--get", default="")
    ap.add_argument("--suggest", action="store_true")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--values", choices=["consensus", "model"], default="consensus",
                    help="points per game from consensus ROS ranks (default) or the model")
    ap.add_argument("--settings", help="JSON of league settings; see draft/settings.py")
    a = ap.parse_args()

    v = load_values(a.season, a.values)
    if a.league:
        lg = json.load(open(a.league))
        teams = {t: resolve(v, r) for t, r in lg["teams"].items()}
        me = lg["me"]
    else:
        teams = {"me": resolve(v, a.mine), "them": resolve(v, a.theirs)}
        me, a.partner = "me", "them"

    if a.suggest:
        return suggest(v, a.season, me, teams, a.partner)

    sim = Season(v, a.season, sims=a.sims,
                 rostered={p for r in teams.values() for p in r})
    trade_window_open(sim.week)      # reported, not enforced: an offer can still be priced
    give, get = resolve(v, a.give), resolve(v, a.get)
    theirs = teams[a.partner]
    for x in give:
        if x not in teams[me]:
            sys.exit(f"{names(v, [x])} isn't on your roster")
    for x in get:
        if x not in theirs:
            sys.exit(f"{names(v, [x])} isn't on {a.partner}'s roster")
    print(f"week {sim.week} of {LAST_WEEK}; league-average lineup scores "
          f"{sim.opp_mu:.0f} a week; waiver line "
          + ", ".join(f"{p} {x:.1f}" for p, x in sim.waiver.items()))
    report(v, evaluate(sim, teams[me], theirs, give, get), give, get)


if __name__ == "__main__":
    main()
