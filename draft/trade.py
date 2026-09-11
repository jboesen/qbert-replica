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
from vbd import LEAGUE
from assistant import match

POS = ["QB", "RB", "WR", "TE"]
FLEX = list(LEAGUE["flex"])
SLOTS = LEAGUE["starters"]
# Weekly PPR sd as a function of the player's mean, per position (player-seasons with 8+
# games, 2018-25). Scoring is right-skewed for RB/WR/TE, so this is a rough spread only.
SD = {"QB": (5.22, .14), "RB": (2.43, .38), "WR": (2.27, .41), "TE": (1.57, .49)}
RECOVER = 0.45         # weekly chance a hurt player returns; mean absence about two games
IR_WEEKS = 4           # players on reserve/PUP sit out at least this long
# Skill players carried per team, for where the waiver wire sits.
ROSTERED = {"QB": 1.5, "RB": 4.5, "WR": 5.0, "TE": 1.5}
REG_END, LAST_WEEK = 14, 17     # fantasy regular season, then playoffs 15-17
# Chance a player misses the week, by game designation (2016-25 play rates). Only used
# when the weekly projection, which carries its own p_play, hasn't been built.
STATUS_OUT = {"Out": 1.0, "Doubtful": 0.99, "Questionable": 0.43}


def sd_of(pos, mean):
    a, b = SD[pos]
    return a + b * mean


# ---------------------------------------------------------------- inputs

def load_values(season):
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
        v["avail"] = b.proj_games / 17
    v["avail"] = v.avail.clip(0.3, 0.98)

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
    try:
        inj = pd.read_parquet(f"data/injuries_{season}.parquet")
    except FileNotFoundError:
        return pd.Series(dtype=float)
    inj = inj[inj.week == inj.week.max()].drop_duplicates("gsis_id", keep="last")
    return inj.set_index("gsis_id").report_status.map(STATUS_OUT).dropna()


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

    def __init__(self, values, season, sims=2000, seed=0, start=None, live=True):
        """`live` reads this week's injury report and matchup projections; a backtest
        of a past season turns it off and starts wherever it likes."""
        self.v, self.sims, self.seed = values, sims, seed
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
        """Best player nobody rosters, per position: a weekly stream, always available."""
        out = {}
        for p in POS:
            ranked = (self.v[self.v.position == p].ppg * self.v.avail).sort_values(ascending=False)
            k = int(round(LEAGUE["teams"] * ROSTERED[p]))
            out[p] = float(ranked.iloc[min(k, len(ranked) - 1)])
        return out

    def _average_team(self):
        """A league-average lineup: each starting slot filled by the average starter."""
        exp = self.v.ppg * self.v.avail
        mu = var = 0.0
        used = {}
        for p, k in SLOTS.items():
            n = LEAGUE["teams"] * k
            top = exp[self.v.position == p].nlargest(n)
            used[p] = n
            mu += k * top.mean()
            var += k * sd_of(p, top.mean()) ** 2
        flex_pool = pd.concat([exp[self.v.position == p].sort_values(ascending=False)
                               .iloc[used[p]:] for p in FLEX])
        top = flex_pool.nlargest(LEAGUE["teams"] * LEAGUE["flex_slots"])
        mu += LEAGUE["flex_slots"] * top.mean()
        var += LEAGUE["flex_slots"] * sd_of("WR", top.mean()) ** 2
        return mu, var

    def player(self, pid):
        """Simulated points-if-started, sims x weeks; NaN when on bye or hurt."""
        if pid in self.cache:
            return self.cache[pid]
        r = self.v.loc[pid]
        rng = np.random.default_rng([self.seed, zlib.crc32(pid.encode())])
        u = rng.random((self.sims, self.W))
        a = r.avail
        hazard = min(RECOVER * (1 - a) / a, 0.9)      # stationary availability = a
        hurt = rng.random(self.sims) < self.hurt0.get(pid, 0.0)
        ir = r.status in ("RES", "PUP", "NFI")
        out = np.empty((self.sims, self.W), bool)
        for i in range(self.W):
            if ir and i < IR_WEEKS:
                hurt = np.ones(self.sims, bool)
            out[:, i] = hurt
            hurt = np.where(hurt, u[:, i] >= RECOVER, u[:, i] < hazard)
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
        cand = np.stack([nxt[p] for p in FLEX], axis=-1)
        pick = cand.argmax(-1)
        flex = np.take_along_axis(cand, pick[..., None], -1)[..., 0]
        mu += flex
        var += np.choose(pick, [sd_of(p, flex) ** 2 for p in FLEX])
        return mu, var

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
    exact version simulates each cut; the quick one drops the lowest surplus."""
    for _ in range(n):
        if exact:
            base = sim.outlook(roster)["wins"].mean()
            cost = {x: base - sim.outlook([y for y in roster if y != x])["wins"].mean()
                    for x in roster}
        else:
            cost = {x: sim.surplus(x) for x in roster}
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
    screen = Season(v, season, sims=300, seed=1)
    exact = Season(v, season, sims=3000, seed=2)
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
    a = ap.parse_args()

    v = load_values(a.season)
    if a.league:
        lg = json.load(open(a.league))
        teams = {t: resolve(v, r) for t, r in lg["teams"].items()}
        me = lg["me"]
    else:
        teams = {"me": resolve(v, a.mine), "them": resolve(v, a.theirs)}
        me, a.partner = "me", "them"

    if a.suggest:
        return suggest(v, a.season, me, teams, a.partner)

    sim = Season(v, a.season, sims=a.sims)
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
