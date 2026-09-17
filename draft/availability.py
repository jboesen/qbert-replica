"""Who can play, defined once.

Availability used to be written down three times and the three did not agree. The
harness read roster status and the injury report directly; lineup.py applied its own
Questionable discount; trade.py ran a two-state injury chain with its own recovery rate
and its own clipping. A player could be unavailable to one tool and fine to another on
the same day.

There is one definition here and every consumer calls it. A player is unavailable for a
week when he is off his team's active roster, or ruled out on the last report that was
published before the decision, or his team is on bye. That predicate is the whole model;
what differs between the harness and the live tools is only where the three inputs come
from, so the two sources are adapters onto the same function rather than two models.

The forward-looking piece, the two-state injury chain the trade evaluator simulates, is
also here, parameterised by the same settings.
"""
import numpy as np
import pandas as pd

import settings as CFG
import weekly as W


# ---------------------------------------------------------------- the one predicate

def is_unavailable(roster_status, hurt, team, plays, s=None):
    """The definition. `hurt` is already resolved against the report that was current.

    A player with no roster row at all counts as unavailable: he is not on anyone's 53
    as far as the decision can see.
    """
    s = s or CFG.get()
    if roster_status is None:
        return True
    return (roster_status not in s.active_statuses) or hurt or (team not in plays)


def sits(status, s=None):
    """Whether a game designation means the player does not play at all."""
    s = s or CFG.get()
    return status in s.out_statuses


def play_discount(status, s=None):
    """What a game designation does to a player's expected points when he is not ruled
    out. Questionable players have played 57% of the time, so they are worth that much."""
    s = s or CFG.get()
    if sits(status, s):
        return 0.0
    return s.p_questionable if status == "Questionable" else 1.0


# ---------------------------------------------------------------- source: past seasons

def harness_grid(ids, y, weeks, s=None):
    """player x week: unavailable at the waiver decision before each week, for a past
    season. Row w-1 is the decision made before week w.

    The harness's own eligibility reads week w's roster status and injury report, which
    publish after waivers clear, so a waiver policy cannot use them. What it can see is
    the schedule and last week's files. Teams on bye have no roster rows, so a player
    whose team was off in week w-1 is read from his last week before that.
    """
    s = s or CFG.get()
    ix = {p: i for i, p in enumerate(ids)}
    n = len(ids)
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG")]
    plays = {(w, t) for w, a, h in zip(g.week, g.away_team, g.home_team) for t in (a, h)}
    inj = pd.read_parquet(f"data/injuries_{y}.parquet")
    out = {(p, w) for p, w, st in zip(inj.gsis_id, inj.week, inj.report_status)
           if st in s.out_statuses}
    ro = pd.read_parquet(f"data/roster_weekly_{y}.parquet")
    ro = ro[(ro.game_type == "REG") & (ro.week <= weeks) & ro.gsis_id.isin(ix)]
    ro = ro.assign(team=W.norm_team(ro.team)).sort_values("week")
    rows = {w: list(zip(x.gsis_id, x.status, x.team)) for w, x in ro.groupby("week")}
    last = {}                                 # player -> (week, status, team), latest seen
    gone = np.ones((n, weeks), bool)          # no roster row yet counts as unavailable
    for w in range(s.first_waiver_week, weeks + 1):
        for p, st, t in rows.get(w - 1, []):
            last[p] = (w - 1, st, t)
        slate = {t for (j, t) in plays if j == w}
        for p, (wk, st, t) in last.items():
            # The injury read is as of the player's own last roster row, which is the
            # last report that mentioned him before the decision.
            gone[ix[p], w - 1] = is_unavailable(st, (p, wk) in out, t, slate, s)
    return gone


# ---------------------------------------------------------------- source: this week

def live_reasons(y, w, ids, rosters, s=None):
    """{player: why} for the live tools: the same predicate on this week's files.

    `rosters` is the weekly roster frame, passed in because the live tools fetch it and
    cache it themselves. Unlike the harness, the report read here is the league-wide
    latest one rather than each player's own last roster row: live, every player has a
    current report, so there is nothing to fall back to.
    """
    s = s or CFG.get()
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG") & (g.week == w)]
    playing = set(g.away_team) | set(g.home_team)
    ro = rosters[(rosters.game_type == "REG") & (rosters.week < w)].sort_values("week")
    last = ro.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
    try:
        inj = pd.read_parquet(f"data/injuries_{y}.parquet")
        inj = inj[inj.week < w]
        inj = inj[inj.week == inj.week.max()] if len(inj) else inj
        hurt = set(inj[inj.report_status.isin(s.out_statuses)].gsis_id)
    except FileNotFoundError:
        hurt = set()
    why = {}
    for p in ids:
        if p not in last.index:
            why[p] = "no roster row"
            continue
        r = last.loc[p]
        team = W.norm_team(pd.Series([r.team])).iloc[0]
        if r.status not in s.active_statuses:
            why[p] = f"roster status {r.status}"
        elif team not in playing:
            why[p] = "bye"
        elif p in hurt:
            why[p] = "out/doubtful last report"
    return why


def report_status(y, s=None, week=None):
    """This week's chance of missing, by game designation, for anyone on the report."""
    s = s or CFG.get()
    try:
        inj = pd.read_parquet(f"data/injuries_{y}.parquet")
    except FileNotFoundError:
        return pd.Series(dtype=float)
    week = inj.week.max() if week is None else week
    inj = inj[inj.week == week].drop_duplicates("gsis_id", keep="last")
    return inj.set_index("gsis_id").report_status.map(s.status_out).dropna()


# ---------------------------------------------------------------- forward simulation

def clip_rate(avail, s=None):
    """Keep a season-long availability rate inside a believable band.

    Nobody is a certainty and nobody is hopeless, and the chain below divides by this
    rate, so the floor also keeps the hazard finite.
    """
    s = s or CFG.get()
    return avail.clip(s.avail_floor, s.avail_ceiling)


def injury_path(rng, sims, weeks, avail, hurt0, on_ir, s=None):
    """Simulated weekly injury states, sims x weeks, as a two-state chain.

    A single weekly coin flip would scatter missed games one at a time; real absences
    run. So a healthy player falls out at a hazard chosen to make the chain's stationary
    availability equal his projected rate, and a hurt one returns with probability
    `recover`. Reserve and PUP players are held out for a fixed spell first.

    The recovery rate is an assumption tuned to a mean absence of about two games, not a
    quantity fit to the injury data.
    """
    s = s or CFG.get()
    u = rng.random((sims, weeks))
    hazard = min(s.recover * (1 - avail) / avail, 0.9)     # stationary availability = avail
    hurt = rng.random(sims) < hurt0
    out = np.empty((sims, weeks), bool)
    for i in range(weeks):
        if on_ir and i < s.ir_weeks:
            hurt = np.ones(sims, bool)
        out[:, i] = hurt
        hurt = np.where(hurt, u[:, i] >= s.recover, u[:, i] < hazard)
    return out
