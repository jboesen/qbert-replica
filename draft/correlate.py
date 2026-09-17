"""Game correlations: a quarterback scores with his own receiver, and a defence
suppresses both.

Every simulator in this repo draws players independently and prices each one against a
league-average opponent. Both are wrong in the same direction. A quarterback and the
receiver he throws to share the same drives, so a lineup holding both swings wider than
its slot-by-slot variance says, which matters for any win probability computed off that
spread. And a week against the best defence in the league is not the same week as one
against the worst.

This is an assumption, not a fix. The correlation below is a plausible figure rather
than one fit to this data, and the defence effect is measured but its size as a
multiplier on a projection is a modelling choice. Switching it on will move results, so
it defaults off in settings and every caller is gated on that flag.
"""
import numpy as np
import pandas as pd

import settings as CFG
import weekly as W

RECEIVERS = ("WR", "TE")


def stacks(position, team, roster):
    """(quarterback, own receivers) for each quarterback on a roster.

    `position` and `team` map a player to his position and NFL team; both are indexable
    the way the live tools' Series are and the harness's arrays are.
    """
    out = []
    qbs = [i for i in roster if position[i] == "QB"]
    for q in qbs:
        mates = [i for i in roster
                 if position[i] in RECEIVERS and team[i] == team[q] and team[i]]
        if mates:
            out.append((q, mates))
    return out


def stack_variance(pairs, sd_of, s=None):
    """Extra lineup variance from starting a quarterback alongside his own receiver.

    `pairs` is a list of (sd-bearing mean of the quarterback, mean of the receiver) for
    each stacked pair actually started. Two correlated scores have variance
    var_a + var_b + 2*rho*sd_a*sd_b, so only the cross term is added here.
    """
    s = s or CFG.get()
    return sum(2 * s.qb_receiver_rho * sd_of(a[0], a[1]) * sd_of(b[0], b[1])
               for a, b in pairs)


def defence_ratings(season, before_week=None, s=None):
    """How many PPR points each defence allows at each position, above or below average.

    For every game already played, an offence's points at a position are compared with
    that offence's own average at the position so far, and a defence's rating is the
    mean of those residuals. Comparing against the offence's own average keeps a defence
    from looking good merely for having faced weak offences. Only games strictly before
    `before_week` are read, so a rating never sees the week it is used to adjust.

    Returns {(defence, position): points above average allowed}.
    """
    st = pd.read_parquet(f"data/stats/w{season}.parquet")
    st = st[(st.season_type == "REG") & st.position.isin(CFG.POSITIONS)]
    if before_week is not None:
        st = st[st.week < before_week]
    if not len(st):
        return {}
    st = st.assign(team=W.norm_team(st.team), opp=W.norm_team(st.opponent_team))
    gm = st.groupby(["week", "team", "opp", "position"]).fantasy_points_ppr.sum().reset_index()
    own = gm.groupby(["team", "position"]).fantasy_points_ppr.transform("mean")
    gm = gm.assign(resid=gm.fantasy_points_ppr - own)
    r = gm.groupby(["opp", "position"]).resid.mean()
    return {k: float(v) for k, v in r.items()}


def defence_multiplier(ratings, position, opponent, base, s=None):
    """What facing `opponent` does to a projection of `base` points at `position`.

    The rating is points above average allowed; scaling it by the effect size and
    dividing by the projection turns it into a multiplier. An effect of 0, the default,
    leaves every projection exactly where it was.
    """
    s = s or CFG.get()
    if not s.correlations or not s.opp_defence_effect or base <= 0:
        return 1.0
    r = ratings.get((opponent, position))
    if r is None:
        return 1.0
    return float(np.clip(1 + s.opp_defence_effect * r / base, 0.5, 1.5))


def team_shock(rng, teams, draws, s=None):
    """A shared per-team scenario shock, as standard normal draws: {team: (draws,)}.

    Blending a share of this into each player's own draw is what makes a quarterback and
    his receiver move together across scenarios, without changing either one's mean.
    """
    s = s or CFG.get()
    return {t: rng.standard_normal(draws) for t in set(teams) if t}


def blend(own, shared, sd, s=None):
    """Mix a player's own residual draw with his team's shared shock, keeping the spread.

    Weighting by sqrt(rho) and sqrt(1-rho) leaves the total variance unchanged, so the
    only thing the correlation changes is how players move together.
    """
    s = s or CFG.get()
    rho = s.qb_receiver_rho
    return np.sqrt(1 - rho) * own + np.sqrt(rho) * sd * shared


def weekly_teams(ids, season, weeks):
    """player x week NFL team, as of the latest weekly roster row at or before the week.

    The same rows the availability grid reads, so a player traded mid-season is on the
    right team in the right week and his stack partners change with him.
    """
    ix = {p: i for i, p in enumerate(ids)}
    ro = pd.read_parquet(f"data/roster_weekly_{season}.parquet")
    ro = ro[(ro.game_type == "REG") & (ro.week <= weeks) & ro.gsis_id.isin(ix)]
    ro = ro.assign(team=W.norm_team(ro.team)).sort_values("week")
    rows = {w: list(zip(x.gsis_id, x.team)) for w, x in ro.groupby("week")}
    cur = np.array([""] * len(ids), dtype=object)
    out = []
    for w in range(1, weeks + 1):
        for p, t in rows.get(w, []):
            cur[ix[p]] = t
        out.append(cur.copy())
    return out
