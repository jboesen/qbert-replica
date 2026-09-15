"""Contingent value of a backup running back: what he would be worth if the back ahead
of him went out (prereg_handcuffs.md).

Consensus rest-of-season ranks price a backup on what he does now, so a manager who
follows them picks him up only after the starter is hurt, and a team already holding him
wins that race. This prices the contingency from box scores alone: who has led each
backfield lately, how often lead backs miss games, how likely the second back is to
inherit the lead role when the first is out, and what he scores in those games, as a
function of how much of the work he got while the first played. Every fit uses only
seasons strictly before the one being decided.

    .venv/bin/python draft/handcuff.py       # fit summaries for each holdout season
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, ".")
from build_data import norm_team

WEEKS = 17
FIRST_FIT = 2015            # the same first fit season as usage.py
RECENT = 4                  # team games in the window that ranks a backfield
TRANSFER = 0.5              # a game counts as inheriting the role at half the RB carries
MIN_P_TRANSFER = 0.5        # stash only backups more likely than not to inherit it


class Usage:
    """Running-back work per team game in one season, and the backfield order it implies.

    Work is carries plus targets. The order is read off the team's last RECENT games
    before the decision, not the season to date, so a lead back who has been out for a
    month stops being the lead and his fill-in stops being a backup.
    """

    def __init__(self, y):
        g = pd.read_csv("data/games.csv")
        g = g[(g.season == y) & (g.game_type == "REG") & (g.week <= WEEKS)]
        self.team_weeks = {}
        for w, a, h in zip(g.week, norm_team(g.away_team), norm_team(g.home_team)):
            for t in (a, h):
                self.team_weeks.setdefault(t, []).append(int(w))
        for t in self.team_weeks:
            self.team_weeks[t].sort()
        st = pd.read_parquet(f"data/stats/w{y}.parquet")
        st = st[(st.season_type == "REG") & (st.week <= WEEKS)]
        self.ppr = {(p, int(w)): v for (p, w), v in
                    st.groupby(["player_id", "week"]).fantasy_points_ppr.sum().items()}
        rb = st[st.position == "RB"]
        rb = rb.assign(team=norm_team(rb.team), opp=rb.carries.fillna(0) + rb.targets.fillna(0),
                       car=rb.carries.fillna(0))
        self.opp, self.car = {}, {}
        sums = rb.groupby(["team", "week", "player_id"])[["opp", "car"]].sum()
        for (t, w, p), o, c in zip(sums.index, sums.opp, sums.car):
            self.opp.setdefault((t, int(w)), {})[p] = o
            self.car.setdefault((t, int(w)), {})[p] = c
        self._states = {}

    def work(self, t, w, p):
        return self.opp.get((t, w), {}).get(p, 0.0)

    def share(self, t, w, p):
        tot = sum(self.opp.get((t, w), {}).values())
        return self.work(t, w, p) / tot if tot else 0.0

    def games_before(self, t, w):
        return [x for x in self.team_weeks.get(t, []) if x < w]

    def state(self, t, w):
        """The backfield before week w: lead, second and third back by work in the last
        RECENT team games, with the lead's share, the second's and third's shares in the
        games the lead played, and whether the lead played the team's last game."""
        if (t, w) in self._states:
            return self._states[(t, w)]
        win = self.games_before(t, w)[-RECENT:]
        tot = {}
        for x in win:
            for p, o in self.opp.get((t, x), {}).items():
                tot[p] = tot.get(p, 0.0) + o
        ranked = sorted((p for p in tot if tot[p] > 0), key=lambda p: (-tot[p], p))
        s = None
        if len(ranked) >= 2:
            lead, back = ranked[0], ranked[1]
            third = ranked[2] if len(ranked) > 2 else None
            played = [x for x in win if self.work(t, x, lead) > 0]
            team_played = sum(sum(self.opp[(t, x)].values()) for x in played)

            def in_lead_games(p):
                if p is None or not team_played:
                    return 0.0
                return sum(self.work(t, x, p) for x in played) / team_played
            s = dict(team=t, lead=lead, back=back, third=third,
                     s1=tot[lead] / sum(tot.values()), s2=in_lead_games(back),
                     s3=in_lead_games(third), last_played=self.work(t, win[-1], lead) > 0)
        self._states[(t, w)] = s
        return s


def fit_rows(y):
    """Fit rows from one past season.

    miss: at every decision where the lead played his team's last game, the fraction of
    the remaining weeks (through week 17) in which he has no work in a game his team
    plays. spells: every game of every absence, a run of team games in which the lead has
    no work right after one in which he played, with the backfield read before the
    absence began, the second back's PPR that game and whether he took the role.
    """
    u = Usage(y)
    miss, spells = [], []
    for t, weeks in u.team_weeks.items():
        for w in range(2, WEEKS + 1):
            s = u.state(t, w)
            if s is None or not s["last_played"]:
                continue
            rest = [x for x in weeks if x >= w]
            missed = sum(u.work(t, x, s["lead"]) == 0 for x in rest)
            miss.append({"week": w, "frac": missed / (WEEKS - w + 1)})
            if not rest or rest[0] != w or u.work(t, w, s["lead"]) > 0:
                continue                      # an absence starts only in a game played at w
            for x in rest:
                if u.work(t, x, s["lead"]) > 0:
                    break
                cars = sum(u.car.get((t, x), {}).values())
                share = u.car.get((t, x), {}).get(s["back"], 0.0) / cars if cars else 0.0
                spells.append({"season": y, "s2": s["s2"], "s3": s["s3"],
                               "ppr": u.ppr.get((s["back"], x), 0.0),
                               "transfer": float(share >= TRANSFER)})
    return pd.DataFrame(miss), pd.DataFrame(spells)


def fit_before(y, first=FIRST_FIT):
    """The three pieces of contingent value, fit only on seasons first..y-1:
    the lead's missed fraction of remaining weeks by decision week, P(the second back
    inherits the role | his and the third back's shares), and his PPR per game in games
    he does and does not inherit it, each linear in his share."""
    parts = [fit_rows(s) for s in range(first, y)]
    miss = pd.concat([p[0] for p in parts], ignore_index=True)
    sp = pd.concat([p[1] for p in parts], ignore_index=True)
    X = lambda d: sm.add_constant(d[["s2", "s3"]], has_constant="add")
    Xs = lambda d: sm.add_constant(d[["s2"]], has_constant="add")
    tr, no = sp[sp.transfer == 1], sp[sp.transfer == 0]
    return {
        "f_miss": miss.groupby("week").frac.mean(),
        "p_transfer": sm.Logit(sp.transfer, X(sp)).fit(disp=0),
        "pts_transfer": sm.OLS(tr.ppr, Xs(tr)).fit(),
        "pts_other": sm.OLS(no.ppr, Xs(no)).fit(),
        "n_spells": len(sp), "transfer_rate": sp.transfer.mean(),
    }


def season_states(y, fits):
    """Every team's backfield at every decision of season y (week = the upcoming week,
    read from games before it), with the second back's fitted P(transfer) and expected
    PPR per game as the fill-in. Only rows where the lead played his team's last game:
    a lead already out has no contingency left to price."""
    u = Usage(y)
    rows = []
    for w in range(2, WEEKS + 1):
        for t in u.team_weeks:
            s = u.state(t, w)
            if s is None or not s["last_played"]:
                continue
            rows.append(dict(s, week=w))
    d = pd.DataFrame(rows)
    p = fits["p_transfer"].predict(sm.add_constant(d[["s2", "s3"]], has_constant="add"))
    xs = sm.add_constant(d[["s2"]], has_constant="add")
    d["p"] = p.values
    d["fill"] = (d.p * fits["pts_transfer"].predict(xs).clip(lower=0)
                 + (1 - d.p) * fits["pts_other"].predict(xs).clip(lower=0)).values
    d["f_miss"] = d.week.map(fits["f_miss"]).values
    return u, d


if __name__ == "__main__":
    for y in range(2021, 2026):
        f = fit_before(y)
        print(f"{y}: fit {FIRST_FIT}-{y - 1}, {f['n_spells']} lead-out games, "
              f"transfer rate {f['transfer_rate']:.2f}")
        print("  P(transfer) logit:", f["p_transfer"].params.round(2).to_dict())
        print("  PPR if transfer:", f["pts_transfer"].params.round(2).to_dict(),
              " otherwise:", f["pts_other"].params.round(2).to_dict())
        print("  lead missed fraction of remaining weeks, by decision week:",
              " ".join(f"{v:.2f}" for v in f["f_miss"].values))
        _, d = season_states(y, f)
        ok = d[d.p >= MIN_P_TRANSFER]
        print(f"  decisions x teams {len(d)}, eligible backups {len(ok)}, "
              f"mean fill {ok.fill.mean():.1f} ppg")
