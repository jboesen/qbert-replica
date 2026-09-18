"""Every league rule and modelling assumption in one place, so the model can be pointed
at a different league and so an assumption can be varied and the change trusted.

These knobs used to live as constants in whichever module needed them, which meant the
same rule was written down more than once and the copies had drifted: the draft capped
rosters at 2 QB and 7 RB, the in-season wire enforced no cap at all, and the trade tool
priced replacement off a third set of numbers again. A single object fixes the drift by
construction, because there is now only one place to write a rule down.

Defaults reproduce the published harness exactly. Nothing here changes a number unless
it is overridden, and anything that would change a result is a switch that defaults to
today's behaviour. Overrides come from a JSON file, either a bare settings object or a
"settings" key inside the existing league.json, selected by `--settings PATH` on the
command line or the FF_SETTINGS environment variable:

    .venv/bin/python draft/waivers.py --league league.json --settings my_league.json

In code, build a variant from the defaults:

    from settings import Settings, activate
    activate(Settings().replace(teams=10, flex_slots=3))

Ordering matters and is load bearing. `starters` is filled in its own insertion order
when a lineup is set, and `flex` is scanned in its own order when replacement level is
computed, so both are kept as ordered structures rather than sets.
"""
import json
import os
import sys
from dataclasses import dataclass, field, replace

POSITIONS = ("QB", "RB", "WR", "TE")


@dataclass(frozen=True)
class Settings:
    """One league's rules plus the modelling assumptions layered on top of them.

    Fields are grouped the way a manager would think about them: the league, the roster,
    the calendar, the wire, trades, injuries, and the model's own choices. A field whose
    value is an assumption rather than a measured quantity says so in its comment.
    """

    # ---- the league
    teams: int = 12
    scoring: str = "ppr"                     # only PPR is implemented; see __post_init__
    starters: dict = field(default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1})
    flex: tuple = ("RB", "WR", "TE")
    flex_slots: int = 1

    # ---- the roster
    rounds: int = 14                         # also the roster size; no IR slot
    roster_size: int = 14
    # At most two QBs and two TEs, at most seven RBs or WRs. This is how draft simulators
    # keep bots sane rather than a rule most real leagues have.
    caps: dict = field(default_factory=lambda: {"QB": 2, "TE": 2, "RB": 7, "WR": 7})
    one_each: tuple = ("QB", "TE")           # positions whose second copy is held back
    second_after: int = 9                    # ...until this round
    # A team may never cut below a legal starting lineup. Defaults to the starters map.
    roster_min: dict = None

    # Caps are enforced at the draft in the published harness and nowhere else. Turning
    # this on enforces them on the wire and in trades too, which is the consistent rule
    # but moves published numbers, so it is off by default.
    enforce_caps_in_season: bool = False
    # When a roster would exceed its size (an add, or a player back from reserve), cut by
    # this rule. "fewest_holes" is the streaming policy's rule; "lowest_value" is the
    # consensus wire's. Only applies where the harness already cuts, unless
    # enforce_caps_in_season is on.
    cut_rule: str = "fewest_holes"

    # ---- the calendar
    reg_weeks: int = 14                      # head-to-head weeks
    weeks: int = 17                          # last week scored
    playoff_teams: int = 6
    playoff_byes: int = 2
    # The published bracket reseeds after the bye round: the lower of the two first-round
    # winners plays the #1 seed. Turning this off keeps a fixed bracket instead.
    playoff_reseed: bool = True
    seasons: tuple = (2021, 2022, 2023, 2024, 2025)

    # ---- the wire
    waiver_mode: str = "priority"            # worst record first; "faab" is not wired up yet
    waiver_moves_per_week: int = 1
    first_waiver_week: int = 2               # the first decision follows week 1
    faab_budget: int = 100                   # only read when waiver_mode == "faab"

    # ---- trades
    trade_first_week: int = 3
    trade_deadline_week: int = 11            # last week a trade may be made
    trade_cap: int = None                    # executed trades per season; None is uncapped
    # Rest-of-season lineup points a capped trade must project before a scarce trade is
    # spent on it. This is tradecap's frozen bar, declared before that run and not tuned.
    trade_gain_min: float = 10.0
    trade_near_best: float = 0.9             # "equal value" band in the tiebreak search
    trade_in_playoffs: bool = False          # no trades once the playoffs start

    # ---- injuries and availability
    out_statuses: tuple = ("Out", "Doubtful")
    active_statuses: tuple = ("ACT", "INA")  # on the 53; INA is the gameday inactive list
    # Share of games a Questionable player has actually played, 2016-25. Measured.
    p_questionable: float = 0.57
    # Weekly chance a hurt player returns. Assumption, tuned to a mean absence of about
    # two games; not fit to the injury data.
    recover: float = 0.45
    ir_weeks: int = 4                        # reserve/PUP players sit out at least this long
    avail_floor: float = 0.30                # availability is clipped into this band so a
    avail_ceiling: float = 0.98              # ...fragile star is not modelled as never or always
    # Chance a player misses the week by game designation (2016-25 play rates). Measured.
    status_out: dict = field(default_factory=lambda: {"Out": 1.0, "Doubtful": 0.99,
                                                      "Questionable": 0.43})

    # ---- the model's own choices
    # "multiplier" prices replacement off an assumed count of players rostered per team,
    # which is what the published results use. "pool" prices it off the best genuinely
    # unrostered player, which is right for a live tool that knows the real rosters.
    replacement: str = "multiplier"
    # Skill players carried per team, for where the waiver wire sits. An assumption, and
    # one that matches neither the draft caps nor the wire's minimums.
    rostered_mult: dict = field(default_factory=lambda: {"QB": 1.5, "RB": 4.5,
                                                         "WR": 5.0, "TE": 1.5})
    # Opponents' deviation from consensus, in units of the experts' own spread.
    noise: float = 1.0
    leagues: int = 20
    schedules: int = 20

    # ---- correlations (off by default: this one changes results by design)
    # A quarterback and his own receiver score together, and both are suppressed by a
    # strong opposing defence. Neither is measured here; the values below are plausible
    # rather than fit, so switching this on is taking an assumption, not a fix.
    correlations: bool = False
    qb_receiver_rho: float = 0.35
    opp_defence_effect: float = 0.0

    def __post_init__(self):
        if self.roster_min is None:
            object.__setattr__(self, "roster_min", dict(self.starters))
        for p in self.flex:
            if p not in self.starters:
                raise ValueError(f"flex position {p!r} has no starting slot")
        if self.playoff_teams > self.teams:
            raise ValueError("more playoff teams than teams")
        if self.playoff_byes >= self.playoff_teams:
            raise ValueError("every playoff team has a bye")
        if self.reg_weeks > self.weeks:
            raise ValueError("regular season runs past the last scored week")
        if self.starting_slots > self.roster_size:
            raise ValueError("a legal starting lineup does not fit on the roster")
        if self.playoff_rounds > self.weeks - self.reg_weeks:
            raise ValueError(
                f"a {self.playoff_teams}-team bracket with {self.playoff_byes} byes needs "
                f"{self.playoff_rounds} weeks, but only {self.weeks - self.reg_weeks} are "
                "left after the regular season")
        if self.replacement not in ("multiplier", "pool"):
            raise ValueError(f"unknown replacement method {self.replacement!r}")
        if self.waiver_mode not in ("priority", "faab"):
            raise ValueError(f"unknown waiver mode {self.waiver_mode!r}")
        # Two knobs are declared because they are real league rules, but nothing reads
        # them yet. Refusing them is better than accepting a setting that does nothing.
        if self.waiver_mode == "faab":
            raise NotImplementedError(
                "the harness runs a priority wire; FAAB bidding lives in sim_faab.py "
                "and is not wired into the shared settings yet")
        if self.scoring != "ppr":
            raise NotImplementedError(
                f"only PPR scoring is implemented; {self.scoring!r} would need a new "
                "points model in fantasy.py")

    # ---- derived

    @property
    def starting_slots(self):
        return sum(self.starters.values()) + self.flex_slots

    @property
    def playoff_rounds(self):
        """How many weeks the bracket takes, byes included. Checked against the calendar
        at construction so a bad league fails when it is loaded, not mid-season."""
        alive, rounds = self.playoff_teams, 0
        while alive > 1:
            byes = self.playoff_byes if rounds == 0 else 0
            rest = alive - byes
            alive = byes + (rest + 1) // 2
            rounds += 1
        return rounds

    @property
    def playoff_weeks(self):
        """The weeks the bracket is played in, straight after the regular season."""
        return tuple(range(self.reg_weeks + 1, self.weeks + 1))

    @property
    def trade_weeks(self):
        """Decision weeks a trade may be made before, deadline included."""
        last = min(self.trade_deadline_week,
                   self.weeks if self.trade_in_playoffs else self.reg_weeks)
        return range(self.trade_first_week, last + 1)

    def league_dict(self):
        """The shape vbd.LEAGUE has always had, for code that still reads a mapping."""
        return dict(teams=self.teams, starters=dict(self.starters), flex=tuple(self.flex),
                    flex_slots=self.flex_slots, rounds=self.rounds)

    def replace(self, **kw):
        """A variant of these settings, e.g. Settings().replace(teams=10, flex_slots=3)."""
        return replace(self, **kw)

    # ---- overrides

    @classmethod
    def from_dict(cls, d):
        """Build from a JSON-shaped dict, leaving anything unmentioned at its default."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        d = dict(d)
        for k in ("flex", "one_each", "out_statuses", "active_statuses", "seasons"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    @classmethod
    def from_file(cls, path):
        """Read a settings JSON, or a league.json carrying a "settings" key.

        A league.json without that key is an old file and stays valid: it describes only
        rosters, so it leaves every assumption at its default.
        """
        raw = json.load(open(path))
        if "settings" in raw:
            raw = raw["settings"]
        elif "teams" in raw and isinstance(raw["teams"], dict):
            raw = {}                          # a rosters-only league.json, no overrides
        s = cls.from_dict(raw)
        object.__setattr__(s, "_named", frozenset(raw))
        return s

    def changes(self):
        """Every field that differs from the defaults, as (name, default, mine)."""
        base = DEFAULTS
        out = []
        for f in self.__dataclass_fields__:
            mine, theirs = getattr(self, f), getattr(base, f)
            if mine != theirs:
                out.append((f, theirs, mine))
        return out

    def summary(self):
        """One line naming every assumption that is not the default, or None if none are.

        A result computed under changed assumptions must never be mistaken for a
        default-settings result, so every tool prints this before it does any work.
        """
        ch = self.changes()
        if not ch:
            return None
        return "settings overrides: " + ", ".join(f"{k}={v!r} (default {d!r})"
                                                  for k, d, v in ch)


DEFAULTS = Settings()
_ACTIVE = DEFAULTS


def get():
    """The settings every module reads. Bound at import, so overrides are applied first."""
    return _ACTIVE


def activate(s, announce=True):
    """Make `s` the active settings and say so if it differs from the defaults."""
    global _ACTIVE
    _ACTIVE = s
    if announce:
        line = s.summary()
        if line:
            print(line, file=sys.stderr, flush=True)
    return s


def announce_run(**knobs):
    """Report run-level knobs a CLI flag set directly, which bypass the settings object.

    `--noise` and `--leagues` rebind the harness's module globals rather than building a
    new Settings, because those two are properties of a run rather than of a league. They
    would otherwise never reach the override summary, so they are reported here instead.
    Printing only; it cannot change a number.
    """
    s = get()
    diff = [f"{k}={v!r} (default {getattr(s, k)!r})" for k, v in knobs.items()
            if getattr(s, k, None) != v]
    if diff:
        print("run overrides: " + ", ".join(diff), file=sys.stderr, flush=True)


# Two assumptions are right for a real league and wrong for the published backtest, so
# they cannot simply be defaults. Replacement should be the best player actually sitting
# in your league's free pool, not a count assumed from roster multipliers, and a roster
# that may not exceed its caps on draft day may not exceed them in November either. The
# harness keeps the published values because changing them would move printed results;
# the tools you point at a real league take these instead, and say so when they do.
LIVE_TOOLS = ("waivers.py", "trade.py", "lineup.py")
LIVE = dict(replacement="pool")


def live_profile(s):
    """`s` with the live-league assumptions, except any the settings file named itself.

    Caps are deliberately not in LIVE. The defaults, 2 QB and 2 TE, are values that keep
    simulated drafters sane rather than a rule real leagues have, and prereg_settings.md
    measured what enforcing them costs: 1.7 to 2.8 points of all-play, negative in nine
    of ten season-designs, because a cap of two forbids carrying the spare quarterback or
    tight end through a bye that hole-aware streaming wins on. A league that really does
    limit rosters says so by naming its own `caps`, and those are then enforced; ESPN's
    limits, 4/8/8/3, were harmless in both designs.
    """
    named = getattr(s, "_named", frozenset())
    out = s.replace(**{k: v for k, v in LIVE.items() if k not in named})
    if "caps" in named and "enforce_caps_in_season" not in named:
        out = out.replace(enforce_caps_in_season=True)
    return out


def _startup():
    """Apply `--settings PATH` or FF_SETTINGS at import, before any module reads a value.

    Modules bind their constants at import time, so the override has to be resolved
    before the first of them is imported. settings.py is the deepest import in the tree,
    which makes import time the one moment that is reliably early enough.
    """
    path = os.environ.get("FF_SETTINGS")
    if "--settings" in sys.argv:
        i = sys.argv.index("--settings") + 1
        if i >= len(sys.argv):
            raise SystemExit("--settings needs a path to a JSON file")
        path = sys.argv[i]
    s = Settings.from_file(path) if path else DEFAULTS
    # A live tool is one pointed at a real league, so it takes the live assumptions
    # unless the caller asked for the backtest's with --backtest-assumptions.
    live = (os.path.basename(sys.argv[0]) in LIVE_TOOLS
            and "--backtest-assumptions" not in sys.argv)
    if live:
        s = live_profile(s)
    if s is not DEFAULTS:
        activate(s)


_startup()
