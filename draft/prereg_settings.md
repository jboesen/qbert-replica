# Preregistration: are the more correct settings also better, or at least harmless?

Frozen 2026-09-18, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check under `--noise 1` had already been run; it printed no outcome metric and
decided nothing. What it checked is listed under "Mechanics check" below.
Any later change to the arms below is a new preregistration, not an edit to this one.

## Why this test

`draft/settings.py` put every league rule and modelling assumption in one object. Three
of its fields are switches that default to the published behaviour because turning them
on would move numbers that are already written down. Defaulting to the published value
is the right call for reproducibility and the wrong call for correctness, so each switch
is currently available but unproven, and nobody can turn one on knowingly.

1. **`enforce_caps_in_season`** (default `False`). The draft caps rosters at 2 QB, 2 TE,
   7 RB and 7 WR. The in-season wire enforces only the starting minimums and never a cap,
   so a simulated team can stream its way to five quarterbacks. Consistency says enforce
   the same rule everywhere.
2. **`replacement`** (default `"multiplier"`). Replacement value is priced by counting
   down the ranked pool to league-wide starter demand (`vbd.replacement_levels`), an
   assumed count of players per team. The honest price for a tool that knows the real
   rosters is the best player nobody actually holds: `replacement="pool"`.
3. **`roster_min`** and forced cuts. The minimums are enforced everywhere and always
   were. `roster.forced_cut` has no caller in the harness, because every add is a 1-for-1
   swap and the 2-for-1 trade does its own cut, so roster size never exceeds 14. There is
   nothing to vary and no arm is run for it. This is stated, not tested.

The question is not whether the corrected settings are more defensible. They are. It is
whether the best realistic policy still holds up in the corrected world, so the default
can be changed with the cost known.

## Scope: the switch applies to the WHOLE league, and what that means

Both switches are properties of the simulated world, not decisions one manager makes.
The wire rule binds all twelve seats; the replacement price feeds the single value array
every seat's wire reads. A seat-only version would be a different and smaller question
("should my tool price replacement this way?"), and for the cap it is not even
well-posed, since a cap only one seat obeys is a handicap rather than a league rule.

**So every arm changes the setting for all twelve seats.** This is preregistered and is
not revisited after the run.

The consequence for the headline number is stated up front. The primary comparison is
between two different leagues, not between two policies in the same league. What is
paired is the *seat*: the same season, the same league index, the same seat, the same
drafted rosters and the same twenty schedules, with the world's rule changed underneath
all twelve teams. The number reported is

> the test seat's all-play win rate against its own eleven opponents under the changed
> rule, minus the same seat's all-play against its own eleven opponents under the
> published rule.

That is a legitimate paired quantity and it answers "does the policy seat do better or
worse, relative to the field it actually faces, once the rule is corrected?". It is not
a head-to-head between the two worlds, and it cannot be: under the changed rule the
eleven opponents are different teams. A change of zero means the correction leaves the
policy's standing alone, which is the outcome that licenses turning the switch on.

Because the opponents move too, a secondary comparison is run inside each world: the
policy's own edge, `mutual_cap2` minus `stream`, computed within that world on the same
seats. That separates "the corrected rule moved everyone's baseline" from "the corrected
rule changed what the policy is worth". A third, context-only comparison reports the
world's effect on the `stream` control alone.

**The drafts are identical in every arm.** `league_backtest.py` binds its draft caps at
import (`CAP`), and no arm rebinds them, so the 2/2/7/7 draft cap stands everywhere and
only the in-season rule varies. This is deliberate: it keeps the pairing exact and it
also means the `caps_espn` arm below is not a looser draft, only a looser wire.

## Arms

The policy in every policy arm is `mutual_cap2` from `sim_tradecap.py`, unchanged: hole-
aware streaming on waivers, at most two executed trades a season, each projecting at
least 10 rest-of-season starting-lineup points, with an opponent who refuses any trade
that lowers its own lineup. `sim_tradecap.simulate`, `sim_trades.search`, `execute`,
`lineup_value`, `ros_points`, `team_at` and `future_avail` are imported and not edited.
New code lives in `draft/sim_settings.py`.

| arm | world (all twelve seats) | policy |
|---|---|---|
| `stream` | published defaults | streaming only |
| `ctrl` (**control**) | published defaults | `mutual_cap2` |
| `caps_on_stream` | `enforce_caps_in_season=True` | streaming only |
| `caps_on` | `enforce_caps_in_season=True` | `mutual_cap2` |
| `pool_stream` | `replacement="pool"` | streaming only |
| `pool` | `replacement="pool"` | `mutual_cap2` |
| `both_stream` | both switches on | streaming only |
| `both` | both switches on | `mutual_cap2` |
| `caps_espn_stream` | caps on, `caps = {QB 4, RB 8, WR 8, TE 3}` | streaming only |
| `caps_espn` | caps on, `caps = {QB 4, RB 8, WR 8, TE 3}` | `mutual_cap2` |

`stream` must reproduce `sim_tradecap.py`'s `stream` arm exactly and `ctrl` must
reproduce its `mutual_cap2` arm exactly (every column, max absolute difference 0.0
against the committed `data/league_tradecap_noise{1,0}.parquet`).

**The realistic caps.** 2/2/7/7 is how a draft simulator keeps bots sane, not a rule a
real league has. `caps_espn` uses ESPN's default position limits for a 12-team league,
QB 4 / RB 8 / WR 8 / TE 3, which is the only widely-used platform default that is a
position limit at all; Sleeper, the platform the trade counts came from, has none, and
an arm with no limits would be identical to the control by construction. These values
come from the platform, not from anything fitted here. On a 14-man roster they are loose,
so this arm is expected to bind rarely; that expectation is written down before the run
and is exactly what the binding diagnostic measures.

## What each switch does mechanically

- **`enforce_caps_in_season`** routes through `roster.add_allowed` (the consensus wire in
  `league_backtest.consensus_move` and the hole-filling add in `streaming_policy`) and
  through `roster.legal` (the trade search's legality test on both sides, for the seat's
  roster after the trade and the opponent's). An add that would break a cap is removed
  from the candidate list; if nothing is left the team makes no move that week. A trade
  whose either side would break a cap is never offered. No roster is ever over its cap at
  the start of a season, since the draft cap is tighter than or equal to every cap tested.
- **`replacement="pool"`** reprices the shared value array once per decision week, before
  that week's wire round, from the rosters as they stand at that moment: the replacement
  level at a position is the consensus rest-of-season points per game of the best player
  no team in the league holds, and every player's value is his points minus that level.
  The same repriced column is what the trade search reads later in the same week, so the
  wire and the trade price players identically, exactly as the fixed multiplier column
  does in the published arms. Only the cross-position comparison moves; the order within
  a position is unchanged, because the level is a constant per position.

## Information timing and leakage control

- Nothing new is read. Both switches are functions of the current rosters and the same
  consensus rest-of-season numbers the published arms already use at the same moment: the
  latest scrape strictly before the upcoming week, through a rank curve fit only on
  seasons before the scored one.
- The pool replacement level is computed from rosters inside the simulated league, which
  are fully observable at decision time to any manager on the platform. It uses no future
  points and no knowledge of any other arm.
- Nothing here is fitted. The ESPN caps come from a platform default; the 2/2/7/7 caps
  and the 10-point trade threshold are `settings.py`'s existing frozen values.
- Stated residual: the same five seasons, leagues and seeds have been used by every
  earlier test in this harness, and `prereg_tradecap.md`'s result was known when this
  test was designed. No parameter here was chosen from an outcome.

## Scoring and decision rule

- Harness: `draft/sim_settings.py`, seasons 2021-25, 20 leagues x 12 seats x 20
  schedules, both opponent designs (`--noise 1` and `--noise 0`), all ten arms in one run
  per design, one design at a time through the shared lock.
- Primary: paired change in all-play win rate of each policy arm against `ctrl`, same
  season, league, seat and schedules. Headline: paired change in title odds. Secondary:
  playoff, wins, points.
- **The three verdict arms are `caps_on`, `pool` and `both`**, each against `ctrl`. Each
  switch is its own decision, so each gets its own verdict rather than a single winner
  picked afterwards. `caps_espn` is a labelled sensitivity.
- **Decision rule (the harness's standard rule), per verdict arm:** it *passes* only if,
  in **both** designs, its pooled all-play change against `ctrl` is positive, positive in
  at least 4 of 5 seasons, and its pooled title change is no worse than -1.0 point.
- **Harmless band, preregistered because a null is the likely and useful answer:** an arm
  that does not pass is called *harmless* if, in both designs, its pooled all-play change
  is within +/- 0.5 points and its pooled title change is no worse than -1.0 point.
- **Recommendation, decided by the above and not afterwards.** For each switch: passes ->
  recommend the default changes to the corrected value; harmless -> recommend the default
  changes anyway, on correctness, with the measured cost quoted; neither -> recommend the
  default stays where it is and quote what the correction costs.
- Seeds: league `lg` of season `y` uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `sim_tradecap.py`; every
  arm shares those draws and the same drafts.
- Bootstrap: `league_backtest.paired`, as in `prereg_tradecap.md` (season means, 2,000
  resamples with `numpy.random.default_rng(1)`, 5th/95th percentiles). The interval is
  reported; the rule decides.
- Run once per design. Reduced (`--leagues` other than 20) runs write `_check` files.

## Reported, not gated: does the unenforced cap ever bind?

Measured in the published control (`ctrl`) only, over all twelve seats, from the rosters
as they stand before each week's wire round (so week w is the roster that played week
w-1), weeks 2-17:

- The distribution of players held at each position, its maximum, and the count and share
  of team-weeks over each of the two cap sets (2/2/7/7 and ESPN's), by position and week.
- The split between the test seat and the eleven consensus seats, since the streaming
  policy is the one most likely to accumulate a position.
- For pool versus multiplier: the mean replacement level at each position under both
  pricings, the mean and 90th percentile of the difference, and how often the best
  available free agent by value is the same player under both. This is the size of the
  thing the switch changes, independent of whether it changes an outcome.

**If the cap does not bind in practice, the switch is cosmetic and will be reported as
cosmetic.** A null dressed up as a finding is worse than a null.

## Known limits

- The primary number compares two leagues. It is a paired seat comparison and is read as
  "the policy's standing against the field it faces", not as one world beating another.
- `roster_min` and `forced_cut` are not exercised by any arm here, because the harness
  never puts a roster over its size. That is a statement about the harness, not evidence
  that forced cuts are harmless in a league that does allow them.
- Pool replacement is priced once per decision week, before the wire runs, so within a
  week the level is stale by up to twelve adds. The multiplier column is stale for the
  whole week, so this is strictly fresher, but it is not a within-round recomputation.
- The pool level ignores availability: the best unrostered player might be on bye that
  week. The multiplier level ignores it too, so the two are compared on equal terms.
- Correlations, the third switch in `settings.py`, is not tested here. Its values are
  stated in `settings.py` to be plausible rather than fitted, so switching it on is
  taking an assumption rather than fixing one, and it is a different kind of question.

## Mechanics check (run before this file was frozen, no outcome metric printed)

At `--leagues 2 --mechanics`, `--noise 1`. Every roster in every league-week was asserted
legal under the world that arm was running (minimums always, caps when enforced) and to
hold exactly 14 players; every executed trade passed `sim_trades.execute`'s existing
invariants and, in the mutual arms, `sim_tradecap`'s assertion that the opponent's
projected lineup does not fall. `stream` and `ctrl` reproduced `sim_tradecap.py`'s
`stream` and `mutual_cap2` rows for those leagues exactly: 120 rows each, every column,
max absolute difference 0.0. No outcome metric was printed or read.

Two decision-level diagnostics were read, and both are disclosed here because they were
seen before the real run. Neither changed anything in the spec above.

- The unenforced cap binds constantly. Over 23,040 team-weeks in the published control,
  33.7% hold more tight ends than the cap of 2 (up to 7), 25.2% more quarterbacks (up to
  6), 10.2% more receivers than 7 (up to 10) and 0.8% more running backs. So the switch
  is not cosmetic, and the real run's diagnostic is a measurement rather than a
  formality. The consistent-rule arm also moves fewer players (9.1 wire adds a season
  against 11.7) and makes fewer trades (1.52 against 1.86), which is the cap refusing
  moves rather than a different objective.
- Pool and multiplier disagree about the best available free agent in about 80% of
  team-weeks, and pool sits 2.0 to 2.6 points per game below the multiplier level at
  every position, so the switch changes the cross-position comparison materially rather
  than shifting every value by a constant.

Both are counts of decisions and inputs. No all-play, title, playoff or win number from
any arm was looked at before this file was frozen.

## Result (added after the single run per design; the spec above is unchanged)

`stream` and `ctrl` reproduce `sim_tradecap.py`'s `stream` and `mutual_cap2` rows exactly
in both designs (1,200 rows each, every column, max absolute difference 0.0).

**Verdicts, by the preregistered rule and the harmless band.**

| switch | noise 1 all-play | seasons + | title | noise 0 all-play | seasons + | title | verdict |
|---|---|---|---|---|---|---|---|
| `caps_on` | -1.7 pp [-2.1, -1.2] | 0/5 | +1.2 | -2.8 pp [-3.9, -1.2] | 1/5 | -2.8 | **fails, and not harmless** |
| `pool` | +1.8 pp [+1.5, +2.1] | 5/5 | +4.2 | +2.3 pp [+0.5, +3.9] | 4/5 | +4.0 | **passes both designs** |
| `both` | +0.2 pp [-0.2, +0.5] | 4/5 | +2.5 | -1.0 pp [-2.7, +0.3] | 1/5 | +1.3 | fails (noise 0) |
| `caps_espn` (sensitivity) | -0.0 pp [-0.1, +0.1] | 3/5 | +1.7 | +0.4 pp [-1.2, +1.9] | 3/5 | -0.3 | fails the rule, harmless in both |

All-play by season 2021-25, each arm against `ctrl`. noise 1: `caps_on` -1.8 -1.7 -1.8
-0.5 -2.6; `pool` +2.3 +1.0 +2.2 +1.8 +1.8; `both` +0.8 +0.0 +0.5 -0.6 +0.3; `caps_espn`
+0.0 -0.3 +0.1 +0.1 -0.0. noise 0: `caps_on` -4.2 -3.3 +0.7 -3.2 -4.1; `pool` -1.0 +3.5
+3.6 +0.1 +5.2; `both` -4.8 -0.2 -0.4 -0.6 +1.2; `caps_espn` -2.5 +1.7 +3.1 -1.5 +1.2.

**Recommendation, by the rule written above.**

- `enforce_caps_in_season` **stays False** at the 2/2/7/7 caps. Enforcing them costs the
  policy seat 1.7 to 2.8 points of all-play and is negative in 9 of 10 season-designs.
- `replacement` **changes to `"pool"`**, with the caveat below, which is large enough that
  the reader should weigh it rather than take the rule's word.
- `roster_min` and forced cuts: nothing to change. `roster.forced_cut` still has no caller
  in the harness.
- A league that really has ESPN-style position limits can turn them on: `caps_espn` is
  inside the harmless band in both designs.

**Why the cap hurts.** It is not a cosmetic switch and it does not bind symmetrically. In
the published control, over 230,400 team-weeks, 33.4% (noise 1) and 35.3% (noise 0) of
rosters hold more tight ends than the cap of 2, up to 7 of them; 25.3% and 24.0% hold more
quarterbacks, up to 7 and 5; 9.6% and 10.8% hold more than 7 receivers; running backs
almost never breach, 0.6% and 0.4%. The breaches start in week 3 and grow all season. The
test seat is the one most often over the cap at exactly the two positions the streaming
policy uses as a lever: it is over the TE cap in 37.4% and 38.5% of its team-weeks against
33.1% and 35.0% for the consensus seats, while it is over the WR cap far *less* often
(3.8% against 10.1% and 11.5%). Hole-aware streaming works by carrying a second and third
quarterback or tight end through a bye or an injury, so the cap takes away precisely the
move the policy passed on. The wire slows from 10.9 to 8.9 adds a season (noise 1) and the
policy's own edge inside the capped world shrinks from +1.1 to +0.6 points of all-play.
A capped league is also worse for the streaming control on its own (-1.2 and -2.0), so the
whole league loses activity, and the policy seat loses more than its opponents.

**Why pool passes, and the caveat.** Pool replacement sits 2.0 to 2.8 points per game
below the multiplier at every position, and the gap is not uniform: quarterbacks and tight
ends fall furthest (-2.5 to -2.8) and running backs least (-2.0), so the switch reprices
positions against each other rather than shifting everything by a constant. The best
available free agent is a different player under the two pricings in about 80% of
team-weeks. But almost none of the gain arrives through the wire. `pool_stream` against
`stream`, which is the pool switch with no trades in play, is +0.2 [-0.1, +0.6] and +0.7
[-0.3, +1.6]: neutral, and it would not pass the rule on its own. The gain arrives through
trades. Under pool pricing the seat executes a trade in 85-95% of the weeks it searches
against 35-48% under the multiplier, takes its full two trades in every seat-season, and
the median projected gain per trade doubles, from 18.5-19.7 to 33.4-40.2 points. The
reason is mechanical: the value-fairness test the opponent applies is a comparison of
replacement-priced values, so repricing the positions changes which offers the model
opponent will accept, and the seat's search finds the offers the new pricing opens up.
`prereg_tradecap.md` already lists that acceptance model as an upper bound, because the
opponent never says no for a reason the harness cannot see. So the honest read is that
pool replacement is the more defensible definition and is measured as neutral-to-slightly
positive where it is cleanly tested (the wire), while its large measured gain runs almost
entirely through a channel the harness is known to flatter. It is recommended on
correctness with the trade gain discounted, not on the size of the number.

**Reported, not gated.**

- Wire adds a season by arm, noise 1 / noise 0: `stream` 10.85 / 12.63, `ctrl` 11.81 /
  12.80, `caps_on_stream` 8.87 / 10.90, `caps_on` 9.11 / 11.25, `pool` 12.01 / 13.18,
  `both` 10.90 / 12.37, `caps_espn` 11.71 / 12.67.
- Trades a season: `ctrl` 1.91 / 1.83, `caps_on` 1.59 / 1.68, `pool` 2.00 / 2.00, `both`
  1.99 / 2.00, `caps_espn` 1.89 / 1.75.
- The policy's own edge inside each world (`mutual_cap2` minus that world's `stream`),
  all-play, noise 1 / noise 0: published +1.1 / +1.9, caps +0.6 / +1.1, pool +2.7 / +3.5,
  both +2.4 / +2.4, ESPN caps +1.4 / +2.8. The policy beats streaming in every world
  tested, which is the one thing every arm agrees on.
- Under the ESPN caps, breaches of the published control's rosters are rare: 0.3-0.6% of
  team-weeks over 4 QB, 8.2-8.9% over 3 TE, 1.9-4.1% over 8 WR, 0.1% over 8 RB. The tight
  end limit is the only one that binds at all often, which is why this arm moves so little.

**Honest read.** Two settings that both look like corrections behave completely
differently. Enforcing the draft's roster caps on the wire is a real rule change that
binds a third of all team-weeks, and it is bad for the policy: streaming a second
quarterback or tight end through a bye is exactly what the cap forbids, and the policy's
edge halves. That is not a reason to think the published harness is right; it is a reason
to say plainly that the published numbers were computed in a league where a team may carry
five quarterbacks, and that a league with the caps on is a harder league for this policy.
The replacement switch passes, but the mechanism is disappointing: the part of it that can
be tested cleanly is a null, and the part that produces the number runs through the trade
acceptance rule that was already flagged as an upper bound. Nothing here changes
`mutual_cap2`'s standing, which beats streaming in all five worlds.

**Known limits beyond those already stated.** The primary comparison is between two
leagues, so the number is the policy seat's standing against the field it faces, not one
world beating another. Pool replacement ignores availability and is priced once per
decision week. No arm exercises forced cuts. The ESPN caps were picked from a platform
default, not from a distribution of what real leagues actually set.
