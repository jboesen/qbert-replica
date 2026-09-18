# Preregistration: stacking, or acting on the quarterback / own-receiver correlation

Frozen 2026-09-18, before the real (LEAGUES=20) runs of these arms. A 2-league mechanics
check under `--noise 1` had already been run; what it showed is under "Mechanics seen
before freezing". It printed no all-play, title, playoff, wins or points outcome for any
arm. The correlation measurement below was made before any arm was coded and is the only
number in this file fitted to data. Any later change to the arms is a new preregistration,
not an edit to this one.

## Why this test

`prereg_winprob.md` and `prereg_variance.md` both failed, and the second measured why.
Week-to-week spread rises with consensus value at every position, so the highest-projected
lineup is already close to the widest one; twelve projected points bought at most +0.33 of
extra spread against -1.63 of floor. There is almost nothing to trade when players are
priced one at a time.

Correlation is the one variance lever an independent-player model cannot see. A
quarterback and the receiver he throws to share the same drives, so a lineup holding both
swings wider than its slot-by-slot variance says, at no cost in projected points. That is
the folk argument for stacking, and `draft/correlate.py` now implements it for the live
tools, gated off behind `settings.correlations`. Its `qb_receiver_rho` is an assumption,
not a measurement, and nobody has tested whether acting on it wins games.

## The measured correlation (estimated before the arms were written)

Weekly PPR box scores, regular season, weeks 1-17, **prior seasons 2015-2020 only** (the
test seasons are 2021-25). For each team-week, the team's primary passer (most attempts,
at least 10) is paired with every pass catcher of his who saw at least one target. Each
player's own weekly expectation is removed before correlating: the primary estimate
subtracts his own mean over the weeks he appeared that season (players with fewer than 6
appearances are dropped), and the check subtracts the harness's own weekly consensus value
(weekly positional ECR rank read through `league_backtest.rank_curve`), which is the
expectation every policy in this repo actually uses. Receiver role is his rank in team
targets over the season.

| estimator | rho | n player-weeks |
|---|---|---|
| own-season-mean residuals, 2015-20 (primary) | **+0.285** | 14,614 (2,736 team-weeks) |
| consensus residuals, 2019-20 (`ecr_weekly` starts 2019) | **+0.249** | 2,855 |

By role, own-mean residuals on 2015-20:

| role | rho | n | receiver residual sd | QB residual sd |
|---|---|---|---|---|
| WR1 | +0.392 | 2,579 | 7.9 | 7.0 |
| WR2 | +0.327 | 2,389 | 7.0 | 7.0 |
| TE1 | +0.290 | 2,359 | 6.0 | 7.0 |
| WR3+ | +0.222 | 4,749 | 5.1 | 7.0 |
| TE2+ | +0.203 | 2,538 | 4.0 | 6.9 |

It is stable: by prior season, 2015 +0.285, 2016 +0.284, 2017 +0.283, 2018 +0.288, 2019
+0.308, 2020 +0.262. (The test seasons, looked at only after the prior-season estimate was
fixed and used for nothing: 2021 +0.281, 2022 +0.249, 2023 +0.280, 2024 +0.271, 2025
+0.287. Pooled +0.274, WR1 +0.355, TE2+ +0.149.)

Opposing offences in the same game, each team's total skill-position PPR against its own
season mean: **rho +0.217** on 1,536 prior-season games (+0.203 on 1,279 test-season
games). Real but smaller than the own-team effect, and it is a whole-game effect rather
than a lineup one.

**`correlate.py`'s default `qb_receiver_rho = 0.35` is too high as a blanket figure.** It
is roughly the WR1 number and about 40% above the pooled estimate. The right blanket value
is **0.25** if the residual is taken against consensus (the harness's own expectation, and
what `correlate.blend` is mixing into), or 0.285 against a player's own mean. A role-aware
version would be WR1 0.39, WR2 0.33, TE1 0.29, other 0.21. This preregistration recommends
0.25 and states it as a recommendation, not as something this test's outcome will settle.

## What the measurement is worth, and therefore what a stack is worth paying

This is the reason the price bar below is small, and it is derived here, before the run.

A stacked pair adds `2 * rho * sd_q * sd_r` to the lineup's variance and nothing to its
mean. At rho = 0.25, sd_q = 7.5 and sd_r = 7.0 that is +26 of variance. A twelve-team PPR
starting lineup has a standard deviation near 25 (variance 625), so stacking takes it to
sqrt(651) = 25.5: **about +0.5 points of lineup standard deviation.**

Against one opponent, with margin standard deviation `sigma` and standardised margin
`z = (my projection - his) / sigma`, the win probability is `Phi(z)`. Its derivative in my
mean is `phi(z)/sigma` and in my own spread is `-z*phi(z)/sigma`, so paying `dmu` of mean
for `dsd` of spread breaks even at `dmu = -z * dsd`. A one-sigma underdog (z = -1) should
pay 0.5 projected points for +0.5 of spread; a two-sigma underdog should pay 1.0. A
favourite should pay nothing and would rather narrow.

So the honest bar for an unconditional stack is about **one projected point**, not the
three that `prereg_variance.md` used as its small budget. That is `C_PLAIN = 1.0` below.
The underdog arm is allowed `C_DOG = 3.0`, which the arithmetic says is too much even for a
large underdog; it is deliberately generous so that the arm has a real chance to show an
effect rather than never firing.

## Arms

Every arm drafts by exact consensus, runs hole-aware streaming on waivers and makes at
most two mutual-benefit trades a season: **`mutual_cap2` from `sim_tradecap.py`**, the
verdict arm of `prereg_tradecap.md` and the best realistic policy in the repo. The eleven
other seats are unchanged. Weeks 15-17 keep consensus lineups in every arm, as in
`prereg_winprob.md` and `prereg_variance.md`.

- **`mutual_cap2`** (control): consensus lineups, consensus-valued wire. Must reproduce
  `sim_tradecap.simulate` under spec `("mutual", 2, 10.0)` exactly in scores, waiver moves
  and final rosters (max abs score difference 0.0).
- **`stack1`** (**the preregistered verdict**): in weeks 1-14, if the roster holds a
  quarterback and a WR or TE on the same NFL team that week with both eligible, start the
  most-projected legal lineup that starts such a pair, provided it gives up at most
  **C_PLAIN = 1.0** projected points against the consensus lineup. Among qualifying pairs
  take the cheapest; ties go to the lineup that starts more stacked pairs. Otherwise play
  the consensus lineup, and score exactly what the control scored that week.
- **`stack_dog`**: the same rule with **C_DOG = 3.0**, but only in a week the seat is the
  underdog, meaning its consensus-lineup projection is below this week's head-to-head
  opponent's consensus-lineup projection. In any other week it plays the consensus lineup
  (C = 0), not `stack1`'s budget.
- **`stack_wire`**: lineups are consensus every week; only the wire changes. When the
  streaming policy decides to add a WR or TE, and a free agent at the **same position**,
  on the NFL team of a quarterback already on the roster, is within **WIRE_EPS = 0.5**
  points of rest-of-season value over replacement of the intended add, take him instead.
  Same drop, same hole closed, same legality. This is "prefer a stack partner at equal or
  near-equal consensus value" and nothing more.

The draft is not touched. A stack-aware draft would change every roster in the league and
is a different test; this is stated as a limit, not done.

## Why `stack1` carries the verdict

It is the direct question: with the measured correlation in hand, is a stack worth a price
set to be worth paying? It is also the cleanest measurement, because its lineup does not
depend on the schedule, so its score is one number a week and the paired contrast against
the control is the lineup decision and nothing else. `stack_dog` and `stack_wire` are
labeled sensitivities and are reported in full; choosing the best of three afterwards is
what this rule exists to prevent.

## The schedule/scoring separation

`stack_dog` needs to know who it plays, which breaks the harness's separation of scoring
from schedule draws. The design is `prereg_variance.md`'s, unchanged and reused through
`sim_variance.path`:

1. The roster path is taken from the control, once per (season, league, seat): draft,
   every waiver move by all twelve teams, and the seat's trades, with a weekly snapshot.
   The mechanics check asserts the path's scores, moves and final rosters are identical to
   `sim_tradecap.simulate`'s.
2. Each seat-week is priced once on that path, schedule-free: the consensus lineup, the
   stacked lineup at each budget, their projected and realised points, and each of the
   eleven possible opponents' consensus-lineup projections.
3. Each schedule is then played week by week; all-play, wins, playoff and title come from
   `season_outcome` on the control's score matrix with the seat's row replaced, averaged
   over the twenty schedules.

The same known bias applies and is restated: an arm whose weekly score differs from the
control's would, in a full re-simulation, sit slightly differently in the waiver queue
later. It inherits the control's queue position instead. The direction is not signable a
priori, and the diagnostics report how often the lineup differs at all, which bounds how
often the channel can fire. `stack1` is schedule-free and so is unaffected by the schedule
part of this; it still inherits the frozen queue position.

`stack_wire` changes who is on rosters, so it is run as a full separate simulation with
its own score matrix for all twelve teams, exactly as `sim_tradecap.py`'s arms are.

## Information timing and leakage control

- The correlation is estimated on 2015-2020 only, strictly before every test season, and
  it sets one preregistered constant (the 1.0 point bar) through the arithmetic above. The
  test-season values are printed for stability and are used for nothing.
- No arm reads `rho` at decision time. The policy is a price in projected points; nothing
  in the simulator is re-drawn with a correlation, and `settings.correlations` stays off.
- Stack membership uses the NFL team on the weekly roster file at or before the week being
  played (`correlate.weekly_teams`), the same rows `Season.elig` reads, so a traded player
  changes partners in the right week. The wire uses `sim_trades.team_at`, which is the
  team as of the decision before the week, so it never sees the week's own roster move.
- Eligibility, weekly consensus values, injury reports and opponent lineups are the
  harness's own pregame quantities, as in `prereg_variance.md`.
- Wire values are `consensus.weekly_curve(2020..y-1)` and the rank curve is fit on 2020,
  both before any test season, unchanged from `sim_tradecap.py`.

## Scoring and decision rule

- Harness: `draft/sim_stack.py`, importing `league_backtest.py`, `sim_trades.py`,
  `sim_tradecap.py`, `sim_variance.py` and `winprob.py`. No existing module is edited.
  Seasons 2021-25, 20 leagues x 12 seats x 20 schedules, both designs (`--noise 1` and
  `--noise 0`), one run per design through the shared lock.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `sim_tradecap.py`. Every
  arm shares those draws and the drafted rosters. No arm here draws any random number of
  its own; every policy is deterministic given the path.
- Primary: paired change in all-play win rate, arm minus `mutual_cap2`, same league, seat
  and schedules. Headline: paired change in title odds. Secondary: playoff, wins, points.
- **Decision rule, on `stack1` only:** it passes only if, in **both** designs, its pooled
  all-play change against `mutual_cap2` is positive, positive in at least 4 of 5 seasons,
  and its pooled title change is no worse than -1.0 point. The rule is printed for the
  other two arms as a sensitivity, not as a verdict.
- Bootstrap: `LB.paired` (season means, 2,000 resamples with `numpy.random.default_rng(1)`,
  5th/95th percentiles). The interval is reported; the rule decides.
- Real results go to `data/league_stack_noise{1,0}.parquet`. Any reduced run writes
  `data/league_stack_check{L}x{S}_noise{N}.parquet` and prints mechanics, never outcomes.

**All-play is schedule-free; a stack's benefit is head-to-head.** That tension is stated in
advance. Widening your own weekly total does not obviously raise the share of the other
eleven teams you beat, because all-play averages over the whole field; it raises the chance
of beating one particular opponent you are behind. The deciding metric is still the
paired all-play change, because that is this repo's rule and changing it for one test would
be picking the metric that flatters the hypothesis. The head-to-head numbers are reported
alongside it in full and given equal room in the write-up.

## Reported, not gated

- How often a stack was available at all (a QB and an own pass catcher both eligible on the
  roster), how often the consensus lineup already started one for free, and how often each
  arm paid to add one, by arm and by season.
- Projected points paid per stack taken.
- Realised weekly spread, as the root mean square of a week's points minus that week's own
  projection, for lineups that started a stack against those that did not, in the control
  and in every arm. This is the direct check that the widening the correlation predicts
  actually shows up in this harness's scores.
- Head-to-head: in weeks an arm paid for a stack, how often it won that week against how
  often the consensus lineup would have.
- How often the seat was the underdog, and how often `stack_dog` found a stack to pay for
  in one.
- `stack_wire`: swaps a season and consensus value given up per swap.

## Known limits

- With fixed weekly values the maximum-points lineup is unique, so a stack changes nothing
  unless the seat is willing to pay. The price is preregistered at 1.0 point and derived,
  but the derivation assumes a normal margin and a fixed opponent spread.
- The correlation is estimated on pass catchers with at least one target and starters with
  at least ten attempts, which is a slightly more usable population than a fantasy roster.
  Demeaning by a player's own season mean over roughly 14 appearances attenuates the
  estimate a little; the consensus-residual check, which does not demean at all, comes in
  lower (0.249), so the two bracket the truth rather than both erring one way.
- The opposing-offence correlation (+0.22) is measured and reported but no arm acts on it.
  Starting a player who faces your head-to-head opponent's player is a further lever and is
  not tested here.
- `stack_wire` never pays more than 0.5 points of value for a partner, so it cannot show
  what a genuinely stack-seeking manager would do, only whether a free preference helps.
- The draft is unchanged in every arm.

## Mechanics seen before freezing

From `sim_stack.py --noise 1 --leagues 2 --schedules 20 --mechanics`, 1,680 seat-weeks
(2 leagues x 12 seats x 14 weeks x 5 seasons). It wrote
`data/league_stack_check2x20_noise1.parquet` and printed no outcome metric.

- The control reproduces `sim_tradecap.simulate` under `("mutual", 2, 10.0)` exactly:
  **max absolute score difference 0**, waiver moves identical, final rosters identical,
  in every seat-season.
- `best_lineup` with nothing forced equals `winprob.consensus_lineup` in **1.000** of the
  1,680 seat-weeks, so the arms' baseline is the harness's own lineup and an arm that
  declines to stack scores exactly what the control scored.
- Every stacked lineup passes `winprob.legality` (1.000). The largest projected points
  given up anywhere is 2.96, inside the 1.0 and 3.0 bars.
- **A stack was available in 33.4% of seat-weeks**, and the consensus lineup already
  started one **for free in 15.8%**. So roughly half the stacks a roster holds are ones
  consensus fields anyway.
- `stack1` pays for a stack in **2.02%** of decisions and `stack_dog` in **1.45%**. Those
  are firing rates, not outcomes, and they are low for the reason the arithmetic above
  gives: the cheapest unstarted stack usually costs more than a projected point. They are
  recorded here because they bound how much any of this can move a season, and they do not
  change the arms.
- `stack_wire` swaps the wire's intended add for a stack partner **1.02 times a
  seat-season** (max 6).
- 0.70 seconds per seat-season for the lineup tables, so the real run is dominated by the
  two full simulations per seat, as `sim_tradecap.py` is.

- Against the **committed** `data/league_tradecap_noise1.parquet`, the control's 120 rows
  for leagues 0-1 match `mutual_cap2` on title, playoff, wins, all-play, regular-season
  points, playoff points and waiver moves, **max absolute difference 0.0**.

The same 2-league check under `--noise 0`, run before that design's real run, gives the
same verdict: control reproduces `mutual_cap2` with **max absolute score difference 0**,
identical moves and rosters, and **max absolute difference 0.0** on all seven outcome
columns against the committed `data/league_tradecap_noise0.parquet`; `best_lineup` matches
the consensus lineup in 1.000 of 1,680 seat-weeks; every stacked lineup legal; stack
available in 30.2% of weeks, started free in 10.4%, `stack1` pays in 2.26% and `stack_dog`
in 3.31%; 0.93 wire swaps a seat-season.

## Result (added after the single run per design; the spec above is unchanged)

Both real runs completed at harness defaults (20 leagues x 12 seats x 20 schedules,
seasons 2021-25), one per design through the shared lock, writing
`data/league_stack_noise1.parquet` and `data/league_stack_noise0.parquet`. The control
reproduces `sim_tradecap`'s `mutual_cap2` exactly in both designs, checked at 2 leagues
against both a live `TC.simulate` and the committed tradecap parquet (max absolute
difference 0.0 on every column).

**The verdict arm, `stack1`, fails the rule in both designs, and so does every other arm.**
Changes against `mutual_cap2`, 90% season-cluster intervals:

| arm | design | title | all-play | all-play by season 2021-25 | seasons positive | rule |
|---|---|---|---|---|---|---|
| `stack1` | noise 1 | -0.0 pp [-0.1, +0.0] | **-0.1 pp** [-0.1, -0.0] | +0.0 -0.0 -0.1 -0.2 -0.1 | 1/5 | fails |
| `stack1` | noise 0 | -0.1 pp [-0.2, +0.0] | **-0.1 pp** [-0.2, -0.0] | -0.0 -0.3 -0.1 -0.3 +0.1 | 1/5 | fails |
| `stack_dog` | noise 1 | -0.0 pp [-0.1, +0.0] | -0.1 pp [-0.1, -0.0] | +0.0 -0.0 -0.1 -0.2 -0.0 | 1/5 | fails |
| `stack_dog` | noise 0 | -0.2 pp [-0.3, -0.1] | -0.3 pp [-0.4, -0.1] | -0.1 -0.5 -0.2 -0.5 -0.1 | 0/5 | fails |
| `stack_wire` | noise 1 | +0.2 pp [+0.1, +0.4] | +0.1 pp [-0.0, +0.2] | +0.1 +0.0 -0.1 +0.3 -0.0 | 3/5 | fails |
| `stack_wire` | noise 0 | +0.9 pp [-0.9, +3.2] | +0.1 pp [-0.2, +0.4] | +0.6 -0.1 +0.2 +0.3 -0.5 | 3/5 | fails |

Title odds by season for `stack_wire` under noise 0 are -0.7, -2.2, +0.2, +1.0, +6.2: the
pooled +0.9 is one season. Nothing here clears the preregistered bar on any arm.

### The correlation is real and it does widen the week. It just doesn't pay.

The chain the hypothesis needs has three links, and this run measures all three. The first
two hold and the third breaks.

1. **The correlation exists**: +0.285 on 14,614 prior-season player-weeks, stable across
   six seasons, and it is not zero by any reading.
2. **It shows up in realised scores.** Root mean square of a week's points minus that
   week's own projection, for the control's own lineups, split by whether they happened to
   start a quarterback with his own receiver:

   | design | started a stack | did not |
   |---|---|---|
   | noise 1 | **23.25** (n=2,701) | 21.78 (n=14,099) |
   | noise 0 | **23.28** (n=1,740) | 21.15 (n=15,060) |

   That is +1.5 to +2.1 points of realised weekly spread, larger than the +0.5 the
   arithmetic predicted, though the split is on rosters rather than randomised, so part of
   it is that a roster holding a startable stack is a different roster. Either way, the
   widening is there and it is not small.
3. **The widening does not win weeks.** In the weeks each arm actually paid for a stack,
   it won its head-to-head week *less* often than the consensus lineup would have:

   | arm | design | h2h win rate when it paid | change vs the consensus lineup |
   |---|---|---|---|
   | `stack1` | noise 1 | 0.580 | **-0.0201** |
   | `stack1` | noise 0 | 0.459 | **-0.0649** |
   | `stack_dog` | noise 1 | 0.370 | -0.0308 |
   | `stack_dog` | noise 0 | 0.322 | -0.0861 |

   So the one metric a stack is supposed to move, the chance of beating the one opponent
   you play, moves the wrong way in both designs and in both lineup arms. All-play, the
   deciding metric, moves the wrong way too and by about the same tiny amount, so the
   schedule-free/head-to-head tension flagged in advance did not end up mattering: neither
   metric rescues the other.

`stack_dog` is the clearer failure and the more informative one. It is the theory's best
case, paying up to three points only when behind, and it is the worst arm in the run
(-0.3 pp of all-play under noise 0, 0 of 5 seasons positive). Its head-to-head win rate
when it pays is 0.32-0.37, which is what an underdog week looks like, and paying for width
made it worse rather than better. Restricting the spend to underdog weeks concentrated the
loss rather than turning it into a gain.

### Why, in one line

Width is cheap to buy and worth very little. A stack adds roughly half a point of lineup
standard deviation on the arithmetic, and around two points in realised terms, against a
weekly total whose spread is already 21-23 points. The cost is certain and the benefit is a
second-order reshaping of a distribution that is already wide relative to the margins being
decided. This is `prereg_variance.md`'s finding again through a different door: there is
almost nothing to buy with projected points in this game, and correlation, the one lever
that test could not see, turns out not to change that.

### Diagnostics, as preregistered

- **A stack was available** (a QB and an own pass catcher both eligible on the roster) in
  **36.3%** of seat-weeks under noise 1 and **30.2%** under noise 0. That is close to the
  32% a back-of-envelope gives for two quarterbacks and six receivers drawn over 32 teams,
  so the harness is not starving the policy of opportunities.
- **The consensus lineup already started one for free** in 16.1% (noise 1) and 10.4%
  (noise 0) of weeks, so roughly a third to a half of available stacks cost nothing and the
  control was already taking them.
- **The policy paid** in 2.65% / 2.26% of decisions for `stack1` and 1.94% / 3.30% for
  `stack_dog`, at a mean cost of 0.33-0.39 projected points (`stack1`) and 1.24-1.34
  (`stack_dog`). The bar bound: most unstarted stacks cost more than a point.
- The seat was the underdog in 28.7% (noise 1) and 44.4% (noise 0) of weeks; `stack_dog`
  found a stack worth paying for in 6.8% / 7.5% of those.
- `stack_wire` swapped the wire's add for a stack partner about once a seat-season, giving
  up a median 0.2 points of rest-of-season value over replacement, never more than 0.5.

### `correlate.py`'s default

The measurement stands on its own and does not depend on this test's outcome.
`qb_receiver_rho = 0.35` is too high as a blanket figure; **0.25** is the right default
against consensus residuals, which is what `correlate.blend` mixes into, and 0.285 against
a player's own mean. A role-aware version would be WR1 0.39, WR2 0.33, TE1 0.29, other
0.21. The setting is left at its committed value here because changing it is a change to
the live tools, not to this test, and `settings.correlations` is off in any case; the
recommendation is recorded for whoever turns it on. The opposing-offence correlation is
+0.217, measured and reported, and no arm acted on it.

### Honest read and residual leakage

Nothing in this run was close. The two lineup arms are negative in both designs on both
metrics and the effect sizes are a tenth of a point of all-play, which is the size of
noise in this harness rather than the size of a finding. `stack_wire` is positive on title
in both designs but positive in only 3 of 5 seasons in each, its all-play interval covers
zero, and its noise-0 title gain is one season out of five; it is the kind of result that
would evaporate on a re-run and should not be read as a lead.

Residual concerns I could not remove:

- The frozen roster path means an arm inherits the control's waiver-queue position. The
  lineup arms differ from the control in under 3% of decisions, so the channel can fire
  rarely, and the direction is not signable. It cannot plausibly account for a result this
  flat.
- The realised-spread split is on rosters that happen to hold a startable stack, not on a
  randomised assignment, so the +1.5 to +2.1 is an upper bound on the correlation's own
  contribution.
- The correlation is estimated on targeted pass catchers and 10-attempt starters, a
  slightly cleaner population than a fantasy roster. Both estimators (own-mean 0.285,
  consensus-residual 0.249) are in the same place, so this is a small effect on the bar.
- The draft was not touched in any arm, so "draft a stack" remains untested. Given that
  paying one projected point a week for a stack is already a loser, a draft-time version
  that pays real draft capital is unlikely to be the thing that works, but it is not what
  was run here.
