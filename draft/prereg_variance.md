# Preregistration: situational, opponent-aware variance in weekly lineups

Frozen 2026-09-15, before the real (LEAGUES=20) runs of these arms. Mechanics checks at
1 and 2 leagues per season had already been run; what they showed is under "Mechanics
seen before freezing". None of them computed or printed an all-play, title, playoff,
wins or points outcome for any arm. Any later change to the arms below is a new
preregistration, not an edit to this one.

## Why this test

`prereg_winprob.md` asked which lineup beats the most of the eleven other teams. It left
the consensus lineup in under 1% of weeks and changed nothing. Its own read was that the
objective, expected count of opponents outscored, is nearly linear in score around the
middle of the field, so there is almost nothing to trade.

The variance argument that managers actually make is not about the field. It is about
one opponent and one situation:

- Head to head, the lineup that maximises your chance of winning depends on the team you
  play this week. A twenty-point underdog should buy variance; a twenty-point favourite
  should buy a floor. Beating the field says nothing about either.
- A seat whose playoff life turns on this week should pay more for a better chance of
  winning it than a seat already locked in or already out.

This test keeps consensus's numbers, keeps the best realistic policy so far, and changes
only the weekly lineup decision.

## Arms

Every arm drafts by exact (noise-free) consensus, runs hole-aware streaming on waivers,
and makes at most two mutual-benefit trades a season: `mutual_cap2` from
`sim_tradecap.py`, the verdict arm of `prereg_tradecap.md`. The eleven other seats are
unchanged from the harness (noisy or exact consensus drafts, consensus lineups, the
consensus wire). Only the test seat's weeks 1-14 lineups differ. Weeks 15-17 keep
consensus lineups in every arm, as in `prereg_winprob.md`: the playoff bracket and its
opponents are decided inside `season_outcome`, and the regular season is where all-play
is measured.

- **mutual_cap2** (control): consensus lineups every week. Must reproduce
  `sim_tradecap.simulate` under spec `("mutual", 2, 10.0)` exactly, in scores, waiver
  moves and final rosters.
- **h2h3**: every week, the legal lineup with the highest probability of outscoring this
  week's head-to-head opponent, among lineups that give up at most **C = 3** projected
  points against the consensus lineup.
- **h2h12**: the same, with **C = 12**.
- **situ** (the preregistered verdict): the same, with C set each week by the situation,
  as defined below.
- **live**: C = 12 while the seat's playoff odds are strictly between 0.05 and 0.90,
  C = 0 otherwise. A coarse version of `situ`: spend everything while alive and
  unsettled, nothing once locked in or out.

## The budget, and why the price is in projected points

The consensus lineup is the legal lineup with the most projected points (that is what
`week_points` builds). So every other lineup costs projected points. **C is the most
projected points an arm will pay to change the shape of its score.** C = 0 is the
consensus lineup. C does not say which direction to move: the win-probability objective
buys width when the seat is an underdog and a floor when it is a favourite, on its own.

## The spread model

Imported unchanged from `winprob.py` (`fit_spread`, `draw`, `candidates`,
`consensus_lineup`, `legality`) and specified in `prereg_winprob.md`: for test season y,
the empirical distribution of actual weekly PPR around the harness's own consensus value
(weekly positional ECR rank read through `rank_curve`), per position and value quintile,
fit on player-weeks of seasons 2020 through y-1 only, each bucket centred to mean zero so
a player's expected points are exactly consensus's. A questionable player is drawn as the
harness's mixture: he plays with probability `weekly.play_probs(y)["Questionable"]` and
then scores like a player of his undiscounted value, otherwise zero.

## The opponent

The opponent is drawn, not treated as a fixed number. This week's head-to-head opponent
starts its consensus lineup (`consensus_lineup`, the lineup `week_points` would field on
its roster in the control's path, which is what it actually starts in the simulation).
Every player in that lineup is drawn from the same spread model, on the same Monte Carlo
scenarios as the seat's own candidates.

`opp_mu`, the opponent's projected total, is that lineup's consensus value; the week's
spread is the seat's consensus-lineup projection minus it. The spread is reported, not a
policy input: it enters only through the win probability.

## The weekly decision

- **Information.** Pregame eligibility exactly as the harness computes it, the weekly
  consensus ranks, the injury report, every roster in the league (visible on any real
  platform), and the results of weeks already played in this schedule. Nothing from week
  w or later.
- **Candidates.** `winprob.candidates`: every legal lineup (1 QB, 2 RB, 2 WR, 1 TE,
  1 FLEX) of eligible players, after dropping players dominated inside their own
  (position, value bucket, questionable) group. The consensus lineup is always
  candidate 0.
- **Objective.** P(my lineup's total > the opponent's total), estimated on **DRAWS =
  2000** independent scenarios per player per week, every candidate and the opponent
  scored on the same scenarios. For each of my draws the probability is the share of the
  opponent's draws below my score; the estimate is the mean over my draws.
- **Choice.** Among candidates whose projected points are at least (consensus minus C),
  take the highest estimate; the first maximum wins, so a tie keeps consensus. Keep
  consensus unless the winner beats it by more than **2.0** paired Monte Carlo standard
  errors, the margin `prereg_winprob.md` froze for the same reason: without it most
  departures chase simulation noise.
- **Seeds.** Each seat-week draws from `numpy.random.default_rng([y, lg, seat, w, 11])`,
  w = 0-13, separate from the harness streams, identical across arms, budget levels and
  both opponent designs. The eleven opponents' scenarios come from the same stream, so
  every arm and every schedule in a seat-week sees the same simulated world.

## Playoff odds and the situational budget

Playoff odds are computed **from games played so far only**, inside the schedule being
played. Before week w (0-indexed), every team's head-to-head wins and points-for through
weeks 0..w-1 are known, and every team has the same number of games left.

- Each remaining game is a coin flip, so a team's final wins are its current wins plus
  Binomial(left, 1/2). The gap between my final wins and team t's is normal with mean
  (my wins - t's wins) and variance left/2, giving
  p_t = Phi((my wins - t's wins) / sqrt(left/2)). With no games left, p_t is 1, 0, or
  the points-for comparison on a tie, which is the harness's own seeding tiebreak.
- Playoff odds = P(at least 6 of the eleven p_t come in behind me), by exact Poisson
  binomial over the eleven, treating them as independent. They are not independent
  (opponents play each other). This number only has to sort weeks into five budget
  levels, and it is stated here as an approximation, not a forecast.
- **Leverage** = (playoff odds if I win this week) - (playoff odds if I lose it), both
  evaluated with `left - 1` games remaining, my wins and my opponent's wins set by the
  result, and the other ten teams carrying half a win for their own unresolved week.
  Points-for is taken through week w-1 in both branches.

`situ` sets **C = 3 x round(4 x leverage)**, clipped to [0, 12]: levels 0, 3, 6, 9, 12.
Leverage is near zero both when the seat is locked in and when it is out, so `situ`
plays consensus in both; it is highest on the bubble and in genuine must-win weeks. A
week is called a **must-win** in the diagnostics when leverage >= 0.20.

An eliminated seat gets C = 0 under both situational arms. That is deliberate. With the
playoffs gone, the only thing left to maximise is points, and the consensus lineup is the
maximum-points lineup; the variance argument has nothing to bite on. It is recorded here
because the opposite ("chase variance when out of it") is the folk version of the rule.

## The schedule/scoring separation

The harness scores every roster once per week and then replays 20 schedule draws, so an
opponent-aware lineup breaks that separation: the lineup, and therefore the seat's score,
depends on which schedule is being played. This test scores the seat **per schedule**, at
the harness's default SCHEDULES = 20, as follows.

1. **The roster path is taken from the control**, once per (season, league, seat): the
   draft, every waiver move by all twelve teams, and the seat's mutual-benefit trades,
   recorded week by week. `sim_variance.path` is `sim_tradecap.simulate` with a weekly
   snapshot added; the mechanics check asserts its scores, moves and final rosters are
   identical.
2. **Each seat-week is priced once**, on that path: the candidate lineups, their
   projected and realised points, and, for each of the eleven possible head-to-head
   opponents and each of the five budget levels, the lineup the policy would start and
   the win probability it and the consensus lineup carry. This table is schedule-free.
3. **Each schedule is then played week by week**: look up the week's opponent, compute
   the situation from the standings so far in that schedule, read the lineup off the
   table, and take its realised points. All-play, wins, playoff berth and title are
   computed per schedule (`season_outcome` on the control's score matrix with the seat's
   row replaced) and averaged over the twenty.

**Why fixing the roster path is valid, and where it is not.** Waiver priority
(`LB.priority`) is deliberately schedule-free already: it ranks teams by all-play record,
not by head-to-head record, precisely so that the wire does not have to be re-run per
schedule draw. Trades key off rosters and consensus rest-of-season value, not off
standings. So the only channel by which a lineup change can move the roster path is the
seat's own weekly score feeding its all-play record and therefore its place in the waiver
queue. Freezing the path at the control's does three things:

- It makes the pairing exact. In a given league and seat, the control and every arm hold
  the same players every week and the eleven opponents score identically; the paired
  difference is the lineup decision and nothing else. Nothing else in this repo's tests
  has that clean a contrast.
- It introduces no future information. The control's path is built from consensus-lineup
  scores alone, with the same weekly information the control had.
- It leaves one known bias, stated and bounded: a variance arm whose weekly scores differ
  from the control's would, in a full re-simulation, have taken a slightly different place
  in the waiver queue in later weeks. The arm inherits the control's queue position
  instead. The direction is not signable a priori (a higher score means a worse waiver
  position, a lower score a better one), and the mechanics check reports how often the
  lineup differs at all, which bounds how often the channel can fire.

A full per-schedule re-simulation would cost twenty times the trade search and twenty
times the wire, per arm, and would still not remove schedule-driven roster divergence; it
would add it. This design was chosen before any outcome was computed.

## Leakage control

- The spread model for season y uses only 2020 through y-1. The rank curve is fit on 2020
  before any test season. Trade and waiver values are `consensus.weekly_curve(2020..y-1)`,
  as in `sim_tradecap.py`.
- Playoff odds and leverage use only results of weeks already played in that schedule.
  No future schedule, no future score, no end-of-season standing.
- Every lineup input is pregame. The candidate set uses `S.elig`, the harness's own
  pregame eligibility (on the 53, not on bye, not Out or Doubtful).
- The budget levels, C_MAX, the step, the margin, the draw count and the must-win
  threshold were all set before any outcome was computed; see below.

## Scoring and decision rule

- Harness: `draft/sim_variance.py`, building on `league_backtest.py` (`Season`, `draft`,
  `schedules`, `season_outcome`, `all_play`, `paired`, `streaming_policy`,
  `waiver_week`), `sim_trades.py`, `sim_tradecap.py` and `winprob.py`. No existing module
  is edited. Seasons 2021-25, 20 leagues x 12 seats x 20 schedules, both opponent designs
  (`--noise 1` and `--noise 0`).
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `prereg.md` and
  `sim_tradecap.py`. Every arm in a league shares those draws, the drafted rosters and
  the roster path.
- Primary: paired change in all-play win rate, arm minus `mutual_cap2`, in the same
  league, seat and schedules. Secondary: title, playoff, wins, points.
- Kept only if, in **both** designs, the pooled all-play change is positive, positive in
  at least 4 of 5 seasons, and the pooled title change is no worse than -1.0 point.
  **`situ` carries the verdict**; the other three arms are sensitivities.
- Bootstrap: the paired differences average within each season to five season means;
  resample the five with replacement 2,000 times with `numpy.random.default_rng(1)`,
  average each resample, report the 5th and 95th percentiles (`LB.paired`). The interval
  is reported; the rule decides.
- Reported, not gated: the share of seat-week-schedule decisions whose lineup differs
  from consensus, overall and by season and by arm; the mean budget spent; the projected
  points given up and the change in lineup standard deviation where it differs; the
  model's expected win-probability gain in those weeks against the realised change in
  head-to-head wins; the same split for must-win weeks; and how often it departs when the
  consensus lineup is the underdog.
- Run once per design, at harness defaults. Real results go to
  `data/league_variance_noise{1,0}.parquet`. Any reduced run (`--leagues` or
  `--schedules` other than 20) writes only `data/league_variance_check{L}x{S}_noise{N}.parquet`
  and prints mechanics, never outcomes.

## Mechanics seen before freezing

- **The spread model has very little shape to sell, and this was known before the run.**
  Residual standard deviation rises monotonically with consensus value in every position
  and both the 2021 and 2025 fits: RB 1.9 -> 8.1, WR 3.1 -> 8.2, TE 1.2 -> 6.8, QB 2.9 ->
  8.2 points from the bottom to the top quintile (2025 fit; the 2021 fit is 2.0 -> 8.3,
  3.6 -> 8.6, 1.3 -> 6.8, 2.1 -> 8.2). So the maximum-points lineup is close to the
  maximum-variance lineup, and paying projected points generally buys a *narrower*
  distribution, not a wider one. The one place the model does see real extra width at
  little cost in expected points is a questionable player, whose harness value is already
  discounted by his chance of playing but whose draw is bimodal.
- Consequently the honest prior for this test is that the underdog half of the argument
  has almost nothing to act on, and what action there is will mostly be favourites and
  locked-in seats buying floors, plus questionable-player swaps. This is recorded so
  neither a pass nor a fail is overread, exactly as `prereg_winprob.md` recorded its own
  forecast. The mechanics check prints, per budget level, the narrowest and widest lineup
  standard deviation the seat could legally field; those numbers are in the Result
  section's diagnostics.
- At the frozen settings, 2 leagues per season x 20 schedules, noise 1 (1,680 seat-weeks,
  470,400 lineup decisions):
  - The control reproduces `sim_tradecap.simulate` under `("mutual", 2, 10.0)` exactly:
    identical weekly scores for all twelve teams, identical waiver-move counts, identical
    final rosters, in every league and seat.
  - Every candidate lineup the policy can start is legal by `winprob.legality` in 1,680
    of 1,680 seat-weeks. Candidates per seat-week: mean 182, max 1,000.
  - 0.188 seconds per seat-week table, max 1.05. A full run is roughly an hour per design.
  - The lineup leaves consensus in 0.87% of decisions for `h2h3` and `h2h12`, 0.69% for
    `situ` and 0.66% for `live`. The binding constraint is the two-standard-error margin,
    not the budget: raising C from 3 to 12 does not raise the departure rate at all.
  - `situ` spends a mean budget of 3.14 projected points and calls 55% of decisions
    must-win at the 0.20 leverage threshold, which is what a six-of-twelve playoff in a
    fourteen-week season produces; it is a label for the diagnostics, not a rare event.
  - The mean-variance frontier, lineup standard deviation by budget level: C = 0 gives
    21.00 to 21.03, C = 3 gives 20.53 to 21.23, C = 6 gives 19.98 to 21.31, C = 9 gives
    19.64 to 21.34, C = 12 gives 19.37 to 21.36. Twelve projected points buys at most
    +0.33 of extra width but up to -1.63 of floor. The frontier is lopsided in exactly
    the direction the bucket standard deviations above predict.
- **What this test can and cannot show, stated before the run.** The departure rate is
  the same order as `winprob`'s, and the available width is small, so the headline all-play
  and title changes will almost certainly be small. What is new here, and what the
  diagnostics are for, is *direction*: whether the departures line up with the situation
  and the spread the way the argument says they should, and whether the model's expected
  win-probability gain in those weeks is realised. A pass on a change this size would not
  be evidence of a useful edge; a fail would not be evidence of harm. It would be evidence
  about whether the situational story survives contact with a model that prices variance.

## Result (added after the single run per design; the spec above is unchanged)

The control reproduces `sim_tradecap`'s `mutual_cap2` exactly in both designs, as the
mechanics check required.

**The verdict arm, `situ`, fails the rule in both designs.** So does every other arm. The
effect is not small and negative; it is indistinguishable from nothing. Pooled paired
changes against `mutual_cap2`, over 336,000 lineup decisions per arm per design (20
leagues x 12 seats x 20 schedules x 14 weeks x 5 seasons):

| design | arm | all-play | seasons + | title | lineup differs |
|---|---|---|---|---|---|
| noise 1 | h2h3 | -0.014 pp | 2/5 | -0.004 pp | 0.69% |
| noise 1 | h2h12 | -0.014 pp | 2/5 | -0.004 pp | 0.69% |
| noise 1 | **situ** | **-0.007 pp** | **3/5** | **+0.004 pp** | **0.53%** |
| noise 1 | live | -0.006 pp | 3/5 | +0.004 pp | 0.51% |
| noise 0 | h2h3 | -0.008 pp | 3/5 | -0.017 pp | 0.74% |
| noise 0 | h2h12 | -0.008 pp | 3/5 | -0.017 pp | 0.74% |
| noise 0 | **situ** | **-0.021 pp** | **0/5** | **-0.013 pp** | **0.66%** |
| noise 0 | live | -0.023 pp | 0/5 | -0.013 pp | 0.64% |

Every bootstrap interval is [-0.1, +0.1] pp or tighter on both metrics. All-play by season
(2021-25), `situ`: noise 1 +0.001 +0.004 -0.044 +0.009 -0.005; noise 0 -0.080 -0.009
-0.007 -0.002 -0.010. The rule fails on its first condition (pooled all-play positive) in
both designs, so nothing downstream matters.

**`h2h3` and `h2h12` are identical in every column, in both designs.** Twelve projected
points of budget buys exactly the same lineups as three. The binding constraint is the
two-standard-error margin, not the price: where a departure is confident enough to survive
the margin, it costs under three projected points anyway. Mean projected points actually
given up where the lineup differs: 0.12 (noise 1) and 0.08 (noise 0), against a budget of
12. The budget lever, the whole mechanism this test added over `winprob`, never fires.

**The departures go the wrong way for the argument.** Where the lineup differs, its
standard deviation *falls* by 0.44 points (noise 1) and 0.17 (noise 0), and it is wider
than consensus in only 33-44% of those weeks. It starts 0.23 fewer questionable players
(noise 1). So the policy is overwhelmingly a favourite shedding a boom-or-bust starter,
not an underdog buying a swing. That is the half of the argument the spread model can
act on, and the mechanics section predicted it: residual standard deviation rises
monotonically with consensus value, so the maximum-points lineup is already close to the
maximum-variance lineup and a budget mostly buys a floor. Measured on the frontier, twelve
projected points bought at most +0.33 of extra width against -1.63 of floor.

**The model's expected gain is not realised.** Where it departs, the model expected
+0.0071 (noise 1) and +0.0075 (noise 0) of head-to-head win probability. Realised, over
those same weeks: -0.026 and +0.009 for the `h2h` arms, -0.013 and -0.015 for `situ`,
-0.011 and -0.019 for `live`. In raw head-to-head results, `situ` won 1,078 of its
departure weeks against consensus's 1,101 on the same weeks (noise 1) and 1,150 against
1,182 (noise 0). The counts are small enough that these are coin flips, but there is no
sign of the expected gain arriving.

**Situation is measured, and it changes almost nothing.** `situ` called 57% (noise 1) and
61% (noise 0) of decisions must-win at the 0.20 leverage threshold, and in those weeks it
left the consensus lineup in 0.7% of them, the same rate as everywhere else. Its expected
gain there was +0.0072 and +0.0076, realised -0.0143 and -0.0080. Scaling the budget by
leverage moved the departure rate from 0.69% to 0.53% (noise 1), which is the whole of the
situational effect: it makes the policy act slightly *less* often, because low-leverage
weeks get budget 0 and fall back to consensus.

### Honest read

This is `winprob`'s result again, reached from the opposite direction and with the
schedule separation actually handled. Making the objective head-to-head instead of
against-the-field, letting the lineup pay projected points for shape, and scaling that
price by playoff leverage all changed the *decision rule* substantially and the *decisions*
almost not at all: under 1% of weeks, and those mostly benching a questionable player
against a favourite.

The reason is the spread model, not the situational argument. Within this model two
players of similar consensus value have nearly the same distribution, and a worse player
is both lower-mean and lower-variance, so the mean-variance frontier available on a real
roster is nearly a point. An underdog who wants a swing has nothing to buy. That was
measured and written into the preregistration before the run, so neither the sign nor the
size here should be read as evidence about whether situational variance matters in real
leagues. It is evidence that this spread model cannot express it.

The lever that would express it is correlation, which `prereg_winprob.md` also named:
stacking a quarterback with his receiver widens a lineup at no cost in expected points,
and starting players who face your opponent's players narrows the margin's spread
directly. Both are invisible to a model that draws every player independently. That is a
separate preregistration, and it is the only version of this test worth running next.

### Known limits

- Players are drawn independently. No stacking, no game correlation, no bring-back. This
  is the binding limit, as above.
- The roster path is the control's, not re-simulated per schedule. The bias channel
  (waiver queue position) can only fire in weeks where the lineup differs, which is under
  1% of them, so it cannot be carrying this result either way.
- Playoff odds treat the eleven head-to-head comparisons as independent and each remaining
  game as a coin flip. A better estimate would change which weeks get which budget, but
  the budget itself never binds, so it would change nothing here.
- Weeks 15-17 keep consensus lineups, so any title effect comes through seeding only.
- One-season spread fit for 2021.
