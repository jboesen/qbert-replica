# Preregistration: win-probability lineups

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. Two mechanics checks
at 2 leagues per season (noise 1) and a Monte Carlo stability check on 2023 had already
been run; what they showed is listed under "Mechanics seen before freezing". None of
them computed or printed an all-play, title, playoff, wins or points outcome for either
arm. Any later change to the arms below is a new preregistration, not an edit to this
one.

## Why this test

Every attempt to out-forecast consensus has failed: our projections, our draft,
market-aware drafts, ADP, usage-based waivers and prediction-market props. Consensus
lineups beat our weekly model's lineups in every season. So this test needs no better
forecast. It keeps consensus's values and changes only the decision.

A consensus manager starts the players with the most expected points. But a week is won
by outscoring other teams, and all-play counts how many of the eleven you outscore. That
count isn't linear in your score. When a team is projected to lose, variance helps it;
when it's projected to win, variance hurts. So the lineup with the most expected wins
can differ from the lineup with the most expected points, even when every player's
expected points are exactly consensus's.

## Arms

Both arms draft by exact (noise-free) consensus. Opponents are unchanged from the
harness: noisy-consensus drafts and consensus lineups. Only the test seat's lineups in
weeks 1-14 differ.

- **cons** (control): consensus lineups every week, as in the harness (`week_points` on
  `ecr_val`). Its scores reproduce the harness's exact-draft, consensus-lineup arm
  exactly (checked at 2 leagues).
- **winprob**: in weeks 1-14, start the legal lineup with the most expected opponents
  outscored that week, as below. Weeks 15-17 keep consensus lineups. That's declared
  here because the harness scores a team's weeks once for all twenty schedule draws, so
  a playoff lineup can't be chosen against a specific opponent, and the regular season
  is where all-play is measured.

## The spread model

- **Value.** The harness's own consensus value: the weekly positional ECR rank (carried
  forward, questionable discount applied) read through `rank_curve`. The spread model
  is fit around the same mapping.
- **Fit data.** For test season y, player-weeks from seasons 2020 through y-1. For 2021
  that is 2020 alone (16 weeks of weekly ranks). A player-week is in the fit pool if he
  is ranked in that week's weekly consensus, is not Out, Doubtful or Questionable on
  the injury report, and his team (his most common team in that season's box scores) is
  not on bye. Roster status (ACT/INA) isn't used, because the weekly roster files start
  in 2021 and the 2021 fit has only 2020. A player-week with no box score counts as
  zero points, as it does in the harness.
- **Form.** Residual = actual PPR minus value. Per position, the pool is split into
  value quintiles (edges from the fit pool, repeated edges merged). A bucket with fewer
  than 30 residuals borrows the pool of the nearest bucket that has 30. Each bucket's
  residuals are centred to mean zero, so a player's expected score is exactly his
  consensus value and the test is only about shape.
- **Why empirical and not normal.** Weekly scoring is right-skewed with a floor near
  zero. On the 2021 fit (2020 only), residual skewness is 1.8 to 4.5 in the bottom
  three quintiles at RB, WR and TE, 0.8 to 0.9 at their top quintile, and near zero
  only for QBs above the bottom two quintiles. At RB, WR and TE the median residual is
  below the mean in every bucket (typically 1 to 2 points below). A normal would put
  too much weight on scores below zero for low-value players and too little on booms,
  and whether a boom-or-bust starter helps is exactly the question. The empirical pool
  keeps the zero mass and the tail. The 2025 fit (2020-24) has the same shape.
- **Questionable players.** The harness values a questionable player at his value times
  P(plays) (`weekly.play_probs(y)`, fit on reports before y). He is drawn as that
  mixture: with probability P(plays) he scores his undiscounted value plus a residual
  from the undiscounted value's bucket, otherwise zero. His expected score is still the
  harness's value.

## The policy

- **Information.** The test seat knows every opponent's roster and that opponents start
  by consensus. That's true in the simulation, and a real manager can see the consensus
  lineup. It uses the harness's pregame eligibility (active or INA, not on bye, not Out
  or Doubtful) and nothing that happens in the week.
- **Candidates.** Every lineup of eligible players with 1 QB, 2 RB, 2 WR, 1 TE and 1
  FLEX (RB, WR or TE), deduplicated as sets. Slots the roster can't fill stay empty, as
  in the harness. Pruning: within a (position, value bucket, questionable or not) group,
  only the top k players by value are kept, where k is the most starters that position
  can take (QB 1, RB 3, WR 3, TE 2). A player below that has the same score distribution
  shifted down, so swapping him for an unused player above him can't lower the
  expected count. The consensus lineup is always a candidate.
- **Objective.** Expected number of the 11 opponents the lineup outscores, strictly (as
  `all_play` counts). Monte Carlo: 10,000 draws per player per week, players drawn
  independently, every candidate lineup and every opponent scored on the same draws.
  For each of my draws, the expected count is the share of all opponents' pooled draws
  below my score; the estimate is the mean over my draws.
- **Choice.** Take the candidate with the highest estimate. Keep the consensus lineup
  unless that candidate's estimated gain over consensus exceeds 2 standard errors (the
  paired standard error of the per-draw gain over my draws). Ties keep consensus.
- **Seeds.** Each decision draws from `numpy.random.default_rng([y, lg, seat, w, 7])`,
  w = 0-13, separate from the harness streams and the same in both opponent designs.

## Leakage control

- The spread model for season y uses only seasons 2020 through y-1. For 2021 that is
  2020 alone. The value curve is the harness's `rank_curve`, fit on 2020 before any
  test season.
- Every input to a week-w decision is pregame: consensus ranks scraped before the
  Sunday slate, the official injury report, and the active roster. No actual points
  from week w or later enter a decision.
- The margin, draw count, bucket count and pruning were set before any outcome was
  computed; see below.

## Scoring and decision rule

- Harness: `draft/winprob.py`, which builds on `league_backtest.py`'s `Season`, `draft`,
  `schedules`, `season_outcome`, `all_play` and `paired`. Seasons 2021-25, 20 leagues x
  12 seats x 20 schedules, under both opponent designs (`--noise 1` and `--noise 0`).
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, lg = 0-19,
  drawing the thirteen noise vectors then the twenty schedules, exactly as `prereg.md`.
  Both arms in a league share those draws and the same drafted rosters.
- Primary: paired change in all-play win rate, winprob minus cons, in the same league,
  seat and schedules. Secondary: title, playoff, wins, points.
- Kept only if, in **both** designs, the pooled all-play change is positive, positive
  in at least 4 of 5 seasons, and the pooled title change is no worse than -1.0 point.
- Bootstrap: the paired differences average within each season to five season means;
  resample the five with replacement 2,000 times with `numpy.random.default_rng(1)`,
  average each resample, report the 5th and 95th percentiles (`numpy.quantile`,
  default interpolation). The interval is reported; the rule decides.
- Reported, not gated: the share of seat-weeks whose lineup differs from consensus
  (overall and by season); the model's expected gain in those weeks against the
  realized change; and the paired weekly all-play change split by whether the consensus
  lineup was projected as an underdog (expected count below 5.5 of 11) or a favourite.
- Run once per design. Real results go to `data/league_winprob_noise{1,0}.parquet`. A
  reduced run (`--leagues` other than 20) writes only
  `data/league_winprob_check{L}_noise{N}.parquet` and prints mechanics, never outcomes.

## Mechanics seen before freezing

- At 2,000 draws and no margin, lineups were legal in 1,680 of 1,680 seat-weeks, and
  departed from consensus in 5-15% of weeks by season. On 84 seat-weeks of 2023,
  rerunning with a second Monte Carlo seed picked a different lineup in 18% of weeks,
  and the estimated gain where it departed was about 0.016 opponents. Most departures
  were chasing simulation noise. That is why the draw count went to 10,000 and the
  2-standard-error margin was added. With both, the same check agreed across seeds in
  99% of weeks.
- At the frozen settings (2 leagues, noise 1): lineups legal in 1,680 of 1,680
  seat-weeks, 0.12 seconds per decision, lineups differing from consensus in 0.3-2.1%
  of weeks by season, and the control arm reproducing the harness's exact-consensus arm
  exactly.
- **The model's own forecast of the effect**, from the same check (a projection, not an
  outcome): the consensus lineup is an underdog in 19% of weeks, the mean projected
  count is 6.65 of 11, the expected gain where the policy departs is 0.05 opponents,
  and the implied pooled all-play gain is about +0.004 points. If the spread model is
  right, the true effect is far below the season-to-season noise, so the sign test
  across seasons is close to a coin flip on the few weeks that differ. A pass would not
  be evidence of a real edge of useful size, and a fail would not be evidence of harm.
  Recorded here so neither result is overread.

## Result (added after the single run; the spec above is unchanged)

The winprob arm fails the rule in both designs. It changes almost nothing, and where it
does change something it does slightly worse, as the model's own forecast above said
was likely.

| design | title odds | all-play | lineup differs | model-expected gain where it differs | realized |
|---|---|---|---|---|---|
| noise 1 | -0.0 pp | -0.0 pp | 0.6% of weeks | +0.039 opponents | -0.224 |
| noise 0 | +0.0 pp | -0.0 pp | 0.4% of weeks | +0.033 opponents | -0.587 |

All-play is non-positive pooled and in most seasons of both designs, so the rule fails
on its first condition. The underdog and favourite splits are both within a few
hundredths of a point of zero.

The read: with spread depending only on position and value bucket, two players of
similar consensus value have nearly the same distribution, so there is almost no
variance to trade and the best-expected-wins lineup is the consensus lineup in more than
99% of weeks. The few departures chase differences the model can estimate but can't
earn. The variance lever that plausibly exists in real leagues is correlation (stacking
a quarterback with his receiver, or playing against your opponent's players), which
this model deliberately doesn't see; that would be a separate preregistration.

## Known limits

- Players are drawn independently. Real scores are correlated within an NFL team (a QB
  and his receivers) and within a game, and that correlation is a real variance lever
  (stacking) this model can't see.
- The spread depends only on position and value bucket, not on role, game script or
  the betting total, so within a bucket every player is equally volatile. A policy can
  only prefer variance it can see, and this model sees little.
- One-season fit for 2021.
- Weeks 15-17 are untouched, so any title effect comes through seeding only.
