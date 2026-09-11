# Preregistration: market-aware draft

Frozen 2026-09-11, before any run of these arms. Any later change is a new
preregistration, not an edit to this one.

## Why this test

In the league backtest nothing in the model beats consensus, and consensus also predicts
season points better than our projection (2023–25, draftable pool, busts included:
correlation QB .39 vs .30, RB .69 vs .66, WR .64 vs .57, TE .51 vs .41). A weighted
stack of the two only ties consensus. So the edge, if there is one, is not a better
forecast. It is a better decision on top of consensus.

Consensus managers draft down one overall list. They don't plan around which players
will still be there at their next pick. So the question is: can a draft that uses
consensus values, but plans picks around predicted availability, beat drafting exactly
by consensus?

## Arms

Lineups are consensus (weekly ECR) for every arm. Only the test seat's draft differs.

- **exact** (control): draft by the noise-free consensus overall rank.
- **market**: value each player by consensus-implied season points (preseason
  positional ECR rank converted to points by a curve), priced over replacement with
  `add_vbd`. At each pick, assume opponents take players in exact consensus overall
  order until my next pick. Choose the position with the Fry–Lundberg–Ohlmann plan
  (`draft_dp.plan` logic and `BENCH_VALUE` unchanged) over those predicted
  availabilities, then take the highest-value player at that position that the roster
  rules allow.
- **market_stack**: the same policy, valuing players by the stacked projection from
  `stack.py`.

## Leakage control

- For each draft season y, the rank-to-points curve and the stack weights are fit only
  on seasons 2020 through y−1. For 2021 that is 2020 alone. Nothing is fit on the season
  being scored.
- Curve: per position, the mean actual season points at each preseason positional rank
  over the fit seasons, smoothed over a 7-rank window and made non-increasing. Ranks
  past 150 take the value at 150. Players who didn't play count as zero.
- Stack: per position OLS, actual ~ const + model + consensus points, on the draftable
  pool (either source ranks the player inside QB 30, RB 70, WR 80, TE 30). Negative
  coefficients are allowed and the output is clipped at zero. A player with no model
  projection takes his consensus points. A player with no consensus rank ranks one past
  the last ranked player at his position.
- `BENCH_VALUE`, the roster rules, eligibility, scoring and opponent noise are unchanged
  from the harness signed off in the earlier debate.

## Scoring and decision rule

- Harness: `league_backtest.py`, seasons 2021–25, 20 leagues × 12 seats × 20 schedules,
  run under both opponent designs (noise 1 × ECR sd, and noise 0).
- Primary: paired change in all-play win rate against **exact**, in the same league,
  seat and schedules. Secondary: title, playoff, wins, points, with a season-cluster
  bootstrap 90% interval.
- An arm is kept only if, in **both** opponent designs, its pooled change in all-play
  is positive, positive in at least 4 of 5 seasons, and its pooled change in title odds
  is no worse than −1.0 point. This is the rule already signed off for the harness.
- Run once.

## Declarations added before the run (Codex review)

- **Consumption forecast.** In both designs the policy forecasts that opponents take
  players in exact consensus overall order (noise-free ECR mean), because a real manager
  can see consensus but not each rival's personal deviations. Under noise 0 the forecast
  matches how opponents actually draft. Under noise 1 the mismatch is deliberate: it is
  the robustness case, where rivals deviate unpredictably. Roster rules (caps and
  forced need-filling) aren't in the forecast either.
- **Objective.** The plan maximizes projected starting-lineup value over replacement:
  the sum over my remaining picks of the value (`add_vbd` over the arm's season points)
  of the best player expected at the chosen position, counting a pick in full when it
  fills a starting slot (flex included) and at `BENCH_VALUE[pos]` otherwise. This is a
  projection of roster value. It isn't tuned to all-play, title odds or any observed
  result. `BENCH_VALUE` is the constant already in `draft_dp.py` from the initial
  commit, set before this study.
- **Seeds.** League lg of season y uses `numpy.random.default_rng([y, lg])`, lg = 0–19.
  It draws the thirteen noise vectors (twelve opponents plus the spare list used by the
  "ecr" arm) and then the twenty schedules, in that order, exactly as the harness
  already does. Every arm in a league shares those draws.
- **Bootstrap.** Per arm: the paired differences are averaged within each season,
  giving five season means. Resample the five with replacement 2,000 times using
  `numpy.random.default_rng(1)`, average each resample, and report the 5th and 95th
  percentiles (`numpy.quantile`, default linear interpolation). The interval is
  reported; the keep rule above decides.

## Result (added after the single run; the spec above is unchanged)

Both arms fail the rule in both designs, so neither is kept. Changes against the exact
consensus draft, consensus lineups, 90% season-cluster intervals:

| arm | opponents | title odds | all-play | all-play by season 2021–25 |
|---|---|---|---|---|
| market | noise 1 | −5.5 pp [−11.5, +0.1] | −6.4 pp [−11.8, −1.4] | −19.6 −5.7 −0.0 −0.2 −6.4 |
| market_stack | noise 1 | +1.2 pp [−5.4, +7.5] | −2.2 pp [−6.2, +0.7] | −11.3 −1.0 +1.1 +1.6 −1.3 |
| market | noise 0 | −1.5 pp [−5.7, +3.6] | −1.8 pp [−7.4, +3.4] | −15.4 +4.0 +6.5 −1.4 −2.5 |
| market_stack | noise 0 | +4.6 pp [−2.6, +11.7] | +0.7 pp [−4.7, +5.1] | −11.7 +4.2 +4.4 −0.6 +7.3 |

2021 is strongly negative for both arms, and it's the only season whose curve and
weights come from a single prior season (2020). That's a hypothesis for a new
preregistration, not a reason to reread this one.

## Known limits

- The plan ignores bye weeks and doesn't remove its own earlier picks from later
  availability (as in the original paper).
- Results for the old arms on 2023–25 were seen before this spec; these arms were not.
