# Preregistration: the draft market against expert consensus

Frozen 2026-09-11, before any run of these arms. Any later change is a new
preregistration, not an edit to this one.

## Why this test

Everything tested so far has been an opinion about players: our projection, a stack of
it with consensus, a plan over consensus availability. All of them lose to drafting by
exact consensus. ADP is a different kind of object. It isn't an opinion, it's a price,
set by what thousands of drafters actually did, and prices aggregate information that no
single panel of experts holds.

Two questions, and they are not the same one:

1. Is the market a better draft signal than the experts? (arm **adp**)
2. Where the two disagree, is the disagreement worth money? (arm **adp_gap**)

The second is the interesting one. A drafter who values players by consensus but pays
market prices should never spend an early pick on a player the market will leave him
anyway. That is the only thing ADP tells you that ECR cannot.

The honest prior is a tie. Over 2021-25, ECR and ADP positional ranks correlate .95 to
.99 by Spearman, and ECR is very slightly the better forecast of season points at every
position. Two signals that collinear can hardly separate in a league simulation. This
test is written to be able to return "no difference", and that is the expected result.

## Data

- ADP: Fantasy Football Calculator, PPR, 12 teams, free API, one snapshot per season
  (`draft/adp.py` -> `data/adp_ffc.parquet`). Attribution requested by FFC.
- The API serves a single window per season, the last days of drafts before the opener;
  `date` and `rounds` parameters are accepted but do not change what is returned. The
  windows are 2021-08-31/09-01, 2022-09-03/04, 2023-08-30/09-01, 2024-08-31/09-01,
  2025-08-25/09-01, against openers 2021-09-09, 2022-09-08, 2023-09-07, 2024-09-05,
  2025-09-04. Every snapshot closes before its season starts. The window is stored in
  the parquet so the check is auditable later.
- This is the same timing as the consensus the harness already uses: the last ECR scrape
  before the opener. Neither signal has seen a down of football. The comparison is fair
  in both directions.
- Mapping to nflverse gsis ids is by normalised name plus position through the
  dynastyprocess crosswalk, as `consensus.py` does; 100% of QB/RB/WR/TE rows match in
  all five seasons after five nickname aliases.
- ADP is on a 12-team board, so an ADP of 30.0 is directly comparable to pick number 30
  in this harness's 12-team snake. No rescaling.

## Arms

Lineups are consensus (weekly ECR) for every arm, throughout. Only the test seat's draft
differs. The control is **exact**, the noise-free consensus draft already in the harness.

- **adp**: draft by the ADP order, noise-free. Players with an ADP come first, ascending;
  then every remaining player with a finite consensus overall rank, ascending by that
  rank. The tail is not cosmetic: the skill-position ADP list runs 146 to 206 deep by
  season, and a 14-round 12-team draft consumes 168 picks, so in 2022 the list is
  exhausted before the draft ends. Ties in ADP break by the order the source lists them
  (stable sort). This is the ADP twin of **exact**, and the answer to question 1.

- **adp_gap**: value players by consensus-implied season points over replacement
  (`S.cons_vbd`, the same quantity the failed `market` arm used: preseason positional ECR
  rank through a curve, priced with `add_vbd`), but pay market prices. At each of my
  picks, let `n` be my next pick number. Among the players available and allowed by the
  roster rules, call a player **unsafe** if his ADP is less than `n`, meaning the market
  says he will be gone before I pick again. A player with no ADP is safe: undrafted in a
  12-team market means he lasts. Let `best` be the highest-value allowed player.
  - If `best` is unsafe, take `best`.
  - Otherwise take the highest-value unsafe player, provided his value is at least
    `value(best) - DELTA`; if there is no such player, take `best`.
  - In the final round there is no next pick, so take `best`.
  `DELTA = 10.0` PPR season points, fixed here before the run. It is the most value the
  policy will give up to take the player the market won't leave it, and it is about 0.6
  points a week over a 17-week season, a small share of the value spread between
  consecutive picks in the early rounds. It is not tuned; no value of DELTA other than
  10.0 will be run.

## Leakage control

- Every fitted quantity uses only seasons before the one being scored. The rank-to-points
  curve behind `cons_vbd` is `stack.fit_before(y)`, fit on 2020 through y-1, unchanged
  from the harness.
- Both signals are pre-opener snapshots of the season being drafted, as above. Nothing
  in either arm is fit on the scored season.
- `BENCH_VALUE` is not used by either arm. Roster rules, caps, eligibility, scoring and
  opponent noise are unchanged from the harness signed off in the earlier debate.
- DELTA is a constant declared in this file, not estimated from data.

## Scoring and decision rule

- Harness: `league_backtest.py`, seasons 2021-25, 20 leagues x 12 seats x 20 schedules,
  run under both opponent designs (noise 1 x ECR sd, and noise 0).
- Primary: paired change in all-play win rate against **exact**, in the same league, seat
  and schedules. Secondary: title, playoff, wins, points, with a season-cluster bootstrap
  90% interval.
- An arm is kept only if, in **both** opponent designs, its pooled change in all-play is
  positive, positive in at least 4 of 5 seasons, and its pooled change in title odds is
  no worse than -1.0 point. This is the rule already signed off for the harness.
- Run once per design. Mechanics may be checked before the run at a reduced league count;
  outcomes may not.

## Declarations added before the run

- **Lineups.** The new arms are scored with consensus lineups only. The other lineup
  policies are not run for them, because this test is about the draft and the harness
  already showed consensus lineups beat ours. This saves three quarters of the new arms'
  runtime and removes no comparison this rule uses.
- **Opponents.** Opponents draft from perturbed ECR in both designs, unchanged. Neither
  new arm tells its opponents anything: **adp** is a fixed list, and **adp_gap**'s notion
  of who will be gone comes from the market snapshot, not from the opponents' actual
  lists. Under noise 0 the opponents follow consensus exactly, so ADP is then a
  deliberately wrong model of the room; that mismatch is the point of running both
  designs, not a defect.
- **Objective.** `adp_gap` maximises consensus-implied roster value subject to not
  spending a pick on a player the market would leave. It is not tuned to all-play, title
  odds or any observed result.
- **Seeds.** League lg of season y uses `numpy.random.default_rng([y, lg])`, lg = 0-19,
  drawing the thirteen noise vectors then the twenty schedules, exactly as the harness
  already does. Every arm in a league shares those draws.
- **Bootstrap.** Per arm: the paired differences are averaged within each season, giving
  five season means. Resample the five with replacement 2,000 times using
  `numpy.random.default_rng(1)`, average each resample, and report the 5th and 95th
  percentiles (`numpy.quantile`, default linear interpolation). The interval is reported;
  the keep rule above decides.

## Descriptive result recorded before the arms were run

Rank correlation with actual season PPR points, stack.py draftable pool, busts counted
as zero, 2021-25, players carrying both signals (n=873):

| | QB | RB | WR | TE | all |
|---|---|---|---|---|---|
| ECR | .403 | .572 | .578 | .434 | .520 |
| ADP | .391 | .559 | .567 | .405 | .509 |
| ECR~ADP | .952 | .975 | .978 | .970 | .984 |

ECR is ahead at all four positions and in four of five seasons. This is a descriptive
statement about forecast accuracy and does not decide the arms, which are about draft
decisions. It was computed before the arms were run and is recorded here so it cannot be
presented afterwards as a prediction.

## Result (added after the single run; the spec above is unchanged)

Both arms fail the rule in both designs, so neither is kept. Changes against the exact
consensus draft, consensus lineups, 90% season-cluster intervals:

| arm | opponents | title odds | all-play | all-play by season 2021-25 |
|---|---|---|---|---|
| adp | noise 1 | -6.1 pp [-10.1, -2.5] | -4.5 pp [-6.2, -2.8] | -8.0 -6.2 -2.4 -2.5 -3.3 |
| adp_gap | noise 1 | -7.0 pp [-12.9, -1.6] | -6.3 pp [-13.7, -0.6] | -23.0 -5.1 -4.1 +4.1 -3.7 |
| adp | noise 0 | -5.3 pp [-6.5, -4.0] | -7.8 pp [-9.6, -5.9] | -10.3 -5.0 -10.3 -4.8 -8.5 |
| adp_gap | noise 0 | +0.6 pp [-5.2, +7.5] | -0.6 pp [-10.0, +8.0] | -21.8 +13.6 -3.2 +8.8 -0.2 |

Both arms are negative in most seasons under both designs, and `adp` is negative in
every season of both. This matches the descriptive result above: ADP correlates with
actual points slightly worse than consensus, so trading consensus's board for a market
board that agrees with it 95-98% of the time but forecasts a little worse cannot help,
and paying market prices for consensus-implied value on top of that (`adp_gap`) does not
recover it either. The market carries nothing here that consensus doesn't already have,
once you're only allowed to draft off one list or the other.

## Known limits

- One ADP snapshot per season, from one site, on drafts concentrated in the final days
  before the opener. Real drafts happen over months at prices that move.
- FFC's pool is one free public source; a different market (Underdog best ball, half-PPR,
  18 rounds, no start/sit) would price differently and was not used for that reason.
- The opponents are synthetic consensus drafters, so a market-exploiting policy is being
  tested against a room that is not the market that set the prices.
