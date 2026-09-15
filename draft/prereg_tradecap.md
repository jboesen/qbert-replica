# Preregistration: roster-fit trades at a realistic rate and with mutual benefit

Frozen 2026-09-15, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check under `--noise 1` had already been run and is not part of the result; it
printed no outcome metrics and decided nothing. It found `stream` and `tradeA_k0`
identical to `sim_trades.py`'s rows for leagues 0-1 (every column, max difference 0.0);
the copied candidate search, with the lineup test off, returning exactly `sim_trades`'
trade in every searched week of every arm (asserted); every executed trade passing the
in-code invariants; no seat exceeding its cap; no capped trade projecting under 10 points;
and no mutual-arm trade lowering the opponent's projected lineup (asserted against the
executed rosters). It also showed that the lineup test binds less than expected: in the
uncapped `mutual` arm a positive value-accepted offer existed in 97% of searched weeks and
one passing the lineup test in 89%, and that arm still averaged 8.0 trades a season. That
is a count of offers, not an outcome, and it changed nothing below. No `--noise 0` check
was run; the real run re-checks both reproductions against `sim_trades.py`'s full result
files in both designs.
Any later change to the arms below is a new preregistration, not an edit to this one.

## Why this test

`prereg_trades.md` found that a seat trading for its own rest-of-season starting lineup,
on top of hole-aware streaming, gains +3.5 (noise 1) and +4.8 (noise 0) points of all-play
over streaming alone. Its own Known limits call that an upper bound on two counts. The seat
traded about 8.3 times a season, while teams in 1,205 real 12-team PPR Sleeper
league-seasons take part in 0.41 trades a season. And the opponents accepted any offer
that was fair by consensus value, even one that emptied their own starting lineup, which
is exactly the surplus a roster-fit seat harvests. The question here is whether trading
for fit still beats streaming when both assumptions are made realistic: few trades a
season, and opponents who only accept trades that don't hurt their own lineup.

## What the real trade counts say

From `data/trade_acceptance.parquet` (skill-player-only completed two-team trades, 1,205
league-seasons, 14,460 team-seasons; read before this file was written, a descriptive
count with no harness outcome in it): 82.1% of teams make no such trade in a season, 10.6%
make one, 3.6% two, 1.6% three, and 1.5% five or more. The 90th percentile team makes 1,
the 95th 2, the 99th 6. In leagues with at least one trade, the busiest team makes a
median of 2. So a seat that trades twice a season is an active manager, at about the 95th
percentile, but a common one: most leagues that trade at all have one.

## Arms

Every arm drafts by exact consensus, starts lineups by weekly consensus, and runs
hole-aware streaming on waivers for the test seat; the eleven other seats run the
consensus wire, exactly as in `prereg_trades.md`. The trade window, the offers searched
(every 1-for-1 and 2-for-1 with each opponent), roster legality, the 2-for-1 free-agent
add and opponent cut, the seat's lineup-value objective, and the tie order are all
unchanged from `prereg_trades.md` test A, with the endowment premium k = 0 (the calibrated
value). The code reuses `sim_trades.py`'s `search`, `execute`, `lineup_value`,
`ros_points`, `team_at` and `future_avail` unchanged; `sim_trades.py` is not edited.

| arm | opponent accepts | trades a season | minimum projected gain |
|---|---|---|---|
| `stream` (control) | no trades | 0 | |
| `tradeA_k0` (context) | fair by value | no cap | > 0 |
| `value_cap1`, `value_cap2`, `value_cap4` | fair by value | 1, 2, 4 | 10 points |
| `mutual` | fair by value and own lineup not lower | no cap | > 0 |
| `mutual_cap1`, `mutual_cap2`, `mutual_cap4` | fair by value and own lineup not lower | 1, 2, 4 | 10 points |
| `mutual_cap2_nomin` | fair by value and own lineup not lower | 2 | > 0 |

`stream` must reproduce `sim_trades.py`'s `stream` arm (itself identical to
`run_streaming`'s) exactly, and `tradeA_k0` must reproduce `sim_trades.py`'s `tradeA_k0`
exactly.

**Which arm carries the verdict: `mutual_cap2`, alone.** Both realism fixes at once is the
question the test asks; judging one arm avoids picking the best of nine afterwards. Two
trades a season is the busiest manager in a typical trading league and the 95th
percentile of all teams, so it is the most a real manager can plausibly be credited with,
not a typical one. All other arms are a labeled sensitivity: the caps 1 and 4 bracket
cap 2 (1 is the 90th percentile team, 4 roughly the 98th); the `value_cap` arms separate
the effect of the cap from that of the acceptance rule; `mutual` shows how much the
acceptance rule alone costs; `mutual_cap2_nomin` checks the minimum-gain rule.

## The cap and the minimum gain

A capped seat searches, each decision week in weeks 3-11, for its best accepted offer
exactly as uncapped test A does (the offer that most raises its rest-of-season starting
lineup value). It executes that offer only if the projected gain is at least **10
rest-of-season starting-lineup points**, and only while it has made fewer trades this
season than its cap. Otherwise it makes no trade that week and searches again the next
week. Once the cap is spent it stops searching. It never looks ahead to compare this
week's offer with later ones.

Why a threshold rather than taking the first positive offers: uncapped, the seat trades
in week 3 in 99% of seat-seasons and then keeps trading small gains most weeks. With a
cap, taking the first N positive offers would spend trades in weeks 3 and 4 on whatever
small edge exists, while a scarce trade should go to a big gain. A threshold is the
simplest rule that does that without lookahead, and it favours early trades only through
their larger gains (more weeks left), which is the right currency, since season points
are what win.

Why 10 points: set from the projected gains (a decision input, not an outcome) in the
uncapped `tradeA_k0` trade log of the frozen `prereg_trades.md` run. Across all its trades
the median projected gain is 6.8 (noise 1) and 5.3 (noise 0), the 75th percentile 13.9
and 10.7. Per seat-season, the median largest trade projects 33 and 25 points, the second
largest 16 and 14, the third 12 and 8, the fourth 8 and 6. So 10 points lets through
roughly the two or three biggest trades a seat makes when unconstrained: a cap of 1 or 2
binds on a seat with the usual offers, a cap of 4 binds on a seat with many. That log's
realised payoffs (`cf_pts`) and every outcome metric were not used to choose it. The same
10 points applies to every capped arm, and `mutual_cap2_nomin` reports what cap 2 does
with no threshold.

## Mutual-benefit acceptance

The opponent accepts an offer only if both hold:

1. **Fair by consensus value** (unchanged, k = 0): V(players it receives) >= V(player it
   gives), V the week-v consensus rest-of-season value over replacement.
2. **Its own lineup is not lower:** its rest-of-season starting-lineup value after the
   trade is at least its value before (tolerance 1e-6). This is the seat's own
   `lineup_value` on the opponent's roster: weeks v..17, best legal lineup on raw
   consensus rest-of-season points per game for week v, zero in a week the player is
   expected unavailable (bye from the schedule, and for week v the streaming policy's
   known-unavailable flags). In a 2-for-1 the opponent's roster after the trade is the one
   `sim_trades.execute` produces, with its lowest-valued cuttable player cut, so the cut
   is part of what it evaluates.

A trade the opponent sees as a zero change in its lineup (for instance bench for bench)
passes 2. The seat takes the best offer among those passing both, with the same tie order.
Under this rule both sides judge by the same consensus numbers: a trade is proposed only
when it is value-neutral and neither side's projected lineup gets worse, and the seat's
gets better.

## Information timing and leakage control

- Unchanged from `prereg_trades.md`: V and points per game are the latest rest-of-season
  consensus scrape strictly before week v through a rank curve fit only on seasons before
  y; availability reads only the schedule and week v-1 (or earlier) roster status and
  injury reports. The opponent's lineup test uses exactly the same inputs as the seat's.
- The cap threshold was chosen from projected gains in the previous run's trade log, which
  are decision inputs at the time of each trade. Stated residual: that log came from the
  same seasons, leagues and seeds this run uses, and its headline all-play result
  (`prereg_trades.md`) was known when this test was designed. The threshold was not tuned
  on any outcome, and a single value is fixed for all caps.
- The cap values come from real 2023-25 Sleeper trade counts, which overlap the test
  seasons 2023-25; they are counts, not fitted to any harness outcome.

## Scoring and decision rule

- Harness: `sim_tradecap.py`, seasons 2021-25, 20 leagues x 12 seats x 20 schedules, both
  opponent designs (`--noise 1` and `--noise 0`), all ten arms in one run per design.
- Primary: paired change in all-play win rate, each arm against `stream`, same league, seat
  and schedules. Secondary: title, playoff, wins, points.
- **Decision rule, on `mutual_cap2` only:** it passes only if, in **both** designs, its
  pooled all-play change against `stream` is positive, positive in at least 4 of 5
  seasons, and its pooled title change is no worse than -1.0 point. The same rule is
  applied to every other arm and printed, but those are sensitivity results, not verdicts.
- Also reported for context: every capped or mutual arm against uncapped `tradeA_k0`.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `sim_trades.py`; every arm
  shares those draws and the same drafts.
- Bootstrap: `LB.paired`, as in `prereg_trades.md` (season means, 2,000 resamples with
  `numpy.random.default_rng(1)`, 5th/95th percentiles). The interval is reported; the rule
  decides.
- Run once per design, one after the other through the shared lock. Reduced (`--leagues`
  other than 20) runs write `_check` files.

## Reported, not gated

- Trades per seat-season, by arm and by season; structure and positions received.
- For every searched week: whether a positive offer the opponent accepts on value alone
  exists, whether one also passing the lineup test exists, and whether one of those clears
  10 points. In the uncapped `mutual` arm, those shares by week, and in week 3 (untraded
  rosters, the same state in every arm) the number of such offers and the median best gain.
- Projected gain for the seat and the opponent per executed trade, and the realised
  starting-lineup points (weeks v..17, both rosters frozen) as in `prereg_trades.md`.
- In uncapped `tradeA_k0`: the share of executed trades that would also have passed the
  opponent's lineup test.

## Known limits

- Mutual benefit is judged on the same consensus numbers by both sides. A real opponent
  has his own opinion and can refuse for reasons the harness can't see (loyalty, dislike
  of a player, not answering). The rule is a necessary condition for a real trade, not a
  sufficient one, so a gain here is still an upper bound, only a tighter one.
- Offers are never refused at random and never countered; an offer that passes is taken.
- The seat can still search every opponent every week; only executed trades are capped.
- The opponent's lineup test uses the same byes-only view of the future as the seat.

## Result (added after the single run per design; the spec above is unchanged)

`stream` and `tradeA_k0` reproduce `sim_trades.py`'s rows exactly in both designs (1,200
rows each, every column, max difference 0.0), and `stream` still matches
`run_streaming`'s stream arm.

**The verdict arm, `mutual_cap2`, passes the rule in both designs.** Changes against
`stream`, 90% season-cluster intervals:

| opponents | title odds | all-play | all-play by season 2021-25 | title by season | reg pts | trades/season |
|---|---|---|---|---|---|---|
| noise 1 | -0.7 pp [-1.8, +0.6] | +1.1 pp [+0.8, +1.4] | +0.3 +1.0 +1.2 +1.4 +1.5 | -2.9 -1.5 -1.2 +1.2 +1.1 | +16 | 1.91 |
| noise 0 | +3.6 pp [-0.2, +7.4] | +1.9 pp [-0.6, +3.5] | +3.4 +2.5 -3.4 +4.3 +2.7 | -0.1 +4.2 -4.7 +9.0 +9.6 | +26 | 1.83 |

It passes narrowly on two counts: with noisy opponents the title change (-0.7) sits
inside the -1.0 guard, and with exact opponents 2023 is negative (-3.4) and the pooled
all-play interval reaches below zero. Under noise 1 it is positive in all five seasons
with an interval clear of zero.

**Sensitivity (not verdicts).** Every arm, against `stream` and against uncapped
`tradeA_k0`, all-play pooled with seasons positive:

| arm | noise 1 all-play | seasons + | title | trades | noise 0 all-play | seasons + | title | trades | vs tradeA_k0 (n1 / n0) |
|---|---|---|---|---|---|---|---|---|---|
| tradeA_k0 | +3.5 | 4 | +0.4 | 8.32 | +4.8 | 4 | +2.5 | 8.33 | |
| value_cap1 | +1.0 | 5 | +0.7 | 1.00 | +2.8 | 5 | -0.2 | 1.00 | -2.5 / -2.0 |
| value_cap2 | +1.7 | 5 | +1.5 | 1.95 | +3.0 | 5 | +0.8 | 1.85 | -1.8 / -1.8 |
| value_cap4 | +2.4 | 5 | +1.2 | 3.27 | +3.6 | 5 | +1.6 | 2.78 | -1.1 / -1.1 |
| mutual | +1.5 | 5 | -0.8 | 7.96 | +2.1 | 4 | +7.3 | 7.97 | -2.0 / -2.7 |
| mutual_cap1 | +0.9 | 5 | +0.8 | 0.99 | +2.0 | 4 | +2.9 | 1.00 | -2.6 / -2.8 |
| **mutual_cap2** | +1.1 | 5 | -0.7 | 1.91 | +1.9 | 4 | +3.6 | 1.83 | -2.4 / -2.9 |
| mutual_cap4 | +1.5 | 5 | +0.1 | 2.96 | +2.1 | 4 | +2.4 | 2.63 | -2.0 / -2.7 |
| mutual_cap2_nomin | +1.0 | 5 | -0.1 | 2.00 | +1.2 | 4 | +3.5 | 2.00 | -2.5 / -3.5 |

By season, all-play against `stream`: value_cap1 n1 +0.5 +0.5 +1.1 +0.9 +2.0, n0 +3.5
+2.6 +2.3 +3.8 +1.8; value_cap2 n1 +1.5 +0.9 +0.9 +2.2 +2.9, n0 +2.4 +2.7 +1.5 +5.2 +3.0;
value_cap4 n1 +1.2 +2.6 +0.6 +4.6 +2.9, n0 +1.5 +5.6 +1.7 +7.0 +2.4; mutual n1 +1.5 +1.8
+0.4 +2.7 +0.9, n0 +2.7 +3.7 -3.6 +4.6 +3.1; mutual_cap1 n1 +0.2 +0.6 +1.6 +0.8 +1.1, n0
+1.8 +3.0 -0.7 +3.4 +2.5; mutual_cap4 n1 +0.3 +1.9 +0.9 +2.3 +1.9, n0 +1.6 +2.0 -2.4 +6.2
+3.0; mutual_cap2_nomin n1 +0.4 +0.8 +1.3 +1.2 +1.4, n0 +1.4 +1.0 -2.7 +3.0 +3.4. Every
arm meets the rule in both designs. Every capped or mutual arm is below uncapped
`tradeA_k0` by 1-3.5 points of all-play pooled.

Reading the sweep: most of the uncapped gain survives the cap. One or two trades a
season, spent on the biggest projected gains, keep about 30-60% of it
(+1.0 to +3.0 against +3.5 and +4.8), because the first trade is by far the largest (median
projected gain 24-31 points for the single trade under cap 1, against 5-7 for an average
uncapped trade). Mutual benefit costs a further 0.6-1.1 points at cap 2, and the
10-point minimum adds a little over taking the first two offers (+1.1 against +1.0, and
+1.9 against +1.2).

**Reported, not gated.**

- Mutual-benefit offers are common. In the uncapped `mutual` arm, a positive offer the
  opponent accepts on value alone existed in 96% of searched weeks, and one that also
  leaves the opponent's lineup no worse in 88% (noise 1) and 89% (noise 0), falling from
  99-100% in week 3 to 75-78% in week 11. On untraded week-3 rosters a seat has a median
  of 474 (noise 1) and 365 (noise 0) positive value-fair offers, of which 331 and 247 are
  mutual; the best mutual offer projects a median 27.4 and 22.9 points against 30.8 and
  25.0 for the best value-only offer. A mutual offer worth at least 10 points existed in
  48% (noise 1) and 35% (noise 0) of the weeks `mutual_cap2` searched.
- In uncapped `tradeA_k0`, only 36% (noise 1) and 41% (noise 0) of executed trades leave
  the opponent's projected lineup no worse; the median opponent change is -3.4 and -1.6
  points. Capped value-only trades cost the opponent more (median -9.6 to -12.8 points
  under noise 1), which is what the mutual rule removes: capped mutual trades project +9 to +15
  points for the opponent as well as +15 to +29 for the seat.
- `mutual_cap2` used its full two trades in 92% (noise 1) and 83% (noise 0) of
  seat-seasons, at a mean week of 4.4 and 4.8; 29% and 63% of its trades were 1-for-1
  (the rest 2-for-1). Realised starting-lineup points per trade, rosters frozen: +12.5
  (noise 1) and +29.7 (noise 0).
- Both designs trade mostly for running backs, then receivers and quarterbacks.

Honest read: a seat that makes two trades a season, both of which its opponent's own
lineup numbers also favour, still beats streaming alone in both designs, by about 1-2
points of all-play, roughly what streaming itself added over the consensus wire. That is
a much smaller edge than the uncapped upper bound, and a weaker pass: the title change is
negative with noisy opponents, and one season is negative with exact ones. Two stated
limits remain. Mutual benefit is judged on the same consensus numbers by both sides, and
the offers are plentiful (most weeks have hundreds), so the harness still assumes an
opponent says yes to any offer his own projections favour; a real manager's no is not
modelled. And two trades a season is an active manager, about the 95th percentile of real
Sleeper teams.
