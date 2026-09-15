# Preregistration: look-ahead streaming on the waiver wire

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check had already been run and is not part of the result; it printed no outcome
metrics and decided nothing. In both designs it found no illegal rosters (no duplicate
players, sizes unchanged, no position below its starting count, after every waiver round),
the horizon's week-w slice identical to `known_unavailable` in every season, and a
control identical to `run_streaming`'s `stream` rows on all seven stored columns. It found
117 (noise 1) and 134 (noise 0) look2 pre-acquisitions over 120 seat-seasons, and 158 and
194 for look3, about two thirds of those two weeks ahead. Adds were mostly RB, then QB and
TE, and drops were mostly surplus WRs. Any later change to the arms below is a new
preregistration, not an edit to this one.

## Why this test

Hole-aware streaming (`prereg_streaming.md`) is the one waiver decision that has beaten the
consensus wire: fill next week's starting hole, even with a player consensus ranks below
the one cut. It only acts once the hole is a week away. By then every team ahead of it in
priority has had that waiver round, and a waiver round earlier, to take the player who
fills it. Streaming practice (Subvertadown's streaming write-ups; r/fantasyfootball threads
recommending 2-4 week holds) says to grab the future fill before the rest of the league
competes for him. The question is whether acting one or two rounds early is worth the
rest-of-season value, and the earlier use of the week's only move, that it costs.

## Arms

Every arm drafts by exact (noise-free) consensus and starts lineups by weekly consensus.
All eleven other seats run the consensus wire policy in every arm. Only the test seat's
waiver decision differs. One add/drop per team per week, priority worst all-play record
first, roster size fixed at 14, no FAAB, no trades, no IR slot, no team may cut a position
below its starting count (`ROSTER_MIN`), all exactly as in `run_streaming`.

- **stream** (control): the test seat runs `streaming_policy`. This is the `stream` arm of
  `run_streaming`, and must reproduce it exactly.
- **look2**: look-ahead streaming with horizon H = 2 (weeks w and w+1).
- **look3**: look-ahead streaming with horizon H = 3 (weeks w, w+1 and w+2).

Two horizons are run because the community range (2-4 week holds) doesn't pick one and
the harness has no basis for choosing. H = 4 is left out: a 17-week season has at most one
bye per team, and a four-week hold on a 14-man roster ties up a bench slot for a quarter of
the season.

## Availability ahead and information timing

The decision before week w (w = 2..17) is made after week w-1 has been played and before
week w's injury report, roster status or weekly consensus rank exists. No week-w-or-later
roster, injury or consensus source is read.

A rostered player is **known unavailable for week w+k** (k = 0..H-1, w+k <= 17) at the
decision before week w if any of these hold, reading the same row `known_unavailable` reads:

1. His team is on bye in week w+k, from the season schedule (`games.csv`), public before the
   season. His team is the one on the row in 2.
2. On the latest weekly roster file at or before week w-1 in which he appears, his status is
   not ACT or INA.
3. He was listed Out or Doubtful on the final injury report for that same week.
4. He has no roster row at all at or before week w-1.

Conditions 2-4 are carried unchanged to every week of the horizon: a current out is assumed
to persist, and no future injury report is used. For k = 0 this is `known_unavailable`
exactly (checked). A consequence, declared here: since 2-4 are the same for every k, a
seat with no hole in week w can only have a future hole from a bye. Look-ahead is in effect
bye planning.

Holes for a week are the starting slots (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX) the players not
known unavailable that week cannot fill, by the harness's `needs()`. A position fills a
hole if it is short at that position, or is RB, WR or TE and the FLEX is short.

## The look-ahead decision (arm with horizon H)

- **Hole in week w:** the streaming decision, exactly, including its fallback to the
  consensus move when no legal move closes the hole. No look-ahead is tried that week.
- **No hole in week w:** for k = 1..H-1 in order (the earliest future week first), skipping
  weeks past 17 and weeks with no hole:
  - The add is the free agent with the highest rest-of-season value (the same
    `over_replacement` consensus value the control uses for week w) among those at a
    position that fills a week-(w+k) hole and not known unavailable in week w+k.
  - The drop comes from the cuttable players (`ROSTER_MIN` on the pre-add counts). A drop
    is legal only if, after the add, week w+k has strictly fewer holes than before and no
    week w..w+H-1 has more holes than before (so week w stays hole-free). Among legal
    drops, minimise in order: holes left in week w+k; total holes across the horizon;
    whether he is a week-w starter (the lineup filled by rest-of-season value from players
    not known unavailable in week w); rest-of-season value (unvalued players first).
  - The move is made regardless of whether the add outranks the drop, as in streaming.
  - If the week has no candidate add or no legal drop, try the next week in the horizon.
- **Otherwise:** the consensus move.

The decision uses the free-agent pool at the moment of the seat's turn in priority order.
look2 and look3 are identical whenever the only future hole is in week w+1, or there is
none within two weeks.

## Leakage control

- Rest-of-season values are the control's: the latest rest-of-season consensus scrape
  strictly before week w (preseason ranks before the first), through the rank curve fit
  only on seasons before y, priced over replacement. Nothing is fit here.
- Availability reads only the schedule and week w-1's (or earlier) roster status and
  injury report, published before the week-w waiver decision.
- Scoring is unchanged: every seat's weekly lineup uses weekly consensus and the harness's
  real eligibility, in every arm.
- Known residuals: a trade or signing after week w-1 is invisible (the bye check uses the
  old team), which can only hurt the policy. The schedule is the final one in `games.csv`;
  a game moved mid-season (rare in 2021-25) is treated as known in advance.

## Scoring and decision rule

- Harness: `draft/sim_lookahead.py`, importing `league_backtest.py` unchanged. Seasons
  2021-25, 20 leagues x 12 seats x 20 schedules, both opponent designs (`--noise 1` and
  `--noise 0`).
- Primary: paired change in all-play win rate, each look arm against `stream`, in the same
  league, seat and schedules. Secondary: title, playoff, wins, points.
- An arm passes only if, in **both** designs, its pooled all-play change is positive,
  positive in at least 4 of 5 seasons, and its pooled title change is no worse than -1.0
  point. Each arm is judged on its own.
- Look-ahead is kept if at least one arm passes. If both pass, the adopted horizon is the
  one with the larger all-play change averaged over the two designs. Two arms are two
  chances to pass by luck; that is declared here rather than corrected, and a pass by only
  one arm should be read with it in mind.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, as `run_streaming` does; all arms of a
  league share those draws and the same drafts.
- Bootstrap: `league_backtest.paired`, as in `prereg_streaming.md`: season means of the
  paired differences, 2,000 resamples of the five with `numpy.random.default_rng(1)` per
  comparison, 5th/95th percentiles. The interval is reported; the rule decides.
- Run once per design. Reduced (`--leagues` other than 20) runs write `_check` files and
  cannot overwrite a real result. Results: `data/league_lookahead_noise{1,0}.parquet`;
  pre-acquisition log: `data/league_lookahead_moves_noise{1,0}.parquet`.

## Reported, not gated

For each look arm and design:

- Pre-acquisitions per seat-season (moves made through the look-ahead branch), by season,
  their lead time, positions, how often consensus would have made the same move, and how
  often the add ranks below the drop.
- **Still needed when his week arrived**, on the roster that played the target week:
  still rostered; *needed* (removing him would reopen a known hole that week, by the
  same availability reading made at the decision before that week); started (in the
  weekly-consensus lineup with real eligibility).
- **Would streaming alone have lost him**: in the `stream` arm of the same league and
  seat, at the test seat's turn in the waiver round before the target week, whether the
  player was free, on the seat's own roster, or on another team (*taken*); also that rate
  restricted to cases where the stream seat then had a hole his position fills, and
  whether he was free in the stream arm when look-ahead took him. The two arms' leagues
  have diverged by then, so this is an approximate counterfactual.
- The target week's paired score, look arm minus `stream`.

## Known limits

- All of `prereg_streaming.md`'s limits (one move a week, all-play priority, Questionable
  counts as available, last week's injury status as a stale proxy).
- Future holes are bye-driven by construction; injury look-ahead would need forecasts
  of return, which this test does not use.
- The opponents run the consensus wire and never stream, so they don't compete for bye
  fills on purpose. They take a bye fill only if consensus ranks him highest. Against
  streaming opponents the early claim could be worth more.

## Result (added after the single run per design; the spec above is unchanged)

Look-ahead streaming fails the rule. look3 passes with noisy opponents but fails with exact
ones; look2 fails both. Changes against streaming on the test seat, 90% season-cluster
intervals:

| comparison | opponents | title odds | all-play | all-play by season 2021-25 | reg pts | rule |
|---|---|---|---|---|---|---|
| look2 minus stream | noise 1 | +0.0 pp [-0.2, +0.4] | +0.0 pp [-0.1, +0.2] | -0.2 -0.0 -0.0 +0.4 -0.0 | +1 | fails |
| look3 minus stream | noise 1 | +0.2 pp [-0.4, +0.8] | +0.2 pp [+0.0, +0.3] | -0.1 +0.3 +0.3 +0.4 +0.1 | +3 | passes |
| look2 minus stream | noise 0 | -1.4 pp [-2.7, -0.1] | -0.2 pp [-0.9, +0.4] | -1.5 +0.3 +0.9 +0.3 -0.9 | -4 | fails |
| look3 minus stream | noise 0 | -0.7 pp [-1.2, -0.3] | -0.3 pp [-0.6, +0.1] | -0.7 -0.3 -0.1 +0.4 -0.7 | -4 | fails |

Title odds by season: look2 noise 1 -0.3 +0.1 +0.1 +0.7 -0.3, noise 0 +0.4 -4.2 +0.6 -1.2
-2.4; look3 noise 1 -0.5 +0.7 +1.5 +0.0 -0.8, noise 0 -0.2 -0.0 -0.6 -1.2 -1.7.

Reported, not gated. The test seat pre-acquires 1.1 players a season under look2 (both
designs) and 1.5 (noise 1) and 1.6 (noise 0) under look3, two thirds of look3's two weeks
ahead. Adds are mostly RB, then QB and TE. Consensus would have made the same move 7-15%
of the time, and the add ranks below the drop in 43-55%.

- **Still needed when his week came.** On the roster that played the target week, the
  pre-acquired player was still rostered 64% (look2, noise 1) / 70% (noise 0) and 55% /
  64% (look3); still needed to cover a known hole 53% / 57% and 42% / 46%; started 54% /
  58% and 44% / 53%. With a two-week lead only 50-56% were still rostered and 37-40%
  still needed. The cut is not traced in the log; the likely route is the consensus move
  the following week, which drops the lowest-valued cuttable player without regard to
  holes, and a bye fill is often exactly that player.
- **Would streaming alone have lost him.** In the stream arm, at its turn before the
  target week, the player was on another team 44-52% of the time, free 40-44%, and
  already on the seat's own roster 7-13%. Where the stream seat then had a hole his
  position fills (63-84% of cases), another team held him 49-55% of the time. He was free
  in the stream arm when look-ahead took him in 91-96% of cases, so this is mostly a
  claim made in between, though the two arms' leagues had diverged by then.
- **Target-week score**, look arm minus stream: look2 -0.2 (noise 1) and -2.8 (noise 0);
  look3 +1.3 and -0.7.
- Total adds per season: stream 10.9, look2 11.1, look3 11.4 (noise 1); 12.6, 13.0, 13.1
  (noise 0).

The race the community advice describes is real here: about half the time the player who
would fill the hole is gone by the week it arrives. Winning that race doesn't pay. A third
to a half of the early claims no longer cover a hole when the week comes, and each one
spends the week's single move and a roster spot on a player consensus ranks lower. The
net is about zero with noisy opponents and slightly negative with exact ones, and the one
arm that passes one design (look3, noise 1, +0.2 pp) is one of two chances. Streaming
stays the policy.
