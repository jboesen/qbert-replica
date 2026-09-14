# Preregistration: hole-aware streaming on the waiver wire

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check (legal rosters, holes detected, moves sane, the control reproducing
`run_waivers`' `cons` arm exactly) had already been run and is not part of the result;
it printed no outcome metrics and decided nothing. It found no illegal rosters, 8-27
hole-driven moves per 12 seats per league, and a control identical to `run_waivers`'
`cons` rows in both designs. Any later change to the arms
below is a new preregistration, not an edit to this one.

## Why this test

`prereg_waivers.md` showed the wire is a real lever (one seat working it alone gains 3-5
points of all-play over a league where nobody does) and that consensus rest-of-season rank
is hard to beat as a signal: reading usage directly lost to it. So this test keeps the
consensus signal and changes the decision. The consensus wire policy ignores need. It adds
the best rest-of-season free agent at any position, even when the team has no starting
hole there and does have one elsewhere, for instance when its only tight end is on bye or
was ruled out last week. A streaming manager fills next week's hole instead, even with a
player consensus ranks below the one he cuts. The question is whether points banked in the
week of a hole are worth more than the rest-of-season value the move gives up.

## Arms

Every arm drafts by exact (noise-free) consensus and starts lineups by weekly consensus,
as in `prereg_waivers.md`. All eleven other seats run the consensus wire policy in both
arms. Only the test seat's waiver decision differs. One add/drop per team per week,
priority worst all-play record first, roster size fixed at 14, no FAAB, no trades, no IR
slot, and no team may cut a position below its starting count (`ROSTER_MIN`), all exactly
as in `run_waivers`.

- **cons** (control): the test seat runs the consensus wire policy. This is the `cons` arm
  of `run_waivers`, and must reproduce it exactly.
- **stream**: the test seat runs hole-aware streaming, defined below.

## The hole definition and information timing

The decision before week w (w = 2..17) is made after week w-1 has been played and before
week w's injury report, roster status or weekly consensus rank exists. None of those three
week-w sources is read.

A rostered player is **known unavailable for week w** if any of these hold:

1. His team is on bye in week w, from the season schedule (`games.csv`), which is public
   before the season.
2. On the latest weekly roster file at or before week w-1 in which he appears, his status
   is not ACT or INA (the harness's own "on the 53" definition; RES, DEV, CUT and the rest
   count as unavailable). The file has no rows for teams on bye, so a player whose team was
   off in week w-1 is read from his last week before that.
3. He was listed Out or Doubtful on the final injury report for that same week (the week
   of the roster row in 2).
4. He has no roster row at all at or before week w-1.

His team for the bye check is the team on that same roster row.

The test seat's **holes for week w** are the starting slots (1 QB, 2 RB, 2 WR, 1 TE,
1 FLEX from RB/WR/TE) that its players not known unavailable cannot fill, computed with
the harness's own `needs()` on the counts of those players. A position **fills** a hole
if it has a shortfall at that position, or if it is RB, WR or TE and the FLEX is short.

## The streaming decision

- **No hole:** do exactly what the consensus policy does (add the best free agent by the
  control's rest-of-season value, drop the lowest-valued cuttable player, only if the add
  is worth more).
- **Hole:** among free agents with a finite rest-of-season value (the same
  `over_replacement` consensus value the control uses for that week) at a position that
  fills a hole and not known unavailable for week w (so not on bye), take the one with the
  highest value. Pick the drop from the cuttable players (the `ROSTER_MIN` rule on the
  pre-add counts, as in `run_waivers`) whose removal, after the add, leaves strictly fewer
  holes than before the move. Among those, the drop minimises, in order: holes left after
  the move; whether he is a week-w starter (in the lineup filled by rest-of-season value
  from players not known unavailable); rest-of-season value (players the policy cannot
  value first). The move is made regardless of whether the add outranks the drop.
- **Hole but no legal filling move** (no such free agent, or no drop that reduces holes):
  fall back to the consensus policy's move for that week.

The decision uses the same free-agent pool at the moment of the seat's turn in priority
order, so a higher-priority team can still take the player it wanted.

## Leakage control

- Rest-of-season values are the control's: the latest rest-of-season consensus scrape
  strictly before week w (preseason ranks before the first scrape), through the rank curve
  fit only on seasons before y, priced over replacement.
- Availability for week w reads only the schedule, and week w-1's (or earlier) roster
  status and injury report. Those are published before week w-1's games, which are played
  before the week-w waiver decision.
- Scoring is unchanged: week-w lineups are set by weekly consensus with the harness's real
  week-w eligibility for every seat and every arm, so no policy gets week-w information
  the others don't.
- Known residual: a mid-week trade or signing between week w-1 and week w is invisible to
  the policy (it reads the old team for the bye), which can only hurt it.

## Scoring and decision rule

- Harness: `league_backtest.py --streaming`, seasons 2021-25, 20 leagues x 12 seats x 20
  schedules, both opponent designs (`--noise 1`, opponents perturbed by 1 x ECR sd, and
  `--noise 0`, exact consensus).
- Primary: paired change in all-play win rate, `stream` against `cons`, in the same
  league, seat and schedules. Secondary: title, playoff, wins, points.
- Kept only if, in **both** designs, the pooled all-play change is positive, positive in
  at least 4 of 5 seasons, and the pooled title change is no worse than -1.0 point.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `prereg.md` and
  `run_waivers` do; both arms of a league share those draws and the same drafts.
- Bootstrap: the paired differences average within each season to five season means;
  resample the five with replacement 2,000 times with `numpy.random.default_rng(1)`,
  average each resample, report the 5th/95th percentiles (`numpy.quantile`, default
  linear interpolation). The interval is reported; the rule decides.
- Run once per design, one after the other. Reduced (`--leagues` other than 20) runs write
  to `_check` files and cannot overwrite a real result.

## Reported, not gated

- How often the test seat makes a hole-driven move, and how often that move differs from
  the move the consensus policy would have made in the same state (same roster, same free
  agents, same week).
- For those differing moves, the change in the test seat's starting-lineup points in week
  w: (a) counterfactual, same roster and week, with the streaming move against with the
  consensus move (or no move, if consensus would not have moved); (b) paired, the `stream`
  arm's week-w score against the `cons` arm's, where the rosters may already differ from
  earlier moves.
- Total adds per season in each arm.

## Known limits

- One add/drop a week and all-play priority, as in `prereg_waivers.md`.
- A Questionable player counts as available; only Out and Doubtful count as holes.
- Last week's injury status is a stale proxy for this week's: a player out in week w-1 can
  be back in week w (a false hole, costing a move) and a player hurt in week w-1's game is
  not yet on any report (a missed hole).
- No lookahead: the policy fills this week's hole and does not plan for next week's byes.

## Result (added after the single run per design; the spec above is unchanged)

Streaming passes the rule in both designs. Changes against the consensus wire policy on
the test seat, 90% season-cluster intervals:

| comparison | opponents | title odds | all-play | all-play by season 2021-25 | reg pts |
|---|---|---|---|---|---|
| stream minus consensus | noise 1 | +1.6 pp [+1.0, +2.0] | +1.0 pp [+0.8, +1.3] | +1.1 +0.8 +1.6 +0.6 +1.0 | +11 |
| stream minus consensus | noise 0 | +1.2 pp [+0.2, +2.0] | +2.0 pp [+1.3, +2.8] | +2.7 +1.8 +3.5 +0.6 +1.5 | +21 |

Title odds by season: noise 1 +1.8 +2.0 +2.3 +0.3 +1.4; noise 0 -1.1 +1.6 +2.0 +2.6 +0.7.

Reported, not gated. The test seat makes 1.7 (noise 1) and 2.1 (noise 0) hole-driven moves
a season, of which 1.5 and 1.7 differ from what consensus would have done in the same
state. In 47% (noise 1) and 34% (noise 0) of hole-driven moves the add is worth less
than the drop by rest-of-season value. On the differing moves, that week's starting
lineup scores +6.3 (noise 1) and +7.8 (noise 0) points more than with consensus's move on
the same roster, and the stream arm outscores the control arm that week by +5.6 and +6.4.
Adds are mostly RB (844 and 840), then TE, QB, WR. Total adds per season rise from 9.9
to 10.9 (noise 1) and from 11.9 to 12.6 (noise 0).

The effect is small, about one to two points of all-play and 10-20 regular-season points,
but it is positive in every season in both designs. It is the first decision in this
harness that beats consensus while using only consensus information.
