# Preregistration: draft structure on top of consensus values

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check (legal rosters, the control reproducing `run_streaming`'s `stream` arm,
the test seat's position mix by round, how often each rule changes a pick) had already
been run in both designs and is not part of the result; it printed no outcome metrics and
decided nothing. It found no illegal roster in any seat or arm, a control identical to the
`stream` rows of `data/league_streaming_noise{1,0}.parquet` (120 of 120 rows each), and no
fallbacks. One parameter was changed after the first check, on position mix alone: the
fall allowance F in `late_qb_te` went from 12 to 24 picks, because with 12 the seat still
took a QB in rounds 3-6 in about 30% of drafts under noise 1, so the arm was not the
late-QB policy it is meant to be. With 24 it takes 0.01 QB and 0.01 TE through round 6
(noise 1) and none (noise 0). Any later change to the arms below is a new preregistration,
not an edit to this one.

## Why this test

Every draft test so far changed the values the seat drafts on (our board, VBD, a
market-aware plan, ADP) and lost to drafting by exact consensus. The one policy that beat
consensus (`prereg_streaming.md`) kept consensus's numbers and changed a decision. So this
test keeps the exact consensus order and changes only the roster structure: which
positions the seat may take in which rounds. Outside evidence points three ways:

- Underdog Best Ball Mania winners took 2 RB and 3 WR through round 6 and no early TE.
- A Reddit simulator of 1M drafts found punting QB lifted title rate to 9.86% from 9.01%
  for ADP drafting, with no waivers in that simulation.
- A 50k-league study found the best time to draft RBs flips from year to year, and
  commenters note it depends on what the other managers do.

None of that is a 12-team PPR head-to-head league with a waiver wire and consensus
opponents, which is what this harness measures.

## Arms

Every arm starts lineups by weekly consensus, and the test seat works the wire with
hole-aware streaming (`streaming_policy`, the `stream` arm of `run_streaming`) while all
eleven other seats run the consensus wire. Waiver rules are unchanged: one add/drop per
team per week, all-play priority, roster 14, no FAAB, trades or IR, `ROSTER_MIN`. Only the
test seat's draft differs. Every arm drafts in exact consensus overall order (`S.exact_order`,
noise-free preseason overall ECR) and takes the first player in that order who is
available, allowed by the harness roster rules (`allowed`: caps QB 2, TE 2, RB 7, WR 7; no
second QB or TE before round 9; forced need-filling at the end), and allowed by the arm's
structural rule. If the structural rule leaves no legal player, the seat takes the exact
consensus pick under the harness rules alone (a fallback, counted).

- **exact** (control): no structural rule. This is `run_streaming`'s `stream` arm and must
  reproduce it exactly.
- **late_qb_te**: no QB before round 9 and no TE before round 8 (rounds 1-indexed),
  unless the player has fallen at least F = 24 picks past his consensus rank, meaning
  `pick number - consensus overall rank >= 24`, where the rank is his 1-based place in
  `S.exact_order` and the pick number is the overall pick (1-168).
- **template**: through round 6 the seat must finish with at least 2 and at most 3 RB, at
  least 3 and at most 4 WR, and at most one QB and TE combined. A pick in rounds 1-6 is
  allowed only if, after it, no maximum is exceeded and the RB and WR minimums can still be
  met with the seat's remaining picks through round 6. So five of the six picks are RB/WR
  and the sixth may be a third RB, a fourth WR, or one QB or TE. From round 7 on, no rule
  beyond the harness's.
- **reactive**: before each of the seat's picks, for each position p with pool size
  N (QB 12, RB 36, WR 36, TE 12), let the pool be the top N players at p in exact consensus
  order. `left` is the number of pool players still available (not taken by any seat,
  the test seat included). `expected` is N minus the number of pool players among the
  first (pick number - 1) entries of `S.exact_order`, what would be left if the room had
  drafted exactly by consensus. The relative surplus is `(left - expected) / N`. A
  position is under-drafted by the room when this exceeds 0.10 (at least 2 QB or TE, or 4
  RB or WR, more left than expected). If any position is under-drafted, take the first
  player in exact consensus order at any under-drafted position (allowed by the harness
  rules); else, or if none is allowed, take the first allowed player in exact order.
  The brief called this "relative scarcity" but asked the seat to lean toward the
  position the room is under-drafting; the statistic is defined in that direction, as the
  surplus the room has left.

Declared expectation, from mechanics only: under `--noise 0` every opponent drafts exactly
by consensus, and in the check `reactive` never picked differently from `exact` (0.00
differing picks per draft). If that holds in the real run, its noise-0 paired change is
exactly zero, which fails the rule (a change must be strictly positive). That is the
correct verdict: a reactive rule has nothing to react to in a room that follows consensus.

## Information timing and leakage control

- The draft reads only the preseason overall consensus list (the last scrape before the
  opener, as every draft arm already does), the picks made so far in this draft, and
  constants fixed above. Nothing is fit, so no in-season or same-season information enters.
- Q = 9, T = 8, F = 24, the template, the pool sizes and the 0.10 threshold come from the
  outside evidence above and the league's starting slots (12 teams x starters, with RB and
  WR sized for the flex). None was chosen by looking at an outcome metric. F was changed
  once on the mechanics check's position mix, as stated at the top.
- The streaming wire and consensus lineups are unchanged from `prereg_streaming.md`,
  including its information timing. Nothing in-season differs by arm except the roster
  the draft produced.

## Scoring and decision rule

- Harness: `draft/sim_draftstruct.py`, importing `league_backtest.py` unchanged, seasons
  2021-25, 20 leagues x 12 seats x 20 schedules, both opponent designs (`--noise 1`,
  opponents perturbed by 1 x ECR sd, and `--noise 0`, exact consensus).
- Primary: paired change in all-play win rate, each arm against `exact`, in the same
  league, seat and schedules. Headline: paired change in title odds. Secondary: playoff,
  wins, points.
- An arm is kept only if, in **both** designs, its pooled all-play change is positive, it
  is positive in at least 4 of 5 seasons, and its pooled title change is no worse than
  -1.0 point. Each arm is judged on its own; no arm is a correction for another.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `run_streaming` does; all
  four arms of a league share those draws. Opponents' drafts can still differ between arms
  because the test seat takes different players.
- Bootstrap: the paired differences average within each season to five season means;
  resample the five with replacement 2,000 times with `numpy.random.default_rng(1)`
  (reset per arm, via `league_backtest.paired`), average each resample, report the
  5th/95th percentiles (`numpy.quantile`, default linear interpolation). The interval is
  reported; the rule decides.
- Run once per design, one after the other. Reduced runs write to `_check` files and
  cannot overwrite `data/league_draftstruct_noise{1,0}.parquet`.

## Reported, not gated

- Round-by-round position mix of the test seat per arm, and mean counts through round 6.
- Consensus value drafted: the sum over the seat's 14 picks of `S.cons_vbd` (preseason
  positional consensus rank through the rank curve fit on seasons before y, priced over
  replacement, unvalued players as 0), per arm and by season.
- Weekly starting-lineup points of the test seat, weeks 1-17, for `exact` and as changes
  from it for each arm.
- Picks per draft that differ from the exact consensus pick in the same state, and
  fallbacks.

## Known limits

- Opponents are consensus plus noise, not real drafters with position tendencies. Under
  noise 1 their deviations are random, not a correlated run on a position, so `reactive`
  reacts to noise. The Reddit and 50k-league findings are about rooms that behave unlike
  this one.
- The best-ball evidence comes from a format with no waivers and no lineup decisions, where
  depth is worth more. Here a streaming wire offsets a thin position, which could favour
  late-QB/TE rules or make them irrelevant.
- The rules are fixed; no Q, T, F, template or threshold other than those above will be run.

## Result (added after the single run per design; the spec above is unchanged)

No arm passes the rule in both designs, so none is kept. Changes against the exact
consensus draft, all with the streaming wire and consensus lineups, 90% season-cluster
intervals:

| arm | opponents | title odds | all-play | all-play by season 2021-25 | reg pts | verdict |
|---|---|---|---|---|---|---|
| late_qb_te | noise 1 | -2.1 pp [-5.7, +0.8] | -2.5 pp [-4.4, -1.1] | -0.9 -6.9 -2.2 -2.0 -0.6 | -24 | fails |
| template | noise 1 | -0.2 pp [-2.4, +1.6] | +0.6 pp [-0.0, +1.4] | -0.2 -0.0 +0.1 +2.4 +0.9 | +9 | fails (3 of 5) |
| reactive | noise 1 | +0.7 pp [-0.3, +1.6] | +0.3 pp [-0.1, +0.6] | -0.1 +0.5 +1.1 -0.2 -0.1 | +3 | fails (2 of 5) |
| late_qb_te | noise 0 | +0.6 pp [-4.6, +6.2] | -1.4 pp [-3.9, +1.0] | -7.2 -3.2 +1.4 +1.4 +0.6 | -14 | fails |
| template | noise 0 | +0.7 pp [-1.9, +3.2] | +0.7 pp [+0.3, +1.3] | -0.2 +0.7 +0.5 +2.0 +0.6 | +8 | passes this design |
| reactive | noise 0 | +0.0 pp | +0.0 pp | identical to exact | 0 | fails (no change) |

Title odds by season: noise 1, late_qb_te -1.3 -10.5 -1.5 +2.0 +1.1, template +0.7 -5.3
+2.8 -0.6 +1.2, reactive -1.4 +2.3 +1.0 -0.1 +1.5; noise 0, late_qb_te -6.4 -4.9 +8.1
-3.3 +9.8, template -3.5 +3.4 +4.2 -3.0 +2.2. The 2022 template all-play under noise 1 is
-0.05 before rounding, so it is a negative season. The control reproduced
`run_streaming`'s `stream` rows exactly in all 1,200 league-seats of each design.

Reported, not gated.

- Position mix through round 6 (mean QB/RB/WR/TE per draft). Noise 1: exact 0.69/1.84/
  2.98/0.48; late_qb_te 0.01/2.33/3.65/0.00; template 0.43/2.13/3.15/0.29; reactive
  0.78/1.78/2.93/0.51. Noise 0: exact 0.58/2.12/2.83/0.47; late_qb_te 0/2.55/3.45/0;
  template 0.35/2.17/3.15/0.33; reactive identical to exact. Late_qb_te then takes its TE
  in round 8 (43% and 35% of drafts) and its QB in rounds 9-13.
- Picks per draft that differ from exact in the same state: late_qb_te 2.75 (noise 1) and
  3.10 (noise 0), template 1.08 and 0.88, reactive 0.97 and 0.00. No fallbacks in any arm.
- Consensus value drafted (sum of `cons_vbd`): noise 1, exact 190, late_qb_te 187,
  template 190, reactive 187; noise 0, exact 26, late_qb_te 9, template 28, reactive 26.
  The structural rules cost little or no drafted value, except late_qb_te under noise 0.
- Weekly starting-lineup points, change against exact averaged over weeks 1-17:
  late_qb_te lower in 13 of 17 weeks (noise 1) and 12 of 17 (noise 0), -1.7 and -1.1 a
  week; template higher in 14 of 17 in both, +0.6 and +0.7 a week; reactive +0.2 a week
  (noise 1) and exactly zero (noise 0).

Read. Waiting on QB and TE loses in both designs. With a streaming wire the late
quarterback is not free: the seat's weekly lineup is lower in most weeks, and 2022 costs
about 7 points of all-play in either design. The best-ball template is the one structural
rule that looks like a small real gain: +0.6 to +0.7 points of all-play and +8 to +9
regular-season points in both designs, lineup points higher in most weeks, at no cost in
drafted consensus value. But it misses the 4-of-5 rule under noise 1 (two seasons at
-0.2 and -0.05) and its title change is -0.2 and +0.7, so it is not shown to help. Most
of its effect comes from 2024. The reactive rule has almost nothing to react to: noisy
opponents' deviations are independent, so the room rarely leaves a whole position on the
board, and exact opponents never do.
