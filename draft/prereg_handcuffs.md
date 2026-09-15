# Preregistration: streaming plus a contingent running-back stash

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check had already been run in both designs and is not part of the result; it
printed no outcome metrics and decided nothing. It found the control identical to
`run_streaming`'s `stream` rows (and the protected streaming copy, with nothing stashed,
identical too), no illegal test-seat roster in any week, no player on two rosters, and a
lineup reconstruction that matches the harness's scores. The first (noise 1) check showed
stashes with slightly negative contingent value being made against deeply negative bench
values, so the requirement CV > 0 below was added before the noise 0 check; nothing else
changed. With it, cross makes 3.1-4.3 stash moves per league-seat season and own 0.5-1.3.
Any later change to the arms below is a new preregistration, not an edit to this one.

## Why this test

Hole-aware streaming (`prereg_streaming.md`) is the one decision that has beaten the
consensus wire, so it is the control here. Fantasy communities and analyst sites
repeatedly recommend one more roster decision: stash contingent backup running backs,
specifically backups behind other teams' starters. The mechanism fits this harness.
Consensus-following managers value players by current rest-of-season rank, so they only
add a backup after the starter ahead of him is hurt and his rank jumps. A team already
holding him wins that race without needing waiver priority. Holding another team's backup
adds a new route to a starter-level player; holding the backup behind your own starter
only insures a slot you already fill. The question is whether that route is worth the
bench slot and the waiver moves it costs, on top of streaming.

## Arms

Every arm drafts by exact (noise-free) consensus and starts lineups by weekly consensus.
All eleven other seats run the consensus wire policy in every arm. One add/drop per team
per week, priority worst all-play record first, roster size fixed at 14, no FAAB, no
trades, no IR slot, no cut below `ROSTER_MIN`, all exactly as in `run_streaming`.

- **stream** (control): the test seat runs `streaming_policy` unchanged. This is the
  `stream` arm of `run_streaming` and must reproduce it exactly.
- **cross** (test): streaming plus up to K = 2 stashed backups behind lead backs of NFL
  teams that none of the test seat's running backs plays for.
- **own** (diagnostic, reported, not gated): the same stash logic restricted to backups
  whose lead back is on the test seat's roster.

Code: `draft/handcuff.py` (contingent value), `draft/sim_handcuffs.py` (policy and run).
`league_backtest.py` is imported, not edited.

## Contingent value

All fits use only seasons 2015 through y-1 for season y (2015 is the first fit season in
`usage.py`, where snap counts begin). Box scores are nflverse weekly stats, regular
season, weeks 1-17, rows with position RB. **Work** is carries plus targets.

1. **Backfield order.** Before week w, a team's backs are ranked by total work in the
   team's last 4 games played before w (bye weeks skipped). Lead = most work, second =
   next, third = the one after (ties by player id). No state if fewer than two backs had
   work. From the same window: s1 = the lead's share of team RB work; s2 and s3 = the
   second's and third's share of team RB work in the window games in which the lead had
   work (0 if none); last_played = the lead had work in the team's last game before w.
   A window rather than the season to date, so a lead who has been out a month stops
   being the lead and his fill-in stops being a backup.
2. **Chance the lead misses time.** At every past decision (team, w = 2..17) with
   last_played, the fraction of the remaining weeks w..17 in which the lead had no work
   in a game his team played. f_miss(w) is the mean of that fraction by decision week w.
   "No work" counts injuries, suspensions, trades, releases and benchings alike, since
   each opens the job.
3. **Role transfer and fill-in points.** An absence is a run of the team's games with no
   work for the lead, starting in the first game after one in which he had work. Each
   game of each absence is a row, with s2 and s3 read at the decision before the absence
   began. The second back **inherits the role** in a game if he has at least 50% of team
   RB carries that game. P(transfer) is a logistic regression of that indicator on
   (constant, s2, s3); the third back's share stands for a competing back. Fill-in PPR is
   two OLS fits of his PPR that game (0 if no box score) on (constant, s2), one on
   transfer games and one on the rest, each prediction clipped at 0. Expected fill-in PPR
   = p x PPR_transfer + (1 - p) x PPR_other.
4. **Contingent value** of the second back at the decision before week w, in points per
   week over replacement: CV = f_miss(w) x (expected fill-in PPR - RB replacement level
   at w). The replacement level is the one `waiver_values` subtracts to price consensus
   rest-of-season values that week (read by wrapping `over_replacement`, which leaves the
   control's numbers unchanged), so CV is on the same scale as the values it competes
   with.

**Eligible stash candidates** at the decision before week w: a team's second back who is
in the harness player pool, whose P(transfer) >= 0.5, whose lead had work in the team's
last game, and whose lead (if in the pool) is not known out for a reason other than a bye
(on his latest roster row at or before week w-1, status not ACT/INA, or Out/Doubtful on
that week's report). Only high-probability transfer cases are stashed, following the
observation (r/Fantasy_Football "Mythbusters #9", 2016-24) that about a third of healthy
handcuffs collapse to a committee role when the starter goes down.

Not used, declared: snap share (the pfr-id bridge maps 89-97% of RB touch volume by season,
worst in 2025, so a missing share would not be missing at random) and depth charts (the
2025 file has a different schema, and their publication time relative to waivers is not
documented). Usage rank stands in for depth-chart rank.

The constants (4-game window, 50% carry threshold, P >= 0.5, K = 2, release at 80% for
2 games) were set before any fit was run. The fit summaries on 2015-2024 were then printed
(transfer rate 0.42-0.44 of lead-out games; f_miss 0.14-0.21; 141-196 eligible team-weeks
per season) and nothing was changed after seeing them. No league outcome of these arms
has been seen.

## The stash decision

Each week the test seat's move is decided in this order:

1. **Releases.** A stashed player is released (untagged; he stays on the roster as an
   ordinary player until some later move cuts him) when either
   - **lead back:** his lead has had a game with no work since the stash was added, and
     in each of the team's last 2 games before w, after that absence, the lead's share of
     team RB work is at least 80% of his share s1 at the time of the stash; or
   - **demoted:** his team's current backfield state exists and he is neither its lead
     nor its second back.
   A lead who stays out simply keeps the stash on the roster, now as the fill-in.
2. **Streaming.** The streaming move is computed exactly as in `streaming_policy`, with
   two changes that only bite when a stash is held: the consensus fallback never cuts a
   stashed player, and among drops tied on holes left and on being a week-w starter, a
   non-stashed player is cut before a stashed one. So a hole-filling move takes a stash's
   slot only when no other drop does as well. If streaming makes a hole-filling move, that
   is the week's move and no stash is considered; a stash cut this way ends the stash.
3. **Stash.** Otherwise (no hole, or a hole with no legal filling move), if fewer than
   K = 2 players are stashed:
   - Candidate: the eligible free agent with the highest CV that week that fits the arm.
     **cross:** his team (from box scores) differs from the team (latest roster row at or
     before w-1) of every running back on the test seat's roster. **own:** his lead back
     is on the test seat's roster.
   - Drop: the lowest rest-of-season-valued (control's consensus value; unvaluable first)
     cuttable player who is not stashed and not a starter, either in the week-w lineup
     from players not known unavailable or in the lineup filled on value alone (a starter
     on bye is still a starter).
   - The stash is made only if CV > 0 and CV > the drop's value, and, when the consensus move would
     otherwise be made that week, CV minus the drop's value exceeds the consensus add's
     value minus its drop's value (ties go to the consensus move).
   - If no stash is made, the week's move is streaming's (the consensus move, with
     stashes protected, or none).

Stashed players are never cut for a consensus move.

## Information timing and leakage control

- The decision before week w is made after week w-1 is played and before week w's injury
  report, roster status or weekly consensus exist. Backfield order, shares and
  last_played read only box scores of weeks before w. Lead availability and roster teams
  read only week w-1 (or earlier) roster rows and injury reports, and the schedule.
- f_miss, P(transfer) and fill-in PPR are fit only on seasons before y.
- Release rules read only box scores before w.
- Rest-of-season values and replacement levels are the control's (latest rest-of-season
  consensus scrape before w, rank curve fit on seasons before y).
- Scoring is unchanged: lineups by weekly consensus with the harness's real week-w
  eligibility, for every seat and arm.
- Known residuals: the nflverse position label (RB) is a season-level label that can be
  assigned with later knowledge, for instance a player converted mid-season; a stats row
  team can differ from the roster team for a player traded between weeks; and the 2016-24
  "Mythbusters" observation that motivated the P >= 0.5 filter overlaps the holdout
  seasons (it informed a constant, not a fit).

## Scoring and decision rule

- Harness: `sim_handcuffs.py`, seasons 2021-25, 20 leagues x 12 seats x 20 schedules,
  both opponent designs (`--noise 1` and `--noise 0`), through the shared run lock.
- Primary: paired change in all-play win rate, `cross` against `stream`, in the same
  league, seat and schedules. Secondary: title, playoff, wins, points.
- Kept only if, in **both** designs, the pooled all-play change is positive, positive in
  at least 4 of 5 seasons, and the pooled title change is no worse than -1.0 point.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `prereg.md` and
  `run_streaming` do; all arms of a league share those draws and the same drafts.
- Bootstrap: the paired differences average within each season to five season means;
  resample the five with replacement 2,000 times with `numpy.random.default_rng(1)`,
  average each resample, report the 5th/95th percentiles (`numpy.quantile`, default
  linear interpolation). The interval is reported; the rule decides.
- Run once per design, noise 1 then noise 0. Reduced (`--leagues` other than 20) runs
  write to `_check` files and cannot overwrite a real result.

## Reported, not gated

- `own` against `stream` and `cross` against `own`, same statistics.
- Stash moves per league-seat season, by season and arm.
- Share of stashes that later started for the test seat while on its roster (in the
  weekly-consensus lineup the harness scores), starts per season, and points scored in
  those starts; share whose lead had a game with no work while the stash was held.
- Release reasons.

## Known limits

- One add/drop a week, so a stash can displace a consensus add; the displacement rule
  prices that but on consensus's own rest-of-season scale.
- No lookahead to byes, playoff weeks or the stash's own bye.
- A backup outside the harness player pool (preseason board and preseason consensus) can
  never be stashed by anyone, so very deep backups are out of reach for every arm.
- The stash's own availability is not checked beyond his being a top-two back by recent
  work.
- Opponents never stash, so the race the test seat wins is against consensus managers
  only; a league of stashers would compete for the same players.

## Result (added after the single run per design; the spec above is unchanged)

The cross-team stash fails the rule. It passes with exact opponents but not with noisy
ones, where it is positive in only 3 of 5 seasons. Changes against streaming on the test
seat, 90% season-cluster intervals:

| comparison | opponents | title odds | all-play | all-play by season 2021-25 | reg pts |
|---|---|---|---|---|---|
| cross minus stream | noise 1 | +0.6 pp [-0.2, +1.4] | +0.8 pp [-0.2, +1.8] | +3.2 +1.0 -0.5 +0.9 -0.7 | +9 |
| cross minus stream | noise 0 | +1.9 pp [-0.5, +5.2] | +1.6 pp [-0.1, +3.3] | +3.7 +0.1 -0.8 +5.0 +0.2 | +12 |
| own minus stream (diagnostic) | noise 1 | -0.0 pp [-0.6, +0.6] | +0.0 pp [-0.1, +0.1] | +0.3 +0.0 -0.0 -0.2 -0.1 | +1 |
| own minus stream (diagnostic) | noise 0 | -1.2 pp [-3.7, +1.3] | -0.1 pp [-0.5, +0.1] | -0.6 +0.1 +0.4 +0.1 -0.6 | +2 |
| cross minus own (diagnostic) | noise 1 | +0.6 pp [-0.5, +1.8] | +0.8 pp [-0.2, +1.8] | +2.9 +0.9 -0.4 +1.0 -0.7 | +8 |
| cross minus own (diagnostic) | noise 0 | +3.0 pp [-0.5, +6.7] | +1.8 pp [-0.1, +3.6] | +4.3 -0.0 -1.2 +4.9 +0.8 | +10 |

Title odds by season, cross minus stream: noise 1 +1.2 +0.6 +0.3 +2.0 -1.2; noise 0
+0.3 -1.6 -0.3 +9.1 +1.8.

Reported, not gated. The cross arm makes 3.9 (noise 1) and 3.7 (noise 0) stash moves a
season, the own arm 1.1. 53% and 57% of cross stashes later start for the test seat,
7.6 and 8.5 starts a season at 11.7 and 12.2 points a start, about 88 and 104 points of
starting-lineup production a season (own: 1.7-1.9 starts, 19-22 points). The lead back
had a game with no work while a cross stash was held in 53% and 60% of stashes, from 29%
(2025) to 77-97% (2023-24). Releases: about a third demoted, under a fifth because the
lead returned, the rest held to season's end.

Read: the mechanism is real. Cross-team stashes do turn into starts, several times more
than own-team handcuffs, which add nothing, as the research claimed. But the gain rides
on seasons with many lead-back absences (2021, 2024) and turns negative in 2023 and
2025 in noise 1, and the pooled intervals include zero in both designs. Not kept.
