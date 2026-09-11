# Preregistration: usage-based waiver wire

Frozen 2026-09-11, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check (roster legality, sane free agents, no crash) had already been run and is
not part of the result; it produced no outcome numbers and decided nothing. Any later
change to the arms below is a new preregistration, not an edit to this one.

## Why this test

Nothing in the league backtest has ever touched the waiver wire; rosters are frozen after
the draft. Consensus rest-of-season ranks are a snapshot, updated on the site's own
schedule, and slow by construction on anything that happened in the last week or two: a
backup who just inherited a backfield, or a receiver whose role jumped because the man
ahead of him got hurt. Box scores carry that information before a rank does. This tests
whether reading role directly from usage catches it early enough to matter.

## Arms

Every arm drafts by exact (noise-free) consensus and starts lineups by consensus, so the
draft and start/sit decisions already shown not to help are held fixed. Only the waiver
policy on the test seat differs. One add/drop per team per week, worst-record-first
priority (all-play through the weeks so far, so it doesn't depend on the schedule draw),
roster size fixed at 14, no FAAB, no trades, no IR slot, and no team may cut below a
legal starting lineup.

- **none** (diagnostic): nobody uses the wire. Shows the baseline before any wire exists.
- **cons** (control): every team, including the test seat, adds the best free agent by
  consensus rest-of-season rank and drops its worst rostered player by the same rank,
  only when the add outranks the drop.
- **usage**: the test seat values free agents (and its own roster) by `draft/usage.py`'s
  signal instead of consensus rank; every other seat still uses `cons`.
- **solo** (diagnostic): only the test seat works the wire (by `cons`); every other seat
  is frozen at its draft-day roster. Shows how much the wire is worth at all, separately
  from whether the usage signal beats consensus at using it.

Both `cons` and `usage` price players through the same consensus rank-to-points curve
(`over_replacement` in `league_backtest.py`), then value over replacement, so the two
policies differ only in the order they rank players, not in how a rank becomes points.
The usage signal enters as an implied rank: it prices a player at the better of his
consensus rank and the rank its own percentile among currently-playing peers would imply
on the consensus scale. It can promote a player above his consensus rank; it never
demotes one below it. Pricing it any other way would be a level shift, not a
disagreement, and the hypothesis is specifically that consensus is slow, not that it's
wrong about who's good.

## Leakage control

- `draft/usage.py`'s regression (rest-of-season PPG on recent usage, team volume,
  red-zone share, and an injury-vacancy flag) is fit only on seasons before the one being
  decided (2015 through y-1; nflverse snap counts start in 2015).
- Every decision for week w uses only games played in weeks before w. The vacancy flag
  for week w reads that week's own injury report, which nflverse publishes before the
  week's games (the same pregame-eligibility assumption the rest of the harness already
  makes for Out/Doubtful designations).
- Consensus rest-of-season ranks for week w use the latest scrape strictly before w
  (preseason ranks stand in before the season's first weekly scrape exists).
- The free-agent pool for a decision is whoever isn't on any of the 12 rosters at that
  moment, from the same season-long player universe (`board.project_upcoming`, plus
  ECR-only players) the rest of the harness uses; no new player universe is introduced.

## Scoring and decision rule

- Harness: `league_backtest.py`'s `run_waivers()`, seasons 2021-25, 20 leagues x 12 seats
  x 20 schedules, both opponent designs (noise 1 x ECR sd, noise 0 exact consensus).
- Primary: paired change in all-play win rate, `usage` against `cons`, in the same
  league, seat and schedules. Secondary: title, playoff, wins, points, season-cluster
  bootstrap 90% interval, as in `prereg.md`.
- Kept only if, in **both** designs, the pooled all-play change is positive, positive in
  at least 4 of 5 seasons, and the pooled title change is no worse than -1.0 point.
- Also reported, not gated: `cons` against `none` (how much the wire is worth when every
  team has it, which should be small by construction, since a shared lever need not move
  relative standings) and `solo` against `none` (how much the wire is worth to one team
  when nobody else uses it, which bounds the size of the whole lever).
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `prereg.md` and the main
  harness already do; the draft in every arm of a league shares those draws.
- Bootstrap: per comparison, the paired differences average within each season to five
  season means; resample the five with replacement 2,000 times with
  `numpy.random.default_rng(1)`, average each resample, report the 5th/95th percentiles.
- Run once per design.

## Result (added after the single run; the spec above is unchanged)

The usage arm fails the rule in both designs. It loses to plain consensus on the wire:

| comparison | opponents | title odds | all-play | all-play by season 2021-25 |
|---|---|---|---|---|
| usage minus consensus | noise 1 | -5.1 pp [-9.0, +0.3] | -3.6 pp [-5.4, -0.5] | -5.5 -4.7 +2.8 -5.9 -4.5 |
| usage minus consensus | noise 0 | -3.0 pp [-5.4, -0.6] | -3.5 pp [-5.0, -2.0] | -3.2 -6.5 -0.2 -3.1 -4.5 |

But the wire itself is a real lever, separate from which policy runs it:

| comparison | opponents | title odds | all-play |
|---|---|---|---|
| solo (only my seat works the wire) minus none | noise 1 | +9.4 pp [+3.1, +17.5] | +2.9 pp [+0.6, +5.2] |
| solo minus none | noise 0 | +5.8 pp [+1.7, +11.6] | +5.3 pp [+1.5, +9.6] |

So the honest read is not "waivers don't matter"; it's "consensus is already good at waivers, and reading usage directly is worse than reading consensus's own rest-of-season rank." The usage arm makes more adds than consensus does in both designs (14.5-14.9 against 9.9-11.9), which is consistent with it chasing short-lived usage spikes that regress before they pay off, rather than the durable role changes it was meant to catch. That's a hypothesis for a follow-up (a persistence filter, or blending the signal with consensus rather than letting it override), not a reason to revisit this result.

## Known limits

- One add/drop a week is a simplification; real waiver systems (FAAB bidding, multiple
  claims) would price contested players differently.
- The usage signal is fit on 2015-24 pooled; it does not treat recent seasons specially.
- The vacancy flag reads "the man ahead of him" from usage rank, not the actual depth
  chart, since usage.py does not carry `role.py`'s week-1 chart forward week to week.
- Waiver priority uses all-play standings, which real leagues don't use; it removes
  schedule-draw dependence from the roster a team ends up holding, at the cost of not
  matching how a real league would order priority.
