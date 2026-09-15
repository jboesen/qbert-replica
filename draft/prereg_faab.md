# Preregistration: bid-shading on FAAB (blind-bid) waivers

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. One mechanics check
at 2 leagues per season (noise 1) had already been run. It printed no all-play, title,
playoff, wins or points outcome for either arm. What it showed is listed under "Mechanics
seen before freezing". Any later change to the arms below is a new preregistration, not
an edit to this one.

## Why this test

The harness's wire runs in priority order, worst record first. Many real leagues use
FAAB instead: each team gets a season budget, bids blind on its claims, and the highest
bid wins and pays what it bid. Streaming (`prereg_streaming.md`) settled whom to claim.
Under FAAB there's a second decision, what to pay, and real bids leave room for it. A
Reddit analysis of real Sleeper transactions (1,658 bids across 371 leagues in one week)
found bids heavily skewed, and field work on auctions (Boudreau and Shunda 2016) found
early overbidding followed by underbidding. A manager who knows who else is likely to
want a player can bid the least that probably wins and keep the rest for later weeks.
This test holds the claim fixed at streaming's and changes only the bid.

## Calibration data (Part A, `draft/sleeper_faab.py`)

- Source: Sleeper's public API. Leagues were found by snowball (seeds: the public users
  `jjzachariason` and `ffballers`, plus the members of 40 of the `scottfishbowl`
  account's 2024 leagues; then each user's leagues, then those leagues' members,
  expanding only through qualifying leagues). 818 users and 17,039 leagues were scanned.
- Qualifying league: 12 teams, redraft (`settings.type` 0), FAAB (`waiver_type` 2) with
  a positive budget, not best ball, full PPR (`rec` 1.0), one QB slot and no superflex,
  season complete. The first 300 per season were kept for 2023, 2024 and 2025. After
  mapping, 296, 293 and 298 have skill-position claims.
- Raw responses are cached in `data/sleeper_faab_raw/` (gitignored). The tidy table
  `data/faab_bids.parquet` has one row per waiver claim, complete or failed.
- Each claim is assigned to the decision before week w, where w is the first NFL week
  whose first kickoff (`games.csv`) comes after the claim's processing time. Sleeper's
  `leg` L maps to w = L+1 for 88 to 100% of claims, depending on the week. Only w = 2..17 and QB/RB/WR/TE
  (Sleeper id to gsis id through Sleeper's player file) are kept: 123,881 claims.
- Rank: the player's positional rest-of-season consensus rank at that decision, by the
  harness's own rule (the latest `ecr_ros` scrape strictly before week w, preseason
  ranks before the first scrape, one past the last ranked player if unranked).
- Bids are rescaled to percent of each league's budget (100: 433 leagues, 1,000: 267,
  300: 119, other: 68).
- Winning bid: the complete claim for that player in that league's processing batch.
  Second-highest bid: the highest failed claim with the note "This player was claimed
  by another owner" in the same batch, observable for 34% of winning claims. Bidders per
  claimed player average 1.64 (66% have one).

What it shows, by week group (2-4, 5-8, 9-12, 13-17) and rank bucket (1-12, 13-24,
25-36, 37-60, 61+ or unranked):

- A single claim's bid: mean 12.1% and median 6.4% for ranks 1-12 in weeks 2-4, falling
  to 4.2% and 0.7% in weeks 13-17. 20 to 50% of claims bid $0.
- Winning bids: mean 7.4%, median 3.0% overall, and 16.4% and 9.0% for ranks 1-12 in
  weeks 2-4. Winning minus second-highest: mean 7.7, median 4.0 points of budget.
- Bids fall through the season, as the field work predicts. Rank is not monotone:
  players ranked 37-60 or unranked draw higher bids than ranks 13-24. Those are
  breakouts whose rest-of-season rank hadn't caught up yet at the harness's info timing.
- A real team spends a mean 55% (median 58%) of its budget on skill players over weeks
  2-17.

## Arms

Both arms draft by exact (noise-free) consensus and start lineups by weekly consensus.
Opponents draft by noisy consensus, set consensus lineups and run the consensus wire's
target, as in `run_streaming`. The waiver mechanism in both arms is FAAB as defined
below. In both arms the test seat claims what `streaming_policy` adds. Only the test
seat's bid differs.

- **ctrl** (control): the test seat bids the way opponents do, a draw from the measured
  distribution.
- **faab**: the test seat bids the least whole dollar that wins with probability P* =
  0.75, under the pacing cap.

## The FAAB mechanism (`sim_faab.faab_week`)

- $100 per team for the season, no refills, no minimum bid ($0 is allowed). One add/drop
  per team per week, roster fixed at 14, no team may cut a position below its starting
  count, no trades, no IR, all as in `run_waivers`.
- Before week w (w = 2..17), in rounds: every team not yet done claims whatever its
  policy would add from the current pool (with the matching drop), and bids on it. The
  claim with the highest bid wins, ties going to waiver priority (worst all-play record
  so far first, `LB.priority`). The winner pays its bid, makes the move and is done for
  the week. A team whose policy makes no move enters the round at $0. Losers claim again
  next round on what's left. The winner's drop goes straight back into the pool, as in
  the harness.
- With every bid equal this is exactly the harness's priority wire. The mechanics check
  confirmed it: $0 bids reproduce `LB.simulate` with the streaming mover, score for
  score, in all 120 seat-seasons.

## Bids

- **Bid distribution.** Every submitted claim, winning or failed, for skill players in
  weeks 2-17, in whole dollars of $100 (`floor(bid_pct + 0.5)`, clipped to 0..100), per
  cell: week group (2-4, 5-8, 9-12, 13-17) × positional rest-of-season rank bucket (1-12,
  13-24, 25-36, 37-60, 61+ or unranked). A simulated opponent is one bidder, so its bid
  comes from all claims, not from winning bids (the top of several real bidders'). A
  cell with fewer than 200 claims would pool its week group. None has fewer than 757.
- **Opponents, and the ctrl test seat:** the bid on player p in week w is a draw from the
  cell distribution by inverse CDF, with a uniform from
  `numpy.random.default_rng([y, league, team, w, p])`. It's drawn once per (team, player,
  week), so a team re-claiming the same player bids the same, and both arms of a seat see
  the same opponent bid for the same claim. The bid is capped at the team's remaining
  budget. The environment's distribution pools 2023-25.
- **faab test seat:** at its first claim on player p in week w, the rivals are the other
  teams still waiting whose current claim is p. Each rival's bid is modelled as a draw
  from the estimated cell distribution, capped at that rival's remaining budget. Budgets
  are public in Sleeper. The win probability at bid b is the product over rivals of
  P(rival bid < b), plus P(rival bid = b) when the test seat is ahead of that rival in
  priority. The bid is the least b in 0..cap with win probability at least 0.75. With no
  rivals that's $0. If no b under the cap reaches 0.75, it bids the cap. The bid is fixed
  for that player and week, and capped at the remaining budget.
- **Pacing cap (faab seat only):** cap = floor(remaining × min(1, 3 / (18 − w))), where
  18 − w is the number of waiver decisions left, this one included (16 before week 2, 1
  before week 17). That's $18 of a full budget before week 2, and a third of what's left
  with nine decisions to go.

## Information timing and leakage control

- Claims (whom to add) use exactly the streaming spec's information: rest-of-season
  consensus before week w, the schedule, and week w-1's roster and injury files.
- Rival identification uses opponents' rosters (public), consensus values (public), and
  the fact that opponents run the consensus wire. That last part is model knowledge: the
  test seat is assumed to know opponents claim consensus's top free agent. Real managers
  can only approximate this.
- The faab seat's bid distribution for season y is fit on the Sleeper seasons other than
  y (2021 and 2022: 2023-25; 2023: 2024-25; 2024: 2023 and 2025; 2025: 2023-24). It never
  includes the scored season's bids. It does include later seasons' bids, and bidding
  behaviour is assumed stationary. The cells are week group and rank bucket, not
  players, so they carry no player-outcome information. Medians by season and week group
  differ by up to 3.3 points of budget (2023 runs higher early: 6.0 vs 3.0 and 2.7).
- The environment (opponent bids) uses the 2023-25 pool for every season. It's the
  simulated world's behaviour model, not a decision input.
- Scoring is unchanged: weekly consensus lineups with the harness's real week-w
  eligibility for every seat.

## Scoring and decision rule

- Harness: `draft/sim_faab.py`, seasons 2021-25, 20 leagues × 12 seats × 20 schedules,
  both opponent designs (`--noise 1` and `--noise 0`).
- Primary: paired change in all-play win rate, `faab` against `ctrl`, in the same league,
  seat and schedules. Secondary: title, playoff, wins, points.
- Kept only if, in **both** designs, the pooled all-play change is positive, positive in
  at least 4 of 5 seasons, and the pooled title change is no worse than -1.0 point.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors and then the twenty schedules, exactly as `run_streaming`. Both
  arms share those draws and the same drafts. Bid uniforms are seeded as above.
- Bootstrap: `LB.paired`. Paired differences average within each season to five season
  means; resample the five with replacement 2,000 times with
  `numpy.random.default_rng(1)`; report the 5th/95th percentiles. The interval is
  reported; the rule decides.
- Run once per design, one after the other, through the run lock. Reduced runs write to
  `_check` files and can't overwrite `data/league_faab_noise{1,0}.parquet`.

## Reported, not gated

- The test seat's cumulative budget used before each week, by arm.
- Win rate on intended claims: the test seat's claim in the first round of each week
  (on the start-of-week pool), won if that player is the one it added that week.
- Average overpay per claim won: bid minus the least bid that would still have won
  against the other claims on that player in that round (the highest rival bid, plus $1
  if that rival was ahead in priority; $0 with no rival).
- Opponents' mean season spend.

## Mechanics seen before freezing (2 leagues per season, noise 1)

- $0 bids reproduce the priority streaming wire exactly (120 of 120 seat-seasons).
  Rosters and budgets are always legal.
- Simulated leagues are far more crowded than real ones. Because all eleven opponents
  value players identically, a won claim had on average 4 to 12 claimants in its round
  (by rank bucket), against 1.64 in the real data, and opponents spend about $99 of $100
  a season, against 55% in real leagues. The environment is therefore a harsher and more
  contested FAAB market than a real league.
- Test seat: intended claims win 11% of the time in ctrl and 47% in faab. Both arms spend
  about $96 by week 17. Average overpay is $5.0 (ctrl) and $3.0 (faab) per claim won.

## Known limits

- The opponents' claim model (identical consensus values, so the same target) is the
  harness's, not real behaviour, and it produces the crowding above. The bid levels are
  real; how many teams bid on one player is not.
- An opponent's bid depends only on week and rank bucket: not on position, its own
  remaining budget (beyond the cap), its needs, or news. In the data, rank explains
  bids weakly and non-monotonically.
- A single claim at a time stands in for Sleeper's ranked claim lists. Dropped players
  clear immediately, not after a waiver period.
- Leagues were found by snowball sampling, which over-represents very active users.

## Result (added after the single run per design; the spec above is unchanged)

Bid shading fails the rule in both designs. All-play is the deciding statistic, and it
isn't positive in 4 of 5 seasons in either design; in the noise 1 design it's negative
overall. Changes against the control (same streaming claims, bids drawn like the
league's), with 90% season-cluster intervals:

| comparison | opponents | title odds | all-play | all-play by season 2021-25 | reg pts |
|---|---|---|---|---|---|
| faab minus ctrl | noise 1 | +1.3 pp [+0.3, +2.3] | -0.4 pp [-1.0, +0.2] | -1.4 -0.5 +0.9 -0.9 -0.2 | -5 |
| faab minus ctrl | noise 0 | +2.4 pp [+1.5, +3.4] | +0.4 pp [-0.2, +1.0] | -0.4 +1.0 +1.5 -0.6 +0.5 | +4 |

Title odds by season: noise 1 +2.2 +1.1 +3.3 -0.5 +0.5; noise 0 +4.4 +1.8 +3.1 +0.8
+2.1. Playoff-week points +4.1 (noise 1) and +7.3 (noise 0) a season.

Reported, not gated (test seat, per league-seat season):

| | noise 1 ctrl | noise 1 faab | noise 0 ctrl | noise 0 faab |
|---|---|---|---|---|
| $ spent by week 5 / 9 / 13 / 17 | 38 / 70 / 88 / 97 | 36 / 68 / 89 / 97 | 48 / 80 / 93 / 99 | 38 / 73 / 92 / 99 |
| intended claims won | 14% | 47% | 12% | 41% |
| claims won a season | 11.1 | 12.3 | 12.8 | 13.5 |
| average overpay per claim won | $5.0 | $2.8 | $4.4 | $2.6 |

Opponents spend about $99 of $100 in every arm.

The bid rule does what it was built to do: it wins its intended claim three to four
times as often, makes about one more add a season, and pays about half as much above the
price that would have won. None of that turned into regular-season points. The title
gain is positive in 9 of 10 season-designs, but title odds aren't the deciding statistic,
and a gain with no all-play behind it is the pattern schedule and playoff-week luck
produce. It rests on 4 to 7 playoff-week points. Both arms end the season with the
budget spent, so what shading changes is timing, not the total spent: less early,
the rest in weeks 5-13.

The simulated market is not a real one. Eleven opponents with identical values claim the
same player (4 to 12 claimants per won claim, against 1.64 in real Sleeper leagues) and
spend their whole budget (real teams spend 55%). In a real league a shaded bid would
more often face no rival at all and win for $0. So the environment is harsher than
reality, and this result doesn't show that bid shading can't help in a real league. It
shows it doesn't move all-play in this harness.
