# Preregistration: roster-fit trades on top of hole-aware streaming

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check had already been run in both designs and is not part of the result; it
printed no outcome metrics and decided nothing. It found the control identical to
`run_streaming`'s `stream` rows (leagues 0-1, every column, max difference 0.0) in both
designs; every executed trade passed the in-code invariants (both rosters keep their size,
no player on two rosters, no position below its starting count); every executed trade
satisfied the opponent's acceptance rule; and trades fell only in weeks 3-11. The part A
crawl was still running when this file was frozen. The calibrated k is therefore not
known here, but the rule that turns the crawl into k, and k into the verdict arms, is
fixed below and leaves no choice to make afterwards. An early look at the first 16
crawled leagues (84 trades) was printed while the summary code was being debugged, after
the estimator was written; it had 16 usable trades and calibrated nothing. Any later
change to the arms below is a new preregistration, not an edit to this one.

## Why this test

Streaming (`prereg_streaming.md`) is the one decision in this harness that beats
consensus, and it wins on need rather than on value. Trades are the other roster lever
the harness has never had. Two managers who price every player identically can both gain
from a trade, because a player's worth to a team depends on who else it starts: a fourth
receiver adds little to a team that starts two, and a lot to a team whose tight end is
hurt or on bye. The question is whether a seat that trades for its own starting lineup,
against opponents who judge offers only on consensus value, gains all-play over a seat
that only streams.

Background, none of it verified here: FantasyCalc prices players from real trades, and a
Reddit post proposes buying where a forecast exceeds that price; trade-calculator studies
credit consolidation (2-for-1), with an unverified claim from 4,500 dynasty trades that a
stud clears at 0.84x the two pieces; a genetic-algorithm study (arXiv 2511.17535) finds
mutually beneficial trades worth about 3 projected points a week to each side, driven by
roster fit; endowment-effect research says owners demand a premium to sell.

## Part A: calibrating acceptance on real trades

`sleeper_trades.py` snowballs from two public Sleeper accounts through league member
lists and keeps every finished 12-team, full-PPR, redraft (settings type 0),
one-quarterback, non-superflex, non-best-ball league from 2023-25, caching the league,
its users, rosters and every leg's transactions under `data/sleeper_trades_raw/`
(gitignored). The crawl stops at 1,200 qualifying league-seasons. `trade_acceptance.py`
keeps completed two-team trades of skill players only (no kickers, defenses, picks or
FAAB), and prices each side the way the harness's consensus managers price players: the
latest rest-of-season positional consensus rank informing the trade's week (preseason
ranks before the first scrape), through `consensus.weekly_curve` fit on seasons before the
trade's season, both as raw points per game (`rank_points`) and over replacement
(`vbd.add_vbd`). The side that accepted is the side that did not create the final offer.

**Estimator, written into `trade_acceptance.py` before any trade was summarised:** over
1-for-1 trades with an identified accepting side and both players above replacement, the
endowment premium k is the median accepting-side ratio (value received over value given,
over replacement) minus 1, clipped to [0, 0.2] and rounded to the nearest of 0, 0.1, 0.2.
Fewer than 200 such trades counts as too thin, and then no k is calibrated.

FantasyCalc: the site's own history endpoint (`/trades/historical/<id>`) starts on
2025-07-01, and redraft values are zero before May 2026, so no 2023-25 redraft market
values can be retrieved. Market minus consensus is not estimated.

The calibration result is appended below the frozen spec, with the Result.

## Arms

Every arm drafts by exact consensus and starts lineups by weekly consensus. All eleven
other seats run the consensus wire policy, as in `prereg_streaming.md`. The test seat runs
hole-aware streaming (`streaming_policy`) on waivers in every arm. Waiver rules are
unchanged.

- **stream** (control): no trades. Must reproduce `run_streaming`'s `stream` arm exactly.
- **tradeA_k{k}**: streaming plus roster-fit trades, defined below.
- **tradeB_k{k}**: streaming plus roster-fit trades, preferring an incoming player with a
  softer weeks 15-17 schedule among near-equal offers, defined below.

Each of A and B is run at k = 0, 0.1 and 0.2 (six trade arms), all in the same run.

**Which arms carry the verdict.** If part A calibrates k (at least 200 usable trades), the
preregistered verdict is `tradeA_k{k}` and `tradeB_k{k}` at that k, each judged on its own;
the other four arms are a labeled sensitivity. If part A is too thin, there is no
preregistered verdict: all six arms are reported as a sensitivity sweep over an
acceptance rule with no empirical basis.

## The trade window

- Decisions before weeks 3 through 11, one per week, made after that week's waiver round
  has cleared (the decision before week v happens after week v-1 is scored and waivers
  for week v have run, exactly where the harness's next scoring week begins).
- Opponents never propose. At most one trade is executed per week.
- **Offers searched:** with each of the eleven opponents, every 1-for-1 (the seat gives
  one, gets one) and every 2-for-1 (the seat gives two of its players, gets one).
- **Acceptance:** the opponent accepts if V(players it receives) / (1 + k) >= V(player it
  gives), where V is the summed consensus rest-of-season value over replacement for week v,
  the same `waiver_values` number every seat already uses on the wire. Nothing else about
  the opponent's roster enters its decision.
- **Legality:** after the trade, both rosters keep every position at or above its starting
  count (`ROSTER_MIN`). In a 2-for-1 the test seat, one player short, adds the best free
  agent by V (the consensus wire's add rule), and the opponent, one player long, cuts its
  lowest-valued cuttable player (the wire's drop rule, which never breaks a minimum).
- **The seat's objective (test A):** among accepted legal offers, the one that most raises
  the seat's own rest-of-season starting-lineup value, and only if that gain is positive
  (above 1e-6). Lineup value is the sum over weeks v..17 of the best legal lineup (1 QB,
  2 RB, 2 WR, 1 TE, 1 FLEX) on raw consensus rest-of-season points per game for week v,
  counting a player as zero in a week when he is expected unavailable: his team is on bye
  that week (schedule, with the team from his latest roster row at or before week v-1),
  or, for week v only, he is known unavailable by the streaming policy's rule. A player
  with no roster row counts as unavailable. The 2-for-1 free-agent add is part of the
  evaluated roster. Ties go to the first offer found (opponents in seat order, 1-for-1
  before 2-for-1 for each incoming player).
- **Test B:** among accepted legal offers whose gain is at least 90% of the best gain, the
  one whose incoming player has the highest weeks 15-17 schedule score; ties by gain.
  Schedule score: from box scores of weeks before v, each game's points by an offense at
  a position minus that offense's mean at the position over its games so far; a defense's
  adjusted points allowed at a position is its mean of those residuals; a player's score is
  the mean over his team's weeks 15-17 opponents (team from the same roster row).

## Information timing and leakage control

- V and the lineup's points per game are the latest rest-of-season consensus scrape
  strictly before week v, through the rank curve fit only on seasons before y; identical to
  what the wire uses, so no seat has information another lacks.
- Availability reads only the season schedule and week v-1 (or earlier) roster status and
  injury reports, as in `prereg_streaming.md`.
- The schedule score reads box scores of weeks before v only.
- Scoring is unchanged: every seat's week-v lineup is set by weekly consensus with the
  harness's real week-v eligibility.
- The acceptance calibration uses 2023-25 real trades, which overlap the test seasons
  2023-25. It sets one scalar (k) that no outcome in the harness is fitted to, and the
  sweep covers the whole declared range, so this is a stated residual, not a fit.

## Scoring and decision rule

- Harness: `sim_trades.py`, seasons 2021-25, 20 leagues x 12 seats x 20 schedules, both
  opponent designs (`--noise 1` and `--noise 0`).
- Primary: paired change in all-play win rate, each trade arm against `stream`, same league,
  seat and schedules. Secondary: title, playoff, wins, points.
- An arm passes only if, in **both** designs, its pooled all-play change is positive,
  positive in at least 4 of 5 seasons, and its pooled title change is no worse than
  -1.0 point.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `run_streaming`; every arm
  of a league shares those draws and the same drafts.
- Bootstrap: the paired differences average within each season to five season means;
  resample the five with replacement 2,000 times with `numpy.random.default_rng(1)`,
  average each resample, report the 5th/95th percentiles (`LB.paired`). The interval is
  reported; the rule decides.
- Run once per design. Reduced (`--leagues` other than 20) runs write `_check` files.

## Reported, not gated

- Trades per league-seat season, by structure, and by season.
- Consensus value given and received per trade.
- Projected rest-of-season lineup gain per trade, and the realised starting-lineup points
  it produced: weeks v..17 actual lineup points (weekly consensus lineups, real
  eligibility) with the post-trade roster minus the pre-trade roster, both held fixed.
- Season points, arm minus control.
- Trade frequency compared with the real Sleeper leagues.

## Known limits

- Opponents are passive value-matchers. A real manager weighs his own lineup too, and
  refuses lopsided-fit offers even at equal value; that is exactly the surplus this seat
  harvests, so any gain is an upper bound on what real opponents would give up.
- The seat may trade every week. Real managers trade far less often (see part A), and
  there is no fatigue, veto or offer limit.
- Ties in value are common: the consensus rank curve is flat past the waiver line, so many
  deep bench players are worth exactly replacement and swap freely at any k.
- Future availability knows only byes; injuries after week v are not anticipated.

## Result (added after the single run per design; the spec above is unchanged)

**Part A.** The crawl kept 1,205 league-seasons, 588 of them with a completed two-team trade;
teams take part in 0.41 trades a season. 435 usable 1-for-1 trades give a median
accepting-side ratio of 0.81 [90% league bootstrap 0.69, 0.91], so k = 0 and the
preregistered verdict arms are `tradeA_k0` and `tradeB_k0`. Real accepters took less
consensus value than they gave in most 1-for-1 trades, so k = 0 is not generous on price.

**Both verdict arms pass the rule in both designs.** Changes against `stream` (which
reproduces `run_streaming`'s stream arm exactly), 90% season-cluster intervals:

| arm | opponents | title odds | all-play | all-play by season 2021-25 | reg pts | trades/season |
|---|---|---|---|---|---|---|
| tradeA_k0 | noise 1 | +0.4 pp [-3.3, +3.7] | +3.5 pp [+1.7, +5.2] | +3.6 +5.5 -0.2 +6.3 +2.3 | +40 | 8.3 |
| tradeA_k0 | noise 0 | +2.5 pp [-3.1, +9.1] | +4.8 pp [+1.4, +8.1] | +8.8 +8.3 +1.5 +7.5 -2.2 | +55 | 8.3 |
| tradeB_k0 | noise 1 | -0.5 pp [-4.2, +2.9] | +3.5 pp [+1.7, +5.0] | +3.8 +5.2 -0.3 +6.0 +2.6 | +41 | 8.3 |
| tradeB_k0 | noise 0 | +0.5 pp [-6.1, +7.7] | +3.8 pp [+1.7, +5.9] | +4.1 +7.5 +0.0 +6.4 +1.2 | +41 | 8.4 |

Title odds by season: A noise 1 -1.1 +6.0 -8.4 +2.5 +3.1, noise 0 +1.0 +16.6 -2.8 -7.9 +5.7;
B noise 1 -4.7 +6.0 -7.5 +1.6 +2.1, noise 0 -5.2 +9.8 -5.5 -9.6 +12.8.

Sensitivity (k = 0.1, 0.2): all-play stays positive pooled in every arm and design
(+0.9 to +2.3), but k = 0.1 is positive in only 3 of 5 seasons under noise 0.

Reported, not gated. The seat trades about 8 times a season, twenty times the real rate.
Under noise 1, `tradeA_k0` made 65% 1-for-1 and 35% 2-for-1 trades; value given and
received are nearly equal (median 4.2 and 4.1 over replacement), as the acceptance rule
forces. The median projected rest-of-season lineup gain per trade is 6.8 points (noise 1)
and 5.3 (noise 0), and the realised starting-lineup gain averages 5.2 and 9.1.

The gain is large, but it's an upper bound: every opponent accepts any value-neutral offer,
every week, with no limit. The next test caps the seat at a realistic trade count.
