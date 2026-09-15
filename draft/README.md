# Draft tool

Turns the projection work into something usable on draft day: a board, a backtest that
says whether the board is worth trusting, and an assistant that recommends picks live.

Three papers are replicated here. One of them turns out to matter enormously, one
matters moderately, and one — the most sophisticated — barely matters at all.

## 1. Hierarchical partial pooling (projections)

`project_season.py`. Follows the hierarchical framing used for fantasy scoring
([Egidi & Gabry, *Frontiers in Sports* 2025](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1486928/full)):
a player's own history is shrunk toward a position/age/experience/pedigree prior by an
amount that depends on how much history he has. Written in closed form (normal-normal
conjugate) rather than sampled — same posterior mean, far cheaper.

Empirical-Bayes shrinkage constants, in games of history needed before a player's own
numbers outweigh the prior:

| QB | RB | WR | TE |
|---|---|---|---|
| 4.3 | 10.6 | 5.4 | 6.2 |

Running backs need twice the history of quarterbacks before their own numbers can be
trusted — the usual folk wisdom that RB production is volatile, falling out of the model
rather than assumed.

Holdout 2021–25, 2,014 player-seasons, projecting full-season PPR points:

| model | RMSE | MAE | corr |
|---|---|---|---|
| hierarchical projection + depth role | **54.7** | **40.2** | **0.800** |
| hierarchical projection alone | 60.2 | 44.9 | 0.756 |
| last season's total | 66.9 | 47.2 | 0.736 |
| own ppg × season games (no shrinkage) | 78.6 | 59.6 | 0.723 |
| position mean | 87.4 | 70.8 | 0.276 |

Games played is modelled separately and matters more than it looks. Players who logged
15+ games last season average **13.2** the next — so the projections price in
availability, which point-estimate projections usually don't.

## 2. Value-based drafting

`vbd.py`. A projection can't rank across positions: 300 points from a QB and 300 from an
RB are worth very different amounts, because the QB you'd have taken instead scores far
more than the RB you'd have taken instead. Every player is priced against the
replacement he displaces at his own position, with flex demand allocated across eligible
positions by who actually fills those slots.

This is the single biggest effect in the whole build — see the backtest.

## 3. Draft as a dynamic program

`draft_dp.py`. Replicates [Fry, Lundberg & Ohlmann, "A Player Selection Heuristic for a
Sports League Draft"](https://www.degruyterbrill.com/document/doi/10.2202/1559-0410.1050/html)
(*JQAS* 2007). Their argument: taking the best available player is wrong, because what
matters is your finished roster, which depends on who survives to your later picks. They
pose it as a stochastic DP, note it's intractable, and reduce it to a deterministic one
by assuming the league consumes the board in a known order. That reduction is what's
implemented — state is (next pick, unfilled starting slots), and the transition inherits
the best player left at each position at each future pick.

## 4. Depth-chart role

`depth_role.py`, `role.py`. The pooled projection knows how good a player has been, not
whether he'll be on the field. A backup quarterback with a good career projected like a
starter, and a receiver who slid to fourth on the chart projected off last year's role.
The depth chart published before week 1 settles that. It's nflverse's, weekly through
2024 and dated snapshots from 2025, back to 2012. Each projection is re-weighted by the
player's slot (starter, second, third or lower, not on the chart), separately for games
played and for points per game, so the trade simulator can tell "rarely plays" from
"plays badly". Rookies get their own coefficients: their projection is the
draft-pedigree prior alone, and that prior undershoots the rookies who win jobs. Fit on
2012–20, scored on 2021–25, with the chart taken from before each season opened:

| | MAE before → after | corr before → after |
|---|---|---|
| QB | 62.7 → **49.0** | 0.746 → **0.830** |
| RB | 49.2 → **42.7** | 0.735 → **0.778** |
| WR | 43.6 → **38.0** | 0.789 → **0.805** |
| TE | 28.7 → **26.0** | 0.790 → **0.806** |

Among fantasy-relevant quarterbacks, correlation goes from 0.41 to 0.59, the largest
single gain anywhere in the build. Relevant receivers don't move (0.57 → 0.56): a top
receiver is a starter either way. It also carries into the weekly model, whose
preseason prior this is: holdout RMSE 6.16 → 6.12, within-week rank 0.597 → 0.607.
With a role to lean on, rookies make the board, 14 of them in 2026.

## Does any of it work?

`backtest.py`. Twelve teams, 14-round snake, PPR, holdout seasons 2021–25. Each strategy
holds three seats, rotated four ways so draft position can't decide the outcome. Rosters
are scored on the best legal lineup each week using what players actually did — so it
measures roster quality, not in-season management.

| strategy | mean season points | sd |
|---|---|---|
| Fry–Lundberg–Ohlmann DP | **1825** | 155 |
| value-based drafting | 1804 | 183 |
| prior-year finish (a naive market) | 1691 | 192 |
| best projected points | 1174 | 109 |

Paired within the same league (n=20 leagues):

| DP vs | difference | std error | DP ahead in |
|---|---|---|---|
| value-based drafting | +21 | 21 | 65% |
| prior-year finish | **+134** | 31 | 80% |
| best projected points | **+651** | 23 | 100% |

(These are with the depth-role projections. Before role, the edge over the market was
+87 ± 26: knowing who starts is worth about 50 points a season on draft day.)

Read honestly:

- **Positional value is everything.** Drafting by raw projected points loses by 651
  points a season — it takes quarterbacks early and never recovers. This is the whole
  ballgame, and it's a 1970s idea, not a modern one.
- **The board beats a naive market by ~134 points a season**, about 8%, at over 4
  standard errors. That's a real edge against a league drafting off last year's finish.
- **The DP paper adds nothing reliable over plain VBD** — +21 ± 21 is noise, though it
  wins 65% of leagues and cuts the spread of outcomes (sd 155 vs 183). Its value here is
  consistency, not upside. Replicated faithfully, and honestly not worth much in this
  setting.

A caveat on the market baseline: no free source of historical ADP exists, so the "market"
is modelled as prior-year fantasy finish. A real league drafting off consensus rankings
is a tougher opponent than that, and one that already knows the depth chart, so treat
+134 as an upper bound.

## Does it win leagues?

`league_backtest.py`, `consensus.py`. Everything above measures accuracy, but you don't
win a league by projecting well. You win by beating eleven other managers, and most of
them draft and set lineups from expert consensus (FantasyPros ECR). A model earns
something only where it disagrees with consensus and is right. So this is the test
that counts. The design was argued out with Codex (gpt-5.6-luna) over five rounds,
including two rounds that blocked it until the flaws below were fixed.

- **Leagues:** 12 teams, PPR, on each holdout season 2021–25. Eleven consensus seats
  draft from the preseason overall ECR, each perturbed by the experts' own spread, and
  start lineups from weekly positional ECR. One test seat swaps in our draft, our
  lineups, or both. Everything else stays the same.
- **Inputs:** everything is as of the decision. Boards are fit only on earlier seasons,
  weekly projections use only earlier weeks, and consensus comes from scrapes dated
  before the Sunday slate. Availability is what's known before kickoff (active roster,
  not on bye, not Out or Doubtful). A starter who doesn't play scores zero, whichever
  policy started him.
- **Scoring:** what players actually did. Weeks 1–14 are head-to-head on random
  schedules, and the top six go to a weeks 15–17 playoff.
- **The metric:** the headline is the paired change in championship probability
  against a consensus seat in the same league. The deciding statistic is the paired
  change in all-play win rate, which is schedule-free and far less noisy. A change is
  kept only if its pooled all-play gain is positive, positive in at least 4 of 5
  seasons, and costs no more than 1 point of title odds. That rule was declared before
  any result was run. Seasons are the independent unit, since the player outcomes are
  fixed.

Paired changes against a consensus seat, 20 drafts × 12 seats × 20 schedules per season:

| the test seat uses | title odds | all-play | all-play by season |
|---|---|---|---|
| our draft, consensus lineups | −3.2 pp | +1.7 pp | +4.7 +9.9 −4.9 −1.4 −0.1 |
| consensus draft, our lineups | −0.8 pp | −1.7 pp | negative in all five |
| consensus draft, blended lineups | +0.2 pp | −0.3 pp | |
| exact consensus draft, consensus lineups | **+9.2 pp** | +8.5 pp | positive in all five |
| blended draft *minus* exact consensus draft | −0.8 pp | +2.6 pp | +6.2 +4.3 −1.7 +3.9 +0.1 |

If every opponent follows consensus exactly (the sensitivity check), our draft loses
7.6 points of title odds and our lineups lose 2.0. The blended draft beats exact
consensus by 6.1 points of all-play but not in title odds (−0.1), and only in
2021–22.

What it says:

- **No current policy is shown to beat consensus.** That holds for this five-season
  simulation under the rule declared up front. The weekly model sets worse lineups
  than weekly ECR in all five seasons under both opponent designs, so for start/sit
  today, use consensus. Blending the two only ties it.
- **The draft board loses title odds.** It takes a quarterback in rounds 1–3 in 58 of
  60 drafts. The likely cause: pricing QBs against the last starter overvalues them in
  a one-QB league, where a streamable QB is always on waivers. Its rosters hold up
  through week 14, then fade in the playoff weeks. It's not a draft recommendation
  until that's fixed.
- **The biggest effect is noise, not modelling.** Giving the test seat noise-free
  consensus rankings, while the opponents keep their noise, raises title odds by about
  9 points. That's largely built in, since the opponents' deviations are pure error
  here. It isolates ranking noise, not an edge a model could carry over.
- **Blending model and consensus** (averaging ranks) adds some all-play over exact
  consensus, but only in some seasons and with no title gain. Not shown to help.
- **The room is in-season.** A simulated ceiling, lineups picked with hindsight,
  sits about 16 points of all-play above consensus lineups. That and waivers are
  where an edge could exist, and where to build next, tested against this harness
  with the same rule.

Caveats: the five seasons are the independent evidence, not the thousands of league
seats. Within one league seat, a paired title difference moves by 9–27 points (sd)
with the schedule alone, so no title effect here is precise. That's why all-play
decides and title odds are only the headline. The opponents are synthetic (consensus
plus noise, not real draft behaviour), and there are no waivers or trades.

**Consensus also out-forecasts the projection.** On 2023–25, over the draftable pool
with busts counted, preseason consensus beats our season projection at every position
(correlation QB .39 vs .30, RB .69 vs .66, WR .64 vs .57, TE .51 vs .41). A weighted stack
of the two (`stack.py`, fit on 2020–22) only ties consensus. Consensus already has what
the model has, plus news, depth charts and team context.

**Planning around the room doesn't help either.** The last idea was a draft that keeps
consensus values but plans picks around when consensus managers will take each player
(Fry–Lundberg–Ohlmann against predicted availability). It was preregistered in
`prereg.md`, frozen and committed before it ran, and run once. Both versions fail the
rule under both opponent designs (all-play −6.4 to +0.7 points against exact
consensus). The details are in that file.

**Real draft-market prices don't help either.** ADP (average draft position, free from
Fantasy Football Calculator) is a different kind of signal from consensus rankings, a
price rather than an opinion, but it correlates slightly worse with actual points (.509
vs .520 overall, 2021–25, ECR ahead at every position). Drafting straight by ADP, and a
policy that pays ADP prices for consensus-implied value, both fail clearly and in both
opponent designs (`prereg_adp.md`): title odds down 5–7 points, all-play down, negative
in nearly every season. The two lists agree 95–98% of the time, so a tie was the honest
prior; it came in worse than a tie.

**Waivers are a real lever, but reading usage doesn't beat reading consensus's own
rankings on the wire.** Nothing had tested in-season transactions before; adding a
weekly add/drop (worst record first, one move a team) to the harness shows the wire is
worth something on its own: a team that works it alone, while everyone else's roster is
frozen, gains 3–5 points of all-play and 6–13 points of title odds over a league where
nobody uses it (`prereg_waivers.md`). But a signal built to catch role changes early
(recent target and carry share, red zone work, an injury-vacated starting job, all fit
on seasons before the one it's scored on) loses to a seat that just adds whoever
consensus rest-of-season ranks highest — by 3–5 points of all-play, in both opponent
designs, in most seasons. It makes more moves than consensus does (14–15 a season
against 10–12), which reads as chasing short usage spikes that don't hold up rather
than catching durable role changes. Consensus is already good at this particular job.

**Filling next week's hole on the wire does beat consensus waivers, by a little.** The
consensus wire policy adds the best rest-of-season free agent at any position, even when
the only tight end is on bye. A seat that instead fills a starting hole it can see at
waiver time (a bye, or a player off the 53 or ruled Out or Doubtful last week), dropping
its least useful bench player even when the add ranks lower, gains 1.0 point of all-play
and 1.6 of title odds with noisy opponents, 2.0 and 1.2 with exact ones, positive in all
five seasons in both (`prereg_streaming.md`). It makes about two such moves a season,
each worth 6-8 lineup points that week. Same signal as consensus, better decision.

**Grabbing next week's or the week after's fill early doesn't add to that.** Streaming
waits until a hole is one week away, and in the harness the player who would fill it is
already on another team about half the time by then. A seat that also pre-acquires the
fill for a bye one or two weeks ahead (`sim_lookahead.py`, `prereg_lookahead.md`) wins
that race but gains nothing: with a one-week horizon all-play moves +0.0 (noisy
opponents) and -0.2 points (exact), and with two weeks +0.2 and -0.3, with title odds down
0.7-1.4 points against exact opponents. Neither horizon passes in both designs. A third
to a half of the early adds no longer cover a hole when the week comes, and each one
costs the week's only move.

**Draft structure on top of consensus doesn't clear the bar either.** Three rules keep
the exact consensus order and only limit which positions the seat takes when, with the
streaming wire in every arm (`sim_draftstruct.py`, `prereg_draftstruct.md`). Waiting on
QB until round 9 and TE until round 8 loses in both designs (all-play -2.5 and -1.4
points, lineups lower most weeks), so a late QB isn't free even with a wire. A best-ball
template (2-3 RB and 3-4 WR through round 6, at most one QB or TE) gains +0.6 and +0.7
points of all-play and about 9 regular-season points in both designs, but only 3 of 5
seasons are positive with noisy opponents, so it isn't shown to help. Leaning toward
positions the room has under-drafted barely changes a pick: consensus-plus-noise
opponents rarely leave a whole position on the board, and exact ones never do.

**Stashing other teams' backup running backs on top of streaming isn't shown to help.**
A seat that streams and also, in weeks with no hole to fill, holds up to two backups
behind lead backs of teams its own backs don't play for (priced by a contingent value fit
on earlier seasons: how often lead backs miss games, how likely the backup is to inherit
the role, what he scores then) gains 0.8 points of all-play with noisy opponents and 1.6
with exact ones, but only in 3 of 5 seasons in the noisy design, so it fails the rule
(`prereg_handcuffs.md`). The stashes do start: about eight starts and 90-100 lineup
points a season, against about two starts for handcuffs behind the seat's own backs,
which add nothing. The gain comes from seasons with many lead-back absences.
**Bidding just enough on FAAB waivers doesn't beat bidding like the league.** Many
leagues run blind-bid budgets instead of priority waivers. `sleeper_faab.py` pulled
every waiver claim from 887 real 12-team redraft PPR FAAB leagues on Sleeper (2023-25,
123,881 claims). Real bids are skewed and fall through the season: winning bids average
7.4% of budget with a median of 3%, and 16% for a top-12 player in weeks 2-4.
`sim_faab.py` gives every team $100 and opponents bids drawn from those claims. The test
seat keeps streaming's claims and bids the least that wins with 75% probability against
the teams it can see chasing the same player, under a pacing cap (`prereg_faab.md`). It
wins its intended claim 41-47% of the time against 12-14% and overpays half as much.
All-play doesn't move (-0.4 and +0.4 points, not positive in 4 of 5 seasons), so it
fails in both designs, though title odds rose 1.3 and 2.4 points. Caveat: the simulated
opponents all chase the same player and spend their whole budget, which real managers
don't.

**Prediction markets are sharp, but consensus already knows what they know.** Kalshi
moves settled markets to a separate historical API, which keeps the whole 2025 season:
receptions, receiving yards, passing yards and anytime touchdowns, as ladders of "X or
more" markets with real volume. `kalshi.py` reads every one at the last hour ending at
least an hour before kickoff (21,123 pregame prices, 494 players, weeks 1–18), and
`kalshi_test.py` turns each ladder into an expected stat and compares against consensus
on 3,465 player-weeks:

- The prices are good. Implied receptions and receiving yards correlate 0.52 and 0.53
  with what happened, and touchdown prices are calibrated (priced 9% scored 8%, priced
  36% scored 37%).
- Adding them to consensus changes almost nothing. Out-of-sample MAE goes 5.17 → 5.15
  at RB, 4.86 → 4.83 at WR, 3.95 → 3.93 at TE, and gets worse at QB (6.38 → 6.43).
- On their own they rank players worse than consensus at RB, WR and TE, even at WR and
  TE, where the market prices nearly the whole PPR score, and tie at QB.

Polymarket gives the same answer on thinner data (`props.py`, `props_test.py`: 1,330
pregame prices from weeks 14–17, R² 0.316 with consensus alone and 0.317 with the market
added). One season is a screen, not a harness verdict, but a gain under 1% of MAE could
not move a league result anyway.

**Lineups that maximise wins instead of points don't help.** A week is won by
outscoring other teams, not by points, so the lineup with the most expected all-play
wins can differ from the one with the most expected points: an underdog should want
variance, a favourite should shed it. `winprob.py` keeps consensus's values, adds a
spread model fit on earlier seasons (the empirical, right-skewed spread of scores around
each consensus value, by position and value tier), and starts the lineup that beats the
most opponents in simulation (`prereg_winprob.md`). It fails in both designs: the chosen
lineup differs from consensus's in under 1% of weeks, all-play and title odds don't
move, and the rare departures do worse than the model expected. Two players with similar
consensus value have nearly the same spread, so there's almost no variance to trade. The
variance lever that plausibly exists is correlation (stacking a quarterback with his
receiver), which this model doesn't see.

**Consensus isn't slow or biased in the ways people assume.** `ecr_bias.py` checks the
2020–25 rankings directly. They don't chase hot or cold weeks. Blending preseason ranks
or season-to-date scoring into the rest-of-season rank doesn't improve it out of sample.
And backups get re-ranked the same week their starter is ruled out. The one bias that
holds in all six seasons is Questionable players: ranked as if they'll play, they sit
20–50% of the time. Preferring an untagged player within about four ranks is worth
roughly 2.5 points per decision, but only before inactives are announced. `lineup.py`
already discounts Questionable players by how often they play.

**Knowing which players a platform's room will leave you doesn't help either.** A 2026
Reddit post argues that drafters anchor on their platform's default list, so a player
consensus likes but the platform buries can be waited on. `platform_ranks.py` recovers
ESPN and Sleeper ADP for 2021-25 from FantasyPros' ADP-by-site table (platform ADP, not
the default lists themselves, which couldn't be found). `sim_platformranks.py` has the
opponents draft off that platform order with the usual noise, and gives the test seat
consensus values plus a one-pick lookahead on the room's order: take someone the room
will take now, and the consensus favourite at the next pick (`prereg_platformranks.md`).
It fails in both designs. All-play moves −0.0 points with noisy opponents and −0.2 with
exact ones, and title odds −0.3 and +0.3. The seat waits 0.4-1.9 times a draft, and
80-84% of the players it waits on do come back to it. But what it takes in the meantime
is only a few consensus places worse than the favourite, so the rosters barely change.
**Trading for your own lineup beats streaming alone, against opponents who accept any fair
offer.** Two managers who price every player the same can both gain from a trade, because
a fourth receiver is worth little to one team and a lot to a team whose tight end is out.
`sim_trades.py` adds a trade each week (weeks 3-11) on top of streaming: the 1-for-1 or
2-for-1 that most raises the seat's rest-of-season starting lineup, among offers the
opponent accepts on consensus value alone (`prereg_trades.md`). Real Sleeper trades set the
acceptance rule: accepters took a median 0.81 of the consensus value they gave, so no
premium is required. It passes in both designs, +3.5 and +4.8 points of all-play over
streaming, positive in 4 of 5 seasons; title odds are noisy (+0.4 and +2.5). It's an upper
bound: the seat trades about 8 times a season against opponents who accept every fair
offer, while real teams trade 0.4 times a season.

**Two trades a season that both teams' numbers favour still beat streaming alone, by
less.** `sim_tradecap.py` reuses that trade search with two realistic limits
(`prereg_tradecap.md`). The seat makes at most one, two or four trades a season, each only
when it projects at least 10 rest-of-season lineup points. And under mutual benefit the
opponent also refuses any trade that lowers his own projected starting lineup, on the same
consensus numbers. The preregistered arm, two such trades a season, makes the busiest
trader in a typical real league (about the 95th percentile of Sleeper teams). It passes in
both designs: +1.1 points of all-play with noisy opponents (positive in all five seasons,
title -0.7) and +1.9 with exact ones (4 of 5, title +3.6). That's roughly what streaming
itself added, and 2-3 points below the uncapped upper bound. Most of the uncapped gain
sits in the first trade or two: one value-fair trade a season keeps +1.0 and +2.8. The
mutual rule costs another 0.6-1.1 points at two trades, and no more than that, because such
offers are common: the seat has one in about
nine weeks of ten, and hundreds of them on week-3 rosters. The harness still assumes an
opponent accepts whatever his own projections favour, which a real manager won't always do.

So the tools now build on consensus. `lineup.py` sets start/sit from weekly consensus
ranks, and `trade.py` values players by consensus rest-of-season ranks (`--values model`
for the old behaviour). Our model supplies what consensus doesn't: availability,
injury runs and byes in the trade simulation. The draft advice is to draft by consensus
and not reach. `waivers.py` runs the one decision that beat consensus: each week it finds
next week's starting holes from what's known when waivers run (byes, players off an
active roster or ruled out last week) and fills them, falling back to the plain
consensus move when there's nothing to fill.

## Using it

```bash
.venv/bin/python draft/seasons.py            # season table for QB/RB/WR/TE
.venv/bin/python draft/depth_role.py         # week-1 depth-chart role, 2012 on
.venv/bin/python draft/project_season.py     # projections (with role) + holdout validation
.venv/bin/python draft/role.py               # what the role adjustment buys, on the holdout
.venv/bin/python draft/vbd.py                # replacement levels and VBD board
.venv/bin/python draft/backtest.py           # simulate drafts, score the strategies
.venv/bin/python draft/board.py 2026         # board for the upcoming season
.venv/bin/python draft/consensus.py          # FantasyPros consensus ranks, as of each decision
.venv/bin/python draft/league_backtest.py    # does it win leagues vs consensus? (--noise 0: exact opponents)
.venv/bin/python draft/league_backtest.py --waivers   # does usage beat consensus on the wire? (prereg_waivers.md)
.venv/bin/python draft/league_backtest.py --streaming # does filling next week's hole beat consensus waivers? (prereg_streaming.md)
.venv/bin/python draft/sim_draftstruct.py --noise 1  # draft structure on consensus: late QB/TE, template, reactive (prereg_draftstruct.md)

.venv/bin/python draft/sim_handcuffs.py --noise 1  # streaming plus a cross-team RB stash (prereg_handcuffs.md)
.venv/bin/python draft/stack.py              # model + consensus season stack, fit 2020-22, checked 2023-25
.venv/bin/python draft/adp.py                # historical ADP, for the market-signal test (prereg_adp.md)
.venv/bin/python draft/lineup.py --mine "..." # this week's start/sit, by consensus
.venv/bin/python draft/waivers.py --league league.json   # this week's waiver move (fill next week's holes)
```

Then on draft day:

```bash
python draft/assistant.py --seat 4 --pick 45 \
  --mine "Bijan Robinson, Puka Nacua, Trey McBride" \
  --taken "Ja'Marr Chase, Justin Jefferson, ..."
```

```
pick 45, seat 4   roster: Bijan Robinson, Puka Nacua, Trey McBride
still to fill: QB, RB, WR, FLEX

>>> take a WR: Davante Adams
    plan for your remaining picks: WR -> RB -> QB -> WR -> RB -> WR ...

best available by position
  QB  Drake Maye (36),  Jalen Hurts (28),  Bo Nix (20)
  RB  Ashton Jeanty (57),  Kyren Williams (51),  Josh Jacobs (48)
  WR  Davante Adams (59),  Chris Olave (49),  A.J. Brown (49)
  TE  Travis Kelce (52),  Brock Bowers (40),  Tyler Warren (29)
```

League settings live in `LEAGUE` at the top of `vbd.py` — teams, starting slots, flex,
rounds. Scoring is full PPR; `seasons.py` also carries standard scoring if you want it.

### During the season

```bash
.venv/bin/python update.py                # pull new stats, schedule, lines, injury reports; rebuild both
.venv/bin/python draft/weekly.py 2026     # next-game projections  -> data/weekly_2026.parquet
.venv/bin/python draft/ros.py 2026        # rest-of-season board   -> data/ros_2026.parquet
.venv/bin/python draft/weekly.py --eval   # train <= 2020, score 2021-25
```

`weekly.py` starts each player at his preseason projection (rookies at a draft-slot
prior) and pools in this season's games as they arrive, then fits that rate to the
game: implied team total from the betting line, the opponent's running points allowed
to the position, home field. `ros.py` projects every remaining game the same way,
discounts each by the chance he plays it, and prices the total by VBD — that's the
board for waivers and trades.

Holdout 2021–25, 29,376 player-games:

| model | RMSE | MAE | corr | rank within position-week |
|---|---|---|---|---|
| weekly model | **6.12** | **4.53** | **0.639** | **0.607** |
| pooled rate only (no matchup) | 6.15 | 4.59 | 0.636 | 0.603 |
| season-to-date average | 6.45 | 4.63 | 0.605 | 0.576 |
| preseason projection | 6.46 | 4.96 | 0.587 | 0.538 |

Scored only on the fantasy-relevant pool, the way published accuracy studies score
experts, correlation is QB .24, RB .43, WR .30, TE .20 — around the best public expert
sources at RB and WR (≈.45 and ≈.30), short of them at QB and TE (≈.30).

- **The in-season update is almost all of it.** The preseason prior is worth only 3–4
  games of this season's evidence, so the board moves quickly in September.
- **Matchups and betting lines add little** on top of a well-pooled rate: dropping both
  costs 0.03 RMSE. Opponent-vs-position is mostly noise, which matches what's been
  published. Usage and red-zone shares were tested and bought 0.016 more; left out.
- **Questionable players play 57% of the time** (2016–25 reports against box scores);
  doubtful 1%.

Limits: projections are conditional on playing. The preseason prior carries each
player's week-1 depth slot, but mid-season depth changes are invisible until they show
up in the box score, so a backup promoted this week isn't flagged. The spread is symmetric while real
weekly scoring is right-skewed, so `proj_sd` is too wide in the middle (75% of games
land inside one sd) and too narrow in the upside tail.

### Trades

`trade.py` values a trade by what it does to each side's season, not by the players'
projections side by side. A player is worth what he adds to the lineup of the team that
gets him: a third running back counts only in the weeks he'd actually start, and only by
his margin over whoever would start instead, down to the waiver wire.

```bash
# a trade someone offered you
.venv/bin/python draft/trade.py --league league.json --with Alex \
  --give "Ja'Marr Chase" --get "Christian McCaffrey, Derrick Henry"

# or without a league file
.venv/bin/python draft/trade.py --mine "..." --theirs "..." --give "..." --get "..."

# trades that help you and that the other side should also want
.venv/bin/python draft/trade.py --league league.json --suggest [--with Alex]
```

`league.json` is `{"me": "John", "teams": {"John": ["name", ...], "Alex": [...], ...}}`.

Points per game come from consensus rest-of-season ranks by default (run
`draft/consensus.py` for fresh ones). The model's availability and injury rates still
drive the simulation. `--values model` uses the model's rates instead.

```
you give: Ja'Marr Chase
you get:  Christian McCaffrey, Derrick Henry

                 wins     pts, weeks to 14    playoff pts
you      +1.26 ± 0.01                +89.0          +19.4
       (must drop Marquise Brown to make room)
them     -0.40 ± 0.01                -28.5           -5.8

>>> accept - it helps you
```

How it works: each side's rest of season is simulated with and without the trade, a few
thousand times. Every week each player is active, on bye, or hurt; injuries follow a
two-state chain whose rates reproduce his projected games, so a fragile player misses
runs of weeks rather than scattered single games, and anyone on this week's injury report
starts at the weekly model's chance of playing. The lineup is set from who's available,
on projections, with the waiver line as a fallback at every slot. Each week's expected
score becomes a win probability against a league-average lineup. Both versions of a
roster see the same simulated injuries, so the difference is the trade and not noise.
When a trade brings in more players than it sends, the evaluator cuts whoever costs the
least.

`--suggest` enumerates 1-for-1, 2-for-1, 1-for-2 and 2-for-2 deals with every other team,
keeps those where both sides gain wins, and re-runs the best with more simulations.
Trades like that exist whenever one roster has depth where the other has a hole. A search
across a whole 12-team league takes about three minutes; `--with` narrows it to one team
(about 15 seconds).

Does it work? `trade_backtest.py` drafts twelve rosters from the preseason board in each
holdout season, makes 400 random trades between them, and values each at week 1. The
trade is then played out on what actually happened: every week, both versions of the
roster start the best lineup by projection among players who played, scored on what they
scored.

2,000 trades, 2021–25:

| valuation | corr with realized change | picks the side that gained |
|---|---|---|
| lineup simulation (`trade.py`) | **0.64** | **71%** |
| value over replacement | 0.47 | 64% |
| projected points in minus out | 0.37 | 62% |

The simulation beats points-in-minus-out by 0.27 in correlation (bootstrap 95% interval
0.24–0.31), and it holds for every trade shape. Of the trades it called good for both
sides, both sides really did come out ahead 37% of the time, against 15% for a random
trade.

Read with care: the realized side uses the same lineup rule and the same waiver line as
the simulation, which flatters it somewhat, and the preseason projections are held fixed
all season. The win estimate is against an average opponent, not your schedule or your
playoff odds. And "the other side should want it" means by this model's valuation. A
manager reading consensus rankings may see it differently.

## Known limits

- **It doesn't beat consensus yet.** In the league backtest neither the draft board
  nor the weekly model improves title odds over expert consensus. See "Does it win
  leagues?".
- **Rookies come from draft slot and depth-chart role only.** Nothing is known about them
  beyond where they were picked and where they sit on the week-1 chart. Rookie tight
  ends still project about 30 points low.
- **Role is read once, from the week-1 depth chart.** The preseason board and prior know
  who's starting in September; a mid-season promotion or benching reaches the numbers
  only through the box scores that follow. Team changes and carry splits within a role
  are projected off the player's own history.
- **The market baseline is weak**, as above.
- Holdouts, suspensions and camp news are invisible to it.

## Not replicated

[Hunter, Vielma & Zaman, *Picking Winners in Daily Fantasy Sports Using Integer
Programming*](https://arxiv.org/abs/1604.01455) solves a genuinely different problem —
selecting a portfolio of lineups for top-heavy DFS contests, where you want correlated
risk rather than the highest expected score. It's the natural next build for DFS, but it
doesn't inform a season-long draft.
