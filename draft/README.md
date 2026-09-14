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

**Prediction markets don't add to consensus either.** Kalshi purges settled markets
after about a month, so its 2025 season is gone. Polymarket keeps 2025, but only listed
player props in volume from week 14, about ten players a game, with no receptions
market. `props.py` pulls every pregame price it has (1,330 prices, 268 players, read at
least an hour before kickoff), and `props_test.py` asks whether they improve on
consensus. They don't: R² 0.316 with consensus alone, 0.317 with the market added, and
out of sample the combination is slightly worse (MAE 5.58 against 5.61). The prices do
carry signal (implied yards correlate 0.45 with actual yards), it's just signal
consensus already has. Four weeks of one season is a screen, not a verdict.

So the tools now build on consensus. `lineup.py` sets start/sit from weekly consensus
ranks, and `trade.py` values players by consensus rest-of-season ranks (`--values model`
for the old behaviour). Our model supplies what consensus doesn't: availability,
injury runs and byes in the trade simulation. The draft advice is to draft by consensus
and not reach.

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
.venv/bin/python draft/stack.py              # model + consensus season stack, fit 2020-22, checked 2023-25
.venv/bin/python draft/adp.py                # historical ADP, for the market-signal test (prereg_adp.md)
.venv/bin/python draft/lineup.py --mine "..." # this week's start/sit, by consensus
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
