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
| hierarchical projection | **60.2** | **44.9** | **0.756** |
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

## Does any of it work?

`backtest.py`. Twelve teams, 14-round snake, PPR, holdout seasons 2021–25. Each strategy
holds three seats, rotated four ways so draft position can't decide the outcome. Rosters
are scored on the best legal lineup each week using what players actually did — so it
measures roster quality, not in-season management.

| strategy | mean season points | sd |
|---|---|---|
| Fry–Lundberg–Ohlmann DP | **1772** | 142 |
| value-based drafting | 1743 | 200 |
| prior-year finish (a naive market) | 1683 | 162 |
| best projected points | 1132 | 164 |

Paired within the same league (n=20 leagues):

| DP vs | difference | std error | DP ahead in |
|---|---|---|---|
| value-based drafting | +16 | 26 | 70% |
| prior-year finish | **+87** | 26 | 75% |
| best projected points | **+627** | 24 | 100% |

Read honestly:

- **Positional value is everything.** Drafting by raw projected points loses by 627
  points a season — it takes quarterbacks early and never recovers. This is the whole
  ballgame, and it's a 1970s idea, not a modern one.
- **The board beats a naive market by ~87 points a season**, about 5%, at roughly 3
  standard errors. That's a real edge against a league drafting off last year's finish.
- **The DP paper adds nothing reliable over plain VBD** — +16 ± 26 is noise, though it
  wins 70% of leagues and cuts the spread of outcomes (sd 142 vs 200). Its value here is
  consistency, not upside. Replicated faithfully, and honestly not worth much in this
  setting.

A caveat on the market baseline: no free source of historical ADP exists, so the "market"
is modelled as prior-year fantasy finish. A real league drafting off consensus rankings
is a tougher opponent than that, so treat +87 as an upper bound.

## Using it

```bash
.venv/bin/python draft/seasons.py            # season table for QB/RB/WR/TE
.venv/bin/python draft/project_season.py     # projections + holdout validation
.venv/bin/python draft/vbd.py                # replacement levels and VBD board
.venv/bin/python draft/backtest.py           # simulate drafts, score the strategies
.venv/bin/python draft/board.py 2026         # board for the upcoming season
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
| weekly model | **6.16** | **4.58** | **0.632** | **0.597** |
| pooled rate only (no matchup) | 6.20 | 4.66 | 0.629 | 0.594 |
| season-to-date average | 6.48 | 4.66 | 0.601 | 0.568 |
| preseason projection | 6.57 | 5.12 | 0.574 | 0.518 |

Scored only on the fantasy-relevant pool, the way published accuracy studies score
experts, correlation is QB .24, RB .42, WR .30, TE .21 — around the best public expert
sources at RB and WR (≈.45 and ≈.30), short of them at QB and TE (≈.30).

- **The in-season update is almost all of it.** The preseason prior is worth only 2–4
  games of this season's evidence, so the board moves quickly in September.
- **Matchups and betting lines add little** on top of a well-pooled rate: dropping both
  costs 0.02 RMSE. Opponent-vs-position is mostly noise, which matches what's been
  published. Usage and red-zone shares were tested and bought 0.016 more; left out.
- **Questionable players play 57% of the time** (2016–25 reports against box scores);
  doubtful 1%.

Limits: projections are conditional on playing, and depth charts are still invisible,
so a backup who won't see the field isn't flagged. The spread is symmetric while real
weekly scoring is right-skewed, so `proj_sd` is too wide in the middle (75% of games
land inside one sd) and too narrow in the upside tail.

## Known limits

- **No rookies.** They have no NFL history to pool, so they're absent from the board
  entirely. In a 2026 draft that's a real hole near the top of round two onward.
- **No team or usage context.** A back who changed teams, or lost his line, or is now
  splitting carries, is projected off his own history alone.
- **The market baseline is weak**, as above.
- Depth charts, holdouts, suspensions and camp news are all invisible to it.

## Not replicated

[Hunter, Vielma & Zaman, *Picking Winners in Daily Fantasy Sports Using Integer
Programming*](https://arxiv.org/abs/1604.01455) solves a genuinely different problem —
selecting a portfolio of lineups for top-heavy DFS contests, where you want correlated
risk rather than the highest expected score. It's the natural next build for DFS, but it
doesn't inform a season-long draft.
