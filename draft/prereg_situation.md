# Preregistration: trades and waivers that read the seat's own situation

Frozen 2026-09-15, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check under `--noise 1` had already been run and is not part of the result; it
printed no outcome metrics and decided nothing. It found the `control` arm identical to
`sim_tradecap.py`'s `mutual_cap2` rows for leagues 0-1 of every season (120 rows, every
column, max difference 0.0); `avail_all` equal to `sim_trades.future_avail` on weeks 3-11
in all five seasons (asserted); every executed trade in the control clearing the 10-point
bar and leaving the opponent's projected lineup no worse (asserted against the executed
rosters); trades falling only in weeks 3-11; and no arm exceeding two trades a seat
(means 1.98 to 2.00, against the control's own). No `--noise 0` check was run; the real
run re-checks the reproduction against `sim_tradecap.py`'s full result files in both
designs. Any later change to the arms below is a new preregistration, not an edit to this
one.

## Why this test

Every policy this harness has tested is stationary. Hole-aware streaming fills the same
hole the same way in week 2 and week 16. `mutual_cap2` applies the same 10-point bar to a
trade in week 3, with fifteen weeks of season left to collect it, as to a trade in week 11
with seven. Neither policy knows whether its seat is 1-5 or 5-1, so neither behaves
differently when the season is already lost or already won. Real managers do both things:
they discount a rest-of-season gain by how much season is left, and they buy or sell
depending on whether they are in the race.

Two pieces of situation are knowable at decision time without touching the future:

1. **Weeks remaining** is on the calendar.
2. **Playoff odds** can be estimated from games already played plus a forward simulation
   of the weeks left, using only consensus's own projections and a spread fitted on
   earlier seasons.

The question is whether conditioning the trade bar and the waiver add on those two things
beats the best realistic policy this harness has, which is `mutual_cap2`.

## Control

**`control` is `sim_tradecap.py`'s `mutual_cap2`, exactly.** Exact-consensus draft, weekly
consensus lineups, eleven opponents on the consensus wire, the test seat on hole-aware
streaming (`LB.streaming_policy`), at most two executed trades a season, each the best
mutual-benefit offer found in weeks 3-11 and each projecting at least
**GAIN_MIN = 10 rest-of-season starting-lineup points**. The trade search, the acceptance
rule, the 2-for-1 mechanics and the tie order are `sim_tradecap.search_mutual` and
`sim_trades.execute` unchanged; neither `sim_tradecap.py`, `sim_trades.py` nor
`league_backtest.py` is edited. The control must reproduce `sim_tradecap.py`'s
`mutual_cap2` rows exactly, checked at 2 leagues before the run and again on the full
files after it.

## The two situational rules

### 1. Weeks-remaining scaling of the trade bar

The 10-point bar prices a rest-of-season gain, so the same number is a much harder test
late than early. The decision before week v covers weeks v..17, that is `18 - v` weeks; at
the first decision (v = 3) that is 15 weeks. The scaled bar is

    bar(v) = 10 * (18 - v) / 15

so it runs 10.0 in week 3, 8.0 in week 6, 6.0 in week 9 and 4.67 in week 11. **This is a
constant 0.667 projected lineup points per remaining week, which is exactly the rate the
frozen flat bar implies at the week it was calibrated on.** That is the parameterisation,
and it introduces no new free number: the level is `sim_tradecap`'s frozen 10 and the
anchor week is the first decision week the trade window has. The alternative of a bar
indexed to a later week would change the level as well as the shape, so it is not run.

The intended effect is not more trades in total (the cap is still two) but a willingness
to spend a trade late on a gain that is large *for the time left*, which the flat bar
refuses.

### 2. Playoff-odds-aware aggressiveness

**How the odds are computed.** At a decision before week v, with weeks 1..v-1 already
played:

- The harness scores twenty schedule draws over one set of weekly scores, so at decision
  time there is no single head-to-head standings table to read. The harness already solves
  that for waiver priority by using all-play instead of head-to-head (`LB.priority`), and
  this uses the same convention: **playoff odds are the probability that the seat's
  all-play wins over weeks 1-14 finish in the top six**, ties broken by points for. This is
  a schedule-free surrogate for "makes the top six", and is stated as such.
- All-play wins and points for over weeks 1..v-1 are the harness's own realised scores,
  which are known at the decision.
- For each remaining regular-season week, each of the twelve teams scores
  `mu[t, j] + sigma[t] * z`, z standard normal and independent across teams and weeks.
  `mu[t, j]` is that team's projected best legal starting lineup in that week, from the
  same consensus rest-of-season points per game and the same bye/availability mask the
  trade search uses (`sim_trades.ros_points` and the `future_avail` rule, extended to
  every waiver week in `avail_all`). `sigma[t]` is the team's week-v projected starters'
  weekly spreads added in quadrature.
- A player's weekly spread is the standard deviation of `winprob.fit_spread`'s residual
  pool for his position and consensus-value bucket, **fitted on seasons 2020..y-1 only**
  and centred to mean zero there, so it never moves anyone's expected points.
- 1,000 Monte Carlo scenarios, drawn from `numpy.random.default_rng([y, lg, seat, v, 11])`:
  a stream of its own, independent of the harness's draws, identical across arms and
  across the two opponent designs, so two arms asking the same question in the same state
  get the same answer.
- At most one estimate a week, made the first time a decision that week asks for one and
  reused by the rest of that week.
- For v >= 15 the regular season is over and the field is decided, so the odds are 1 or 0
  from the realised week-14 all-play standings.

**Buyer and seller.** A seat is a **seller** when its odds are below
**ODDS_CUT = 0.25**, and a **buyer** otherwise. The cut is one chance in four of a
six-of-twelve playoff, a seat that needs an unusual run; spending a scarce trade there on
a marginal rest-of-season gain, or the week's only waiver move on a slightly better floor,
is the wrong use of both. It is a round number fixed before the run and not swept.

- **Trade bar.** A buyer's bar is multiplied by **BUY_MULT = 0.5** (trade on half the
  usual gain, because the berth is live and the two trades are worth spending), a seller's
  by **SELL_MULT = 2.0** (hold out for something large). The cap stays 2 and every other
  part of the trade rule, including mutual benefit, is unchanged.
- **Waiver add.** Among free agents the streaming policy is choosing between, take the
  best by rest-of-season value unless another is within **NEAR_ADD = 1.0 points per game**
  of it; among those within the band, a **seller** takes the one whose weekly score
  distribution has the highest 90th percentile (upside for a team that needs variance) and
  a **buyer** the one with the highest 10th percentile (floor for a team protecting a
  berth). Percentiles come from the same prior-seasons residual pools, which are centred,
  so the tilt trades expected points for shape and never claims a better forecast. Ties go
  to the higher value, then to the lower index, so with a single candidate in the band the
  choice is exactly `LB.streaming_policy`'s own argmax and no odds estimate is even made.
  The hole rule, the fallback to the consensus move, the drop rule and the
  "add must outrank the drop" rule in the no-hole branch are all unchanged.
  NEAR_ADD is about a quarter of a typical rostered player's value over replacement and
  small next to the 8-10 point weekly spread it is being traded for.

## Arms

| arm | trade bar | waiver add |
|---|---|---|
| `control` | flat 10 | hole-aware streaming |
| `scaled` | `10 * (18-v)/15` | hole-aware streaming |
| `odds` | `10 * (0.5 buyer / 2.0 seller)` | hole-aware streaming |
| `situation` | `10 * (18-v)/15 * (0.5 buyer / 2.0 seller)` | odds-tilted |

**Which arm carries the verdict: `situation`, alone.** It is the whole hypothesis, both
levers on the trades and the waivers too; judging one arm avoids picking the best of four
afterwards. `scaled` and `odds` are a labeled sensitivity that separates the two trade
levers from each other and from the waiver tilt.

An end-of-season all-in rule (after week 9, value players only for weeks 15-17 when
already playoff-bound) is **not run**. `mutual_cap2` spends both its trades at a mean week
of 4.4-4.8 and uses them both in 83-92% of seat-seasons, so such a rule would fire in a
fraction of a percent of seat-seasons and could not move a result. It is dropped rather
than reported as a null with no power.

## Information timing and leakage control

- The trade search, its values and its availability mask are `sim_trades`' and
  `sim_tradecap`'s unchanged, with the leakage controls stated in `prereg_trades.md`:
  consensus scrapes strictly before week v, a rank curve fit only on seasons before y, and
  availability from the schedule and week v-1 or earlier roster and injury files.
- The odds estimate reads realised scores for weeks 1..v-1 only, and projects the rest
  from the same decision-time consensus numbers. **No future week's result, and no
  schedule outcome, enters it.** The regular-season schedule structure is not used either:
  the all-play convention needs no schedule at all.
- The spread model is `winprob.fit_spread(y, curves)`, fit on seasons 2020..y-1, and its
  pools are centred, so it contributes shape only. `winprob.py` is imported, not edited.
- `avail_all` extends `sim_trades.future_avail`'s rule to weeks 2-17 and is asserted equal
  to it on weeks 3-11 in the mechanics check.
- Stated residual: the three situational numbers (10 scaled from week 3, ODDS_CUT 0.25,
  the 0.5/2.0 multipliers, NEAR_ADD 1.0) were chosen by argument, not fitted, but they
  were chosen by an author who had already read `prereg_tradecap.md`'s Result. None is
  swept, and one arm carries the verdict.
- Stated approximation, not leakage: the odds are a probability of finishing top six in
  all-play, not in the head-to-head standings any one of the twenty schedule draws would
  produce. The two agree on who is good and disagree on schedule luck, which is exactly
  the noise the harness's primary metric already removes.
- Second stated approximation: `mu` is consensus rest-of-season points per game, which is
  not calibrated to actual weekly team totals, so the odds' absolute level is only roughly
  right. The ordering across teams, which is what the buyer/seller split turns on, uses
  one measure applied identically to all twelve.

## Scoring and decision rule

- Harness: `sim_situation.py`, seasons 2021-25, 20 leagues x 12 seats x 20 schedules, both
  opponent designs (`--noise 1` and `--noise 0`), all four arms in one run per design.
- Primary: paired change in all-play win rate, each arm against `control`, same league,
  seat and schedules. Secondary: title, playoff, wins, points.
- **Decision rule, on `situation` only:** it passes only if, in **both** designs, its
  pooled all-play change against `control` is positive, positive in at least 4 of 5
  seasons, and its pooled title change is no worse than -1.0 point. The same rule is
  applied to `scaled` and `odds` and printed, but those are sensitivity results, not
  verdicts.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `sim_trades.py` and
  `sim_tradecap.py`; every arm shares those draws and the same drafts. The odds Monte
  Carlo has its own stream, given above.
- Bootstrap: `LB.paired`, as in `prereg_tradecap.md` (season means, 2,000 resamples with
  `numpy.random.default_rng(1)`, 5th/95th percentiles). The interval is reported; the rule
  decides.
- Run once per design, one after the other through the shared lock. Reduced (`--leagues`
  other than 20) runs write `_check` files.

## Reported, not gated

- How often the situational rule changed a decision: per searched trade week, whether the
  arm's execute/wait differs from what the flat 10-point bar would have done in the same
  state; per waiver decision in `situation`, whether the add differs from streaming's.
- The distribution of playoff odds at trade decisions, and separately at the decisions the
  rule changed, plus the share of decisions made as a seller.
- Trades per seat-season, by season and by arm, and the week they fall in.
- Positions added when the waiver tilt changes the add.

## Known limits

- Everything `prereg_tradecap.md` lists still holds: mutual benefit is judged on the same
  consensus numbers by both sides, offers are never refused at random, and two trades a
  season is about the 95th percentile of real Sleeper teams.
- The opponents are still stationary. Only the test seat reads its situation, so no
  opponent becomes a seller in week 10 and dumps a star, which is where much of the real
  value of being a buyer lives.
- Playoff odds are a top-six-in-all-play surrogate and are not calibrated against realised
  berths in this harness.
- The waiver tilt can only choose among near-equal free agents, and `prereg_winprob.md`
  already found that two players of similar consensus value have nearly the same spread.
  A small effect there is the prior, not a surprise.

## Result (added after the single run per design; the spec above is unchanged)

`control` reproduces `sim_tradecap.py`'s `mutual_cap2` exactly in both designs (1,200
rows each, every column, max difference 0.0).

**The verdict arm, `situation`, fails the rule in both designs.** Changes against
`control`, 90% season-cluster intervals:

| opponents | title odds | all-play | all-play by season 2021-25 | title by season | reg pts | trades/season |
|---|---|---|---|---|---|---|
| noise 1 | -0.3 pp [-1.1, +0.6] | -0.4 pp [-1.0, +0.2] | +0.9 -0.8 -1.4 +0.0 -0.6 | +1.9 -0.6 -1.1 -1.4 -0.5 | -2 | 1.99 |
| noise 0 | -0.9 pp [-4.1, +2.9] | -0.5 pp [-2.2, +2.2] | -1.1 -2.3 +5.2 -2.0 -2.2 | -3.8 -5.8 +7.9 -3.2 +0.5 | -10 | 1.93 |

All-play is negative pooled in both designs and positive in 2 of 5 seasons with noisy
opponents and 1 of 5 with exact ones, so it fails on two counts before the title guard is
reached. Conditioning on the situation the way this test conditions on it does not beat
the stationary policy.

**Sensitivity (not verdicts).** Each trade lever on its own, against `control`:

| arm | noise 1 all-play | seasons + | title | trades | noise 0 all-play | seasons + | title | trades | rule |
|---|---|---|---|---|---|---|---|---|---|
| `scaled` | +0.1 [-0.0, +0.1] | 4 | +0.6 | 1.98 | -0.1 [-0.4, +0.2] | 1 | +0.8 | 1.98 | passes n1 only |
| `odds` | +0.1 [-0.0, +0.2] | 3 | +0.5 | 1.98 | +0.1 [-0.6, +0.8] | 3 | +0.5 | 1.90 | fails both |
| **`situation`** | -0.4 | 2 | -0.3 | 1.99 | -0.5 | 1 | -0.9 | 1.93 | **fails both** |

By season, all-play: `scaled` n1 +0.2 -0.2 +0.1 +0.1 +0.1, n0 -0.5 -0.3 +0.6 -0.0 -0.2;
`odds` n1 +0.2 -0.1 -0.1 +0.3 +0.1, n0 -1.2 -0.5 +1.7 +0.1 +0.6. Title by season:
`scaled` n1 +0.1 +1.1 +1.2 -0.1 +0.4, n0 +0.1 -1.9 +3.6 +0.0 +2.3; `odds` n1 +1.4 +0.9
+1.0 -0.0 -0.8, n0 -1.1 -2.8 +5.0 +1.2 +0.3.

The two trade levers are each worth about +0.1 points of all-play, inside their own
intervals and not positive in four seasons except `scaled` under noise 1. The combined
arm is worse than either, and the difference between `odds` (+0.1, +0.1) and `situation`
(-0.4, -0.5) is the waiver tilt: it costs about half a point of all-play in both designs
and is the only part of this test that clearly does harm.

**Reported, not gated.**

- **How often the rule changed a trade decision.** Against the flat bar in the same
  state, `scaled` executes or waits differently in 5.6% (noise 1) and 9.1% (noise 0) of
  searched weeks, `odds` in 16.6% and 20.8%, `situation` in 16.8% and 19.8%. The mean bar
  falls from 10.0 to 8.8 and 8.5 (`scaled`), 5.6 and 9.0 (`odds`), 5.1 and 7.5
  (`situation`). Trades executed per searched week rise from 0.48 and 0.35 to 0.70 and
  0.51, the mean trade week falls from 4.4 and 4.8 to 4.0 and 4.4, and the median
  projected gain per executed trade falls from 19.7 and 18.5 to 16.4 and 15.9. Realised
  starting-lineup points per trade with both rosters frozen fall the same way: 12.5 and
  29.7 in the control against 9.8 and 25.9 in `situation`. A lower bar buys trades
  earlier and buys worse ones.
- **The odds distribution is the heart of the null.** At the trade decisions that were
  made, playoff odds have deciles 0.48 / 0.74 / 0.90 / 0.97 / 0.99 with noisy opponents
  and 0.05 / 0.22 / 0.50 / 0.77 / 0.92 with exact ones. Only 3.8-4.3% (noise 1) and
  25-27% (noise 0) of those decisions are made as a seller. Trades happen in weeks 3-5 in
  about 90% of seat-seasons, three to five weeks in, when almost nobody in a league where
  six of twelve make the playoffs is out of it yet. **So the odds tilt is, in practice,
  an almost-flat halving of the bar, not a buyer/seller split.** The odds at the changed
  decisions (0.81-0.83 and 0.44-0.54) look like the odds everywhere else. The noise-1
  design is the more extreme case because eleven noisy opponents make the test seat, which
  drafts exact consensus, a favourite nearly everywhere.
- **The waiver tilt fires constantly and removes moves.** In `situation` there are two or
  more free agents within NEAR_ADD of the best in 97.7% (noise 1) and 96.2% (noise 0) of
  waiver decisions, and the tilt changes the add in 65.8% and 64.0% of all decisions. Only
  313 and 480 of those are hole-filling weeks; the other 8,669 and 9,312 are the no-hole
  branch, where the change is a floor-seeking swap on a bench spot. Because the tilted add
  is worth less by rest-of-season value, it more often fails the consensus rule that the
  add must outrank the drop, so the seat makes fewer moves at all: 10.13 against 11.81 a
  season (noise 1) and 12.28 against 12.80 (noise 0). Positions added when it differs are
  mostly RB, then TE/WR and QB.
- Trades per season are 1.83-1.91 in the control and 1.90-1.99 in the situational arms:
  the lower bar mostly means the cap is spent rather than more trades.

**Honest read.** The gap this test set out to close is real, but the test does not close
it, and two of the three reasons are visible in the diagnostics rather than in the
outcome. First, the weeks-remaining scaling is almost inert because `mutual_cap2` already
spends both its trades by week 5 in nine seat-seasons out of ten; a bar that only gets
softer after week 5 has very little left to decide. Second, playoff odds three to five
weeks into a season where half the league makes the playoffs are nearly uninformative, so
the buyer/seller split degenerates into a uniformly lower bar, and a uniformly lower bar
is exactly what `prereg_tradecap.md` already showed to be worse (`mutual_cap2_nomin`,
no threshold at all, was below `mutual_cap2` in both designs). Third, the waiver tilt
touches two decisions in three, which is not a situational rule in any meaningful sense;
it is a standing preference for floor over value that the harness dislikes, and
`prereg_winprob.md` had already found that two players of similar consensus value have
nearly the same spread, so the preference buys almost no variance while giving up real
value. A situational rule that would have had something to bite on would need either a
later decision window, where odds separate, or a trade budget that is not already spent
by week 5.

**Leakage I could not rule out.** None in the odds estimate itself: it reads realised
scores only for weeks already played, projects the rest from decision-time consensus, and
the spread model is fitted on 2020..y-1 and centred. The stated residual is the one in the
spec: the author had read `prereg_tradecap.md`'s Result before choosing 0.25, 0.5/2.0 and
1.0, and those numbers are arguments rather than fits. Since the verdict arm fails, that
residual could only have flattered a result that did not appear.
