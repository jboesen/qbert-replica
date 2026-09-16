# Preregistration: rival-aware waivers and trades

Frozen 2026-09-15, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check under `--noise 1` had already been run; it is not part of the result, it
printed no outcome metrics, and it decided nothing beyond the two thresholds below, which
are set from the distribution of projected blocking gains, a decision input. What the
check found is recorded under "Mechanics check". Any later change to the arms below is a
new preregistration, not an edit to this one.

## Mechanics check (2 leagues per season, `--noise 1`, no outcome metrics)

Every in-code assertion passed: the target rival always sits later in the week's waiver
order than the seat; the `rival` trade rule returns an offer the mutual rule accepts,
never beating the best own gain and never falling below 0.9 of it; every executed trade
leaves both rosters legal at 14 players and leaves the receiving side's projected lineup
no lower; and, with the search run through this module's own copy of the enumeration, the
`mine` rule returns exactly `sim_tradecap.search_mutual`'s offer in every searched week of
every arm. The control makes zero blocks. Trades per seat-season are 1.73-1.86 across
arms, in line with `mutual_cap2`'s 1.91. The unbarred `block` arm makes 5.77 blocks a
season and the barred one 2.43; blocks run from week 2 to week 17 (mean week 7.3), 98% of
them are a move consensus would not have made, 71% add a player worth less than the one
dropped by wire value, and the positions blocked are RB 239, TE 230, QB 190, WR 33. Those
counts are inputs and structure, not outcomes; they changed nothing except the two
threshold values, as described under "The thresholds". Exact-consensus (`--noise 0`)
mechanics were not checked separately; the real run re-checks the control's reproduction
against `sim_tradecap.py`'s full result files in both designs.

## Why this test

Every policy this harness has tested reads one roster: its own. Streaming asks whether
the seat has a hole; the trade search asks what raises the seat's own lineup; the wire
policy ranks free agents by value to nobody in particular. But the harness publishes
what a real league publishes. At the moment a manager acts he can see all twelve rosters,
the standings, the waiver order those standings imply, and the free agents still
unclaimed. A league is a zero-sum race for six playoff seeds and one title, so a point
taken off the team you are racing is worth roughly as much as a point added to yourself,
and it may be far cheaper to buy.

Two decisions are available that only a rival-aware seat can make.

1. **Blocking.** In a week where the seat has no starting hole, `mutual_cap2` makes the
   plain consensus add: the best free agent by rest-of-season value, dropped for the
   worst bench player, a move usually worth a point or two. That same move could instead
   deny a specific rival the free agent who would most raise *his* starting lineup.
2. **Rival-need-aware trading.** The mutual-benefit rule already produces hundreds of
   accepted offers a week (`prereg_tradecap.md`: a median 331 on week-3 rosters). The
   seat currently takes the one that helps it most and is indifferent to which rival it
   arms. Among offers that are near-equivalent to the seat, it can pick the one that does
   the receiving side the least good.

The honest prior is that blocking transfers value rather than creating it. Points denied
to one of eleven rivals do not raise the seat's own score, and all-play scores the seat
against all eleven, so a block that costs the seat anything at all should show up as a
small all-play loss and can only pay in the head-to-head outcomes: seeding, playoff berth
and title. That is the opposite shape from every policy that has passed here, and the
decision rule still gates on all-play, which makes this a hard test to pass by design.
It is the right rule to keep: an arm that loses all-play and gains title odds over five
seasons is not distinguishable from schedule luck at this sample size.

## The seat and the control

Every arm drafts by exact consensus, sets lineups by weekly consensus, runs hole-aware
streaming on the waiver wire, and makes at most two mutual-benefit trades a season with
a 10-point minimum projected gain: that is `mutual_cap2` from `prereg_tradecap.md`, the
best realistic policy this harness has. The other eleven seats run the consensus wire in
every arm. Waiver rules, roster legality, the trade window (weeks 3-11), the candidate
offers searched, the acceptance test and the tie order are all unchanged.

- **`mutual_cap2` (control).** Exactly `sim_tradecap.py`'s `mutual_cap2`. It must
  reproduce that module's rows for the same seasons, leagues and seats exactly, in both
  designs; the real run asserts this against `data/league_tradecap_noise{1,0}.parquet`.

## What the seat is allowed to read

At the moment of each decision, and nowhere else:

| information | when it is read | visible to a consensus manager then? |
|---|---|---|
| all twelve rosters as they stand | at the seat's turn in the waiver order, and at the trade decision | yes, rosters are public and update as claims process |
| standings through the weeks already played | before the week's waivers | yes |
| this week's waiver order | before the week's waivers | yes, it is a published function of the standings |
| free agents still unclaimed at the seat's turn | at its turn | yes |
| consensus rest-of-season ranks scraped strictly before week w | at the decision | yes, same source every arm uses |
| bye weeks from the season schedule | any time | yes |
| week w-1 roster status and injury report | at the decision | yes |

No week-w information of any kind is read, by any arm, including the arms' own
evaluations of rival rosters. The rival-roster evaluation uses the same `pts` and `avail`
tables the trade arms already use, built from the same pre-week-w scrapes, and the same
`lineup_value` function; a rival is valued exactly the way the seat values itself.

## Standings, the target rival, and why it is blockable

The standings are the harness's own: all-play record through the weeks played, ties by
points for, which is what `league_backtest.priority` sorts on. Waiver order is that list
reversed (worst record first). Therefore:

- The **target rival** is the team immediately **above** the seat in the standings. If the
  seat is first in the standings, there is no target and no block is attempted that week.
- Because waiver order is exactly reverse standings, the target always acts **after** the
  seat, so the seat can always reach a free agent before he can. A rival below the seat
  acts before it and cannot be blocked at all, which is why the rule looks up and not
  down. The mechanics check asserts the ordering.

This is the sharpest available definition of "closest to you in the standings" in this
harness: it is the team the seat is directly racing for a seed, and the only adjacent
team it has the power to act against.

**A limitation to state plainly.** The brief asked for an arm that treats waiver priority
as a consumable resource, spent by a claim and restored by waiting. This harness does not
work that way: `priority()` recomputes the order from the standings every week, so a
claim costs nothing in future priority and there is nothing to time. Changing that would
mean changing the waiver rule for all twelve seats, which is a harness change, not a
policy. What is genuinely scarce here is the **move**: one add/drop a week, so a block
spends the week's only transaction and forgoes the consensus add. The `block_bar` arm
below is that scarcity rule, and it is labelled as such rather than as priority timing.

## The blocking decision

Before week w, at the seat's turn in the waiver order, in an arm with a block spec:

1. If the seat has any starting hole for week w (`league_backtest.holes` on its own roster
   with the streaming policy's known-unavailable flags), it runs the streaming policy
   unchanged. Its own lineup always comes first; blocking is only ever done with a spare
   move.
2. If there is no target rival, it runs the streaming policy unchanged (which, with no
   hole, is the consensus move).
3. Candidates are the **40 highest-value free agents** still unclaimed at the seat's turn,
   by the same `over_replacement` wire value the control uses for week w. (Forty is a
   bound on work, not a tuned parameter; the wire pool is in the hundreds and the players
   who matter to anyone's lineup are at the top of it.)
4. **Harm to the target** of candidate f: the target's rest-of-season starting-lineup
   value (`sim_trades.lineup_value` over weeks w..17, on consensus rest-of-season points
   per game, zero in a week the player is expected unavailable) after it adds f and cuts
   its own lowest-valued cuttable player, minus that value now. That cut is the drop its
   consensus policy would make, so the comparison is against what the target would
   actually do with him. This is the same rest-of-season value the trade arms use on both
   sides of a trade.
5. The block add A is the candidate with the largest harm. Blocking is attempted only if
   harm > 0, and in `block_bar` only if harm >= **BLOCK_MIN**.
6. The seat's drop D is chosen among its own cuttable players, excluding any whose removal
   would open a hole in the seat's own week-w lineup, as the one that costs the seat's own
   rest-of-season starting-lineup value the least. Own cost is that change (negative or
   zero). In `block_bar` the block is abandoned if the cost is worse than
   **-BLOCK_COST**.
7. If no legal drop survives 6, the seat runs the streaming policy unchanged.
8. Otherwise it plays (A, D) as its week's move. This may be a move consensus would not
   have made at all, and the add may be worth less than the drop by wire value, exactly as
   a hole-filling stream may be.

Blocking is attempted every week from 2 to 17, whenever the seat has no hole.

### The thresholds

`BLOCK_MIN` = **8.0** and `BLOCK_COST` = **4.0** rest-of-season starting-lineup points.
Both are set from the 2-league mechanics check's log of the unbarred `block` arm, which
records, for every block it made, the projected harm to the target and the projected cost
to the seat. Those two numbers are decision inputs available to the seat at the moment it
acts, and the check printed no outcome metric of any kind.

The 692 unbarred blocks in that check project a median harm of 6.3 points and quartiles of
4.6 and 11.8, so 8.0 keeps the larger 39% and spends the week's move only when the denial
is worth about half a point a week for the rest of the season. `BLOCK_COST` is a safety
rail rather than a binding rule: because the drop is chosen to be the cheapest legal one
and a 14-man roster nearly always holds a bench player worth nothing to the
rest-of-season lineup, the own cost of a block in the check was never negative (88% of
blocks cost exactly zero and the rest were positive, the added player improving the seat's
own lineup as well). It is preregistered anyway so that the arm is fully specified, and it
is stated here in advance that it is expected never to bind. Both values are frozen and
apply to both designs.

## The trade decision

Unchanged from `mutual_cap2` except for which accepted offer is taken. The seat still
searches every 1-for-1 and 2-for-1 with every rival in weeks 3-11, still executes only
offers the rival accepts on consensus value and that leave the rival's own projected
lineup no lower, still requires a 10-point projected gain and still stops at two trades.

- Rule **`mine`** (control): take the offer with the largest own gain, with
  `sim_tradecap`'s tie order. Identical to `mutual_cap2`.
- Rule **`rival`**: among accepted offers whose own gain is at least **0.9** times the
  best available own gain, take the one ranked first by, in order: the receiving team is
  not the target rival; the receiving team's projected lineup gain is smallest; the seat's
  own gain is largest. The 0.9 tolerance is `sim_trades.NEAR_BEST`, already used in this
  codebase for "equal value to me", and is not re-tuned here.

The rule expresses both halves of the idea at once. Among deals the seat values the same,
it prefers to give away a player the receiving rival will not actually start (its lineup
gain is near zero) and prefers not to arm the team it is racing.

## Arms

| arm | blocking | trade pick |
|---|---|---|
| `mutual_cap2` (control) | none | `mine` |
| `block` | every positive-harm block | `mine` |
| `block_bar` | harm >= BLOCK_MIN, cost <= BLOCK_COST | `mine` |
| `trade_rival` | none | `rival` |
| `block_bar_trade` | harm >= BLOCK_MIN, cost <= BLOCK_COST | `rival` |

**Which arm carries the verdict: `block_bar`, alone.** It is the disciplined form of the
question the test asks: spend the week's spare move on a rival only when the denial is
worth something and the cost to the seat is small. `block` is the unbarred version and
shows what happens with no discipline, `trade_rival` isolates the trade rule, and
`block_bar_trade` is both levers at once. Those three are labelled sensitivity, not
verdicts; judging one arm avoids picking the best of four afterwards.

## Scoring and decision rule

- Harness: `sim_blocking.py`, seasons 2021-25, 20 leagues x 12 seats x 20 schedules, both
  opponent designs (`--noise 1` and `--noise 0`), all five arms in one run per design.
- Primary: paired change in all-play win rate, each arm against `mutual_cap2`, same
  league, seat and schedules. Secondary: title, playoff, wins, points.
- **Decision rule, on `block_bar` only:** it passes only if, in **both** designs, its
  pooled all-play change against `mutual_cap2` is positive, positive in at least 4 of 5
  seasons, and its pooled title change is no worse than -1.0 point. The same rule is
  printed for every other arm as sensitivity.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `sim_tradecap.py`; every
  arm shares those draws and the same drafts.
- Bootstrap: `LB.paired`, as in `prereg_tradecap.md` (season means, 2,000 resamples with
  `numpy.random.default_rng(1)`, 5th/95th percentiles). The interval is reported; the rule
  decides.
- Run once per design, one after the other through the shared lock. Reduced (`--leagues`
  other than 20) runs write `_check` files and cannot overwrite a real result.

## Reported, not gated

- Blocking moves per seat-season, by arm and season, and the position blocked.
- Per block: the projected harm denied to the target, the projected cost to the seat,
  whether the blocked player was also the single best free agent by wire value (so the
  player the target's consensus policy would have claimed at its own turn), whether he
  would have been in the target's week-w starting lineup, and whether the block differs
  from the move consensus would have made in the same state.
- Trades: count, median projected own gain, median projected gain to the receiving rival,
  the share of executed trades made with the target rival, and the size of the near-best
  set the `rival` rule chooses from.
- The control's reproduction of `sim_tradecap.py`'s `mutual_cap2` rows.

## Known limits

- Blocking transfers rather than creates. The denied points go nowhere; they are simply
  not scored by the target. All-play compares the seat with all eleven rivals at once, so
  a block can only help through one of them, while any cost to the seat's own roster is
  paid against all eleven. The rule gates on all-play, so this arm is being asked to clear
  a bar shaped against it, and that is on purpose.
- The target rival is recomputed every week from the current standings, so early-season
  blocks can land on a team that is not a genuine rival by December. No arm tries to
  forecast the final standings.
- Blocking only ever denies one rival. A free agent kept off the target's roster remains
  available to the other ten, who may claim him next week, so the denial can be one week
  long. The diagnostics report whether the blocked player was the top free agent, which is
  the case where the denial is most likely to be real.
- The target evaluates the block the way the seat does, on the same consensus numbers. A
  real manager might not have wanted that player.
- Waiver priority is not consumable in this harness, as stated above, so the scarcity the
  bar rations is the weekly move, not the priority.
- No arm blocks a rival by trade (for instance by trading a player to the rival's rival).
  Only the choice among already-accepted offers changes.

## Result (added after the single run per design; the spec above is unchanged)

The control reproduces `sim_tradecap.py`'s `mutual_cap2` rows exactly in both designs
(1,200 rows each, every column, max difference 0.0).

**The verdict arm, `block_bar`, fails the rule in both designs, and fails it clearly.**
Changes against `mutual_cap2`, 90% season-cluster intervals:

| opponents | title odds | all-play | all-play by season 2021-25 | title by season | reg pts | blocks/season |
|---|---|---|---|---|---|---|
| noise 1 | -0.3 pp [-1.0, +0.5] | -0.6 pp [-1.1, -0.1] | -1.1 -0.7 +0.6 -0.8 -1.2 | -0.6 +0.1 +1.5 -1.6 -0.9 | -7 | 2.74 |
| noise 0 | -1.6 pp [-3.2, +0.0] | -1.7 pp [-2.6, -0.8] | -3.1 -1.1 +0.2 -2.9 -1.7 | +1.6 -3.0 -0.2 -4.8 -1.7 | -18 | 3.42 |

All-play is negative pooled and in four of five seasons in both designs, with an interval
clear of zero, so the rule is missed on two counts at once. Title odds do not rescue it:
they are negative in both designs, and against exact opponents the title loss (-1.6) also
breaches the -1.0 guard. Blocking was expected to transfer rather than create value, and
to show up in title odds sooner than in all-play; it shows up in neither.

**Sensitivity (not verdicts).** Every arm against `mutual_cap2`:

| arm | noise 1 all-play | seasons + | title | noise 0 all-play | seasons + | title | blocks | trades |
|---|---|---|---|---|---|---|---|---|
| block | -1.2 | 1 | -1.0 | -3.2 | 1 | -1.9 | 6.2 / 7.4 | 1.80 / 1.68 |
| **block_bar** | -0.6 | 1 | -0.3 | -1.7 | 1 | -1.6 | 2.7 / 3.4 | 1.86 / 1.78 |
| trade_rival | -0.2 | 2 | +0.7 | -0.3 | 1 | -0.6 | 0 | 1.90 / 1.78 |
| block_bar_trade | -0.4 | 2 | +0.5 | -3.0 | 1 | -1.3 | 2.7 / 3.6 | 1.86 / 1.80 |

By season, all-play: `block` n1 -2.6 -1.2 +0.5 -1.5 -1.2, n0 -6.9 -3.6 +1.1 -2.8 -4.1;
`trade_rival` n1 -0.1 -0.9 +0.1 +0.3 -0.3, n0 -1.6 -0.3 +2.8 -2.1 -0.6;
`block_bar_trade` n1 -0.8 -0.8 +0.8 +0.0 -1.2, n0 -5.2 -1.5 +0.4 -6.3 -2.2. No arm passes
in either design. Every arm's all-play is negative pooled in both designs, and 2023 is the
only season where any of them is reliably positive.

Reading the sweep: the damage scales with how much blocking an arm does. Unbarred `block`
makes 6.2 and 7.4 blocks a season and loses 1.2 and 3.2 points of all-play; the barred arm
makes 2.7 and 3.4 and loses 0.6 and 1.7. The bar works as intended (it removes the
cheapest blocks and raises mean projected harm denied from 10.0 to 16.8 under noise 1),
and it halves the loss, but it does not reach zero, which says the loss is proportional
to blocking itself rather than to bad block selection.

`trade_rival` is the only arm that is close to free. It loses 0.2 and 0.3 points of
all-play, inside both intervals, and gains 0.7 points of title odds with noisy opponents
while losing 0.6 with exact ones. It fails on seasons positive rather than on magnitude,
and is best read as: giving up the top tenth of a trade's own gain to hand the other side
less costs about what it looks like it should cost, and buys nothing reliable.

**Reported, not gated.**

- Blocks per seat-season: 2.74 (noise 1) and 3.42 (noise 0) in `block_bar`, run from week
  2 to week 17 at a mean week of 7.0 and 7.6; 98% are moves the consensus policy would not
  have made. Total adds a season rise from 11.8 to 13.3 (noise 1) and 12.8 to 14.2
  (noise 0), so the blocks are almost all weeks the control would have stood pat or made a
  marginal add.
- The blocks are real denials, not phantom ones. The blocked player projects to raise the
  target's rest-of-season starting lineup by a mean 16.8 (noise 1) and 19.3 (noise 0)
  points, and he would have been in the target's week-w starting lineup 42% and 52% of the
  time. But he was the single best free agent by wire value only 18% and 20% of the time,
  which is the case where the target loses him for good; in the other four fifths the
  target's own consensus policy was going to take someone else that week anyway, so the
  denial is mostly of a player the target would have gotten to later, if at all.
- The cost to the seat's own roster is, by projection, zero or positive: mean +0.9 and
  +1.1 rest-of-season lineup points, and `BLOCK_COST` never bound, exactly as
  preregistered. The realised cost is not zero: regular-season points fall by 7 and 18.
  The gap is the projection's blind spot. A block adds the player who fits the target's
  roster, which is a player the seat does not need, and drops the cheapest bench player,
  whose value the rest-of-season lineup measure prices at nothing because he is currently
  behind a starter. When a starter is later hurt or on bye, that bench player is the one
  streaming would have used. The barred arm blocks quarterbacks above all (1.7 and 1.8 a
  season, against 0.8 and 1.3 running backs), which is the clearest version of it: a
  second quarterback is worth nothing to the seat in a one-QB league and takes a roster
  spot all season.
- Trades barely move. `trade_rival` executes 1.90 and 1.78 a season against the control's
  1.91 and 1.83, at a near-identical median projected own gain (19.6 and 19.3 against 19.7
  and 18.5), and it chooses from a near-best set of only 3.3 and 3.4 offers. It does what
  it was built to do: the median projected gain to the receiving rival falls from 10.8 to
  6.2 (noise 1) and 14.1 to 6.4 (noise 0), and the share of trades made with the
  standings-closest rival falls from 6% to 3% and 12% to 5%.

Honest read: rival-aware management does not work here, and the reason is not that the
blocks miss. They land on the right players, they deny the target a starter about half the
time, and the bar sharpens them. The trouble is arithmetic that blocking cannot escape in
this harness. All-play scores the seat against all eleven rivals every week, so a point
denied to one of them is worth a eleventh of a point to the seat, while the roster spot
and the week's move are paid in full against all eleven. Head-to-head title odds should in
principle weight the target more heavily, and they do not here either: the target is
recomputed weekly from the current standings, so the denial is spread across whichever
team happened to be one seed ahead in October, which is often not the team the seat plays
in the playoffs. Against that, the seat gives up its own depth. The rest-of-season lineup
measure prices a healthy bench player at zero and so reports a block as free, and it is
not: the same bench player is what hole-aware streaming, the one policy that passed here,
exists to use. Blocking and streaming are competing for the same scarce thing, and
streaming is the better use of it.

Two limits on that conclusion. Blocking one rival at a time in a twelve-team league is
close to the worst case for the idea; in a four- or six-team playoff race late in the
season, or in a league where the wire is thin, the arithmetic could look different, and
nothing here tests that. And the harness's waiver order is recomputed from the standings
every week, so a claim costs no future priority. In a real league with consumable
priority, a block is strictly more expensive than it is here, which pushes in the same
direction as this result rather than against it.
