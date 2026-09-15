# Preregistration: planning the draft on a platform room's order

Frozen 2026-09-14, before the real (LEAGUES=20) run of these arms. A 2-league-per-season
mechanics check had already been run and is not part of the result; it printed no outcome
metrics and decided nothing. With weight 0 (plain consensus rooms) the `cons` arm
reproduced `run_streaming`'s `stream` rows exactly, all 120 league-seat-seasons, in both
designs. With weight 1 the test seat waited 0.72 (noise 1) and 1.17 (noise 0) times a draft,
mostly in rounds 6-13, giving up a median 4-6 consensus places at the pick, and 83-84% of
waited-on players ended up on its roster (the misses come from the room's roster rules and
noise, which the planner ignores). Sample waits looked sane (for example Cooper Kupp waited
on for CeeDee Lamb in a 2023 Sleeper round 1). Any later change to the arms below is a new
preregistration, not an edit to this one.

## Why this test

A 2026 r/fantasyfootball post, "Abusing Draft Rankings 2026 (ESPN, Sleeper, Yahoo, CBS)",
argues that drafters anchor on their platform's default list, so a player consensus likes
and the platform buries lasts longer in that platform's rooms, and you can wait on him.
`prereg_adp.md` failed, but it asked something else: whether a market board beats the
experts' board, against rooms that drafted by consensus. Here consensus still decides who
is good in both arms. The platform only predicts who will still be there, against rooms
that really do draft off the platform. Streaming (`prereg_streaming.md`) showed that
decisions made on consensus's own numbers can beat managers who follow consensus; this is
the same idea at the draft.

## Data

- `draft/platform_ranks.py` -> `data/platform_ranks.parquet`: ESPN and Sleeper PPR ADP,
  each as an overall rank, from FantasyPros' "ADP by site" page for 2021-25. The full
  tables are CSV exports of that page committed to GitHub by other people, pinned by
  commit. Each season's file reproduces, exactly, the five rows FantasyPros still serves
  anonymously for that year, and agrees with a second, independently downloaded export on
  every player in the top 180 at both sites. FantasyPros' Sleeper column matches Sleeper's
  own API season ADP (Spearman .999 to 1.000 over the top 150).
- Snapshot dates, as FantasyPros prints them: ESPN 2021-09-07, 2022-09-07, 2023-09-04,
  2024-09-03, 2025-09-02; Sleeper 2021-09-08, 2022-09-07, 2023-09-05, 2024-09-03,
  2025-09-02. Openers 2021-09-09, 2022-09-08, 2023-09-07, 2024-09-05, 2025-09-04. Every
  snapshot predates its season.
- One patch: the 2025 export omits Austin Ekeler (FantasyPros' current export drops
  players who have since retired). His row comes from an export of the same page
  committed 2025-08-30. One alias: the site's "William Fuller" is Will Fuller.
- Mapping to gsis ids through the dynastyprocess crosswalk as `adp.py` does: 100% of each
  site's top 200 skill players map in every season. ESPN ranks 187-426 skill players a
  season, Sleeper 241-295; 85-90% (ESPN) and 97-100% (Sleeper) of consensus's top 200
  carry a rank.
- **This is platform ADP, not the platform's default ranking.** It is where that
  platform's rooms actually took players, anchoring and all. No preseason copy of ESPN's,
  Yahoo's or Sleeper's default list for these seasons could be recovered: ESPN's API
  returns past-season ranks that were overwritten in-season, Yahoo appears only in
  exports of unknown date that match no reference, the Reddit post series could not be
  read, and the Internet Archive was offline. Yahoo is not in the test.

Descriptive, recorded before any arm was run. Over consensus's top 168 (the players a
14-round, 12-team draft takes), the platform's order differs from consensus's by 9-19
places on average (median 7-11), with 33-45% of players a full round (12 places) or more
apart; Spearman .95-.98. ESPN and Sleeper differ from each other by 8-9 places on average.

## Arms

Every seat, in both arms, sets lineups by weekly consensus. In season the test seat runs
hole-aware streaming (`streaming_policy`, as frozen in `prereg_streaming.md`) and the other
eleven seats run the consensus wire, exactly as in `run_streaming`. Only the test seat's
draft differs between arms. The opponents are identical in both arms.

- **Rooms.** League lg of each season is an ESPN room when lg is even and a Sleeper room
  when lg is odd, so each platform gets 10 of the 20 leagues. The platform's order is put
  on the consensus scale: the platform's k-th skill player (by site rank; players the site
  did not rank come after every ranked player, in consensus order; ties by consensus) gets
  the k-th smallest ECR mean of the season. Call that `key`. Opponent k's list is the
  harness's noisy-consensus list with `key` in place of the consensus mean:
  `score = key + NOISE x ECR sd x noise[k]`, sorted ascending, with the harness's own draws.
- **Weight.** The opponents' list is weight 1.0 on the platform order and 0 on consensus.
  This is the weight the data support, not a choice among blends: platform ADP is already
  the outcome of real drafters mixing their platform's list with whatever consensus they
  follow, so the room's expected order is the platform ADP order itself. A further blend
  with consensus would count consensus twice. Drafter-to-drafter deviation comes from the
  harness's noise, as it always has. No other weight is run as a result.
- **cons** (control): the test seat drafts by exact consensus (`S.exact_order`).
- **avail**: the test seat values players by exact consensus (ECR mean, lower is better)
  and plans with `key`. At a pick with `m` other picks before its next one:
  1. `b` = the allowed available player with the best consensus rank. At its last pick it
     takes `b`.
  2. The room is predicted to take the `m` available players with the lowest `key`,
     ignoring the room's roster rules. `s` = the best-consensus player, other than `b`,
     not predicted taken when the seat takes `b` now, and allowed at its next pick with `b`
     on the roster (none if there is no such player).
  3. Going through allowed players in consensus order, the first `u` other than `b` with
     consensus rank better than `s` (any `u`, if `s` is none) such that: the room is
     predicted to take `u` if the seat takes `b`; the room is predicted to leave `b` if the
     seat takes `u`; and `b` is allowed at the next pick with `u` on the roster. The seat
     takes that `u`, planning to take `b` next.
  4. If no such `u`, it takes `b`.
  The comparison is between two plans that both end with `b`: {`u`, `b`} against {`b`,
  `s`}. It has no parameter. It looks one pick ahead only, and it does not re-plan around
  the waited-on player afterwards: at the next pick the same rule runs from scratch.
- **Mechanics identity.** With weight 0 the rooms are the plain consensus rooms, and
  `cons` must reproduce `run_streaming`'s `stream` arm exactly, row for row.

## Information timing and leakage control

- Platform ADP is each site's last pre-season snapshot (dates above), the same timing as
  the preseason consensus the harness drafts from. Neither signal has seen a down.
- The test seat knows `key` exactly, but not the opponents' noise, their roster rules or
  their actual lists. The opponents know nothing about the test seat.
- Nothing is fitted in the draft policy. The in-season values are `run_streaming`'s:
  rest-of-season consensus scrapes strictly before each week, through a curve fit on
  seasons before y.
- Known residual: the rooms draft off the realized ADP of the season being scored. That is
  pre-season information, not an outcome, but it means the test seat's model of the room
  is as good as it could possibly be. The test is a best case for the idea.

## Scoring and decision rule

- Harness: `draft/sim_platformranks.py`, seasons 2021-25, 20 leagues x 12 seats x 20
  schedules, both designs (`--noise 1` and `--noise 0`), weight 1.0.
- Primary: paired change in all-play win rate, `avail` against `cons`, same league, seat,
  rooms and schedules. Headline: title odds. Secondary: playoff, wins, points.
- Kept only if, in **both** designs, the pooled all-play change is positive, positive in
  at least 4 of 5 seasons, and the pooled title change is no worse than -1.0 point.
- Seeds: league lg of season y uses `numpy.random.default_rng([y, lg])`, drawing the
  thirteen noise vectors then the twenty schedules, exactly as `run_streaming`; both arms
  share those draws.
- Bootstrap: paired differences averaged within each season to five season means,
  resampled with replacement 2,000 times with `numpy.random.default_rng(1)`, 5th/95th
  percentiles (`numpy.quantile`, linear). The interval is reported; the rule decides.
- Run once per design. Reduced or non-1.0-weight runs write to `_check` files and cannot
  overwrite `data/league_platformranks_noise{1,0}.parquet`.

## Reported, not gated

- The same paired change within ESPN rooms and within Sleeper rooms.
- Waits per draft, by season and round; the share of waited-on players that end up on the
  test seat's roster; the consensus places given up at the pick (`u` minus `b`).

## Known limits

- Platform ADP stands in for the platform's default list (see Data). If real rooms anchor
  on the default list, ADP already shows the result; what it cannot show is a room whose
  drafters differ in how much they anchor.
- Every opponent in a room anchors on the same platform with the same weight; real rooms
  mix drafters who import rankings, use other sites, or draft by feel.
- One snapshot per site per season; real drafts happen over weeks at moving prices.
- Opponent noise is the experts' spread, which was built for consensus drafters.
- The planner ignores opponents' roster rules and looks one pick ahead.

## Result (added after the single run per design; the spec above is unchanged)

The availability planner fails the rule in both designs, so it is not kept. Changes against
the exact-consensus draft in the same platform-anchored rooms, both arms streaming, 90%
season-cluster intervals:

| comparison | opponents | title odds | all-play | all-play by season 2021-25 | reg pts |
|---|---|---|---|---|---|
| avail minus cons | noise 1 | -0.3 pp [-1.2, +0.5] | -0.0 pp [-0.2, +0.2] | +0.3 -0.4 +0.1 -0.1 +0.1 | +0 |
| avail minus cons | noise 0 | +0.3 pp [-0.9, +1.4] | -0.2 pp [-2.1, +1.8] | +0.0 -0.4 +4.5 -3.7 -1.5 | +0 |

Title odds by season: noise 1 -0.2 -2.1 -1.0 +0.1 +1.5; noise 0 -1.5 +1.0 +0.8 -1.2 +2.5.
Noise 1 is positive in 3 of 5 seasons with a pooled change of zero. Noise 0 is positive in
1 of 5, and its pooled all-play change is negative.

Reported, not gated. ESPN rooms only: noise 1 title -0.3, all-play -0.0; noise 0 title
+0.8, all-play -1.2 [-2.7, -0.0]. Sleeper rooms only: noise 1 -0.4 and +0.0; noise 0 -0.2
and +0.8. The seat waits 0.40-1.11 times a draft by season under noise 1 (800 waits in all)
and 0.58-1.88 under noise 0 (1,400). 80.5% and 83.6% of the players waited on end up on
its roster. It gives up 5.4 and 8.3 consensus places on average at the pick where it waits.
Waits cluster in rounds 6-13. Under noise 1, 68% of league-seats finish with outcomes
identical to the control's.

A reading, not part of the rule. The mechanism works as the post describes: the room
does leave the players the plan expects it to leave. It just isn't worth anything. The
players you take early are only a few consensus places worse than the ones you waited
on, and the rest of the draft reshuffles around them, so the rosters come out almost the
same. Where there is any sign, it's noise: the noise-0 season means swing from +4.5 to
-3.7 because those drafts are deterministic, so one different pick moves every league the
same way. This setup is a best case for the idea, since the seat knows each room's exact
expected order from the realized ADP of the season being scored. A real drafter would
predict less well.

A side observation from the control arm, not tested: with exact opponents drafting
platform ADP, the exact-consensus seat had all-play of 39-53% by season (46.0% pooled) and
5.5% title odds, below a 1-in-12 share even with streaming. Under noise 1 it was 53.9% and
14.4%. That doesn't square with `prereg_adp.md` (ADP forecasts slightly worse than
consensus) unless roster rules interact with eleven identical lists. It wasn't
investigated, and it affects both arms equally.
