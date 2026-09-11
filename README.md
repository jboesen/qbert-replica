# QBERT-replica

A local reimplementation of Silver Bulletin's **QBERT** quarterback rating, validated
against the QBERT numbers that have been published, with a fantasy-points projection
on top.

The real QBERT is closed: no code, no weights, no data export, and the ratings tables
are paywalled. But the [methodology post](https://www.natesilver.net/p/best-quarterbacks-of-all-time-qbert-elway)
describes the construction in enough detail to rebuild it from public data, and enough
individual ratings have been published in free posts to check the rebuild against.

## How it works

By Silver's own description, QBERT is a regression that reproduces **ESPN's QBR** from
statistics available much further back than QBR itself — QBR starts in 2006, QBERT runs
to 1950. So:

1. **Raw rating** — regress game-by-game QBR on per-play box-score rates: completions,
   TDs, interceptions, fumbles, completed air yards (YAC credited to receivers), net QB
   rush yards, first downs, carries, pressure faced, RB yards per carry, game script.
   Fit on the log-odds of QBR, since QBR is bounded 0–100; weighted by plays.
2. **Adjustments** — opponent-defense rolling rating (read pre-game, so no leakage),
   home field, cold and wind for outdoor games, per-season recentering.
3. **Scale** — 80 = league average by construction. The *spread* is set by the one
   season rating Silver has published outright, Rodgers' 2011 at 108.6.
4. **WAR** — quarterbacks take ~30% of wins above a replacement-level baseline.
5. **Projection** — rolling play-weighted form (10-game half-life), aging curve peaking
   at 28–29, experience term over a QB's first ~20 starts.
6. **Fantasy** — projected rating, expected volume and rushing share to a projected
   fantasy-point total for the next start.

## Does it track QBERT?

One published number is used to build the model (Rodgers 2011). Everything else is a
test. `validate.py` runs them all:

| published QBERT claim | value | replica |
|---|---|---|
| replacement level (undrafted QB's first start) | 68 | **65.8** |
| league average | 80 | 80.0 (by construction) |
| Brady 2007 = top season by WAR, all time | #1 | **#2** in 2006–25 |
| Lamar Jackson 2024 = 10th all time by WAR | #10 | **#5** in 2006–25 |
| preseason 2025 projections (Lamar / Allen / Mahomes) | 104.5 / 100.9 / 94.4 | 98.6 / 97.3 / 90.7 — **order matches**, leave-one-out error 1.5 |
| Super Bowl LX (Darnold / Maye) | 82.3 / 75.9 | 90.2 / 68.8 — order matches |
| career WAR (Brady / Manning / Brees) | 114.3 / 94.8 / 86.5 | 73.2 / 41.3 / 55.4 over 2006+ only |

Replacement level landing at 65.8 is the strongest single result — the published anchor
is 68 and nothing in the build targets it.

The Super Bowl LX rows are worth a look. ESPN's raw QBR had Maye at **16.3** and Darnold
at **53.0**; QBERT published 75.9 and 82.3. The replica gives 68.8 and 90.2. All three
systems agree the game was far less lopsided than raw QBR says, which is evidence the
box-score reconstruction behaves the way QBERT's does rather than merely tracking QBR.

The one systematic miss is projections, uniformly ~4 points low: the replica regresses
toward average slightly harder than QBERT does. A constant offset is all three anchors
can support — fitting a slope too fails leave-one-out (10.4 points of error on the
held-out anchor), so the code applies the offset only.

## Where the remaining gap comes from

`benchmark.py`, holdout 2021–25, play-weighted R² against ESPN QBR:

| model | R² |
|---|---|
| published QBERT (Silver's reported figure) | 0.750 |
| replica, WLS on logit(QBR) | 0.677 |
| ridge regression | 0.671 |
| gradient boosting | 0.662 |
| random forest | 0.661 |
| linear + gradient boosting blend | 0.682 |

**No open-source ML model beats the linear fit.** The shortfall is a data problem, not a
model-class problem. Which the next test confirms: QBERT deliberately restricts itself to
statistics available back to 1950, and a modern-era build has no such constraint —
EPA exists from 1999.

| model, modern-era inputs allowed | R² |
|---|---|
| linear + EPA + clutch flags | 0.727 |
| gradient boosting + EPA | 0.720 |
| blend | **0.734** |

That essentially closes the gap to Silver's reported 0.75. Run `run.py --modern` for
this variant. It is a better retrodictive rating and, notably, **not** a better fantasy
projection — see below.

## Independent check on WAR

`war_compare.py` builds a second QB WAR from scratch following the
[nflWAR](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2018-0010/html)
approach (Yurko, Ventura & Horowitz, *JQAS* 2019) — expected points above a
replacement-level baseline, divided by points per win — using EPA, so it shares no
inputs with the QBR reconstruction. Agreement across 660 qualified seasons:
**Spearman 0.942, Pearson 0.947.**

## Fantasy projection

`benchmark.py`, projecting next-start QB fantasy points, holdout 2021–25, 2,825 starts
(actual points sd = 7.83):

| model | RMSE | MAE | corr |
|---|---|---|---|
| QBERT regression | **7.23** | **5.80** | **0.389** |
| random forest | 7.27 | 5.83 | 0.373 |
| gradient boosting | 7.30 | 5.86 | 0.365 |
| rolling fantasy PPG | 7.54 | 6.05 | 0.341 |
| season-to-date mean | 7.89 | 6.27 | 0.000 |

Three things stand out. The rating-based projection beats the standard rolling-average
baseline on every metric. Tree models lose to the linear one here too. And the modern
EPA variant, despite reconstructing QBR far better, projects fantasy points no better at
all (RMSE 7.25 vs 7.24) — week-to-week QB fantasy scoring is mostly volume and noise,
and rating precision washes out through the rolling average. Only about 15% of the
variance in a QB's next-week fantasy total is predictable at all.

## Related work

- Yurko, Ventura & Horowitz, [nflWAR: a reproducible method for offensive player
  evaluation in football](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2018-0010/html),
  *JQAS* 2019 — the canonical open-source WAR framework; multilevel models over
  multinomial-logistic expected points, with resampling-based uncertainty. Used above
  as the independent check.
- Wolfson, Addona & Schmicker, [The Quarterback Prediction
  Problem](https://www.degruyterbrill.com/document/doi/10.2202/1559-0410.1302/html),
  *JQAS* 2011 — forecasting NFL QB success from college and combine data; the source of
  the draft-position term in the projection model.
- [Quarterback evaluation using tracking
  data](https://link.springer.com/article/10.1007/s10182-021-00406-8), *AStA* 2021 —
  what becomes available with player-tracking data, which the public feed lacks.
- [Moving from Machine Learning to Statistics: the case of Expected Points in American
  football](https://arxiv.org/pdf/2409.04889) — argues for statistical over ML framing
  of EPA; consistent with the benchmark result that boosting does not beat regression.
- Egidi & Gabry, [Bayesian hierarchical models for predicting individual performance in
  football](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1486928/full)
  — hierarchical pooling for fantasy scoring, the natural next step for per-player
  uncertainty here.
- [nflfastR](https://nflfastr.com/) / [nflverse](https://github.com/nflverse/nflverse-data)
  — the open play-by-play, EPA and CPOE pipeline everything here is built on.

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas numpy scikit-learn statsmodels pyarrow requests scipy
.venv/bin/python build_data.py      # weekly QB stats + playoffs, joined to ESPN QBR
.venv/bin/python pbp_extra.py       # pressure + game script from play-by-play
.venv/bin/python run.py             # ratings, WAR, leaderboards  (--modern for the EPA variant)
.venv/bin/python fantasy.py         # projections
.venv/bin/python fantasy_proj.py    # fantasy points
.venv/bin/python validate.py        # check against published QBERT numbers
.venv/bin/python benchmark.py       # model comparison
.venv/bin/python war_compare.py     # independent WAR cross-check
.venv/bin/python update.py          # in season: refresh data, rebuild weekly + rest-of-season
```

`anchors.py` holds every published QBERT figure found, with its source.

Coverage is 2006–2025, bounded by the QBR training data; 1999 is reachable from the same
play-by-play. Data comes from [nflverse](https://github.com/nflverse/nflverse-data)
releases. Nothing here is scraped from behind Silver Bulletin's paywall.

## Draft tool

See [`draft/README.md`](draft/README.md) — season-long projections for QB/RB/WR/TE,
value-based draft board, a backtested pick recommender, and three replicated papers.
