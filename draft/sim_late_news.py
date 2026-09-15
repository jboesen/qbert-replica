"""Does reacting to the 90-minute inactive list beat stale consensus lineups?

The baseline follows the historical weekly consensus ranking and may start someone
who was later declared inactive.  The test seat receives only the official inactive
list immediately before each kickoff, then fills its normal lineup from the same
rankings and roster.  No projections, waivers, or opponent choices change.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
import league_backtest as LB

STREAMING = "--streaming" in sys.argv


def run():
    curves, wk_proj = LB.rank_curve(), LB.weekly_projections()
    rows = []
    for year in LB.SEASONS:
        season = LB.Season(year, wk_proj, curves)
        if STREAMING:
            curve = C.weekly_curve(range(2020, year))
            values, _ = LB.waiver_values(season, year, curve, None)
        else:
            values = None
        for league in range(LB.LEAGUES):
            rng = np.random.default_rng([year, league])
            noise = rng.standard_normal((LB.TEAMS + 1, len(season.ids)))
            orders = []
            for draw in noise[:LB.TEAMS]:
                score = season.ecr_mean + LB.NOISE * season.ecr_sd * draw
                available = np.where(np.isfinite(score))[0]
                orders.append(list(available[np.argsort(score[available])]))
            schedules = LB.schedules(rng, LB.SCHEDULES)
            for seat in range(LB.TEAMS):
                draft_orders = list(orders)
                draft_orders[seat] = season.exact_order
                drafted = LB.draft(season, draft_orders)
                scores = {}
                if STREAMING:
                    gone = LB.known_unavailable(season, year)
                    arms = (("stream", frozenset()), ("stream_late_news", frozenset({seat})))
                else:
                    arms = (("consensus", frozenset()), ("late_news", frozenset({seat})))
                for arm, late in arms:
                    movers = ({seat: LB.streaming_policy(season, gone, [])} if STREAMING else None)
                    score, _, _ = LB.simulate(season, drafted,
                                               [values] * LB.TEAMS if STREAMING else [None] * LB.TEAMS,
                                               set(range(LB.TEAMS)) if STREAMING else set(), movers,
                                               late_seats=late)
                    scores[arm] = score[seat]
                    result = [LB.season_outcome(score, s) for s in schedules]
                    rows.append({"season": year, "league": league, "seat": seat, "arm": arm,
                                 "title": np.mean([r[2][seat] for r in result]),
                                 "playoff": np.mean([r[1][seat] for r in result]),
                                 "wins": np.mean([r[0][seat] for r in result]),
                                 "all_play": LB.all_play(score, seat),
                                 "pts_reg": score[seat, :LB.REG].sum(),
                                 "pts_playoff": score[seat, LB.REG:].sum()})
            print(f"{year} league {league + 1}/{LB.LEAGUES}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if "--noise" in sys.argv:
        LB.NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:
        LB.LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    check = "" if LB.LEAGUES == 20 else f"_check{LB.LEAGUES}"
    result = run()
    name = "late_news_stream" if STREAMING else "late_news"
    path = f"data/league_{name}{check}_noise{LB.NOISE:g}.parquet"
    result.to_parquet(path)
    if check:
        print(f"wrote {path}")
        raise SystemExit()
    test, control = ("stream_late_news", "stream") if STREAMING else ("late_news", "consensus")
    print("\nlate inactive news on top of streaming" if STREAMING else
          "\nlate inactive news versus stale weekly consensus")
    LB.paired(result, test, control, f"{test} minus {control}")
