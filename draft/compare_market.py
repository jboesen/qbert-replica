"""Compare a market board (site projections, in rank order) against the model's board.

The market's rank order is what the room will draft by, so the difference between it and
the model's own valuation is the only thing that can produce an edge: players the model
likes far more than the room are the ones worth waiting on or reaching for.
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
from vbd import LEAGUE, add_vbd

SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$", re.I)


def norm(s):
    s = s.str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False)
    return s.str.replace(SUFFIX, "", regex=True).str.strip()


def load_market(path="data/market/espn_2026.csv"):
    m = pd.read_csv(path)
    m["market_ppr"] = (m.pass_yds * .04 + m.pass_td * 4 - m.pass_int * 2
                       + m.rush_yds * .1 + m.rush_td * 6
                       + m.rec + m.rec_yds * .1 + m.rec_td * 6)
    m["key"] = norm(m.player)
    return m


def main():
    mk = load_market()
    mine = pd.read_parquet("data/board_2026.parquet").reset_index()
    mine["key"] = norm(mine.player_display_name)

    j = mk.merge(mine[["key", "player_display_name", "position", "age", "proj_games",
                       "proj_points", "vbd"]], on="key", how="left")

    missing = j[j.proj_points.isna()]
    print(f"market board: {len(mk)} players;  model can project {len(j) - len(missing)}")
    print(f"no model projection ({len(missing)}) - no NFL history to pool from:")
    for r in missing.itertuples():
        print(f"   #{r.rank:<4} {r.player} ({r.pos}, {r.team})")

    ok = j[j.proj_points.notna()].copy()

    # Price both sets of projections the same way, so the comparison is like for like.
    mk_board = ok.rename(columns={"market_ppr": "proj_points_market"})
    tmp = ok.assign(proj_points=ok.market_ppr, position=ok.pos)
    market_vbd, _, _ = add_vbd(tmp.set_index("key"))
    ok["market_vbd"] = ok.key.map(market_vbd.vbd)

    ok["market_rank"] = ok["rank"].rank()
    ok["model_rank"] = ok.vbd.rank(ascending=False)
    ok["gap"] = ok.market_rank - ok.model_rank          # positive = model likes him more
    ok["pts_gap"] = ok.proj_points - ok.market_ppr

    cols = ["rank", "player", "pos", "age", "proj_games", "market_ppr", "proj_points",
            "pts_gap", "model_rank", "gap"]
    fmt = lambda v: f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"

    print("\n=== Model likes these far more than the board does (wait on them) ===")
    print(ok.nlargest(15, "gap")[cols].to_string(index=False, float_format=fmt))

    print("\n=== Board likes these far more than the model does (let someone else) ===")
    print(ok.nsmallest(15, "gap")[cols].to_string(index=False, float_format=fmt))

    ok.to_parquet("data/market_compare_2026.parquet")
    print(f"\nrank correlation between the two boards: "
          f"{ok.market_rank.corr(ok.model_rank, method='spearman'):.3f}")
    print(f"model projects fewer points on average by "
          f"{-ok.pts_gap.mean():.0f} (it prices injury risk into games played)")


if __name__ == "__main__":
    main()
