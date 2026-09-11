"""Project the upcoming season and print a draft board.

Every player with NFL history gets a row for the target season - one year older, one
year more experienced - and the projection model is refit on everything through the
previous season. Rookies are absent by construction: they have no history to pool.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import project_season as P
from vbd import LEAGUE, add_vbd


def project_upcoming(target):
    s = P.build()
    prior = s[s.season < target]
    if not len(prior[prior.season == target - 1]):
        raise SystemExit(f"no {target - 1} data; cannot project {target}")

    # Carry each active player forward a year.
    recent = prior[prior.season >= target - 2]
    last = recent.sort_values("season").groupby("player_id").last().reset_index()
    nxt = last.copy()
    nxt["season"] = target
    nxt["age"] = nxt.age + (target - last.season)
    nxt["exp"] = nxt.exp + (target - last.season)
    nxt["games"] = np.nan
    nxt["ppr"] = np.nan
    nxt["ppr_pg"] = np.nan
    nxt["season_games"] = 17
    nxt["prev_games"] = last.games

    both = pd.concat([prior, nxt], ignore_index=True)
    hist = P.player_history(both)
    both = both.drop(columns=[c for c in ("own_ppg", "own_w", "own_touch") if c in both])
    both = both.merge(hist, on=["player_id", "season"], how="left")
    both["prev_games"] = both.groupby("player_id").games.shift(1).fillna(both.prev_games)

    out, _ = P.fit_predict(both, both.season < target)
    return out[out.season == target]


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    proj = project_upcoming(target)
    proj = proj[proj.own_w > 4]                      # needs some history to project
    board, levels, need = add_vbd(proj.set_index("player_id"))
    board.to_parquet(f"data/board_{target}.parquet")

    print(f"=== {target} draft board, 12-team PPR ===")
    print("replacement level:", {k: round(v) for k, v in levels.items()})
    print("(returning players only - rookies have no history to project)\n")
    show = board.head(40)[["player_display_name", "position", "age",
                           "proj_games", "proj_points", "vbd"]].reset_index(drop=True)
    show.index += 1
    print(show.to_string(float_format=lambda v: f"{v:.1f}"))
