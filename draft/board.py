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


def rookie_rows(target, known):
    """This year's rookies at the four positions, as rows with no history: the
    projection falls back to the draft-pedigree prior, re-weighted by depth role."""
    p = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id")
    p = p[(p.rookie_season == target) & p.position.isin(P.POSITIONS)
          & ~p.gsis_id.isin(known)]
    born = pd.to_datetime(p.birth_date, errors="coerce").dt.year
    return pd.DataFrame({
        "player_id": p.gsis_id, "player_display_name": p.display_name,
        "position": p.position, "season": target, "age": target - born - 0.3,
        "exp": 0.0, "draft_pick": p.draft_pick.fillna(260), "rookie_season": target,
        "season_games": 17,
    })


def project_upcoming(target, rookies=False):
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
    if rookies:
        nxt = pd.concat([nxt, rookie_rows(target, set(s.player_id))], ignore_index=True)

    both = pd.concat([prior, nxt], ignore_index=True)
    hist = P.player_history(both)
    both = both.drop(columns=[c for c in ("own_ppg", "own_w", "own_touch") if c in both])
    both = both.merge(hist, on=["player_id", "season"], how="left")
    both["prev_games"] = both.groupby("player_id").games.shift(1).fillna(both.prev_games)

    out, _ = P.fit_predict(both, both.season < target)
    return out[out.season == target]


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    proj = project_upcoming(target, rookies=True)
    # Returning players need some history to project; rookies need a spot on the chart,
    # since the role fit only ever saw rookies who got on the field.
    proj = proj[(proj.own_w > 4) | ((proj.exp == 0) & proj.role.isin(["d1", "d2"]))]
    board, levels, need = add_vbd(proj.set_index("player_id"))
    board.to_parquet(f"data/board_{target}.parquet")

    print(f"=== {target} draft board, 12-team PPR ===")
    print("replacement level:", {k: round(v) for k, v in levels.items()})
    print(f"({(proj.exp == 0).sum()} rookies included, projected from draft slot and "
          f"depth-chart role)\n")
    show = board.head(40)[["player_display_name", "position", "age",
                           "proj_games", "proj_points", "vbd"]].reset_index(drop=True)
    show.index += 1
    print(show.to_string(float_format=lambda v: f"{v:.1f}"))
