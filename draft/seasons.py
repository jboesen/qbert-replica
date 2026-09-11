"""Season-level fantasy table for the skill positions, with age and games played."""
import glob
import numpy as np
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]


def build():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/stats/w*.parquet"))])
    d = d[(d.season_type == "REG") & (d.position.isin(POSITIONS))]

    s = d.groupby(["player_id", "player_display_name", "position", "season"]).agg(
        games=("week", "count"),
        ppr=("fantasy_points_ppr", "sum"),
        std=("fantasy_points", "sum"),
        # Volume drives fantasy scoring and is stickier than efficiency.
        targets=("targets", "sum"), carries=("carries", "sum"),
        attempts=("attempts", "sum"), rush_yards=("rushing_yards", "sum"),
        rec_yards=("receiving_yards", "sum"), pass_yards=("passing_yards", "sum"),
        tds=("rushing_tds", "sum"), rec_tds=("receiving_tds", "sum"),
        pass_tds=("passing_tds", "sum"), target_share=("target_share", "mean"),
    ).reset_index()
    s["touches"] = s.targets.fillna(0) + s.carries.fillna(0) + s.attempts.fillna(0)
    s["ppr_pg"] = s.ppr / s.games
    s["touch_pg"] = s.touches / s.games

    players = pd.read_parquet("data/players.parquet")[
        ["gsis_id", "birth_date", "draft_pick", "rookie_season"]].rename(
        columns={"gsis_id": "player_id"})
    s = s.merge(players, on="player_id", how="left")
    s["birth_date"] = pd.to_datetime(s.birth_date, errors="coerce")
    s["age"] = s.season - s.birth_date.dt.year - 0.3
    s["exp"] = s.season - s.rookie_season.fillna(s.season)
    s["draft_pick"] = s.draft_pick.fillna(260)

    # A 17th regular-season game arrived in 2021; put every season on a 17-game basis.
    s["season_games"] = np.where(s.season >= 2021, 17, 16)
    return s.sort_values(["player_id", "season"])


if __name__ == "__main__":
    s = build()
    s.to_parquet("data/draft_seasons.parquet")
    print(f"{len(s)} player-seasons, {s.season.min()}-{s.season.max()}")
    print(s.groupby("position").agg(n=("ppr", "size"), ppg=("ppr_pg", "mean")).round(2))
