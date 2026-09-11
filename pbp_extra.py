"""Pull the two pbp-only QBERT inputs: pressure faced, and true Q2/Q3 game script."""
import io, requests, numpy as np, pandas as pd, pyarrow.parquet as pq

COLS = ["game_id", "season", "week", "posteam", "passer_player_id", "qb_hit", "sack",
        "qtr", "game_seconds_remaining", "posteam_score", "defteam_score", "pass", "qb_scramble"]
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"

out = []
for y in range(2006, 2026):
    raw = requests.get(URL.format(y), timeout=180).content
    t = pq.read_table(io.BytesIO(raw), columns=COLS).to_pandas()
    t = t[t.posteam.notna()]
    # Pressure faced: QB hits (sacks included) per dropback, by team-game.
    db = t[t["pass"] == 1]
    press = db.groupby(["season", "week", "posteam"]).agg(
        dropbacks=("pass", "sum"), qb_hits=("qb_hit", "sum")).reset_index()
    # Game script: net score at the end of Q2 and Q3, as Silver describes.
    halves = []
    for q in (2, 3):
        e = (t[t.qtr == q].sort_values("game_seconds_remaining")
             .groupby(["season", "week", "posteam"]).first().reset_index())
        e[f"net_q{q}"] = e.posteam_score - e.defteam_score
        halves.append(e[["season", "week", "posteam", f"net_q{q}"]])
    m = press.merge(halves[0], how="left").merge(halves[1], how="left")
    out.append(m)
    print(y, len(m), flush=True)

pd.concat(out, ignore_index=True).to_parquet("data/pbp_extra.parquet")

# Join the pbp-only features onto the game table the rest of the pipeline reads.
extra = pd.read_parquet("data/pbp_extra.parquet").rename(columns={"posteam": "team"})
full = pd.read_parquet("data/qb_games.parquet").merge(
    extra, on=["season", "week", "team"], how="left")
full.to_parquet("data/qb_games_full.parquet")
print(f"qb_games_full: {len(full)} rows, {(full.week > 18).sum()} playoff, "
      f"pressure coverage {full.dropbacks.notna().mean():.3f}")
