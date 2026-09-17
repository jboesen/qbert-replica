"""Value-based drafting: convert projected points into draft value.

A projection alone cannot rank across positions - 300 points from a quarterback and 300
from a running back are worth very different amounts, because the quarterback you could
have had instead scores far more than the running back you could have had instead. VBD
prices every player against the replacement he displaces at his own position.
"""
import numpy as np
import pandas as pd

import settings as CFG

# The league's shape, from the one settings object every tool reads. Defaults are the
# 12-team PPR league with one flex that every published result was run on.
LEAGUE = CFG.get().league_dict()


def replacement_levels(proj, league=LEAGUE):
    """Where the waiver wire starts, per position.

    Counting only nominal starters understates demand at the flex positions, so flex
    slots are allocated across the eligible positions in proportion to how often each
    actually fills one.
    """
    n = league["teams"]
    need = {p: n * c for p, c in league["starters"].items()}
    flex_total = n * league["flex_slots"]

    # Allocate flex by which positions supply the best players just past the starters.
    pool = []
    for p in league["flex"]:
        extra = proj[proj.position == p].nlargest(int(need[p]) + flex_total, "proj_points")
        pool.append(extra.iloc[int(need[p]):][["position", "proj_points"]])
    flex_fill = pd.concat(pool).nlargest(flex_total, "proj_points").position.value_counts()
    for p, c in flex_fill.items():
        need[p] += c

    levels = {}
    for p, c in need.items():
        ranked = proj[proj.position == p].nlargest(int(c) + 1, "proj_points")
        levels[p] = ranked.proj_points.iloc[-1]
    return levels, need


def add_vbd(proj, league=LEAGUE):
    levels, need = replacement_levels(proj, league)
    proj = proj.copy()
    proj["replacement"] = proj.position.map(levels)
    proj["vbd"] = proj.proj_points - proj.replacement
    return proj.sort_values("vbd", ascending=False), levels, need


if __name__ == "__main__":
    s = pd.read_parquet("data/draft_projected.parquet")
    y = s[(s.season == 2025) & s.proj_points.notna()]
    board, levels, need = add_vbd(y)
    print("replacement level (projected season points):",
          {k: round(v, 1) for k, v in levels.items()})
    print("starter demand incl. flex:", {k: int(v) for k, v in need.items()})
    print("\n=== 2025 draft board, top 20 by VBD ===")
    out = board.head(20)[["player_display_name", "position", "age", "proj_points", "vbd"]]
    out = out.reset_index(drop=True); out.index += 1
    print(out.to_string(float_format=lambda v: f"{v:.1f}"))
