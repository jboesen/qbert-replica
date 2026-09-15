"""Turn paired factor experiments into the usable fantasy-policy shortlist.

An experiment is *kept* only when it beats its declared control in both realistic
and exact-consensus opponent designs: positive pooled all-play, positive all-play in
at least four of five seasons, and title odds no worse than one percentage point down.
Anything failing either available design is rejected immediately.  This makes the
portfolio selection reproducible instead of depending on a prose recap.

    .venv/bin/python draft/factor_portfolio.py
"""
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent
OUT = ROOT / "data" / "factor_portfolio.json"
RULE = {
    "pooled_all_play_gt": 0.0,
    "positive_seasons_at_least": 4,
    "title_floor": -0.01,
}

# The control is the best policy available when a factor was tested, not necessarily
# raw consensus.  Paths deliberately point at the separate worktrees that produced the
# run artifacts, preserving the original full-result files.
CANDIDATES = {
    "hole_aware_streaming": {
        "description": "Use the waiver wire to fill a known upcoming starting-lineup hole.",
        "control": "cons", "arm": "stream",
        "files": {"realistic": CODE / "qbert-replica/data/league_streaming_noise1.parquet",
                  "exact": CODE / "qbert-replica/data/league_streaming_noise0.parquet"},
    },
    "late_inactive_news": {
        "description": "Replace a player only when the official late inactive list confirms he is out.",
        "control": "stream", "arm": "stream_late_news",
        "files": {"realistic": ROOT / "data/league_late_news_stream_noise1.parquet",
                  "exact": ROOT / "data/league_late_news_stream_noise0.parquet"},
    },
    "kdst_rotation": {
        "description": "Cycle handcuff RBs through kickoff windows after streaming.",
        "control": "control", "arm": "rotate",
        "files": {"realistic": CODE / "qbert-replica-kdst/data/league_kdst_noise1.parquet",
                  "exact": CODE / "qbert-replica-kdst/data/league_kdst_noise0.parquet"},
    },
    "early_rb_risk_discount": {
        "description": "Discount RB consensus value by 15% in the first six rounds.",
        "control": "exact", "arm": "d15",
        "files": {"realistic": CODE / "qbert-replica-rbrisk/data/league_rbrisk_noise1.parquet",
                  "exact": CODE / "qbert-replica-rbrisk/data/league_rbrisk_noise0.parquet"},
    },
    "bye_week_clustering": {
        "description": "Break close draft ties toward projected starters sharing a bye week.",
        "control": "exact", "arm": "cluster",
        "files": {"realistic": CODE / "qbert-replica-byecluster/data/league_byecluster_noise1.parquet",
                  "exact": CODE / "qbert-replica-byecluster/data/league_byecluster_noise0.parquet"},
    },
    "late_round_expert_disagreement": {
        "description": "Use the most optimistic expert rank to break late-round consensus ties.",
        "control": "exact", "arm": "upside",
        "files": {"realistic": CODE / "qbert-replica-lateupside/data/league_lateupside_noise1.parquet",
                  "exact": CODE / "qbert-replica-lateupside/data/league_lateupside_noise0.parquet"},
    },
}


def score(path, arm, control):
    """Read one paired run and apply the fixed portfolio rule."""
    if not path.exists():
        return {"state": "missing", "file": str(path)}
    d = pd.read_parquet(path)
    key = ["season", "league", "seat"]
    a = d[d.arm == arm].set_index(key)
    b = d[d.arm == control].set_index(key)
    if len(a) == 0 or len(b) == 0:
        return {"state": "invalid", "file": str(path), "arms": sorted(d.arm.unique().tolist())}
    diff = a[["all_play", "title"]] - b.loc[a.index, ["all_play", "title"]]
    season = diff.groupby("season").all_play.mean()
    all_play, title = float(diff.all_play.mean()), float(diff.title.mean())
    passed = (all_play > RULE["pooled_all_play_gt"]
              and int((season > 0).sum()) >= RULE["positive_seasons_at_least"]
              and title >= RULE["title_floor"])
    return {
        "state": "pass" if passed else "fail", "file": str(path),
        "all_play_pp": round(100 * all_play, 3), "title_pp": round(100 * title, 3),
        "all_play_by_season_pp": {str(k): round(100 * float(v), 3) for k, v in season.items()},
        "positive_seasons": int((season > 0).sum()),
    }


def main():
    portfolio = {"selection_rule": RULE, "factors": {}}
    for name, spec in CANDIDATES.items():
        runs = {kind: score(path, spec["arm"], spec["control"])
                for kind, path in spec["files"].items()}
        states = [x["state"] for x in runs.values()]
        status = "keep" if states == ["pass", "pass"] else (
            "reject" if "fail" in states else "pending")
        portfolio["factors"][name] = {
            "status": status, "description": spec["description"], "runs": runs,
        }
    portfolio["usable_shortlist"] = [name for name, item in portfolio["factors"].items()
                                      if item["status"] == "keep"]
    OUT.write_text(json.dumps(portfolio, indent=2) + "\n")
    print(json.dumps(portfolio, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
