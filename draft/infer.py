"""Structured, live inference from the validated fantasy policy.

The model does not pretend to find a better player projection than expert consensus.
It uses consensus rest-of-season values, then chooses the waiver transaction that fills
an upcoming starting-lineup hole.  The output is JSON so a UI, agent, or script can use
the recommendation without parsing terminal prose.

Examples:
  python draft/infer.py --league my_league.json
  python draft/infer.py --mine "Player A,Player B" --available "Player C,Player D"

A league file has ``me`` and ``teams`` keys.  Supplying ``--available`` limits inference
to the actual waiver pool; otherwise every player not rostered by a league-file team is
considered available.
"""
import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, "draft")
import waivers as W


def resolve_names(names, players):
    """Resolve names without emitting prose into this command's JSON output."""
    eligible = players[players.position.isin(W.POS)].reset_index()
    low = eligible.display_name.str.lower()
    resolved, unresolved = [], []
    for name in (str(x).strip() for x in names if str(x).strip()):
        hit = eligible[low == name.lower()]
        if not len(hit):
            hit = eligible[low.str.contains(name.lower(), regex=False)]
        if len(hit):
            resolved.append(hit.sort_values("last_season", ascending=False).gsis_id.iloc[0])
        else:
            unresolved.append(name)
    return resolved, unresolved


def named(pid, players, pos, rank, value, reason=None):
    item = {
        "player_id": pid,
        "name": str(players.display_name.get(pid, pid)),
        "position": str(pos.get(pid, "")),
        "consensus_ros_rank": int(rank[pid]) if pid in rank else None,
        "consensus_ros_value": round(float(value.get(pid, 0.0)), 2),
    }
    if reason:
        item["unavailable_reason"] = reason
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", help="JSON with {me, teams}; identifies all rostered players")
    parser.add_argument("--mine", help="comma-separated roster, if no league file is supplied")
    parser.add_argument("--available", help="comma-separated available-player pool")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, help="upcoming scoring week; inferred when omitted")
    args = parser.parse_args()
    if not args.league and not args.mine:
        parser.error("supply --league or --mine")

    players = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
    unresolved = []
    if args.league:
        with open(args.league) as f:
            league = json.load(f)
        mine, missing_mine = resolve_names(league["teams"][league["me"]], players)
        unresolved.extend(missing_mine)
        rostered = set()
        for team in league["teams"].values():
            players_on_team, missing = resolve_names(team, players)
            rostered.update(players_on_team)
            unresolved.extend(missing)
        owner = league["me"]
    else:
        mine, unresolved = resolve_names(args.mine.split(","), players)
        rostered, owner = set(), "you"

    y, w = args.season, args.week or W.upcoming_week(args.season)
    values, ranks, positions = W.values(y, w, players.position)
    value, rank, pos = values.to_dict(), ranks.to_dict(), positions.to_dict()
    if args.available:
        available, missing = resolve_names(args.available.split(","), players)
        unresolved.extend(missing)
        free = [player for player in available if player in value and player not in rostered]
    else:
        free = [player for player in value if player not in rostered and pos.get(player) in W.POS]

    known_out = W.unavailable(y, w, set(mine) | set(free))
    free = [player for player in free if player not in known_out]
    stream, holes = W.streaming_move(mine, free, pos, value, known_out)
    consensus = W.consensus_move(mine, free, pos, value)

    result = {
        "policy": "hole_aware_streaming",
        "evidence": {
            "realistic_opponents_all_play_pp": 1.03,
            "exact_consensus_all_play_pp": 2.035,
            "positive_seasons": "5 of 5 in both designs",
        },
        "season": y, "week": w, "manager": owner,
        "inputs": {
            "roster_size": len(mine), "available_pool_size": len(free),
            "unresolved_names": list(dict.fromkeys(unresolved)),
        },
        "starting_holes": {slot: int(count) for slot, count in holes.items() if count},
        "known_unavailable": [named(player, players, pos, rank, value, known_out[player])
                              for player in mine if player in known_out],
        "recommendation": None,
        "consensus_alternative": None,
    }
    if stream:
        result["recommendation"] = {
            "action": "add_drop",
            "add": named(stream[0], players, pos, rank, value),
            "drop": named(stream[1], players, pos, rank, value),
            "reason": "This is the highest consensus-valued available player who closes an upcoming starting hole.",
            "trades_ros_value_for_lineup": bool(value.get(stream[0], -1e9) < value.get(stream[1], -1e9)),
        }
    elif consensus:
        result["recommendation"] = {
            "action": "add_drop",
            "add": named(consensus[0], players, pos, rank, value),
            "drop": named(consensus[1], players, pos, rank, value),
            "reason": "The hole-aware rule found no available transaction that reduces the lineup shortfall, so it falls back to the normal consensus value move.",
        }
    else:
        result["recommendation"] = {"action": "hold", "reason": "No hole needs filling and no available player improves consensus rest-of-season value."}
    if consensus and consensus != stream:
        result["consensus_alternative"] = {
            "add": named(consensus[0], players, pos, rank, value),
            "drop": named(consensus[1], players, pos, rank, value),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
