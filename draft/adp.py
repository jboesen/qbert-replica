"""Average draft position: what the market actually paid, as against what the experts
ranked.

The league backtest already knows what consensus (FantasyPros ECR) thought. ADP is the
other public signal, and a different kind of thing: not an opinion but a price, set by
what thousands of drafters actually did. The two are close cousins, so the question is
whether the market carries anything consensus doesn't.

Source: the Fantasy Football Calculator API, free, PPR, 12 teams. It serves one snapshot
per season, the last few days of drafts before that season's opener, which is the same
timing as our preseason ECR snapshot (the last scrape before the opener). So neither
signal sees a down of football. Attribution requested by FFC; noted here and in the
README.

FFC identifies players by its own id and a display name, so the mapping to nflverse gsis
ids goes through names: normalise both sides the same way and match on name plus
position, as consensus.py does through the dynastyprocess crosswalk.

    .venv/bin/python draft/adp.py
"""
import re
import sys
import time

import pandas as pd
import requests

API = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
IDS = "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv"
SEASONS = range(2021, 2026)
POS = ["QB", "RB", "WR", "TE"]
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
# FFC lists a handful of players under the name they go by rather than the one the
# crosswalk carries. These are the only ones that failed to match on name plus position;
# each is a nickname or a shortened first name, not a different player.
ALIAS = {
    "hollywood brown": "marquise brown",
    "gabe davis": "gabriel davis",
    "kenny gainwell": "kenneth gainwell",
    "jeff wilson": "jeffery wilson",
    "chig okonkwo": "chigoziem okonkwo",
}


def norm(name):
    """One spelling for both sides: lowercase, no punctuation, no generational suffix."""
    s = str(name).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = SUFFIX.sub("", s)
    s = " ".join(s.split())
    return ALIAS.get(s, s)


def fetch_season(y):
    """One season's ADP table, with the snapshot window the API reports for it."""
    r = requests.get(API, params={"teams": 12, "year": y}, timeout=120)
    r.raise_for_status()
    d = r.json()
    if d.get("status") != "Success":
        raise ValueError(f"FFC returned {d.get('status')} for {y}")
    meta = d["meta"]
    t = pd.DataFrame(d["players"])
    t["season"] = y
    # The window is the evidence that the snapshot predates the season; carry it in the
    # data rather than in a comment, so the leakage check is auditable later.
    t["snapshot_start"] = meta["start_date"]
    t["snapshot_end"] = meta["end_date"]
    t["total_drafts"] = meta["total_drafts"]
    return t


def crosswalk():
    """gsis id per (normalised name, position), from the dynastyprocess id table."""
    ids = pd.read_csv(IDS, low_memory=False)
    ids = ids[ids.gsis_id.notna() & ids.name.notna()]
    ids = ids.assign(key=ids.name.map(norm), pos=ids.position.astype(str))
    # Keep the most recent row per player so a name that moved position maps once.
    ids = ids.sort_values("db_season").drop_duplicates(["key", "pos"], keep="last")
    return ids


def main():
    ids = crosswalk()
    by_key_pos = ids.set_index(["key", "pos"]).gsis_id
    solo = ids.drop_duplicates("key", keep=False).set_index("key").gsis_id
    frames = []
    for y in SEASONS:
        t = fetch_season(y)
        frames.append(t)
        time.sleep(1)                      # be polite to a free API
    adp = pd.concat(frames, ignore_index=True)
    adp["key"] = adp.name.map(norm)
    adp["player_id"] = pd.MultiIndex.from_arrays([adp.key, adp.position]).map(by_key_pos)
    # A few players are listed at a position the crosswalk disagrees with; fall back to
    # the name alone when that name belongs to exactly one player.
    miss = adp.player_id.isna()
    adp.loc[miss, "player_id"] = adp.loc[miss, "key"].map(solo)

    flex = adp[adp.position.isin(POS)]
    print(f"rows {len(adp)}, of them {len(flex)} at QB/RB/WR/TE")
    print("\nper season: rows, skill rows, matched, match rate, snapshot window")
    for y, g in adp.groupby("season"):
        f = g[g.position.isin(POS)]
        ok = f.player_id.notna().sum()
        print(f"  {y}  {len(g):4d}  {len(f):4d}  {ok:4d}  {ok / len(f):6.1%}  "
              f"{f.snapshot_start.iloc[0]} to {f.snapshot_end.iloc[0]}  "
              f"({f.total_drafts.iloc[0]} drafts)")
    bad = flex[flex.player_id.isna()]
    if len(bad):
        print("\nunmatched skill players:")
        for r in bad.itertuples():
            print(f"  {r.season} {r.name:28s} {r.position} {r.team} adp {r.adp}")

    keep = ["season", "player_id", "name", "position", "team", "adp", "times_drafted",
            "high", "low", "stdev", "snapshot_start", "snapshot_end", "total_drafts"]
    adp[keep].to_parquet("data/adp_ffc.parquet")
    print("\nwrote data/adp_ffc.parquet")


def positional_rank(adp, y):
    """ADP as a rank within position, for the season's snapshot."""
    a = adp[(adp.season == y) & adp.player_id.notna() & adp.position.isin(POS)]
    a = a.drop_duplicates("player_id").set_index("player_id")
    return a.assign(adp_rank=a.groupby("position").adp.rank(method="first"))


def compare():
    """Descriptive: does the market forecast the season better than the experts?

    Same pool and the same busts-count-as-zero convention as stack.py, so this sits
    beside the consensus-vs-model comparison already in the README. Rank correlation
    rather than points, because ADP is only an ordering; the sign is flipped so that
    higher always means the signal ranked the season better.
    """
    sys.path.insert(0, "draft")
    import stack as K

    adp = pd.read_parquet("data/adp_ffc.parquet")
    rows = []
    for y in SEASONS:
        t = K.table(y)
        a = positional_rank(adp, y)
        t["adp"] = a.adp.reindex(t.index)
        t["adp_rank"] = a.adp_rank.reindex(t.index)
        # Same convention stack.py uses for a missing consensus rank: one past the last
        # ranked player at the position. Undrafted is information, not absence.
        worst = t.groupby("pos").adp_rank.max()
        t["adp_filled"] = t.adp_rank.fillna(t.pos.map(worst) + 1)
        t = t[K.in_pool(t)]
        rows.append(t)
    t = pd.concat(rows)

    def sp(x, col, on):
        return -x[col].corr(x[on], method="spearman")

    print("ADP vs ECR as a forecast of season PPR points, 2021-25")
    print("stack.py draftable pool, players who never played count as zero")
    print("Spearman of the signal's rank with actual points, sign flipped so higher is "
          "better\n")
    both = t[t.adp_rank.notna()]
    print(f"{'pos':4s} {'n both':>7s} {'ECR':>6s} {'ADP':>6s} | {'n pool':>7s} "
          f"{'ECR':>6s} {'ADP*':>6s} | {'ECR~ADP':>8s}")
    for p in POS + ["all"]:
        x = both if p == "all" else both[both.pos == p]
        f = t if p == "all" else t[t.pos == p]
        # Across positions, compare on the overall ordering rather than the positional one.
        c1, c2 = ("rank", "adp_rank") if p != "all" else ("rank", "adp_rank")
        print(f"{p:4s} {len(x):7d} {sp(x, c1, 'actual'):6.3f} {sp(x, c2, 'actual'):6.3f} | "
              f"{len(f):7d} {sp(f, 'rank', 'actual'):6.3f} "
              f"{sp(f, 'adp_filled', 'actual'):6.3f} | "
              f"{x[c1].corr(x[c2], method='spearman'):8.3f}")
    print("\n* ADP with undrafted players ranked one past the last drafted at the position")

    print("\nby season, players with both signals (ECR / ADP, and their rank correlation):")
    for y, g in both.groupby("season"):
        print(f"  {y}  n={len(g):4d}   ECR {sp(g, 'rank', 'actual'):6.3f}   "
              f"ADP {sp(g, 'adp_rank', 'actual'):6.3f}   "
              f"ECR~ADP {g['rank'].corr(g.adp_rank, method='spearman'):.3f}")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
    else:
        main()
