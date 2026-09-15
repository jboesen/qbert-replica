"""Real FAAB bids from Sleeper leagues, to calibrate the blind-bid waiver test.

The harness's waivers run one add a week in worst-record-first order, but many leagues
bid a budget instead, and whether bidding well is a lever depends on how real managers
bid. Nothing in the repo says that, and the one public analysis (1,658 bids, one week)
is too thin to condition on week and player quality. Sleeper's public API returns every
waiver claim in a league, winning and failed, with its bid, so this crawls a few hundred
12-team redraft PPR one-QB FAAB leagues per season for 2023-25 and turns their claims
into a tidy table: one row per claim, with the week it was decided for and the player's
consensus rest-of-season positional rank at that point.

League discovery is a snowball (users -> their leagues -> those leagues' users), which
over-represents people in many leagues. That is a known bias, reported, not corrected.

    .venv/bin/python draft/sleeper_faab.py crawl       # cached under data/sleeper_faab_raw/
    .venv/bin/python draft/sleeper_faab.py summarize   # -> data/faab_bids.parquet
"""
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API = "https://api.sleeper.app/v1"
RAW = Path("data/sleeper_faab_raw")
SEASONS = (2023, 2024, 2025)
TARGET = 300                    # qualifying leagues per season
LEGS = 18
# Seeds: a public analyst account and the Scott Fish Bowl account, whose 2024 leagues
# hold a few thousand active players who also run ordinary home leagues.
SEED_USERS = ("jjzachariason", "ffballers", "scottfishbowl")
# Sleeper asks for under 1,000 calls a minute; stay well below it.
MIN_GAP = 0.07

_lock = threading.Lock()
_last = [0.0]
_session = requests.Session()


def get(path):
    """GET an API path, cached on disk so a crawl can resume and the summary is offline."""
    f = RAW / (path.strip("/").replace("/", "__") + ".json.gz")
    if f.exists():
        with gzip.open(f, "rt") as fh:
            return json.load(fh)
    for attempt in range(5):
        with _lock:
            wait = _last[0] + MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = _session.get(API + path, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    else:
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(f, "wt") as fh:
        json.dump(data, fh)
    return data


def qualifies(lg):
    """12-team redraft, full PPR, one QB and no superflex, FAAB, not best ball, finished.
    That is the harness's league, so the bids come from the setting being simulated."""
    s, rp = lg.get("settings") or {}, lg.get("roster_positions") or []
    return (lg.get("total_rosters") == 12 and s.get("type") == 0
            and s.get("waiver_type") == 2 and (s.get("waiver_budget") or 0) > 0
            and not s.get("best_ball") and lg.get("status") == "complete"
            and (lg.get("scoring_settings") or {}).get("rec") == 1.0
            and rp.count("QB") == 1 and "SUPER_FLEX" not in rp)


def crawl():
    pool = ThreadPoolExecutor(8)
    seen_users, seen_leagues = set(), set()
    found = {y: {} for y in SEASONS}          # season -> league id -> FAAB budget
    queue = []
    for name in SEED_USERS:
        u = get(f"/user/{name}")
        if u:
            queue.append(u["user_id"])
    # The SFB account's own leagues are special-format, but their members are the seed.
    sfb = get(f"/user/{get('/user/scottfishbowl')['user_id']}/leagues/nfl/2024") or []
    for users in pool.map(lambda lg: get(f"/league/{lg['league_id']}/users") or [], sfb[:40]):
        queue += [x["user_id"] for x in users]

    while queue and min(len(v) for v in found.values()) < TARGET:
        batch, queue = [u for u in queue[:200] if u not in seen_users], queue[200:]
        seen_users.update(batch)
        jobs = [(u, y) for u in batch for y in SEASONS if len(found[y]) < TARGET]
        new = []
        for leagues in pool.map(lambda j: get(f"/user/{j[0]}/leagues/nfl/{j[1]}") or [], jobs):
            for lg in leagues:
                lid = lg["league_id"]
                if lid in seen_leagues:
                    continue
                seen_leagues.add(lid)
                if qualifies(lg) and len(found[int(lg["season"])]) < TARGET:
                    found[int(lg["season"])][lid] = lg["settings"]["waiver_budget"]
                    new.append(lid)
        # Expand only through qualifying leagues: their members play the format wanted.
        for users in pool.map(lambda lid: get(f"/league/{lid}/users") or [], new):
            queue += [x["user_id"] for x in users if x["user_id"] not in seen_users]
        print(f"users {len(seen_users)}  leagues seen {len(seen_leagues)}  qualifying "
              + " ".join(f"{y}:{len(v)}" for y, v in found.items()), flush=True)

    (RAW / "leagues.json").write_text(json.dumps(found))
    jobs = [(lid, w) for v in found.values() for lid in v for w in range(1, LEGS + 1)]
    for k, _ in enumerate(pool.map(lambda j: get(f"/league/{j[0]}/transactions/{j[1]}"), jobs)):
        if k % 1000 == 0:
            print(f"transactions {k}/{len(jobs)}", flush=True)
    get("/players/nfl")
    print("done")


# ---------------------------------------------------------------- summary

def week_starts(y):
    """First kickoff of each regular-season week, in UTC epoch ms."""
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == y) & (g.game_type == "REG")].copy()
    et = ZoneInfo("America/New_York")
    ts = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00"))
    g["ms"] = [int(t.replace(tzinfo=et).timestamp() * 1000) for t in ts]
    return g.groupby("week").ms.min().sort_index()


def ros_ranks(y, w):
    """Positional rest-of-season rank at the decision before week w, the harness's rule:
    the latest scrape strictly before w, preseason ranks before the first one."""
    ros = pd.read_parquet("data/ecr_ros.parquet")
    ros = ros[(ros.season == y) & (ros.week < w)]
    src = ros[ros.week == ros.week.max()] if len(ros) else (
        lambda p: p[p.season == y])(pd.read_parquet("data/ecr_preseason.parquet"))
    src = src.assign(rank=src.groupby("pos").ecr.rank(method="first"))
    return src.set_index("player_id")[["pos", "rank"]], src.groupby("pos")["rank"].max()


def summarize():
    found = json.loads((RAW / "leagues.json").read_text())
    players = get("/players/nfl")
    ids = pd.read_csv("data/playerids.csv", dtype=str)
    alt = dict(zip(ids.sleeper_id, ids.gsis_id))
    gsis = {k: (v.get("gsis_id") or "").strip() or alt.get(k) for k, v in players.items()}
    pos_of = {k: v.get("position") for k, v in players.items()}
    rows = []
    for y, leagues in found.items():
        y = int(y)
        starts = week_starts(y)
        for lid, budget in leagues.items():
            for leg in range(1, LEGS + 1):
                for t in get(f"/league/{lid}/transactions/{leg}") or []:
                    if t.get("type") != "waiver" or not t.get("adds"):
                        continue
                    bid = (t.get("settings") or {}).get("waiver_bid")
                    pid = next(iter(t["adds"]))
                    if bid is None or not pid.isdigit():
                        continue                                  # team defenses
                    ms = t.get("status_updated") or t.get("created")
                    later = starts[starts > ms]
                    rows.append({
                        "season": y, "league": lid, "leg": leg, "ms": ms,
                        "week": int(later.index[0]) if len(later) else 99,
                        "sleeper_id": pid, "player_id": gsis.get(pid),
                        "sleeper_pos": pos_of.get(pid), "budget": budget, "bid": bid,
                        "bid_pct": 100.0 * bid / budget, "status": t["status"],
                        "notes": (t.get("metadata") or {}).get("notes", ""),
                        "roster": t["roster_ids"][0] if t.get("roster_ids") else None,
                    })
    d = pd.DataFrame(rows)
    d = d.drop_duplicates(["league", "ms", "sleeper_id", "roster", "bid", "status"])
    d["outbid"] = d.notes.str.contains("claimed by another", na=False)

    # Consensus rank at the decision point, the harness's way (one past the last ranked).
    parts = []
    for (y, w), g in d[d.week.between(2, 17)].groupby(["season", "week"]):
        rk, worst = ros_ranks(y, w)
        g = g.join(rk, on="player_id")
        g["pos"] = g.pos.fillna(g.sleeper_pos)
        g = g[g.pos.isin(["QB", "RB", "WR", "TE"])]
        g["rank"] = g["rank"].fillna(g.pos.map(worst) + 1)
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)

    # Per player-batch: the winning bid and, when someone was outbid, the best losing bid.
    key = ["league", "ms", "sleeper_id"]
    win = d[d.status == "complete"].groupby(key).bid_pct.max().rename("win_pct")
    second = d[d.outbid].groupby(key).bid_pct.max().rename("second_pct")
    n = d[(d.status == "complete") | d.outbid].groupby(key).size().rename("n_bidders")
    d = d.join(win, on=key).join(second, on=key).join(n, on=key)
    d = d.drop(columns=["notes"])
    d.to_parquet("data/faab_bids.parquet")
    print(f"{len(d)} claims, {d.league.nunique()} leagues, "
          + " ".join(f"{y}:{g.league.nunique()}" for y, g in d.groupby("season")))


if __name__ == "__main__":
    {"crawl": crawl, "summarize": summarize}[sys.argv[1]]()
