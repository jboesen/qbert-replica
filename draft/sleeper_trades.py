"""Crawl completed trades from real Sleeper redraft leagues (prereg_trades.md, part A).

The trade test needs a rule for when an opponent accepts an offer, and a rule with no
empirical basis would make any result about trades a result about the rule. Sleeper's
API is free and public, so the value ratios real managers accept can be measured
directly: snowball from a few users to their leagues, keep 12-team PPR redraft
one-quarterback leagues from 2023-25, and save every completed trade.

Raw responses are cached under data/sleeper_trades_raw/ (gitignored), one bundle per
league, so the summary in trade_acceptance.py can be rebuilt without the network.

    .venv/bin/python draft/sleeper_trades.py [--max-leagues 1500]
"""
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://api.sleeper.app/v1"
RAW = "data/sleeper_trades_raw"
SEASONS = ("2023", "2024", "2025")
# Two public accounts that hold 12-team leagues; the snowball does the rest.
SEEDS = ("matt", "ryan")
LEGS = range(1, 19)
# Sleeper asks for under 1000 calls a minute; stay well below it, other agents share the box.
RATE = 6.0


class Throttle:
    """A shared minimum gap between requests across worker threads."""

    def __init__(self, per_sec):
        self.gap, self.next, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


THROTTLE = Throttle(RATE)
SESSION = requests.Session()


def get(path):
    for attempt in range(5):
        THROTTLE.wait()
        try:
            r = SESSION.get(f"{API}/{path}", timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def qualifies(lg):
    """12 teams, full PPR, redraft (settings.type 0), one QB and no superflex, finished."""
    s, sc, pos = lg.get("settings") or {}, lg.get("scoring_settings") or {}, lg.get("roster_positions") or []
    return (lg.get("total_rosters") == 12 and s.get("type") == 0 and sc.get("rec") == 1.0
            and pos.count("QB") == 1 and "SUPER_FLEX" not in pos
            and lg.get("status") == "complete" and s.get("best_ball", 0) == 0)


def crawl_league(lg):
    """League, users, rosters and every leg's transactions, as one cached bundle."""
    path = f"{RAW}/league_{lg['league_id']}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    lid = lg["league_id"]
    bundle = {"league": lg, "users": get(f"league/{lid}/users") or [],
              "rosters": get(f"league/{lid}/rosters") or [],
              "transactions": {str(w): get(f"league/{lid}/transactions/{w}") or [] for w in LEGS}}
    with open(path + ".tmp", "w") as f:
        json.dump(bundle, f)
    os.replace(path + ".tmp", path)
    return bundle


def league_users(lid):
    """Members of any NFL league, qualifying or not: the snowball needs a wide frontier,
    since 12-team PPR redraft leagues are a minority of what users belong to."""
    path = f"{RAW}/users_{lid}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = get(f"league/{lid}/users") or []
    with open(path, "w") as f:
        json.dump([{"user_id": u["user_id"]} for u in out], f)
    return out


def user_leagues(uid):
    path = f"{RAW}/user_{uid}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = {y: get(f"user/{uid}/leagues/nfl/{y}") or [] for y in SEASONS}
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def main(max_leagues):
    os.makedirs(RAW, exist_ok=True)
    queue = deque(get(f"user/{u}")["user_id"] for u in SEEDS)
    seen_users, seen_leagues = set(queue), set()
    done, trades = 0, 0
    with ThreadPoolExecutor(8) as pool:
        while queue and done < max_leagues:
            batch = [queue.popleft() for _ in range(min(16, len(queue)))]
            found, other = [], []
            for leagues in pool.map(user_leagues, batch):
                for y in SEASONS:
                    for lg in leagues.get(y) or []:
                        if lg["league_id"] in seen_leagues:
                            continue
                        seen_leagues.add(lg["league_id"])
                        (found if qualifies(lg) else other).append(lg)
            members = []
            for b in pool.map(crawl_league, found[:max_leagues - done]):
                done += 1
                trades += sum(t.get("type") == "trade" and t.get("status") == "complete"
                              for txs in b["transactions"].values() for t in txs)
                members += b["users"]
            # Grow the frontier through other leagues only while it is thin.
            if len(queue) < 2000:
                for us in pool.map(league_users, [lg["league_id"] for lg in other[:40]]):
                    members += us
            for u in members:
                if u["user_id"] not in seen_users:
                    seen_users.add(u["user_id"])
                    queue.append(u["user_id"])
            print(f"leagues {done}  trades {trades}  users queued {len(queue)}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--max-leagues") + 1]) if "--max-leagues" in sys.argv else 1500
    main(n)
