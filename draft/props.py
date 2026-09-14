"""Prediction-market player props, as pregame prices, from Polymarket.

The league backtest keeps finding that consensus is hard to beat, and every signal we
own is one consensus already knows. A betting market is the one source that isn't an
opinion: it is money, and it moves until kickoff. So the question worth asking is
narrow and cheap: do prop prices carry anything about a player's week that the
consensus ranking doesn't already have?

What's actually available is thin, and the design follows from that. Polymarket only
started listing NFL player props in volume around week 14 of 2025, carries about ten
players a game, and has no receptions market at all. Receptions are a point each in
PPR, so props cannot reconstruct a fantasy score on their own. They don't have to: the
test is whether they add anything on top of consensus, not whether they replace it.

Prices are read at a declared cutoff before kickoff, never after. A market that only
lists once the game is under way is dropped rather than read late, since an in-game
price knows what it is supposed to be predicting.

    .venv/bin/python draft/props.py            # pull weeks 14-17 of 2025 -> data/props_2025.parquet
"""
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com/events?slug="
CLOB = "https://clob.polymarket.com/prices-history"
SEASON, WEEKS = 2025, range(14, 18)
CUTOFF_MIN = 60                  # read the price at least this long before kickoff
# Polymarket slugs use the lowercase nflverse abbreviation for most teams; these differ.
TEAM = {"LA": "la", "LAC": "lac", "LV": "lv", "NE": "ne", "NO": "no", "NYG": "nyg",
        "NYJ": "nyj", "SF": "sf", "TB": "tb", "GB": "gb", "KC": "kc", "JAX": "jax",
        "WAS": "was"}
KINDS = {"Anytime Touchdown": "td", "Receiving Yards": "rec_yds",
         "Rushing Yards": "rush_yds", "Passing Yards": "pass_yds"}


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(1 + i)


def kickoffs():
    """Kickoff in epoch seconds per game, with the slug Polymarket files it under."""
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == SEASON) & (g.game_type == "REG") & g.week.isin(WEEKS)]
    out = []
    for r in g.itertuples():
        when = pd.Timestamp(f"{r.gameday} {r.gametime}", tz="US/Eastern")
        away, home = TEAM.get(r.away_team, r.away_team.lower()), TEAM.get(r.home_team, r.home_team.lower())
        out.append({"week": r.week, "away": r.away_team, "home": r.home_team,
                    "slug": f"nfl-{away}-{home}-{r.gameday}", "kick": when.timestamp()})
    return pd.DataFrame(out)


def parse(question):
    """'Jahmyr Gibbs: Receiving Yards O/U 63.5' -> (name, kind, line)."""
    if ":" not in question:
        return None
    who, rest = question.split(":", 1)
    rest = rest.strip()
    for label, kind in KINDS.items():
        if rest.startswith(label):
            tail = rest[len(label):].strip()
            line = np.nan
            if tail.startswith("O/U"):
                try:
                    line = float(tail.split()[-1])
                except ValueError:
                    return None
            elif tail:
                return None                      # "2+ Touchdowns" and friends: skip
            return who.strip(), kind, line
    return None


def pregame_price(token, kick):
    """The last traded price at least CUTOFF_MIN before kickoff, or None."""
    cut = kick - CUTOFF_MIN * 60
    h = get(f"{CLOB}?market={token}&startTs={int(kick - 8 * 86400)}"
            f"&endTs={int(cut)}&fidelity=1")
    if not h:
        return None
    pts = h.get("history", [])
    pts = [p for p in pts if p["t"] <= cut]
    return pts[-1]["p"] if pts else None


def pull():
    rows = []
    games = kickoffs()
    for i, g in enumerate(games.itertuples(), 1):
        ev = get(GAMMA + g.slug)
        if not ev:
            continue
        for m in ev[0].get("markets", []):
            p = parse(m.get("question", ""))
            if not p:
                continue
            name, kind, line = p
            toks = m.get("clobTokenIds")
            toks = json.loads(toks) if isinstance(toks, str) else toks
            if not toks:
                continue
            price = pregame_price(toks[0], g.kick)
            if price is None:                    # listed only once the game began
                continue
            rows.append({"season": SEASON, "week": g.week, "player": name, "kind": kind,
                         "line": line, "p_over": float(price),
                         "away": g.away, "home": g.home})
        print(f"  {i}/{len(games)} {g.slug}: {len(rows)} rows so far", flush=True)
    return pd.DataFrame(rows)


def main():
    d = pull()
    if not len(d):
        sys.exit("no prop rows retrieved")
    d.to_parquet("data/props_2025.parquet")
    print(f"\ndata/props_2025.parquet: {len(d)} prop prices, "
          f"{d.player.nunique()} players, weeks {sorted(d.week.unique())}")
    print(d.groupby(["week", "kind"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
