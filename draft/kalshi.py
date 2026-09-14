"""Kalshi player props for the 2025 season, as pregame prices.

Kalshi's live API only lists recently settled markets; everything settled before its
historical cutoff lives under /historical, and that includes the whole 2025 NFL season.
Kalshi is the better source for this question than Polymarket: it has a receptions
market (a point each in PPR, and what Polymarket lacked), and its prop volume is real
(tens of thousands of dollars on a starter's touchdown market, against a few hundred).

Each stat is a ladder of markets ("Khalil Shakir: 80+", "70+", ...), so the prices
trace out the market's whole distribution for that player-game, and the expected value
is the area under it: E[X] = sum over k of P(X >= k).

Prices are read from hourly candlesticks at the last hour ending at least an hour before
kickoff, as the midpoint of the best bid and ask (the last trade if the book was empty).
Nothing at or after kickoff is used.

The pull is about 130 markets a game and there is no batch endpoint, so it runs one
request per market, several at a time, and saves each game as it finishes; rerunning
skips games already saved.

    .venv/bin/python draft/kalshi.py [--games N]     # -> data/kalshi_2025.parquet
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

B = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"KXNFLREC": "rec", "KXNFLRECYDS": "rec_yds", "KXNFLANYTD": "td",
          "KXNFLPASSYDS": "pass_yds"}
SEASON = 2025
CUTOFF = 3600                      # seconds before kickoff; later prices know the game
LOOKBACK = 4 * 86400               # markets list a day or two ahead
RAW = "data/kalshi_raw"
# Kalshi's team codes, where they differ from nflverse's.
CODE = {"JAX": "JAC", "LA": "LAR"}


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            return json.load(urllib.request.urlopen(req, timeout=40))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** i)             # 429 and 5xx: back off and retry
        except Exception:
            time.sleep(2 ** i)
    return None


def games():
    g = pd.read_csv("data/games.csv")
    g = g[(g.season == SEASON) & (g.game_type == "REG")].copy()
    when = pd.to_datetime(g.gameday + " " + g.gametime).dt.tz_localize("US/Eastern")
    g["kick"] = when.map(lambda t: int(t.timestamp()))
    g["code"] = [f"{pd.Timestamp(d):%y%b%d}".upper() + CODE.get(a, a) + CODE.get(h, h)
                 for d, a, h in zip(g.gameday, g.away_team, g.home_team)]
    return g[["game_id", "week", "gameday", "away_team", "home_team", "kick", "code"]]


def pregame(ticker, kick):
    """Implied probability of 'yes' at the last hour ending an hour before kickoff."""
    d = get(f"{B}/historical/markets/{ticker}/candlesticks?start_ts={kick - LOOKBACK}"
            f"&end_ts={kick - CUTOFF}&period_interval=60")
    if not d:
        return None
    for c in reversed(d.get("candlesticks", [])):
        if c["end_period_ts"] > kick - CUTOFF:
            continue
        bid, ask = c["yes_bid"].get("close"), c["yes_ask"].get("close")
        bid, ask = (float(bid) if bid else 0.0), (float(ask) if ask else 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        if c["price"].get("close"):
            return float(c["price"]["close"])
    return None


def pull_game(g):
    path = f"{RAW}/{g.code}.json"
    if os.path.exists(path):
        return json.load(open(path))
    markets = []
    for s, kind in SERIES.items():
        d = get(f"{B}/historical/markets?event_ticker={s}-{g.code}&limit=1000") or {}
        for m in d.get("markets", []):
            name = (m.get("yes_sub_title") or "").split(":")[0].strip()
            markets.append((m["ticker"], kind, name, m.get("floor_strike"),
                            float(m.get("volume_fp") or 0)))
    with ThreadPoolExecutor(2) as ex:
        prices = list(ex.map(lambda m: pregame(m[0], g.kick), markets))
    rows = [{"game_id": g.game_id, "week": int(g.week), "ticker": t, "kind": k,
             "player": n, "strike": s, "volume": v, "p_yes": p}
            for (t, k, n, s, v), p in zip(markets, prices)]
    json.dump(rows, open(path, "w"))
    return rows


def main():
    os.makedirs(RAW, exist_ok=True)
    g = games()
    if "--games" in sys.argv:
        g = g.head(int(sys.argv[sys.argv.index("--games") + 1]))
    out = []
    for i, r in enumerate(g.itertuples(), 1):
        rows = pull_game(r)
        out += rows
        priced = sum(1 for x in rows if x["p_yes"] is not None)
        print(f"  {i}/{len(g)} wk{r.week} {r.code}: {len(rows)} markets, {priced} priced "
              f"pregame", flush=True)
    d = pd.DataFrame(out)
    d.to_parquet("data/kalshi_2025.parquet")
    p = d[d.p_yes.notna()]
    print(f"\n{len(d)} markets, {len(p)} with a pregame price, "
          f"{p.player.nunique()} players, weeks {sorted(p.week.unique())}")


if __name__ == "__main__":
    main()
