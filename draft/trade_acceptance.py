"""What value ratio do real managers accept in a trade? (prereg_trades.md, part A)

The trade test needs opponents who accept some offers and refuse others. Rather than
invent that rule, this measures it on completed trades from real 12-team PPR redraft
Sleeper leagues (crawled by sleeper_trades.py). Each side of a trade is priced the way
the harness's consensus managers price players: the latest rest-of-season consensus rank
before the trade's week, through the weekly rank curve fit on earlier seasons only
(consensus.rank_points), both as raw points per game and over replacement.

The side that accepted is the side that did not create the final offer (a counter-offer
is created by the side that countered, so the other side still accepted). An endowment
premium k means an owner accepts only when he receives at least (1 + k) times what he
gives. If managers priced players at consensus with private noise and no premium, the
accepting side's ratio would centre on 1; a premium pushes the centre above 1.

Estimator, declared before any trade was summarised: over 1-for-1 trades whose accepting
side is identified and whose two players are both worth more than replacement, k is the
median accepting-side ratio (value received / value given, over replacement) minus 1,
clipped to [0, 0.2] and rounded to the nearest of 0, 0.1, 0.2. With fewer than 200 such
trades the data count as too thin and no k is calibrated.

    .venv/bin/python draft/trade_acceptance.py
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import consensus as C
from league_backtest import POS, WEEKS
from vbd import add_vbd

RAW = "data/sleeper_trades_raw"
OUT = "data/trade_acceptance.parquet"
MIN_TRADES = 200
K_GRID = (0.0, 0.1, 0.2)


def sleeper_to_gsis():
    ids = pd.read_csv("data/playerids.csv", dtype=str)
    ids = ids[ids.sleeper_id.notna() & ids.gsis_id.notna()]
    return dict(zip(ids.sleeper_id, ids.gsis_id))


def trades():
    """One row per completed two-team trade: who gave what, and who accepted."""
    rows = []
    for path in sorted(glob.glob(f"{RAW}/league_*.json")):
        with open(path) as f:
            b = json.load(f)
        lg = b["league"]
        owners = {}
        for r in b["rosters"]:
            for u in [r.get("owner_id")] + (r.get("co_owners") or []):
                if u:
                    owners[u] = r["roster_id"]
        for leg, txs in b["transactions"].items():
            for t in txs:
                if t.get("type") != "trade" or t.get("status") != "complete":
                    continue
                rids = t.get("roster_ids") or []
                if len(rids) != 2:
                    continue
                creator = owners.get(t.get("creator"))
                acc = [r for r in rids if r != creator][0] if creator in rids else None
                got = {r: [p for p, to in (t.get("adds") or {}).items() if to == r] for r in rids}
                gave = {r: [p for p, fr in (t.get("drops") or {}).items() if fr == r] for r in rids}
                rows.append({
                    "season": int(lg["season"]), "league_id": lg["league_id"],
                    "week": int(t.get("leg") or leg), "transaction_id": t["transaction_id"],
                    "side_a": rids[0], "side_b": rids[1], "accepter": acc,
                    "a_gets": got[rids[0]], "b_gets": got[rids[1]],
                    "a_gives": gave[rids[0]], "b_gives": gave[rids[1]],
                    "picks": len(t.get("draft_picks") or []),
                    "faab": len(t.get("waiver_budget") or []),
                    "start_week": (lg.get("settings") or {}).get("start_week", 1),
                })
    return pd.DataFrame(rows).drop_duplicates("transaction_id")


def week_values(season, week, crv):
    """Consensus rest-of-season value per player as of a trade in `week`: the latest
    rest-of-season scrape informing that week (preseason ranks before the first one),
    unranked players one past the last ranked, as in league_backtest.waiver_values."""
    ros = pd.read_parquet("data/ecr_ros.parquet")
    ros = ros[(ros.season == season) & (ros.week <= week)]
    if len(ros):
        src = ros[ros.week == ros.week.max()]
    else:
        src = pd.read_parquet("data/ecr_preseason.parquet")
        src = src[src.season == season]
    src = src[src.pos.isin(POS)].drop_duplicates("player_id")
    src = src.assign(rank=src.groupby("pos").ecr.rank(method="first"))
    pts = C.rank_points(crv, src.pos.values, src["rank"].values)
    b, _, _ = add_vbd(pd.DataFrame({"position": src.pos.values, "proj_points": pts},
                                   index=src.player_id.values))
    worst = src.groupby("pos")["rank"].max().to_dict()
    floor = {p: C.rank_points(crv, [p], [worst[p] + 1])[0] for p in worst}
    return (dict(zip(b.index, b.proj_points)), dict(zip(b.index, b.vbd)),
            floor, dict(zip(b.position, b.replacement)))


def price(d):
    g2 = sleeper_to_gsis()
    players = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id")
    pos_of = dict(zip(players.gsis_id, players.position))
    out = []
    for (y, w), grp in d.groupby(["season", "week"]):
        if y not in (2023, 2024, 2025) or w > WEEKS:
            continue
        crv = C.weekly_curve(range(2020, y))       # fit on earlier seasons only
        raw, vor, floor, repl = week_values(y, w, crv)
        for r in grp.itertuples():
            rec = r._asdict()
            ok = True
            for side in ("a_gets", "b_gets"):
                ids = [g2.get(p) for p in getattr(r, side)]
                pos = [pos_of.get(i) for i in ids]
                if any(i is None or p not in POS for i, p in zip(ids, pos)):
                    ok = False                    # a kicker, defence or unmapped player
                    continue
                # A player consensus doesn't rank sits one past the last ranked player.
                rec[side + "_raw"] = sum(raw.get(i, floor[p]) for i, p in zip(ids, pos))
                rec[side + "_vor"] = sum(vor.get(i, floor[p] - repl[p]) for i, p in zip(ids, pos))
                rec[side + "_pos"] = "/".join(sorted(pos))
            rec["skill_only"] = ok
            out.append(rec)
    return pd.DataFrame(out)


def structure(n1, n2):
    lo, hi = sorted((n1, n2))
    return f"{hi}-for-{lo}" if (hi, lo) in ((1, 1), (2, 1), (2, 2)) else "other"


def summarise(p):
    p = p.assign(n_a=p.a_gets.str.len(), n_b=p.b_gets.str.len())
    p["structure"] = [structure(a, b) for a, b in zip(p.n_a, p.n_b)]
    clean = p[p.skill_only & (p.picks == 0) & (p.faab == 0) & (p.n_a > 0) & (p.n_b > 0)].copy()
    # Accepter's view: what the accepting side received over what it gave.
    acc_a = clean.accepter == clean.side_a
    for s in ("raw", "vor"):
        got = np.where(acc_a, clean[f"a_gets_{s}"], clean[f"b_gets_{s}"])
        gave = np.where(acc_a, clean[f"b_gets_{s}"], clean[f"a_gets_{s}"])
        clean[f"acc_gets_{s}"] = np.where(clean.accepter.notna(), got, np.nan)
        clean[f"acc_gives_{s}"] = np.where(clean.accepter.notna(), gave, np.nan)
        # Consolidation price in 2-for-1s: the one player over the two he cost.
        one = np.where(clean.n_a == 1, clean[f"a_gets_{s}"], clean[f"b_gets_{s}"])
        two = np.where(clean.n_a == 1, clean[f"b_gets_{s}"], clean[f"a_gets_{s}"])
        clean[f"one_over_two_{s}"] = np.where(clean.structure == "2-for-1", one / two, np.nan)
    return p, clean


def report(p, clean):
    print(f"completed two-team trades in qualifying leagues: {len(p)} "
          f"({p.league_id.nunique()} league-seasons); by season "
          + str(p.groupby('season').size().to_dict()))
    n_lg = len(glob.glob(f"{RAW}/league_*.json"))
    print(f"of {n_lg} crawled league-seasons; trades per team-season "
          f"{len(p) / (12 * n_lg):.2f}, so a team takes part in {2 * len(p) / (12 * n_lg):.2f}")
    print(f"skill players only, no picks or FAAB: {len(clean)}; by structure "
          + str(clean.structure.value_counts().to_dict()))
    print(f"accepting side identified: {clean.accepter.notna().sum()}")
    qs = [.1, .25, .5, .75, .9]
    for s, lab in (("raw", "raw points"), ("vor", "over replacement")):
        print(f"\naccepting side's ratio, value received / value given ({lab}); quantiles "
              + " ".join(f"{q:.2f}" for q in qs))
        for st, g in clean[clean.accepter.notna()].groupby("structure"):
            ok = g[(g[f"acc_gets_{s}"] > 0) & (g[f"acc_gives_{s}"] > 0)]
            if not len(ok):
                continue
            r = ok[f"acc_gets_{s}"] / ok[f"acc_gives_{s}"]
            print(f"  {st:8s} n={len(ok):5d} (of {len(g)})  "
                  + " ".join(f"{v:5.2f}" for v in r.quantile(qs))
                  + f"   share >= 1: {(r >= 1).mean():.2f}")
        g = clean[(clean.structure == "2-for-1")]
        x = g[f"one_over_two_{s}"]
        x = x[np.isfinite(x) & (x > 0)]
        if len(x):
            print(f"  2-for-1: one player / the two he cost, n={len(x)}  "
                  + " ".join(f"{v:5.2f}" for v in x.quantile(qs)))

    ones = clean[(clean.structure == "1-for-1") & clean.accepter.notna()
                 & (clean.acc_gets_vor > 0) & (clean.acc_gives_vor > 0)]
    print(f"\ncalibration (declared estimator): {len(ones)} usable 1-for-1 trades")
    if len(ones) < MIN_TRADES:
        print(f"  too thin (< {MIN_TRADES}); k is not calibrated and is swept as a sensitivity")
        return None
    med = float((ones.acc_gets_vor / ones.acc_gives_vor).median())
    k = min(K_GRID, key=lambda g: abs(g - float(np.clip(med - 1, 0, 0.2))))
    # How stable is it? A bootstrap over leagues, since trades in a league share managers.
    rng = np.random.default_rng(1)
    lg = ones.league_id.unique()
    boot = []
    for _ in range(2000):
        pick = rng.choice(lg, len(lg))
        take = ones.set_index("league_id").loc[pick]
        boot.append(float((take.acc_gets_vor / take.acc_gives_vor).median()))
    lo, hi = np.quantile(boot, [.05, .95])
    print(f"  median accepting-side ratio {med:.3f} [90% league bootstrap {lo:.3f}, {hi:.3f}]"
          f"  -> k = {k:g}")
    return k


if __name__ == "__main__":
    d = trades()
    p = price(d)
    p, clean = summarise(p)
    k = report(p, clean)
    out = clean.assign(k_calibrated=np.nan if k is None else k)
    out.to_parquet(OUT)
    print(f"\nwrote {OUT}")
