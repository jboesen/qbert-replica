"""Platform-anchored draft rooms: does knowing who a platform's room will leave you win
leagues, when consensus still decides who is good? (prereg_platformranks.md)

The failed ADP test (prereg_adp.md) asked whether the market is a better opinion than the
experts, and paid market prices against rooms that drafted by consensus. This asks a
narrower thing. Real rooms draft off their platform's list; if they do, a player consensus
likes and the platform buries lasts longer there, and a drafter who knows it can take
someone else now and still get him at the next pick. Consensus sets every value; the
platform only predicts availability.

  - Opponents draft the harness's noisy-consensus way, but the list they perturb is the
    platform's: each season's ESPN or Sleeper ADP order (data/platform_ranks.parquet),
    mapped onto the consensus scale so the experts' spread means the same thing.
  - The test seat values players by exact consensus and plans with the room's expected
    order: it waits on the best player the room is predicted to leave and takes the best
    player it is predicted not to, when that player beats what would otherwise be left.
  - Control: exact consensus draft against the same rooms. Both arms stream on waivers.

    .venv/bin/python draft/sim_platformranks.py --noise 1     (through the run lock)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "draft")
import league_backtest as LB
import settings as CFG
import consensus as C
from draft_dp import snake_picks

SET = CFG.get()
NOISE = SET.noise
LEAGUES = LB.LEAGUES
# Weight on the platform order in the opponents' list; 0 is the plain consensus room of
# run_streaming (used only for the mechanics check that the control reproduces it).
WEIGHT = 1.0
# Even leagues are ESPN rooms, odd leagues Sleeper rooms.
PLATFORMS = ("espn", "sleeper")


def platform_key(S, site):
    """The platform's order on the consensus scale.

    A site rank is an ordinal; ECR mean is on a different scale (an average of experts'
    overall ranks). Reading the platform's k-th player at the k-th smallest ECR mean puts
    both on one scale, so a blend is a blend of orders and the ECR spread used for noise
    keeps its meaning. Players the site never ranked come after every ranked player, in
    consensus order: a room that did not draft a player buries him.
    """
    pr = pd.read_parquet("data/platform_ranks.parquet")
    pr = pr[(pr.season == S.y) & pr.player_id.notna()].drop_duplicates("player_id")
    rank = pr.set_index("player_id")[f"{site}_rank"].reindex(S.ids).values
    ok = np.isfinite(S.ecr_mean)
    idx = np.where(ok)[0]
    # Ranked first by site rank, then the unranked by consensus; ties by consensus.
    order = idx[np.lexsort((S.ecr_mean[idx], np.nan_to_num(rank[idx], nan=1e9)))]
    key = np.full(len(S.ids), np.nan)
    key[order] = np.sort(S.ecr_mean[idx])
    return key


def predicted_gone(key, pool, m):
    """The m players in pool the room is predicted to take next: lowest key first."""
    if m <= 0:
        return set()
    return set(pool[np.argsort(key[pool], kind="stable")[:m]].tolist())


def availability_policy(S, seat, key, log):
    """Value by exact consensus; plan with the room's expected order `key`.

    At a pick with m other picks before my next one: b is the best allowed player by
    consensus. If the room is predicted to leave b, compare two plans that both end with
    b: take the best-consensus allowed player u the room is predicted to take (then b at
    the next pick), or take b now (then s, the best player predicted to be left besides
    b). Wait on b only when u beats s by consensus. Otherwise, and at the last pick, take b.
    """
    mine = snake_picks(seat, LB.TEAMS, LB.ROUNDS)

    def pick(taken, c, rnd, pick_no):
        avail = np.where(~taken & np.isfinite(S.ecr_mean))[0]
        ok = [x for x in avail if LB.allowed(S.pos[x], c, rnd)]
        if not ok:
            return None
        ok = np.array(ok)
        b = int(ok[np.argmin(S.ecr_mean[ok])])
        later = [p for p in mine if p > pick_no]
        if not later:
            return b
        m = later[0] - pick_no - 1
        rest = avail[avail != b]
        gone_if_b = predicted_gone(key, rest, m)
        # s: what taking b now leaves me at the next pick, by the room's prediction.
        c_b = dict(c); c_b[S.pos[b]] += 1
        left = [x for x in rest if x not in gone_if_b and LB.allowed(S.pos[x], c_b, rnd + 1)]
        s = min(left, key=lambda x: S.ecr_mean[x]) if left else None
        entry = {"round": rnd + 1, "pick": pick_no, "best": S.ids[b], "waited": False}
        for u in ok[np.argsort(S.ecr_mean[ok], kind="stable")]:
            u = int(u)
            if u == b:
                continue
            if s is not None and S.ecr_mean[u] >= S.ecr_mean[s]:
                break                          # sorted: no later u beats s either
            if u not in gone_if_b:
                continue                       # the room leaves u too: nothing to gain now
            pool = avail[avail != u]
            if b in predicted_gone(key, pool, m):
                continue                       # taking u changes the room's picks; b goes
            c_u = dict(c); c_u[S.pos[u]] += 1
            if not LB.allowed(S.pos[b], c_u, rnd + 1):
                continue
            entry.update(waited=True, took=S.ids[u], next_pick=later[0],
                         gap=float(S.ecr_mean[u] - S.ecr_mean[b]))
            log.append(entry)
            return u
        log.append(entry)
        return b

    return pick


def room_orders(S, key, noise):
    """Twelve opponents' lists and the spare, as in run_streaming but on the blended key."""
    base = (1 - WEIGHT) * S.ecr_mean + WEIGHT * key if WEIGHT else S.ecr_mean
    out = []
    for k in range(LB.TEAMS + 1):
        score = base + NOISE * S.ecr_sd * noise[k]
        ok = np.where(np.isfinite(score))[0]
        out.append(list(ok[np.argsort(score[ok])]))
    return out


def outcome_row(y, lg, seat, arm, site, sc, scheds, moves):
    res = [LB.season_outcome(sc, s) for s in scheds]
    return {
        "season": y, "league": lg, "seat": seat, "arm": arm, "platform": site,
        "title": np.mean([r[2][seat] for r in res]),
        "playoff": np.mean([r[1][seat] for r in res]),
        "wins": np.mean([r[0][seat] for r in res]),
        "all_play": LB.all_play(sc, seat),
        "pts_reg": sc[seat, :LB.REG].sum(),
        "pts_playoff": sc[seat, LB.REG:].sum(),
        "moves": moves[seat],
    }


def run():
    curves = LB.rank_curve()
    wk_proj = LB.weekly_projections()
    rows, picks = [], []
    for y in LB.SEASONS:
        S = LB.Season(y, wk_proj, curves)
        crv = C.weekly_curve(range(2020, y))
        cons_val, _ = LB.waiver_values(S, y, crv, None)
        gone = LB.known_unavailable(S, y)
        keys = {site: platform_key(S, site) for site in PLATFORMS}
        every = set(range(LB.TEAMS))
        base = [cons_val] * LB.TEAMS
        for lg in range(LEAGUES):
            site = PLATFORMS[lg % 2]
            # The same draws in the same order as run_streaming, so with WEIGHT 0 the
            # control is run_streaming's stream arm exactly.
            rng = np.random.default_rng([y, lg])
            noise = rng.standard_normal((LB.TEAMS + 1, len(S.ids)))
            orders_room = room_orders(S, keys[site], noise)
            scheds = LB.schedules(rng, LB.SCHEDULES)
            for seat in range(LB.TEAMS):
                log = []
                for arm, own in (("cons", S.exact_order),
                                 ("avail", availability_policy(S, seat, keys[site], log))):
                    orders = list(orders_room[:LB.TEAMS])
                    orders[seat] = own
                    drafted = LB.draft(S, orders)
                    movers = {seat: LB.streaming_policy(S, gone, [])}
                    sc, moves, _ = LB.simulate(S, drafted, base, every, movers)
                    rows.append(outcome_row(y, lg, seat, arm, site, sc, scheds, moves))
                    if arm == "avail":
                        # Did the player waited on survive? Taken players are in rosters.
                        held = {S.ids[i]: t for t, r in enumerate(drafted) for i in r}
                        for e in log:
                            e.update(season=y, league=lg, seat=seat, platform=site,
                                     got_best=held.get(e["best"]) == seat)
                        picks += log
            print(f"{y} league {lg + 1}/{LEAGUES} ({site})", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(picks)


def mechanics(d, pk):
    """Outcome-free checks for the reduced run: legal rosters are enforced by draft();
    here, how often the policy waits and whether waits resolve as planned."""
    print("\nmechanics (no outcome metrics):")
    print(f"  rows per arm: {d.arm.value_counts().to_dict()}")
    w = pk[pk.waited]
    n = d[d.arm == "avail"].shape[0]
    print(f"  waits per draft {len(w) / max(n, 1):.2f}; waited-on player still got by me: "
          f"{w.got_best.mean() if len(w) else float('nan'):.1%}")
    if len(w):
        print("  waits by round: " + " ".join(f"{r}:{k}" for r, k in
                                            w["round"].value_counts().sort_index().items()))
        print(w.head(8)[["season", "platform", "round", "pick", "best", "took", "gap",
                         "got_best"]].to_string(index=False))


def report(d, pk):
    print("\n=== draft: availability planning against platform-anchored rooms ===")
    print(f"opponent list: {WEIGHT:g} x platform order + {1 - WEIGHT:g} x consensus, "
          f"noise {NOISE:g} x ECR sd; streaming in both arms\n")
    print(f"{'arm':6s} {'title':>7s} {'playoff':>8s} {'wins':>6s} {'all-play':>9s} "
          f"{'reg pts':>8s} {'po pts':>7s}")
    for arm, g in d.groupby("arm", sort=False):
        print(f"{arm:6s} {100 * g.title.mean():6.1f}% {100 * g.playoff.mean():7.1f}% "
              f"{g.wins.mean():6.2f} {100 * g.all_play.mean():8.1f}% "
              f"{g.pts_reg.mean():8.0f} {g.pts_playoff.mean():7.0f}")
    print("\npreregistered test (prereg_platformranks.md), season-cluster bootstrap 90% interval:")
    keep = LB.paired(d, "avail", "cons", "avail minus consensus")
    print(f"  -> {'passes this design' if keep else 'fails this design'}")
    print("\nreported, not gated:")
    for site in PLATFORMS:
        LB.paired(d[d.platform == site], "avail", "cons", f"  {site} rooms only")
    w = pk[pk.waited]
    n = d[d.arm == "avail"].groupby("season").size()
    print(f"  waits per draft by season: "
          + " ".join(f"{y} {(w.season == y).sum() / k:.2f}" for y, k in n.items()))
    print(f"  waited-on player landed on my roster: {w.got_best.mean():.1%} of {len(w)} waits")
    print(f"  mean consensus gap given up at the pick (u minus b, ECR ranks): {w.gap.mean():+.1f}")
    print("  waits by round: " + " ".join(f"{r}:{k}" for r, k in
                                        w["round"].value_counts().sort_index().items()))


if __name__ == "__main__":
    if "--noise" in sys.argv:
        NOISE = float(sys.argv[sys.argv.index("--noise") + 1])
    if "--leagues" in sys.argv:
        LEAGUES = int(sys.argv[sys.argv.index("--leagues") + 1])
    if "--weight" in sys.argv:
        WEIGHT = float(sys.argv[sys.argv.index("--weight") + 1])
    real = LEAGUES == LB.LEAGUES and WEIGHT == 1.0
    tag = "" if real else f"_check{LEAGUES}_w{WEIGHT:g}"
    d, pk = run()
    d.to_parquet(f"data/league_platformranks{tag}_noise{NOISE:g}.parquet")
    pk.to_parquet(f"data/league_platformranks_picks{tag}_noise{NOISE:g}.parquet")
    print(f"\nopponent noise: {NOISE:g} x ECR sd; weight {WEIGHT:g}")
    if real:
        report(d, pk)
    else:
        mechanics(d, pk)
