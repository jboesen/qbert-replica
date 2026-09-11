"""Season-ahead fantasy projections with partial pooling.

Follows the hierarchical framing used for fantasy scoring (Egidi & Gabry, Frontiers in
Sports 2025): a player's own history is not trusted on its own but shrunk toward a
position/age/experience prior, by an amount that depends on how much history he has.
Written in closed form - the normal-normal conjugate case - rather than sampled, which
gives the same posterior mean and a usable predictive spread at a fraction of the cost.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import role

RECENCY = 0.65          # weight decay per season going back
POSITIONS = ["QB", "RB", "WR", "TE"]


def player_history(s):
    """Recency-weighted mean of each player's own prior seasons, and its weight.

    Strictly backward-looking: season t sees seasons < t only.
    """
    rows = []
    for pid, g in s.sort_values("season").groupby("player_id", sort=False):
        gm, pg, yr = g.games.values, g.ppr_pg.values, g.season.values
        tp = g.touch_pg.values
        for i in range(len(g)):
            if i == 0:
                rows.append((pid, yr[i], np.nan, 0.0, np.nan))
                continue
            back = yr[i] - yr[:i]
            w = RECENCY ** (back - 1) * gm[:i]
            rows.append((pid, yr[i], np.average(pg[:i], weights=w), w.sum(),
                         np.average(tp[:i], weights=w)))
    return pd.DataFrame(rows, columns=["player_id", "season", "own_ppg", "own_w", "own_touch"])


def prior_model(train):
    """The population prior: what a player of this position, age and pedigree scores."""
    fits = {}
    for pos in POSITIONS:
        t = train[(train.position == pos) & train.own_ppg.notna()]
        X = pd.DataFrame({
            "age": t.age - 26, "age2": (t.age - 26) ** 2,
            "exp": np.log1p(t.exp), "pedigree": -np.log(t.draft_pick),
        }, index=t.index)
        fits[pos] = sm.WLS(t.ppr_pg, sm.add_constant(X), weights=t.games).fit()
    return fits


def prior_predict(fits, d):
    out = pd.Series(np.nan, index=d.index)
    for pos, m in fits.items():
        x = d[d.position == pos]
        if not len(x):
            continue
        X = pd.DataFrame({
            "age": x.age - 26, "age2": (x.age - 26) ** 2,
            "exp": np.log1p(x.exp), "pedigree": -np.log(x.draft_pick),
        }, index=x.index)
        out.loc[x.index] = m.predict(sm.add_constant(X, has_constant="add"))
    return out


def aging_deltas(train, min_games=6):
    """Aging curve by the delta method: within-player year-over-year change.

    Fitting age cross-sectionally understates decline, because the players still in the
    league at 33 are the ones who aged well. Comparing each player against himself a
    year earlier removes that selection.
    """
    curves = {}
    for pos in POSITIONS:
        t = train[(train.position == pos) & (train.games >= min_games)].sort_values(
            ["player_id", "season"])
        d = t.assign(prev_ppg=t.groupby("player_id").ppr_pg.shift(1),
                     prev_season=t.groupby("player_id").season.shift(1))
        d = d[(d.season - d.prev_season == 1) & d.prev_ppg.notna()]
        d["delta"] = d.ppr_pg - d.prev_ppg
        d["bucket"] = d.age.round().clip(22, 35)
        by_age = d.groupby("bucket").apply(
            lambda g: np.average(g.delta, weights=g.games), include_groups=False)
        # Chain the yearly changes into a level curve, centred at 26.
        ages = np.arange(21, 39)
        step = pd.Series(by_age).reindex(ages).ffill().bfill().fillna(0.0)
        curve = step.cumsum()
        curves[pos] = (curve - curve.loc[26]).to_dict()
    return curves


def age_value(curves, position, age):
    out = np.empty(len(age))
    for i, (p, a) in enumerate(zip(position, age)):
        c = curves[p]
        k = int(np.clip(np.round(a) if np.isfinite(a) else 26, 21, 38))
        out[i] = c[k]
    return out


def shrinkage_k(train):
    """Empirical Bayes: how many games of history equal one prior's worth of evidence.

    k = (noise variance) / (talent variance), in games. Small k means a position's
    numbers stabilise fast and history can be trusted; large k means regress hard.
    """
    ks = {}
    for pos in POSITIONS:
        t = train[(train.position == pos) & train.own_ppg.notna() & (train.own_w > 8)]
        resid = t.ppr_pg - t.prior                       # deviation from the population
        own_dev = t.own_ppg - t.prior_own
        # Regress this season's deviation on prior deviation, weighted by evidence.
        # The slope of that relation implies k through slope = w / (w + k).
        b = np.polyfit(own_dev, resid, 1, w=np.sqrt(t.games))[0]
        w_bar = np.average(t.own_w, weights=t.games)
        ks[pos] = max(w_bar * (1 - b) / max(b, 1e-6), 1.0)
    return ks


def games_model(train):
    """Expected games played: availability is a large part of season-long value.

    Last season's games total on its own reads an injury as a lasting condition. It
    isn't - good players return to full roles, while the players who genuinely lose
    games year after year are the ones losing jobs. Conditioning on talent as well as
    on last year's availability cuts the coefficient on last year by more than half.
    """
    fits = {}
    for pos in POSITIONS:
        t = train[(train.position == pos) & train.own_ppg.notna()]
        fits[pos] = sm.OLS(t.games, sm.add_constant(games_features(t))).fit()
    return fits


def games_features(x):
    return pd.DataFrame({
        "age": x.age - 26, "age2": (x.age - 26) ** 2,
        "prior_games": x.prev_games.fillna(12),
        "ppg": x.own_ppg.fillna(0),                  # talent
        "touch": x.own_touch.fillna(0),              # role
        "own_w": np.log1p(x.own_w.fillna(0)),        # how established he is
    }, index=x.index)


def build():
    s = pd.read_parquet("data/draft_seasons.parquet")
    s = s.merge(player_history(s), on=["player_id", "season"], how="left")
    s["prev_games"] = s.groupby("player_id").games.shift(1)
    return s


def fit_predict(s, train_mask):
    tr = s[train_mask]
    fits = prior_model(tr)
    s["prior"] = prior_predict(fits, s)
    # The prior evaluated at the player's age when his history was accumulated, so the
    # deviation compares like with like.
    hist = s.assign(age=s.age - 1.5, exp=(s.exp - 1.5).clip(lower=0))
    s["prior_own"] = prior_predict(fits, hist)

    ks = shrinkage_k(s[train_mask])
    k = s.position.map(ks)
    w = s.own_w.fillna(0)
    # A player's history was accumulated at a younger age than the season being
    # projected, so carry it forward along the aging curve before blending. Without
    # this the curve only ever touches the population prior, which leaves the model
    # systematically low on ascending players and high on declining ones.
    # Two aging curves, each biased in a known direction: the cross-sectional one
    # understates decline (only good old players are still playing), the delta one
    # overstates improvement (only good young players stick around). Blend them, with
    # the weight chosen on training seasons alone.
    curves = aging_deltas(s[train_mask])
    delta_shift = (age_value(curves, s.position.values, s.age.values)
                   - age_value(curves, s.position.values, (s.age - 1.5).values))
    cross_shift = s.prior - s.prior_own

    def blended(alpha):
        shift = alpha * delta_shift + (1 - alpha) * cross_shift
        own_aged = s.own_ppg + shift
        return np.where(s.own_ppg.notna(), (w * own_aged + k * s.prior) / (w + k), s.prior)

    tr_rows = train_mask & s.own_ppg.notna() & s.games.notna() & (s.games >= 4)
    grid = np.linspace(0, 1, 21)
    err = [np.sqrt(np.average((s.ppr_pg[tr_rows] - blended(a)[tr_rows]) ** 2,
                              weights=s.games[tr_rows])) for a in grid]
    alpha = float(grid[int(np.argmin(err))])
    s["proj_ppg"] = blended(alpha)
    s.attrs["alpha"] = alpha


    gm = games_model(s[train_mask])
    s["proj_games"] = np.nan
    for pos, m in gm.items():
        x = s[s.position == pos]
        s.loc[x.index, "proj_games"] = m.predict(
            sm.add_constant(games_features(x), has_constant="add"))
    s["proj_games"] = s.proj_games.clip(1, s.season_games)
    s["proj_points"] = s.proj_ppg * s.proj_games

    # History says how good he is; the week-1 depth chart says whether he'll play.
    # Re-weight by role (draft/role.py), fit on the training seasons only.
    s = role.attach(s)
    s = role.apply(s, role.fit(s, train_mask))
    s.attrs["alpha"] = alpha                # the merge in attach drops attrs
    return s, ks


if __name__ == "__main__":
    s = build()
    s, ks = fit_predict(s, s.season <= 2020)
    print(f"aging-curve blend (1 = pure delta method): alpha = {s.attrs['alpha']:.2f}")
    print("shrinkage k, in games of history needed to half-trust a player:")
    print({p: round(v, 1) for p, v in ks.items()})
    s.to_parquet("data/draft_projected.parquet")

    te = s[(s.season > 2020) & s.own_ppg.notna() & (s.games >= 4)].copy()
    te["last_total"] = s.groupby("player_id").ppr.shift(1).reindex(te.index)
    te = te[te.last_total.notna()]                 # common subset, so RMSE compares
    print(f"\nholdout 2021-25, n={len(te)} player-seasons with prior-year history")
    for name, p in [("hierarchical projection", te.proj_points),
                    ("last season's total", te.last_total),
                    ("own ppg x season games", te.own_ppg * te.season_games),
                    ("position mean", te.groupby("position").ppr.transform("mean"))]:
        print(f"  {name:26s} RMSE {np.sqrt(np.mean((te.ppr - p) ** 2)):6.1f}   "
              f"MAE {np.mean(np.abs(te.ppr - p)):5.1f}   corr {np.corrcoef(te.ppr, p)[0, 1]:.3f}")
