"""QBERT-replica: a local reimplementation of Silver Bulletin's QBERT quarterback rating.

Pipeline, following the published methodology:
  1. raw rating   - regress ESPN QBR on widely-available box-score rates, logit link
  2. adjustments  - opponent-defense Elo, home field, weather, era recentering
  3. scale        - 80 = league average, 68 = replacement level
  4. WAR          - ~30% of wins above replacement credited to the quarterback
  5. projection   - rolling form + aging/experience curve for the next start
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

AVG, REPL = 80.0, 68.0          # published QBERT anchors
QB_WIN_SHARE = 0.30             # share of wins above replacement credited to the QB
PTS_PER_WIN = 35.0              # NFL points-to-wins conversion


# --------------------------------------------------------------------- features
def rates(d):
    """Every count becomes a per-play rate; QBERT is an efficiency measure."""
    p = d.plays.clip(lower=1)
    f = pd.DataFrame(index=d.index)
    f["comp"] = d.completions.fillna(0) / p
    f["td"] = (d.passing_tds.fillna(0) + d.rushing_tds.fillna(0)) / p
    f["int"] = d.passing_interceptions.fillna(0) / p
    f["fum"] = (d.sack_fumbles.fillna(0) + d.rushing_fumbles.fillna(0)) / p
    f["fum_lost"] = (d.sack_fumbles_lost.fillna(0) + d.rushing_fumbles_lost.fillna(0)) / p
    f["air"] = (d.passing_yards.fillna(0) - d.passing_yards_after_catch.fillna(0)) / p
    f["net_rush"] = (d.rushing_yards.fillna(0) - d.sack_yards_lost.fillna(0)) / p
    f["first"] = (d.passing_first_downs.fillna(0) + d.rushing_first_downs.fillna(0)) / p
    f["carry"] = d.carries.fillna(0) / p
    f["sack"] = d.sacks_suffered.fillna(0) / p
    f["rb_ypc"] = d.rb_ypc.fillna(4.2)
    f["log_plays"] = np.log(p)
    f["intended_air"] = d.passing_air_yards.fillna(0) / p
    f["deep20"] = d.passing_20.fillna(0) / p
    f["deep40"] = d.passing_40.fillna(0) / p
    f["cpoe"] = d.passing_cpoe.fillna(0)
    f["press"] = (d.qb_hits.fillna(0) / d.dropbacks.clip(lower=1)).fillna(0)
    f["net_q2"], f["net_q3"] = d.net_q2.fillna(0), d.net_q3.fillna(0)
    f["script"] = d.result.fillna(0)
    return f


def rates_modern(d):
    """Feature set for the modern-era variant.

    QBERT restricts itself to statistics available back to 1950. Nothing forces a
    1999-onward build to do the same, and EPA plus explicit clutch flags recover
    most of the remaining gap to ESPN's QBR.
    """
    f = rates(d)
    p = d.plays.clip(lower=1)
    f["pass_epa"] = d.passing_epa.fillna(0) / p
    f["rush_epa"] = d.rushing_epa.fillna(0) / p
    f["comeback"] = ((d.net_q3.fillna(0) < 0) & (d.result.fillna(0) > 0)).astype(float)
    f["ot"] = (d.result.fillna(0).abs() <= 3).astype(float)
    return f


def fit_raw(train, feats=rates):
    """QBR is bounded [0, 100], so fit its log-odds and invert."""
    y = train.qbr_total.clip(0.5, 99.5)
    z = np.log(y / (100 - y))
    return sm.WLS(z, sm.add_constant(feats(train)),
                  weights=train.plays.clip(lower=1)).fit()


def predict_qbr(model, d, feats=rates):
    X = sm.add_constant(feats(d), has_constant="add")[model.params.index]
    return 100 / (1 + np.exp(-model.predict(X)))


# ------------------------------------------------------------------ adjustments
def defense_elo(d, k=0.06, regress=0.25):
    """Rolling rating for each defense: QBR allowed relative to average.

    Walked in chronological order and read *before* the game is applied, so a
    quarterback's adjustment never uses the game he is being rated on.
    """
    d = d.sort_values(["season", "week"])
    elo, out, season = {}, np.empty(len(d)), None
    for i, r in enumerate(d.itertuples()):
        if r.season != season:                        # regress toward mean each year
            elo = {t: v * (1 - regress) for t, v in elo.items()}
            season = r.season
        cur = elo.get(r.opponent_team, 0.0)
        out[i] = cur
        elo[r.opponent_team] = cur + k * ((r.qbr_hat - 50.0) - cur)
    return pd.Series(out, index=d.index)


def adjust(d):
    """Strip out context the QB did not control: defense faced, venue, weather."""
    d = d.copy()
    d["def_elo"] = defense_elo(d)
    cold = (d.temp.fillna(65) - 65).clip(upper=0)     # only cold hurts
    wind = d.wind.fillna(0).clip(upper=25)
    indoors = d.roof.isin(["dome", "closed"])
    d["weather_pen"] = np.where(indoors, 0.0, 0.06 * cold - 0.25 * wind)
    d["home_pen"] = np.where(d.home == 1, 1.0, -1.0)
    # Adjusted QBR: add back what the context took away.
    d["qbr_adj"] = d.qbr_hat - d.def_elo - d.weather_pen - 1.0 * d.home_pen
    return d


def to_scale(d, anchor=("Aaron Rodgers", 2011, 108.6)):
    """Map adjusted QBR onto the published 80-average / 68-replacement scale.

    The spread of the scale is set by the one season rating Silver Bulletin has
    published outright - Rodgers' 2011 MVP year at 108.6 - rather than by a guess at
    where replacement level sits. That leaves the 68 anchor free to serve as a check
    on the result instead of an input to it (see calibrate.py).
    """
    league = d.groupby("season").apply(
        lambda g: np.average(g.qbr_adj, weights=g.plays), include_groups=False)
    d = d.join(league.rename("league_qbr"), on="season")
    d["qbr_centered"] = d.qbr_adj - d.league_qbr

    name, season, rating = anchor
    a = d[(d.player_display_name == name) & (d.season == season)]
    k = (rating - AVG) / np.average(a.qbr_centered, weights=a.plays)
    d["qbert"] = AVG + k * d.qbr_centered
    d["qbert_raw"] = AVG + k * (d.qbr_hat - d.league_qbr)
    return d


def war(d):
    """Points above replacement per play, converted to wins at ~35 points a win."""
    d = d.copy()
    pts_per_play = (d.qbert - REPL) * 0.021          # calibrated below against pts_added
    d["paa"] = pts_per_play * d.plays
    d["war"] = d.paa / PTS_PER_WIN * (QB_WIN_SHARE / 0.30)
    return d
