"""Where each draft platform's rooms actually took players, 2021-25: ESPN and Sleeper.

A 2026 r/fantasyfootball post ("Abusing Draft Rankings 2026") argues that drafters anchor
on their platform's default rankings, so a player consensus likes and the platform buries
lasts longer on that platform. Testing that needs, per season, the order each platform's
rooms draft in. What could be recovered for 2021-25 is FantasyPros' "ADP by site" table,
which gives each platform's ADP as an overall rank, with the date of the site's last
snapshot before the season. It is platform ADP, not the platform's default ranking: the
outcome of drafting in that room, anchoring included, rather than the list that anchors it.

Sources, and why each is trusted:

  - FantasyPros still serves the historical page (ppr-overall.php?year=Y), but anonymous
    visitors get only the first five rows. Those rows, and the per-site snapshot dates in
    the same page, are fetched live here and used as the reference.
  - The full tables come from CSV exports of that page that other people committed to
    GitHub, pinned to a commit. Exports downloaded at different times can hold different
    snapshots, so a file is used only if it reproduces the reference rows exactly. The
    chosen files also agree with an independently downloaded export on every player in
    the top 180 at both sites (checked while building this; see the report).
  - Sleeper's own projections API carries a season adp_ppr. Its order is compared to the
    FantasyPros Sleeper column as a third, first-party check.

Not recovered: Yahoo. Yahoo columns exist only in exports of unknown date that do not
match any reference, and FantasyPros no longer lists Yahoo for these seasons. ESPN's API
returns rankings for past seasons, but they were overwritten in-season (2024's list has
Malik Nabers sixth), so they are not preseason defaults. Wayback captures could not be
checked: the Internet Archive was offline while this was built.

    .venv/bin/python draft/platform_ranks.py
"""
import csv
import io
import json
import re
import sys
import urllib.parse

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "draft")
from adp import crosswalk, norm

SEASONS = range(2021, 2026)
POS = ["QB", "RB", "WR", "TE"]
SITES = {"ESPN": "79", "Sleeper": "4350"}          # FantasyPros expert ids
FP = "https://www.fantasypros.com/nfl/adp/ppr-overall.php?year={y}"
RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
# One export per season, each pinned to the commit it was read at.
FILES = {
    2021: ("juliancanaless/fantasy-rl-draft", "bca6bf3129fb842ea1728aaefbac24c0f5c55cdc",
           "data/raw/adp/adp_2021.csv"),
    2022: ("juliancanaless/fantasy-rl-draft", "bca6bf3129fb842ea1728aaefbac24c0f5c55cdc",
           "data/raw/adp/adp_2022.csv"),
    2023: ("juliancanaless/fantasy-rl-draft", "bca6bf3129fb842ea1728aaefbac24c0f5c55cdc",
           "data/raw/adp/adp_2023.csv"),
    2024: ("juliancanaless/fantasy-rl-draft", "bca6bf3129fb842ea1728aaefbac24c0f5c55cdc",
           "data/raw/adp/adp_2024.csv"),
    # The 2025 file in that repo was exported in June 2025, before the final snapshot.
    2025: ("wambozi/fantasy-football", "bc05b00ef9cf95b5c17f5054ea050e5e99561d0d",
           "data/FantasyPros_2025_Overall_ADP_Rankings (1).csv"),
}
# Players the chosen export lacks although the site ranked them. FantasyPros' current export
# drops players who have since retired, so the 2025 file (downloaded in 2026) has no Austin
# Ekeler (ECR 115th). His row comes from an export of the same page committed 2025-08-30,
# four days before the final snapshot; no other player is patched.
PATCH = {2025: [("Austin Ekeler", "cliffdboan/fantasy_ranking",
                 "85b565a1da5d4851e4b5008261726c549187dd90",
                 "FantasyPros_2025_Overall_ADP_Rankings.csv")]}
# The site's spelling where the crosswalk has another (as ALIAS in adp.py).
RENAME = {"William Fuller": "Will Fuller"}
SLEEPER = ("https://api.sleeper.app/projections/nfl/{y}?season_type=regular"
           "&position[]=QB&position[]=RB&position[]=WR&position[]=TE")
UA = {"User-Agent": "Mozilla/5.0"}


def reference(y):
    """FantasyPros' own first rows and each site's snapshot date, from the live page."""
    h = requests.get(FP.format(y=y), headers=UA, timeout=120).text
    i = h.find('"table":{"fields"')
    table, _ = json.JSONDecoder().raw_decode(h[i + len('"table":'):])
    top = {norm(r["player"]["name"]): {s: r[f"src_{k}"] for s, k in SITES.items()}
           for r in table["rows"]}
    dates = {}
    # The date as the page prints it (M/D); the sort key beside it is a UTC timestamp that
    # can land on the next day.
    for k, md in re.findall(r'name="expert\[\]" value="(\d+)".*?data-sort="\d+">([\d/]+)<', h, re.S):
        for s, sk in SITES.items():
            if k == sk:
                m, d = md.split("/")
                dates[s] = f"{y}-{int(m):02d}-{int(d):02d}"
    return top, dates


def export(y, repo=None, sha=None, path=None):
    """The pinned CSV export: player, position, and each site's overall ADP rank."""
    if repo is None:
        repo, sha, path = FILES[y]
    text = requests.get(RAW.format(repo=repo, sha=sha, path=urllib.parse.quote(path)),
                        timeout=120).content.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    head, body = rows[0], rows[1:]
    # A handful of rows per file are broken by an unescaped quote in the name (Ty'Son
    # Williams); none is a draftable player, and they are counted in the report.
    good = [dict(zip(head, r)) for r in body if len(r) == len(head)]
    d = pd.DataFrame(good)
    name = d["Player"] if "Player" in d else d["Player (Bye)"].str.replace(r"\s{2,}.*$", "", regex=True)
    out = pd.DataFrame({
        "season": y, "name": name.replace(RENAME),
        "position": d.POS.str.extract(r"^([A-Z]+)")[0],
    })
    for s in SITES:
        out[f"{s.lower()}_rank"] = pd.to_numeric(d[s], errors="coerce")
    out["source"] = f"{repo}@{sha[:7]}:{path}"
    return out, len(body) - len(good)


def sleeper_api(y):
    js = requests.get(SLEEPER.format(y=y), timeout=180).json()
    t = pd.DataFrame([(norm(x["player"]["first_name"] + " " + x["player"]["last_name"]),
                       x["player"].get("position"), x["stats"].get("adp_ppr"))
                      for x in js], columns=["key", "position", "api_adp"])
    return t[t.api_adp.notna() & (t.api_adp < 999)].drop_duplicates(["key", "position"])


def main():
    ids = crosswalk()
    by_key_pos = ids.set_index(["key", "pos"]).gsis_id
    solo = ids.drop_duplicates("key", keep=False).set_index("key").gsis_id
    frames = []
    print("season  ref rows matched  ESPN date   Sleeper date  bad rows  Sleeper col vs API rho")
    for y in SEASONS:
        top, dates = reference(y)
        t, bad = export(y)
        for who, *src in PATCH.get(y, []):
            other, _ = export(y, *src)
            t = pd.concat([t, other[other.name == who]], ignore_index=True)
        t["key"] = t.name.map(norm)
        ok = 0
        for k, want in top.items():
            got = t[t.key == k]
            ok += len(got) == 1 and all(got[f"{s.lower()}_rank"].iloc[0] == v
                                        for s, v in want.items())
        if ok != len(top):
            raise ValueError(f"{y}: export does not reproduce FantasyPros' reference rows")
        api = sleeper_api(y)
        m = t.merge(api, on=["key", "position"])
        m = m[m.sleeper_rank <= 150]
        rho = m.sleeper_rank.corr(m.api_adp, method="spearman")
        t["espn_date"], t["sleeper_date"] = dates.get("ESPN"), dates.get("Sleeper")
        print(f"  {y}  {ok}/{len(top)}  {dates.get('ESPN')}  {dates.get('Sleeper')}   "
              f"{bad:4d}      {rho:.4f} (n={len(m)})")
        frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.position.isin(POS)].copy()
    d["player_id"] = pd.MultiIndex.from_arrays([d.key, d.position]).map(by_key_pos)
    miss = d.player_id.isna()
    d.loc[miss, "player_id"] = d.loc[miss, "key"].map(solo)
    report_coverage(d)
    keep = ["season", "player_id", "name", "position", "espn_rank", "sleeper_rank",
            "espn_date", "sleeper_date", "source"]
    d[keep].to_parquet("data/platform_ranks.parquet")
    print("\nwrote data/platform_ranks.parquet")


def report_coverage(d):
    """How much of the draftable pool the mapping covers, judged where it matters: the
    players a 12-team, 14-round room could take, and the consensus pool the harness uses."""
    ecr = pd.read_parquet("data/ecr_preseason_overall.parquet")
    print("\ncoverage (QB/RB/WR/TE):")
    print("  season  site      ranked  mapped  top-200 mapped  ECR top-200 with a rank")
    for y, g in d.groupby("season"):
        e = ecr[ecr.season == y].nsmallest(200, "ecr")
        for s in ("espn", "sleeper"):
            r = g[g[f"{s}_rank"].notna()]
            top = r.nsmallest(200, f"{s}_rank")
            have = set(r.player_id.dropna())
            print(f"  {y}  {s:8s} {len(r):6d}  {r.player_id.notna().mean():6.1%}  "
                  f"{top.player_id.notna().mean():14.1%}  {e.player_id.isin(have).mean():12.1%}")
    bad = d[d.player_id.isna() & (d[["espn_rank", "sleeper_rank"]].min(axis=1) <= 200)]
    if len(bad):
        print("\nunmapped players ranked in a site's top 200:")
        for r in bad.itertuples():
            print(f"  {r.season} {r.name:28s} {r.position} espn {r.espn_rank} sleeper {r.sleeper_rank}")


if __name__ == "__main__":
    main()
