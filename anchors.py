"""Published QBERT values, scraped from the free portions of Silver Bulletin and Nate
Silver's posts. These are the only ground truth available - the tables are paywalled -
so they serve as the calibration and tracking targets for the replica.
"""

# Season-level adjusted QBERT ratings.
SEASON = {
    ("Aaron Rodgers", 2011): 108.6,     # his MVP year, quoted in the all-time piece
}

# Preseason 2025 projected ratings, from Nate Silver's post announcing them.
PROJECTED_2025 = {
    "Lamar Jackson": 104.5,
    "Josh Allen": 100.9,
    "Patrick Mahomes": 94.4,
}

# Single-game adjusted ratings, Super Bowl LX.
GAMES = {
    ("Sam Darnold", 2025, "SB"): 82.3,
    ("Drake Maye", 2025, "SB"): 75.9,
}

# Career WAR, full careers (so not directly comparable to a 2006+ build).
CAREER_WAR = {"Tom Brady": 114.3, "Peyton Manning": 94.8, "Drew Brees": 86.5}

# Qualitative claims that the replica can be checked against.
CLAIMS = [
    ("Brady 2007 is the top season by WAR, all time", "brady_2007_war_rank_1"),
    ("Lamar Jackson 2024 ranks 10th all time in seasonal WAR", "lamar_2024_war_top10"),
    ("Montana 1989, Young 1994, Manning 2004 tie as most efficient seasons", "pre2006"),
    ("Rodgers' first-start projection was 88.2 adjusted QBERT", "rookie_projection"),
    ("Drake Maye was the 2025 MVP by QBERT", "maye_2025_mvp"),
    ("Sam Darnold ranked 14th in forward-looking QBERT, 8th by 2025 WAR", "darnold_2025"),
]

# Published scale anchors.
AVERAGE, REPLACEMENT = 80.0, 68.0
