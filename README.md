# CurbAI

**Block-level scoring for autonomous mobility.** Tells you where in a city the next autonomous ride service should launch, where an autonomous delivery robot can actually drop a package, and which ride drop-offs have the biggest food-delivery conversion upside — one map, three scores, built on open data. Scope: San Francisco v1.

## What it shows

Three tabs, one H3 res-9 grid (1,112 cells, ~150 m edge), three scoring functions:

1. **AV Rider Launch Readiness** — where should an autonomous ride service launch next? Scores road simplicity, curb availability, rider demand, and transit-gap fit per cell.
2. **Autonomous Delivery Handoff** — where can a delivery robot actually drop a package and have a human find it? Scores curb access, building line-of-sight, pedestrian path density, and safety proxies.
3. **Rides → Eats Conversion Upside** — which ride drop-off zones have the highest food-delivery cross-sell opportunity? Scores restaurant supply, underserved home zones, and evening gravity.

Click any hex to select it — the side panel updates with the score breakdown and the top five most-similar cells elsewhere in the city (FAISS nearest-neighbor over a z-scored feature matrix). Search any SF address to zoom. See the **Methodology** page in the app sidebar for full data-source and scoring-formula details.

## Live demo

[URL TBD — coming once deployed to Streamlit Community Cloud.]

## Reproduce locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# One-time data bootstrap (reads sibling geofm-global project once, then independent)
python scripts/bootstrap_data.py

# Fetch open-source OSM data for SF (~1-2 min)
python scripts/fetch_osm.py

# Compute features and scores (~30 sec)
python scripts/build_sf.py

# Boot
streamlit run app.py
```

## Data sources

All open, all no-auth, all cached locally:

- **H3 cell grid + 51,572 Overture POIs**: bootstrapped once from a sibling project's processed parquet outputs (filtered to the SF bbox). After the bootstrap runs, CurbAI has no runtime dependency on any sibling project.
- **Roads, buildings, transit stops, amenities**: OpenStreetMap via OSMnx — 9,890 road-graph nodes / 27,261 edges, 158,765 buildings, 7,415 transit points, 15,571 amenities.
- **San Francisco bbox**: `[-122.52, 37.71, -122.36, 37.83]` (west, south, east, north).

## Architecture

```
curbai/
├── app.py                      Streamlit entrypoint (main map page)
├── pages/
│   └── 1_Methodology.py        Dedicated methodology page
├── curbai/
│   ├── loader.py               Cached parquet loader
│   ├── scoring.py              Three scoring functions
│   └── similarity.py           FAISS nearest-neighbor
├── scripts/
│   ├── bootstrap_data.py       One-time sibling-project bootstrap
│   ├── fetch_osm.py            OSMnx SF fetch
│   └── build_sf.py             Feature engineering + scoring precompute
├── data/
│   ├── raw/                    gitignored, re-creatable
│   └── sf_scored.parquet       gitignored, re-creatable
├── .streamlit/config.toml      Warm beige/cocoa theme
├── requirements.txt
└── README.md
```

## License

MIT.
