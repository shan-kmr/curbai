# CurbIndex v0.2 — Three enhancements for the geo data platform demo

## Context

CurbIndex is a Streamlit app (live at https://huggingface.co/spaces/skay97/curbai, source at https://github.com/shan-kmr/curbai) that demonstrates what block-level geospatial intelligence looks like built on open data (OSM + Overture). Three tabs: Site Intelligence, Brand Location Planner, Neighborhood Character. 1,112 H3 res-9 cells over San Francisco.

This plan adds three features that strengthen the "geo data as a platform" pitch — each one demonstrates a commercially valuable capability where the open-data version is useful but the first-party-data version (i.e. Snap's 470M MAU device-level location signals) is dramatically better. The gap between the two IS the platform value.

Audience: Snap Map product team first, then cross-functional leadership. All open-source data, public HF Space.

---

## Enhancement 1: Temporal Intelligence — "When is this block alive?"

### What it does

Adds a predicted hourly activity profile per cell. Category composition tells you WHEN a block is busy:
- Office POIs + transit → morning/evening commute peaks
- Restaurants + retail → midday lunch + weekend peaks
- Nightlife + bars → evening/late-night peaks
- Parks + leisure → weekend + afternoon peaks
- Mixed-use (high entropy) → active across the day

### Where it shows up

- New enhancement to the Site Intelligence tab OR a standalone 4th tab called "Temporal Patterns"
- Side panel: when a cell is clicked, show a small bar chart with 4 time buckets (morning / midday / evening / late night) estimated from the POI category mix
- Each bar is a 0-1 "expected activity" score for that time window

### How to implement

1. In `curbai/scoring.py` (or a new `curbai/temporal.py`), add a function `estimate_temporal_profile(row) -> dict[str, float]` that returns:

```python
{
    "morning": norm(transit_stop_count) * 0.5 + norm(intersection_count) * 0.3 + norm(shop_count) * 0.2,
    "midday": norm(restaurant_count) * 0.4 + norm(shop_count) * 0.3 + norm(amenity_count) * 0.3,
    "evening": norm(nightlife_count) * 0.4 + norm(restaurant_count) * 0.3 + norm(amenity_count) * 0.3,
    "late_night": norm(nightlife_count) * 0.7 + norm(restaurant_count) * 0.3,
}
```

2. In `scripts/build_sf.py`, compute these 4 columns per cell and add to `sf_scored.parquet`:
   - `activity_morning`, `activity_midday`, `activity_evening`, `activity_late_night`

3. In `app.py`, in the side panel breakdown for Site Intelligence (or all tabs), add a `st.bar_chart` or 4x `st.progress` bars showing the temporal profile of the selected cell.

4. In `pages/1_Methodology.py`, add the temporal formula to the scoring docs and add to the "What First-Party Data Unlocks" table:
   - Open proxy: "Category-derived time-of-day estimates (office → morning, nightlife → evening)"
   - First-party upgrade: "Actual device-density per cell per hour, refreshed every 15 minutes, from 470M MAU"

### Why this is the strongest Snap pitch

The open-data version is obviously crude — it infers WHEN from WHAT (POI types) instead of measuring directly. The first-party version is Snap Map's heat map: real-time crowd density. The gap between the two is the widest of any feature in CurbIndex. When the team sees the primitive category-derived sparkline and knows what the heat map actually has, the pitch writes itself.

Buyers: out-of-home advertising (when to show the digital billboard), retail staffing (how many cashiers at 2pm vs 6pm), delivery fleet positioning (where to pre-stage at 5:30pm), event planning, city emergency management.

### Estimated effort: ~1-2 hours

---

## Enhancement 2: Nearest Competitors in the Brand Planner

### What it does

When a category is selected in the Brand Location Planner and a cell is clicked, the side panel already shows "3 coffee shops within 500m" as a number. This enhancement surfaces the actual NAMES and approximate distances of those competitors from the Overture POI data.

### Where it shows up

Side panel of the Brand Location Planner tab, under the white-space analysis section. Example:

```
Opportunity: 87 / 100
Coffee Shop in this cell: 0
Coffee Shop within 500m: 3
  Nearest competitors:
  - Philz Coffee         ~220m south
  - Blue Bottle Coffee   ~340m east
  - Starbucks            ~480m northwest
```

### How to implement

1. In `curbai/brand_planner.py`, add a function `nearest_competitors(category, h3_id, pois_df, k=5) -> list[dict]` that:
   - Filters `pois_sf.parquet` to the selected category
   - Computes haversine distance from the selected cell's center to each POI
   - Returns the k nearest with name, distance in meters, and cardinal direction

2. Load `pois_sf.parquet` at app boot (cached via `@st.cache_data`). It's 3.7 MB, already in `data/raw/pois_sf.parquet`. For deployment, either:
   - Copy it to `data/pois_sf.parquet` during `build_sf.py` so it's alongside the other committed data files
   - OR commit `data/raw/pois_sf.parquet` to git (3.7 MB — borderline but acceptable)

3. In `app.py`, in `render_brand_breakdown()`, after the existing white-space metrics, call `nearest_competitors()` and render the results as a small table or bullet list.

4. Cardinal direction: compute bearing from cell center to POI lat/lon, bucket into N/NE/E/SE/S/SW/W/NW.

### Why it matters

Turns an abstract score into something a franchise operator can act on TODAY. The first-party upgrade: "we don't just know the competitors exist — we know their visit-share. Philz gets 45% of coffee visits, Blue Bottle gets 30%."

### Estimated effort: ~30-45 minutes

---

## Enhancement 3: Walk-Time Catchment

### What it does

When a cell is selected on any tab, overlay a shaded "walk-time ring" showing which other cells are reachable within 5, 10, and 15 minutes on foot. Uses the OSM road graph already loaded by `scripts/fetch_osm.py`.

### Where it shows up

- Visual: additional pydeck layer on the main map. Selected cell is bright; cells within 5 min are medium bright; 10 min are dim; 15 min are faint. Cells beyond 15 min are not highlighted.
- Side panel: "~742 POIs and ~12,000 estimated residents within a 10-minute walk."

### How to implement

1. Create `curbai/catchment.py` with a function `walk_time_cells(h3_id, road_graph, cells_df, max_minutes=15) -> dict[str, float]` that:
   - Finds the nearest road graph node to the cell center
   - Runs `networkx.single_source_dijkstra_path_length(G, source, cutoff=max_minutes * 80)` where 80m/min is average walking speed (~4.8 km/h)
   - For each reachable node, maps it to an H3 cell via `h3.geo_to_h3(lat, lon, 9)`
   - Returns `{h3_index: walk_time_minutes}` for all reachable cells

2. Load the road graph at app boot. Currently it's in `data/raw/osm/sf_roads.gpkg`. Options:
   - Load via `geopandas.read_file()` + `osmnx.graph_from_gdfs()` at boot (cached)
   - OR save the graph as a pickle during `build_sf.py` and load the pickle (faster boot)
   The pickle approach is better for Streamlit Cloud / HF deploy performance.

3. In `app.py`, when a cell is selected, compute `walk_time_cells()` and add a second pydeck H3HexagonLayer with the reachable cells colored by walk time (bright = close, dim = far). This layer sits under the main hex layer.

4. In the side panel, show a summary: "Within 5 min walk: X cells, Y POIs. Within 10 min: ..."

5. In `pages/1_Methodology.py`, add to the first-party upgrade table:
   - Open proxy: "Road-network walking distance (OSMnx shortest path, 4.8 km/h)"
   - First-party upgrade: "Actual origin-destination trip flows — we don't estimate who COULD walk here, we know who DOES walk here"

### Performance consideration

NetworkX single-source Dijkstra on a 9,890-node graph with cutoff is ~10-50ms. Fast enough for interactive use. Cache the result per selected cell with `@st.cache_data`.

The graph needs to have edge weights in meters. OSMnx stores `length` on edges by default, so `weight='length'` in the Dijkstra call. Cutoff = `max_minutes * 80` meters (walking speed ~80m/min).

### Why it matters

Catchment / trade-area analysis is the core of what Placer.ai charges six figures for. The open-data version uses road-network walking distance. The first-party version uses actual trip data. Showing the primitive version makes the upgrade path visceral.

### Estimated effort: ~1-1.5 hours

---

## Files to modify

| File | Changes |
|---|---|
| `curbai/scoring.py` or new `curbai/temporal.py` | Add `estimate_temporal_profile()` |
| `curbai/brand_planner.py` | Add `nearest_competitors()` with haversine + cardinal direction |
| New: `curbai/catchment.py` | Walk-time Dijkstra on OSM road graph |
| `curbai/loader.py` | Add cached loader for road graph (pickle) and pois_sf |
| `scripts/build_sf.py` | Compute temporal columns, pickle the road graph, copy pois_sf to data/ |
| `app.py` | Temporal bar chart in side panel, competitor list in Brand Planner, catchment overlay layer |
| `pages/1_Methodology.py` | Add temporal + catchment + competitor docs, expand first-party upgrade table |
| `data/sf_scored.parquet` | 4 new columns: activity_morning/midday/evening/late_night |
| New: `data/road_graph.pkl` | Pickled NetworkX graph for fast boot |
| `data/pois_sf.parquet` | Copy from data/raw/ so it's available in deployed environment |

## Build and deploy sequence

1. Update `scripts/build_sf.py` — add temporal columns, pickle road graph, copy pois_sf
2. Run `python scripts/build_sf.py`
3. Implement `curbai/temporal.py`, `curbai/catchment.py`, update `curbai/brand_planner.py`
4. Update `curbai/loader.py` with new cached loaders
5. Update `app.py` — temporal bars in side panel, competitor names in Brand Planner, catchment overlay
6. Update `pages/1_Methodology.py` — new sections
7. Update `.gitignore` to allow `data/pois_sf.parquet` and `data/road_graph.pkl`
8. Boot streamlit locally, test all three enhancements
9. `git add -A && git commit && git push origin main`
10. `upload_folder` to HF Space, poll until RUNNING
11. Verify live URL

## Verification

- Click a cell on any tab → temporal bar chart appears in side panel with 4 bars
- Brand Planner: select "coffee_shop" → click a cell → see nearest competitor names + distances + directions
- Click a cell → walk-time catchment ring appears on the map (3 concentric shading levels)
- Side panel shows "X POIs within 5/10/15 min walk"
- Methodology page has updated formulas and 3 new rows in the first-party upgrade table
- HF Space health: HTTP 200
