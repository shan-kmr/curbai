"""
Build the full United States for the Hex Atlas — two resolutions:

  data/us_r5.parquet       national OVERVIEW. One row per res-5 cell (~252 km²),
                           ~38.6k cells covering the lower-48 + AK/HI. Raw
                           metrics rolled up from the res-9 children. Renders
                           the whole country in a single pydeck payload.

  data/us_r9_base.parquet  DETAIL. Every res-9 cell (~0.105 km²) the geofm
                           stores carry inside the US, ~4.67M rows, with the
                           same raw card fields as the city atlas plus a `res5`
                           parent column so the app can pull one res-5 cell's
                           children on a click (the drill-down).

No scores — raw open-data fields joined by h3_index. Rollups are plain sums
(counts/areas) and means (intensities); nothing is blended.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

FE = (Path.home() / "Downloads/Final Semester/geofm-global/data/processed").as_posix()
CELLS = f"{FE}/cells.parquet"
OUTDIR = Path(__file__).resolve().parents[1] / "data"
BOUNDARY = OUTDIR / "us_boundary.wkt"

# Cheap bbox pre-filter (lower-48 + AK + HI) before the exact polygon clip.
BBOX = (
    "((center_lat BETWEEN 24 AND 49.5 AND center_lon BETWEEN -125 AND -66.5) "
    "OR (center_lat BETWEEN 51 AND 72 AND center_lon BETWEEN -170 AND -129) "
    "OR (center_lat BETWEEN 18 AND 23 AND center_lon BETWEEN -161 AND -154))"
)


def rp(rel: str) -> str:
    return f"read_parquet('{FE}/{rel}')"


def us_where() -> str:
    """bbox pre-filter AND exact point-in-US-polygon clip (drops Toronto,
    Monterrey, Havana, ... that fall inside the rectangle)."""
    wkt = BOUNDARY.read_text().replace("'", "''")
    return (f"{BBOX} AND ST_Contains(ST_GeomFromText('{wkt}'), "
            f"ST_Point(center_lon, center_lat))")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3;")
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def build_overview(con: duckdb.DuckDBPyConnection) -> None:
    """res-5 rollup of the whole US — the national overview layer."""
    df = con.execute(f"""
        WITH us AS (
          SELECT h3_index, h3_cell_to_parent(h3_index, 5) AS res5
          FROM read_parquet('{CELLS}') WHERE {us_where()}
        ),
        joined AS (
          SELECT us.res5,
            pop.population, kon.kontur_population, nl.nightlight_2021,
            b.building_count, b.total_building_area,
            poi.poi_count, poi.count_transit,
            rd.road_count, wt.wt_visit_count
          FROM us
          LEFT JOIN {rp('features/population_features.parquet')} pop ON pop.h3_index=us.h3_index
          LEFT JOIN {rp('features/kontur_population.parquet')} kon ON kon.h3_index=us.h3_index
          LEFT JOIN {rp('features/nightlight_features.parquet')} nl ON nl.h3_index=us.h3_index
          LEFT JOIN {rp('features/building_features.parquet')} b ON b.h3_index=us.h3_index
          LEFT JOIN {rp('features/poi_features.parquet')} poi ON poi.h3_index=us.h3_index
          LEFT JOIN {rp('features/road_features.parquet')} rd ON rd.h3_index=us.h3_index
          LEFT JOIN {rp('features/worldtrace_features.parquet')} wt ON wt.h3_index=us.h3_index
        )
        SELECT res5 AS h3_index,
          h3_cell_to_lat(res5) AS center_lat,
          h3_cell_to_lng(res5) AS center_lon,
          count(*)                        AS n_res9,
          sum(population)                 AS population,
          sum(kontur_population)          AS kontur_population,
          avg(nightlight_2021)            AS nightlight_2021,
          sum(building_count)             AS building_count,
          sum(total_building_area)        AS total_building_area,
          sum(poi_count)                  AS poi_count,
          sum(count_transit)              AS count_transit,
          sum(road_count)                 AS road_count,
          sum(wt_visit_count)             AS wt_visit_count
        FROM joined GROUP BY res5
    """).df()
    out = OUTDIR / "us_r5.parquet"
    df.to_parquet(out, index=False)
    mb = out.stat().st_size / 1e6
    print(f"[overview] {len(df):,} res-5 cells | {df.shape[1]} cols | {mb:.1f} MB -> {out.name}")
    print(f"[overview] pop total {int(df['population'].fillna(0).sum()):,} | "
          f"POIs {int(df['poi_count'].fillna(0).sum()):,} | "
          f"buildings {int(df['building_count'].fillna(0).sum()):,}")


def build_detail(con: duckdb.DuckDBPyConnection) -> None:
    """Full res-9 detail store with a res5 parent column for drill-down."""
    df = con.execute(f"""
        WITH us AS (
          SELECT h3_index, center_lat, center_lon,
                 h3_cell_to_parent(h3_index, 5) AS res5
          FROM read_parquet('{CELLS}') WHERE {us_where()}
        )
        SELECT h.h3_index, h.res5, h.center_lat, h.center_lon,
          pop.population, kon.kontur_population, nl.nightlight_2021,
          b.building_count, b.max_height, b.avg_floors, b.total_building_area,
          poi.poi_count, poi.top_category,
          poi.has_hospital, poi.has_school, poi.has_park, poi.has_pharmacy, poi.has_worship, poi.count_transit,
          inf.dist_hospital_km, inf.dist_school_km, inf.dist_park_km, inf.dist_transit_km, inf.dist_pharmacy_km,
          rd.road_count, rd.road_primary, rd.road_secondary, rd.road_residential, rd.road_footway,
          cl.annual_mean_temp, cl.annual_precipitation,
          wt.wt_visit_count, wt.wt_peak_hour, wt.wt_night_fraction, wt.wt_radius_of_gyration_km,
          txt.llmgeovec_text
        FROM us h
        LEFT JOIN {rp('features/population_features.parquet')} pop ON pop.h3_index=h.h3_index
        LEFT JOIN {rp('features/kontur_population.parquet')} kon ON kon.h3_index=h.h3_index
        LEFT JOIN {rp('features/nightlight_features.parquet')} nl ON nl.h3_index=h.h3_index
        LEFT JOIN {rp('features/building_features.parquet')} b ON b.h3_index=h.h3_index
        LEFT JOIN {rp('features/poi_features.parquet')} poi ON poi.h3_index=h.h3_index
        LEFT JOIN {rp('features/infrastructure_distances.parquet')} inf ON inf.h3_index=h.h3_index
        LEFT JOIN {rp('features/road_features.parquet')} rd ON rd.h3_index=h.h3_index
        LEFT JOIN {rp('features/climate_features.parquet')} cl ON cl.h3_index=h.h3_index
        LEFT JOIN {rp('features/worldtrace_features.parquet')} wt ON wt.h3_index=h.h3_index
        LEFT JOIN {rp('features/llmgeovec_text.parquet')} txt ON txt.h3_index=h.h3_index
    """).df()
    out = OUTDIR / "us_r9_base.parquet"
    df.to_parquet(out, index=False)
    mb = out.stat().st_size / 1e6
    print(f"[detail] {len(df):,} res-9 cells | {df.shape[1]} cols | {mb:.1f} MB -> {out.name}")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    con = connect()
    if what in ("all", "overview"):
        build_overview(con)
    if what in ("all", "detail"):
        build_detail(con)


if __name__ == "__main__":
    main()
