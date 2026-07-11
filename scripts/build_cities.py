"""
Globalise the Hex Atlas — build a base-card parquet per deep city from the
geofm global feature stores. Same raw fields as NYC (minus crashes, which are
NYC-only). Cities come from cells.parquet's deep_city label.

Writes: curbai/data/{city}_base.parquet  for each deep city.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

FE = (Path.home() / "Downloads/Final Semester/geofm-global/data/processed").as_posix()
CELLS = f"{FE}/cells.parquet"
OUTDIR = Path(__file__).resolve().parents[1] / "data"

CITIES = ["nyc", "la", "chicago", "houston", "sf",
          "delhi", "mumbai", "bangalore", "hyderabad", "chennai"]


def rp(rel: str) -> str:
    return f"read_parquet('{FE}/{rel}')"


def main() -> None:
    con = duckdb.connect()
    for city in CITIES:
        df = con.execute(f"""
            WITH hexes AS (
              SELECT h3_index, center_lat, center_lon
              FROM read_parquet('{CELLS}') WHERE deep_city = '{city}'
            )
            SELECT h.h3_index, h.center_lat, h.center_lon,
              pop.population, kon.kontur_population, nl.nightlight_2021,
              b.building_count, b.max_height, b.avg_floors, b.total_building_area,
              poi.poi_count, poi.top_category,
              poi.has_hospital, poi.has_school, poi.has_park, poi.has_pharmacy, poi.has_worship, poi.count_transit,
              inf.dist_hospital_km, inf.dist_school_km, inf.dist_park_km, inf.dist_transit_km, inf.dist_pharmacy_km,
              rd.road_count, rd.road_primary, rd.road_secondary, rd.road_residential, rd.road_footway,
              cl.annual_mean_temp, cl.annual_precipitation,
              wt.wt_visit_count, wt.wt_peak_hour, wt.wt_night_fraction, wt.wt_radius_of_gyration_km,
              cb.nycsw_sidewalk_length_m,
              txt.llmgeovec_text
            FROM hexes h
            LEFT JOIN {rp('features/population_features.parquet')} pop ON pop.h3_index=h.h3_index
            LEFT JOIN {rp('features/kontur_population.parquet')} kon ON kon.h3_index=h.h3_index
            LEFT JOIN {rp('features/nightlight_features.parquet')} nl ON nl.h3_index=h.h3_index
            LEFT JOIN {rp('features/building_features.parquet')} b ON b.h3_index=h.h3_index
            LEFT JOIN {rp('features/poi_features.parquet')} poi ON poi.h3_index=h.h3_index
            LEFT JOIN {rp('features/infrastructure_distances.parquet')} inf ON inf.h3_index=h.h3_index
            LEFT JOIN {rp('features/road_features.parquet')} rd ON rd.h3_index=h.h3_index
            LEFT JOIN {rp('features/climate_features.parquet')} cl ON cl.h3_index=h.h3_index
            LEFT JOIN {rp('features/worldtrace_features.parquet')} wt ON wt.h3_index=h.h3_index
            LEFT JOIN {rp('features/res9_curbside.parquet')} cb ON cb.h3_index=h.h3_index
            LEFT JOIN {rp('features/llmgeovec_text.parquet')} txt ON txt.h3_index=h.h3_index
        """).df()
        out = OUTDIR / f"{city}_base.parquet"
        df.to_parquet(out, index=False)
        pop_cov = int(df["population"].notna().sum())
        txt_cov = int(df["llmgeovec_text"].notna().sum())
        print(f"[{city:9s}] {len(df):>6,} hexes | pop {pop_cov:,} | model-text {txt_cov:,} -> {out.name}")


if __name__ == "__main__":
    main()
