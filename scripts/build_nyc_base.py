"""
Base-context card for the Hex Atlas — pull EVERYTHING the geofm model carries
per hex for the NYC crash cells, straight from the per-hex feature stores.
No scores; raw fields joined by h3_index.

Writes: curbai/data/nyc_base_h3.parquet
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

FE = (Path.home() / "Downloads/Final Semester/geofm-global/data/processed").as_posix()
CRASH = Path(__file__).resolve().parents[1] / "data" / "nyc_crashes_h3.parquet"
OUT = Path(__file__).resolve().parents[1] / "data" / "nyc_base_h3.parquet"


def rp(rel: str) -> str:
    return f"read_parquet('{FE}/{rel}')"


def main() -> None:
    hexes = pd.read_parquet(CRASH, columns=["h3_index"])
    con = duckdb.connect()
    con.register("hexes", hexes)
    df = con.execute(f"""
        SELECT h.h3_index,
          pop.population, kon.kontur_population, nl.nightlight_2021,
          b.building_count, b.max_height, b.avg_floors, b.total_building_area,
          poi.poi_count, poi.top_category,
          poi.has_hospital, poi.has_school, poi.has_park, poi.has_pharmacy, poi.has_worship, poi.count_transit,
          inf.dist_hospital_km, inf.dist_school_km, inf.dist_park_km, inf.dist_transit_km, inf.dist_pharmacy_km,
          rd.road_count, rd.road_primary, rd.road_secondary, rd.road_residential, rd.road_footway,
          tr.transit_stops,
          cl.annual_mean_temp, cl.annual_precipitation,
          pm.pm25_annual,
          wt.wt_visit_count, wt.wt_peak_hour, wt.wt_night_fraction, wt.wt_median_dwell_seconds, wt.wt_radius_of_gyration_km,
          cb.nycsw_sidewalk_length_m, cb.nycsw_sidewalk_present,
          cb.mly_sign_regulatory_stop_g1, cb.mly_sign_warning_pedestrians_crossing_g1,
          cb.mly_sign_regulatory_maximum_speed_limit_25_g1, cb.mot_motion_speed_mps_median,
          txt.llmgeovec_text
        FROM hexes h
        LEFT JOIN {rp('features/population_features.parquet')} pop ON pop.h3_index=h.h3_index
        LEFT JOIN {rp('features/kontur_population.parquet')} kon ON kon.h3_index=h.h3_index
        LEFT JOIN {rp('features/nightlight_features.parquet')} nl ON nl.h3_index=h.h3_index
        LEFT JOIN {rp('features/building_features.parquet')} b ON b.h3_index=h.h3_index
        LEFT JOIN {rp('features/poi_features.parquet')} poi ON poi.h3_index=h.h3_index
        LEFT JOIN {rp('features/infrastructure_distances.parquet')} inf ON inf.h3_index=h.h3_index
        LEFT JOIN {rp('features/road_features.parquet')} rd ON rd.h3_index=h.h3_index
        LEFT JOIN {rp('features/transit_features.parquet')} tr ON tr.h3_index=h.h3_index
        LEFT JOIN {rp('features/climate_features.parquet')} cl ON cl.h3_index=h.h3_index
        LEFT JOIN {rp('features/pm25_features.parquet')} pm ON pm.h3_index=h.h3_index
        LEFT JOIN {rp('features/worldtrace_features.parquet')} wt ON wt.h3_index=h.h3_index
        LEFT JOIN {rp('features/res9_curbside.parquet')} cb ON cb.h3_index=h.h3_index
        LEFT JOIN {rp('features/llmgeovec_text_nyc.parquet')} txt ON txt.h3_index=h.h3_index
    """).df()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[base] {len(df):,} hexes, {df.shape[1]} cols")
    cov = {c: int(df[c].notna().sum()) for c in df.columns if c != "h3_index"}
    print("[base] non-null coverage (of 6,511):")
    for c, n in cov.items():
        print(f"    {c:38s} {n:>6,}")


if __name__ == "__main__":
    main()
