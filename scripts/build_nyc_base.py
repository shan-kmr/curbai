"""
Base-context card for the Hex Atlas — pull population + buildings + night-lights
per NYC crash hex, straight from the geofm-global per-hex feature stores
(17.5M global res-9 hexes). No scores; raw fields joined by h3_index.

Writes: curbai/data/nyc_base_h3.parquet
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

FEAT = Path.home() / "Downloads/Final Semester/geofm-global/data/processed/features"
CRASH = Path(__file__).resolve().parents[1] / "data" / "nyc_crashes_h3.parquet"
OUT = Path(__file__).resolve().parents[1] / "data" / "nyc_base_h3.parquet"


def main() -> None:
    hexes = pd.read_parquet(CRASH, columns=["h3_index"])
    con = duckdb.connect()
    con.register("hexes", hexes)
    pop = (FEAT / "population_features.parquet").as_posix()
    bld = (FEAT / "building_features.parquet").as_posix()
    nl = (FEAT / "nightlight_features.parquet").as_posix()
    df = con.execute(f"""
        SELECT h.h3_index,
               p.population, p.pop_density,
               b.building_count, b.total_building_area, b.max_height, b.avg_floors,
               n.nightlight_2021
        FROM hexes h
        LEFT JOIN read_parquet('{pop}') p ON p.h3_index = h.h3_index
        LEFT JOIN read_parquet('{bld}') b ON b.h3_index = h.h3_index
        LEFT JOIN read_parquet('{nl}')  n ON n.h3_index = h.h3_index
    """).df()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    matched = int(df["population"].notna().sum())
    print(f"[base] {len(df):,} hexes | population matched: {matched:,} | "
          f"buildings matched: {int(df['building_count'].notna().sum()):,}")
    print(f"[base] median pop {df['population'].median():.0f} | "
          f"median buildings {df['building_count'].median():.0f} | wrote {OUT.name}")


if __name__ == "__main__":
    main()
