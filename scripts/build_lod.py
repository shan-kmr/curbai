"""
LOD ladder for the tiled Janus map — res-6/7/8 rollups of the US res-9 base,
so the map can split tiles progressively as you zoom (5 → 6 → 7 → 8 → 9).

Each file keeps a `res5` ancestor column so the app can fetch only the cells
under the viewport's res-5 parents (parquet predicate pushdown).

Writes: data/us_lod/r6.parquet, r7.parquet, r8.parquet
"""

from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
R9_GLOB = str(ROOT / "data" / "us_r9" / "*.parquet")
OUTDIR = ROOT / "data" / "us_lod"

# metrics carried up the ladder (sums for counts, mean for nightlight)
SUMS = ["kontur_population", "poi_count", "building_count",
        "wt_visit_count", "road_count"]
MEANS = ["nightlight_2021"]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3;")
    agg = ", ".join([f"sum({c}) AS {c}" for c in SUMS] +
                    [f"avg({c}) AS {c}" for c in MEANS])
    for res in (6, 7, 8):
        out = OUTDIR / f"r{res}.parquet"
        con.execute(f"""
            COPY (
              SELECT h3_cell_to_parent(h3_index, {res}) AS h3_index,
                     any_value(res5)                    AS res5,
                     count(*)                           AS n_res9,
                     {agg}
              FROM read_parquet('{R9_GLOB}')
              GROUP BY h3_cell_to_parent(h3_index, {res})
              ORDER BY res5
            ) TO '{out}' (FORMAT parquet, ROW_GROUP_SIZE 120000)
        """)
        n, mb = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0], out.stat().st_size / 1e6
        print(f"[lod] r{res}: {n:>9,} cells  {mb:5.1f} MB -> {out.name}")


if __name__ == "__main__":
    main()
