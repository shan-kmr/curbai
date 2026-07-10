"""
Effective sidewalk width per hex — NYC planimetrics, zero new ingestion.

The geofm res9_curbside store carries per-hex sidewalk polygon AREA and
SHAPE_Leng-derived length. SHAPE_Leng on a polygon is its PERIMETER
(verified in geofm src/curbside/nyc_sidewalk.py:149), so for long thin
ribbon polygons:  width ≈ 2 * area / perimeter.

Sanity anchors: Midtown 6th Ave ≈ 4 m, residential Queens ≈ 2.8 m,
citywide median ≈ 3.1 m — consistent with NYC's 3–4 m standard.

Writes: data/nycsw_width_h3.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = (Path.home() / "Downloads/Final Semester/geofm-global/data/processed"
       / "features/res9_curbside.parquet")
OUT = Path(__file__).resolve().parents[1] / "data" / "nycsw_width_h3.parquet"

MIN_PERIM_M = 80.0   # need a real ribbon to divide by
W_LO, W_HI = 0.5, 15.0


def main() -> None:
    d = pd.read_parquet(SRC, columns=[
        "h3_index", "nycsw_sidewalk_area_m2",
        "nycsw_sidewalk_length_m", "nycsw_sidewalk_segment_count"])
    d = d[d.nycsw_sidewalk_length_m.fillna(0) > MIN_PERIM_M].copy()
    d["swd_width_eff_m"] = (2 * d.nycsw_sidewalk_area_m2
                            / d.nycsw_sidewalk_length_m).clip(W_LO, W_HI)
    out = d.rename(columns={"nycsw_sidewalk_area_m2": "swd_area_m2",
                            "nycsw_sidewalk_length_m": "swd_perimeter_m",
                            "nycsw_sidewalk_segment_count": "swd_segments"})
    out.to_parquet(OUT, index=False)
    q = out.swd_width_eff_m.quantile
    print(f"[swwidth] {len(out):,} hexes | med {q(.5):.2f} m | p10 {q(.1):.2f} | p90 {q(.9):.2f}")

    import h3
    for name, (la, lo) in {"Midtown 6th Ave": (40.758, -73.985),
                           "Residential Queens": (40.727, -73.885),
                           "FiDi narrow": (40.707, -74.011)}.items():
        hx = h3.geo_to_h3(la, lo, 9)
        r = out[out.h3_index == hx]
        print(f"[swwidth]   {name}: {r.swd_width_eff_m.iloc[0]:.1f} m" if len(r)
              else f"[swwidth]   {name}: no data")


if __name__ == "__main__":
    main()
