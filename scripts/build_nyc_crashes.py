"""
Aggregate NYC Vision Zero collisions to H3 res-9 — RAW per-hex crash detail.
No scores: just counts, by mode / severity / hour / day-of-week / factor.

Source: janus-pitch research-harness vision_zero_collisions.parquet (867k crashes,
already downloaded from NYC Open Data h9gi-nx95).
Writes: curbai/data/nyc_crashes_h3.parquet — one row per hex.
"""

from __future__ import annotations

import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd

SRC_CANDIDATES = [
    Path.home() / "Downloads/janus-pitch/research-harness/papers/nyc-risk-mobility-atlas/data/vision_zero_collisions.parquet",
    Path.home() / "Downloads/janus-pitch/research-harness/code/data_pipeline/data/cache/collisions_raw.parquet",
]
OUT = Path(__file__).resolve().parents[1] / "data" / "nyc_crashes_h3.parquet"
RES = 9

INJ_COLS = [
    "number_of_persons_injured", "number_of_persons_killed",
    "number_of_pedestrians_injured", "number_of_pedestrians_killed",
    "number_of_cyclist_injured", "number_of_cyclist_killed",
    "number_of_motorist_injured", "number_of_motorist_killed",
]


def parse_hour(t) -> int:
    try:
        return int(str(t).split(":")[0])
    except Exception:
        return -1


def main() -> None:
    src = next((p for p in SRC_CANDIDATES if p.exists()), None)
    if src is None:
        raise SystemExit(f"no collisions parquet found in {SRC_CANDIDATES}")
    print(f"[crashes] reading {src.name}")
    df = pd.read_parquet(src)

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df.latitude.between(40.4, 41.0) & df.longitude.between(-74.3, -73.6)]
    print(f"[crashes] geocoded NYC rows: {len(df):,}")

    df["h3_index"] = [h3.geo_to_h3(la, lo, RES) for la, lo in zip(df.latitude.values, df.longitude.values)]
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df["dow"] = df["crash_date"].dt.dayofweek           # 0=Mon … 6=Sun
    df["hour"] = df["crash_time"].apply(parse_hour)
    for c in INJ_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # vectorised histograms
    hour_ct = pd.crosstab(df.h3_index, df.hour).reindex(columns=range(24), fill_value=0)
    dow_ct = pd.crosstab(df.h3_index, df.dow).reindex(columns=range(7), fill_value=0)

    rows = []
    for h, g in df.groupby("h3_index"):
        hh = hour_ct.loc[h].tolist()
        dh = dow_ct.loc[h].tolist()
        fac = g["contributing_factor_vehicle_1"].dropna()
        fac = fac[~fac.isin(["Unspecified", "", "Other Vehicular"])].value_counts().head(3)
        la, lo = h3.h3_to_geo(h)
        rows.append({
            "h3_index": h,
            "crashes": int(len(g)),
            "killed": int(g.number_of_persons_killed.sum()),
            "injured": int(g.number_of_persons_injured.sum()),
            "ped_inj": int(g.number_of_pedestrians_injured.sum()),
            "ped_kill": int(g.number_of_pedestrians_killed.sum()),
            "cyc_inj": int(g.number_of_cyclist_injured.sum()),
            "cyc_kill": int(g.number_of_cyclist_killed.sum()),
            "mot_inj": int(g.number_of_motorist_injured.sum()),
            "mot_kill": int(g.number_of_motorist_killed.sum()),
            "peak_hour": int(np.argmax(hh)) if sum(hh) else -1,
            "peak_dow": int(np.argmax(dh)) if sum(dh) else -1,
            "hour_hist": json.dumps([int(x) for x in hh]),
            "dow_hist": json.dumps([int(x) for x in dh]),
            "top_factors": json.dumps([[str(k), int(v)] for k, v in fac.items()]),
            "center_lat": float(la),
            "center_lon": float(lo),
        })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[crashes] wrote {OUT.name}: {len(out):,} hexes, {out.crashes.sum():,} crashes, "
          f"{out.killed.sum():,} killed")
    top = out.nlargest(3, "crashes")[["h3_index", "crashes", "killed"]]
    print("[crashes] worst cells:\n", top.to_string(index=False))


if __name__ == "__main__":
    main()
