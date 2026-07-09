"""
Build the traffic-exposure (AADT) layer for the Janus Hex Atlas.

AADT (Annual Average Daily Traffic) is the exposure denominator that turns crash
COUNTS into crash RATES. Source is real FHWA / state-reported government data,
served by BTS/NTAD as ArcGIS FeatureServers on services.arcgis.com/xOi1kZaI0eWDREZv.

Two stages, merged (2023 is primary; 2022 only fills hexes 2023 doesn't cover):

  STAGE A -- primary   Roadways_(HPMS_2023)   FHWA HPMS 2023 submittal (line geom)
      where AADT>0 AND F_SYSTEM IN (1,2,3,4) -> Interstate, Other Freeways/Expr,
      Other Principal Arterial, Minor Arterial. Carries the truck breakdown
      (AADT_COMBINATION + AADT_SINGLE_UNIT). This layer is only populated for ~31
      states/territories (much of the Midwest/Plains/Mountain West is absent).

  STAGE B -- fill      HPMS_FULL_US_2022_Sysnomulti_view   HPMS 2022 (line geom)
      where AADT>0 AND F_SYSTEM IN (1,2,3,4). ~233k segs, essentially the
      Interstate + principal-arterial backbone, AADT only (NO truck fields).
      Used ONLY for hexes STAGE A does not already cover, so it backfills the
      states missing from 2023 (IL, OH, MI, AZ, CO, MN, IN, MO, TN, ...).

Result: near-national AADT on the Interstate + arterial network. Every hex is
tagged hpms_source so consumers know whether truck data is real (2023) or the
cell is a 2022 backbone fill (hpms_pct_truck is NaN there).

Method (line -> hex): the server reprojects geometry to WGS84 (outSR=4326) and
lightly simplifies it; we densify each polyline to ~100 m spacing (res-9 edge is
~174 m, so no traversed hex is skipped), snap each sample to
h3.geo_to_h3(lat, lon, 9), and aggregate per hex counting each segment ONCE per
hex it touches:
    hpms_aadt_max    max segment AADT in the hex
    hpms_aadt_mean   segment-weighted mean AADT
    hpms_seg_count   # distinct HPMS segments touching the hex
    hpms_pct_truck   100 * sum(truck AADT) / sum(AADT)  (2023 hexes only; else NaN)
    hpms_fsystem_min lowest (= most major) functional class in the hex
    hpms_source      'hpms2023' (full arterial net + truck) | 'hpms2022' (fill)

Writes data/hpms_us_h3.parquet keyed by h3_index (string, res-9).

Usage:
    python scripts/build_hpms.py          # full build (both stages, merged)
    python scripts/build_hpms.py --test   # quick LA-bbox validation of stage A
    python scripts/build_hpms.py --fresh  # ignore stage checkpoints, redownload
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ORG = "https://services.arcgis.com/xOi1kZaI0eWDREZv/ArcGIS/rest/services"
SVC_2023 = f"{ORG}/Roadways_%28HPMS_2023%29/FeatureServer/0/query"
SVC_2022 = f"{ORG}/HPMS_FULL_US_2022_Sysnomulti_view/FeatureServer/0/query"
WHERE = "AADT>0 AND F_SYSTEM IN (1,2,3,4)"
PAGE = 2000
STEP_M = 100.0
RES = 9
OUTDIR = Path(__file__).resolve().parents[1] / "data"
OUT = OUTDIR / "hpms_us_h3.parquet"
SCRATCH = Path("/private/tmp/claude-501/-Users-shantanukumar-Downloads/"
               "bd4cbfb1-819b-40f9-a0a2-6d81e6038edc/scratchpad")


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def sample_hexes(paths) -> set[str]:
    """Densify every path to ~STEP_M spacing; return the set of res-9 hexes."""
    hexes: set[str] = set()
    for path in paths:
        if not path:
            continue
        pts = np.asarray(path, dtype=float)          # columns: lon, lat
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        lon, lat = pts[:, 0], pts[:, 1]
        if pts.shape[0] == 1:
            hexes.add(h3.geo_to_h3(float(lat[0]), float(lon[0]), RES))
            continue
        seg_d = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        lat_s, lon_s = [], []
        for i in range(pts.shape[0] - 1):
            n = max(1, int(seg_d[i] // STEP_M))
            fr = np.arange(n) / n
            lat_s.append(lat[i] + (lat[i + 1] - lat[i]) * fr)
            lon_s.append(lon[i] + (lon[i + 1] - lon[i]) * fr)
        lat_s.append(lat[-1:])
        lon_s.append(lon[-1:])
        for a, o in zip(np.concatenate(lat_s), np.concatenate(lon_s)):
            hexes.add(h3.geo_to_h3(float(a), float(o), RES))
    return hexes


def num(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def fetch_page(service: str, where: str, fields: str, extra=None) -> dict:
    cmd = [
        "curl", "-s", "--max-time", "120", service,
        "--data-urlencode", f"where={where}",
        "--data-urlencode", "orderByFields=OBJECTID",
        "--data-urlencode", f"outFields={fields}",
        "--data-urlencode", "outSR=4326",
        "--data-urlencode", "returnGeometry=true",
        "--data-urlencode", "maxAllowableOffset=0.0001",
        "--data-urlencode", f"resultRecordCount={PAGE}",
        "--data-urlencode", "f=json",
    ]
    if extra:
        cmd += extra
    for attempt in range(6):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=140)
            d = json.loads(r.stdout)
            if "features" in d:
                return d
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"page fetch failed after retries: {service} {where!r}")


def update_agg(agg: dict, feats: list, truck: bool) -> None:
    for feat in feats:
        g = feat.get("geometry") or {}
        paths = g.get("paths") or []
        if not paths:
            continue
        a = feat["attributes"]
        aadt = num(a.get("AADT"))
        if aadt <= 0:
            continue
        tr = (num(a.get("AADT_COMBINATION")) + num(a.get("AADT_SINGLE_UNIT"))) if truck else 0.0
        fsys = a.get("F_SYSTEM")
        fsys = int(fsys) if fsys is not None else 9
        for h in sample_hexes(paths):
            rec = agg.get(h)
            if rec is None:
                agg[h] = [aadt, aadt, tr, 1, fsys]     # max, sum, truck_sum, segs, fsys_min
            else:
                if aadt > rec[0]:
                    rec[0] = aadt
                rec[1] += aadt
                rec[2] += tr
                rec[3] += 1
                if fsys < rec[4]:
                    rec[4] = fsys


def run_stage(tag, service, fields, truck, fresh, extra=None, where=WHERE) -> dict:
    ckpt = SCRATCH / f"hpms_{tag}.pkl"
    if ckpt.exists() and not fresh:
        agg = pickle.loads(ckpt.read_bytes())
        print(f"[{tag}] loaded checkpoint: {len(agg):,} hexes", flush=True)
        return agg
    agg: dict = {}
    last_oid, page_no = 0, 0
    t0 = time.time()
    while True:
        d = fetch_page(service, f"{where} AND OBJECTID>{last_oid}", fields, extra)
        feats = d.get("features", [])
        if not feats:
            break
        update_agg(agg, feats, truck)
        last_oid = feats[-1]["attributes"]["OBJECTID"]
        page_no += 1
        if page_no % 20 == 0:
            print(f"[{tag} p{page_no}] last_oid={last_oid} hexes={len(agg):,} "
                  f"{time.time()-t0:0.0f}s", flush=True)
        if len(feats) < PAGE and not d.get("exceededTransferLimit"):
            break
    print(f"[{tag}] done: {page_no} pages, {len(agg):,} hexes, {time.time()-t0:0.0f}s",
          flush=True)
    ckpt.write_bytes(pickle.dumps(agg))
    return agg


FIELDS_2023 = "OBJECTID,AADT,AADT_COMBINATION,AADT_SINGLE_UNIT,F_SYSTEM"
FIELDS_2022 = "OBJECTID,AADT,F_SYSTEM"


def finalize(agg23: dict, agg22: dict, out: Path) -> pd.DataFrame:
    rows = []
    for h, (mx, ssum, tsum, segs, fsys) in agg23.items():
        la, lo = h3.h3_to_geo(h)
        # truck share is a % of AADT; a handful of HPMS segments report the truck
        # subfields inconsistently with mainline AADT (>100%), so cap at the valid bound.
        pct = round(min(100.0, 100.0 * tsum / ssum), 2) if ssum > 0 else np.nan
        rows.append((h, int(round(mx)), round(ssum / segs, 1), int(segs),
                     pct, int(fsys), "hpms2023", float(la), float(lo)))
    fill = 0
    for h, (mx, ssum, tsum, segs, fsys) in agg22.items():
        if h in agg23:                                # 2023 is primary; skip overlap
            continue
        fill += 1
        la, lo = h3.h3_to_geo(h)
        rows.append((h, int(round(mx)), round(ssum / segs, 1), int(segs),
                     np.nan, int(fsys), "hpms2022", float(la), float(lo)))
    df = pd.DataFrame(rows, columns=[
        "h3_index", "hpms_aadt_max", "hpms_aadt_mean", "hpms_seg_count",
        "hpms_pct_truck", "hpms_fsystem_min", "hpms_source",
        "center_lat", "center_lon"])
    df = df.sort_values("hpms_aadt_max", ascending=False).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[merge] {len(agg23):,} primary(2023) + {fill:,} fill(2022) "
          f"= {len(df):,} hexes", flush=True)
    return df


def verify(df: pd.DataFrame, out: Path) -> None:
    print("\n================= VERIFICATION =================")
    print(f"hexes with AADT : {len(df):,}   ({out.stat().st_size/1e6:0.1f} MB)")
    print(f"schema          : {list(df.columns)}")
    print(f"source mix      : {df.hpms_source.value_counts().to_dict()}")
    print(f"AADT max range  : {df.hpms_aadt_max.min():,} .. {df.hpms_aadt_max.max():,}"
          f"  (median {int(df.hpms_aadt_max.median()):,})")
    print(f"fclass mix      : {df.hpms_fsystem_min.value_counts().sort_index().to_dict()}")
    print("\nBusy-corridor spot checks (max AADT within ~2-ring of point):")
    probes = [
        ("LA  I-10 downtown",      34.0339, -118.2280),
        ("LA  I-405 Sepulveda",    34.1010, -118.4700),
        ("Houston I-45/I-610",     29.8090, -95.3790),
        ("Atlanta Connector I-75/85", 33.7710, -84.3900),
        ("Chicago Dan Ryan I-90/94",  41.8340, -87.6300),
        ("Phoenix I-10",           33.4460, -112.0670),
        ("Denver I-25",            39.7000, -104.9880),
        ("NYC Cross Bronx I-95",   40.8480, -73.9110),
    ]
    idx = {h: i for i, h in enumerate(df.h3_index.values)}
    aadt = df.hpms_aadt_max.values
    src = df.hpms_source.values
    for name, la, lo in probes:
        ring = h3.k_ring(h3.geo_to_h3(la, lo, RES), 2)
        hits = [(aadt[idx[h]], src[idx[h]]) for h in ring if h in idx]
        if hits:
            v, s = max(hits)
            print(f"  {name:28s} -> {v:>8,} vpd  [{s}]")
        else:
            print(f"  {name:28s} -> no HPMS hex nearby")
    print("================================================")


def run_test() -> None:
    env = ["--data-urlencode", "geometry=-118.7,33.7,-117.9,34.35",
           "--data-urlencode", "geometryType=esriGeometryEnvelope",
           "--data-urlencode", "inSR=4326",
           "--data-urlencode", "spatialRel=esriSpatialRelIntersects"]
    agg = run_stage("test23", SVC_2023, FIELDS_2023, True, True, extra=env)
    out = OUT.with_name("hpms_us_h3_test.parquet")
    df = finalize(agg, {}, out)
    verify(df, out)


def main() -> None:
    if "--test" in sys.argv:
        run_test()
        return
    fresh = "--fresh" in sys.argv
    agg23 = run_stage("2023", SVC_2023, FIELDS_2023, True, fresh)
    agg22 = run_stage("2022", SVC_2022, FIELDS_2022, False, fresh)
    df = finalize(agg23, agg22, OUT)
    verify(df, OUT)


if __name__ == "__main__":
    main()
