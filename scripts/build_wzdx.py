#!/usr/bin/env python
"""
Build the real-time work-zone layer for the Janus Hex Atlas from WZDx feeds.

This is the "is this closed right now" bit for robots/AVs.

Source: the WZDx (Work Zone Data Exchange) Feed Registry on USDOT datahub —
Socrata dataset 69qe-yiui:
  https://datahub.transportation.gov/resource/69qe-yiui.json
Each registry row lists a live feed URL, state/agency, spec version, and
whether an API key is required. ONLY feeds that need no key are used
(needapikey empty/false); key-required feeds are counted and listed, never
fetched. Feeds are GeoJSON FeatureCollections per the WZDx spec (v3.x has
event fields flat on properties; v4.x nests event_type/road_names under
properties.core_details). Per-feed failures (timeouts, 5xx, bad JSON) are
tolerated and recorded — the layer is built from whatever answered.

Active filter ("now-ish"): keep an event iff start_date <= now <= end_date
for whichever of the two dates parse; events with no parseable dates are
kept (they are in the live feed, which is itself a statement of currency).

Hex mapping per event (H3 res 9):
  LineString  -> buffer ~40 m to a ribbon polygon
  Point       -> buffer 60 m to a disc
  Polygon     -> as-is
Buffers are computed in a local projected frame (meters-per-degree at the
geometry's latitude — shapely buffering raw lon/lat degrees would be wrong),
then h3.polyfill at res 9 with geo_json_conformant=True (lon/lat order).
Long lines are densified and chunked (<= ~8 km) before buffering so polyfill
stays on small bounding boxes; the hex union is the same ribbon. Wherever a
polyfill comes back empty (sliver thinner than the res-9 center lattice),
we fall back to the hexes of that part's vertex points, so no event with
geometry ever maps to zero hexes.

Aggregation per res-9 hex — counts with real units, no composite scores:
  wzdx_zones        distinct active work-zone events touching the hex
  wzdx_lane_impact  events whose vehicle_impact indicates a lane closure
                    ('some-lanes-closed' or 'all-lanes-closed')
  wzdx_top_type     modal WZDx event_type string ('work-zone', 'detour', ...)
  wzdx_feeds        distinct source feeds touching the hex
  wzdx_snapshot     ISO UTC timestamp of this run (same on every row)

SNAPSHOT SEMANTICS — this layer is a point-in-time snapshot of live feeds.
The snapshot time is baked into the wzdx_snapshot column and printed at the
end of the run. RE-RUNNING THIS SCRIPT REFRESHES THE SNAPSHOT (the parquet
is overwritten with the new "now"). Nothing here is historical.

Output: data/wzdx_us_h3.parquet keyed h3_index (string); counts int32.
Cache:  data/raw/wzdx/ (gitignored via data/raw/) — registry.json, one raw
        .geojson per fetched feed, feed_status.json for the run.

Usage:
  python scripts/build_wzdx.py                      # full build
  python scripts/build_wzdx.py --limit-feeds 3      # smoke test
  python scripts/build_wzdx.py --out /tmp/wz.parquet
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import mapping, shape
from shapely.ops import transform

import h3

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "wzdx"
OUT_DEFAULT = ROOT / "data" / "wzdx_us_h3.parquet"

REGISTRY_URL = "https://datahub.transportation.gov/resource/69qe-yiui.json?$limit=1000"
UA = {"User-Agent": "JanusHexAtlas/1.0 (wzdx snapshot builder)"}

H3_RES = 9
LINE_BUFFER_M = 40.0
POINT_BUFFER_M = 60.0
CHUNK_M = 8_000.0        # max chunk length for line ribbons before polyfill
DENSIFY_M = 2_000.0      # max vertex spacing after densification
TIMEOUT_S = 30
RETRIES = 2              # retries after the first attempt (3 tries total)
LANE_CLOSED = {"some-lanes-closed", "all-lanes-closed"}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def fetch_registry() -> list[dict]:
    r = requests.get(REGISTRY_URL, timeout=TIMEOUT_S, headers=UA)
    r.raise_for_status()
    rows = r.json()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "registry.json").write_text(json.dumps(rows, indent=1))
    return rows


def partition_feeds(rows: list[dict]):
    """Split registry rows into (open_feeds, key_required, no_url)."""
    open_feeds, key_required, no_url = [], [], []
    for r in rows:
        name = r.get("feedname") or r.get("issuingorganization") or "?"
        url = (r.get("url") or {}).get("url") if isinstance(r.get("url"), dict) else r.get("url")
        rec = {
            "feedname": name,
            "state": (r.get("state") or "n/a").strip(),
            "url": url,
            "version": r.get("version", "?"),
        }
        if r.get("needapikey") in (True, "true", "True"):
            key_required.append(rec)
        elif not url:
            no_url.append(rec)
        else:
            open_feeds.append(rec)
    return open_feeds, key_required, no_url


# --------------------------------------------------------------------------
# feed fetching
# --------------------------------------------------------------------------

def fetch_feed(feed: dict) -> dict:
    """GET one feed with retries; cache raw bytes; return feed + status/data."""
    out = dict(feed, status="failed", error=None, data=None, bytes=0)
    last_err = "unknown"
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(feed["url"], timeout=TIMEOUT_S, headers=UA)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
                last_err = "not-a-FeatureCollection"
                break
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in feed["feedname"])
            (RAW_DIR / f"{safe}.geojson").write_bytes(r.content)
            out.update(status="ok", data=data, bytes=len(r.content))
            return out
        except requests.exceptions.Timeout:
            last_err = f"timeout>{TIMEOUT_S}s"
        except requests.exceptions.HTTPError as e:
            last_err = f"http-{e.response.status_code}"
        except (requests.exceptions.RequestException, ValueError) as e:
            last_err = f"{type(e).__name__}"
        if attempt < RETRIES:
            time.sleep(2 * (attempt + 1))
    out["error"] = last_err
    return out


# --------------------------------------------------------------------------
# WZDx event field extraction (v3 flat / v4 core_details), active filter
# --------------------------------------------------------------------------

def event_fields(feature: dict) -> dict:
    p = feature.get("properties") or {}
    cd = p.get("core_details") or {}
    road_names = cd.get("road_names") or p.get("road_names") or p.get("road_name") or []
    if isinstance(road_names, str):
        road_names = [road_names]
    return {
        "event_type": (cd.get("event_type") or p.get("event_type") or "unknown"),
        "road_names": [str(x) for x in road_names],
        "vehicle_impact": p.get("vehicle_impact") or cd.get("vehicle_impact") or "unknown",
        "start_date": p.get("start_date") or cd.get("start_date"),
        "end_date": p.get("end_date") or cd.get("end_date"),
    }


def parse_dt(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_active(fields: dict, now: datetime) -> bool:
    """start <= now <= end for whichever dates parse; keep if none parse."""
    s, e = parse_dt(fields["start_date"]), parse_dt(fields["end_date"])
    if s is not None and now < s:
        return False
    if e is not None and now > e:
        return False
    return True


# --------------------------------------------------------------------------
# geometry -> res-9 hexes
# --------------------------------------------------------------------------

def _local_frame(geom):
    """Forward/inverse transforms lon/lat <-> local meters at the geometry's latitude."""
    minx, miny, maxx, maxy = geom.bounds
    lat0 = (miny + maxy) / 2.0
    m_lat = 111_320.0
    m_lon = max(m_lat * math.cos(math.radians(lat0)), 1e-6)

    def fwd(x, y, z=None):
        return (x * m_lon, y * m_lat)

    def inv(x, y, z=None):
        return (x / m_lon, y / m_lat)

    return fwd, inv


def _polyfill(poly_lonlat) -> set:
    """Polyfill one lon/lat shapely Polygon (holes included) at H3_RES."""
    try:
        return set(h3.polyfill(mapping(poly_lonlat), H3_RES, geo_json_conformant=True))
    except Exception:
        return set()


def _vertex_hexes(coords_lonlat) -> set:
    return {h3.geo_to_h3(lat, lon, H3_RES) for lon, lat in coords_lonlat}


def _line_hexes(line_lonlat, fwd, inv) -> set:
    """Densify, chunk to <= CHUNK_M, buffer LINE_BUFFER_M, polyfill each chunk.
    Empty chunk polyfills (slivers) fall back to that chunk's vertex hexes."""
    line_m = transform(fwd, line_lonlat)
    if line_m.length > DENSIFY_M:
        line_m = line_m.segmentize(DENSIFY_M)
    coords = list(line_m.coords)
    chunks, cur, acc = [], [coords[0]], 0.0
    for a, b in zip(coords, coords[1:]):
        acc += math.dist(a, b)
        cur.append(b)
        if acc >= CHUNK_M:
            chunks.append(cur)
            cur, acc = [b], 0.0
    if len(cur) > 1:
        chunks.append(cur)
    if not chunks:  # degenerate single-vertex line
        chunks = [coords + coords]
    hexes: set = set()
    from shapely.geometry import LineString as _LS

    for ch in chunks:
        ribbon = transform(inv, _LS(ch).buffer(LINE_BUFFER_M, quad_segs=4))
        got = _polyfill(ribbon)
        if not got:
            got = _vertex_hexes(transform(inv, _LS(ch)).coords)
        hexes |= got
    return hexes


def geom_to_hexes(geom_dict: dict) -> set:
    """Map one GeoJSON geometry to a set of res-9 hexes per the layer rules."""
    try:
        g = shape(geom_dict)
    except Exception:
        return set()
    if g.is_empty:
        return set()
    fwd, inv = _local_frame(g)
    hexes: set = set()

    def eat(geom):
        t = geom.geom_type
        if t == "Point":
            disc = transform(inv, transform(fwd, geom).buffer(POINT_BUFFER_M, quad_segs=4))
            got = _polyfill(disc)
            hexes.update(got or {h3.geo_to_h3(geom.y, geom.x, H3_RES)})
        elif t == "MultiPoint":
            for pt in geom.geoms:
                eat(pt)
        elif t == "LineString":
            hexes.update(_line_hexes(geom, fwd, inv))
        elif t == "MultiLineString":
            for ln in geom.geoms:
                hexes.update(_line_hexes(ln, fwd, inv))
        elif t == "Polygon":
            got = _polyfill(geom)
            hexes.update(got or _vertex_hexes(geom.exterior.coords))
        elif t == "MultiPolygon":
            for pg in geom.geoms:
                eat(pg)
        elif t == "GeometryCollection":
            for sub in geom.geoms:
                eat(sub)

    eat(g)
    return hexes


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--limit-feeds", type=int, default=0, help="debug: only first N open feeds")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    snapshot = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- registry ----------------------------------------------------------
    registry = fetch_registry()
    open_feeds, key_required, no_url = partition_feeds(registry)
    print(f"registry rows: {len(registry)}")
    print(f"  open (no key): {len(open_feeds) + len(no_url)}  "
          f"(fetchable: {len(open_feeds)}, no URL listed: {len(no_url)})")
    print(f"  key-required (SKIPPED): {len(key_required)}")
    for r in key_required:
        print(f"    skip[key] {r['state']:16s} {r['feedname']}")
    for r in no_url:
        print(f"    skip[no-url] {r['state']:16s} {r['feedname']}")
    if args.limit_feeds:
        open_feeds = open_feeds[: args.limit_feeds]

    # ---- fetch -------------------------------------------------------------
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_feed, open_feeds))
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    print(f"\nfetched OK: {len(ok)} / {len(open_feeds)}   failed: {len(failed)}")
    for r in failed:
        print(f"    FAIL {r['state']:16s} {r['feedname']:20s} {r['error']}")

    # ---- events -> hexes ---------------------------------------------------
    hex_zones = defaultdict(int)
    hex_lane = defaultdict(int)
    hex_types = defaultdict(Counter)
    hex_feeds = defaultdict(set)
    state_zones = Counter()
    feed_status = []

    total_features = total_active = no_geom = 0
    spot = None  # best all-lanes-closed example: (hexcount, road, state, feed)

    for r in ok:
        feats = r["data"].get("features") or []
        n_active = 0
        for ft in feats:
            if not isinstance(ft, dict):
                continue
            total_features += 1
            fields = event_fields(ft)
            if not is_active(fields, now):
                continue
            total_active += 1
            n_active += 1
            hexes = geom_to_hexes(ft.get("geometry") or {})
            if not hexes:
                no_geom += 1
                continue
            state_zones[r["state"]] += 1
            closed = fields["vehicle_impact"] in LANE_CLOSED
            for hx in hexes:
                hex_zones[hx] += 1
                if closed:
                    hex_lane[hx] += 1
                hex_types[hx][fields["event_type"]] += 1
                hex_feeds[hx].add(r["feedname"])
            if fields["vehicle_impact"] == "all-lanes-closed":
                road = fields["road_names"][0] if fields["road_names"] else "?"
                if spot is None or len(hexes) > spot[0]:
                    spot = (len(hexes), road, r["state"], r["feedname"])
        feed_status.append({k: r[k] for k in ("feedname", "state", "url", "version", "status", "error", "bytes")}
                           | {"features": len(feats), "active_events": n_active})

    for r in failed:
        feed_status.append({k: r[k] for k in ("feedname", "state", "url", "version", "status", "error")})
    (RAW_DIR / "feed_status.json").write_text(json.dumps(
        {"snapshot": snapshot, "feeds": feed_status}, indent=1))

    # ---- aggregate + write -------------------------------------------------
    if not hex_zones:
        print("FATAL: no hexes produced — refusing to write an empty layer.")
        return 1
    rows = [{
        "h3_index": hx,
        "wzdx_zones": hex_zones[hx],
        "wzdx_lane_impact": hex_lane[hx],
        "wzdx_top_type": hex_types[hx].most_common(1)[0][0],
        "wzdx_feeds": len(hex_feeds[hx]),
        "wzdx_snapshot": snapshot,
    } for hx in hex_zones]
    df = pd.DataFrame(rows).sort_values("h3_index").reset_index(drop=True)
    for c in ("wzdx_zones", "wzdx_lane_impact", "wzdx_feeds"):
        df[c] = df[c].astype("int32")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    # ---- verify ------------------------------------------------------------
    print(f"\n=== SNAPSHOT {snapshot} (re-run to refresh) ===")
    print(f"feeds: {len(registry)} in registry | {len(open_feeds) + len(no_url)} open (no key) | "
          f"{len(ok)} fetched OK | {len(failed) + len(no_url)} failed/unfetchable | "
          f"{len(key_required)} skipped (key)")
    print(f"events: {total_features} in feeds, {total_active} active now, "
          f"{no_geom} active-but-unmappable (null/bad geometry)")
    print(f"hexes: {len(df)} res-{H3_RES} cells | lane-closure hexes: {(df.wzdx_lane_impact > 0).sum()}")
    print("top-5 states by active zones:")
    for st, n in state_zones.most_common(5):
        print(f"    {st:16s} {n}")
    if spot:
        print(f"spot check (all-lanes-closed): road={spot[1]!r} state={spot[2]} "
              f"feed={spot[3]} -> {spot[0]} hexes")
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    print(df.dtypes.to_string())
    print(df.head(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
