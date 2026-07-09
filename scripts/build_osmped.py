#!/usr/bin/env python3
"""Build the global-schema pedestrian/curb attribute layer from OpenStreetMap
for the continental US, aggregated to H3 res-9 hexes.

Pipeline, per Geofabrik US regional extract (west/midwest/northeast/south/pacific):
  1. wait for / perform download (a sibling download.sh prefetches sequentially)
  2. osmium tags-filter -> small us-<region>-ped.osm.pbf
       nodes: kerb=*, tactile_paving=yes, highway=crossing
       ways : footway=sidewalk|crossing  (referenced nodes kept -> geometry)
  3. delete the raw multi-GB pbf (disk is tight); filtered pbf kept as cache
  4. pyosmium pass (locations via flex_mem index):
       nodes -> kerb classes, tactile, crossing classification -> hex counts
       ways  -> sample geometry every ~50m -> per-hex sidewalk segs + length,
                crossing-way count, surface votes, rough (smoothness) segs
  5. global dedup of node/way IDs across regions (extracts overlap at borders)

Output: data/osmped_us_h3.parquet keyed h3_index (res 9), int32 counts.
"""
import math
import os
import subprocess
import sys
import time
from collections import defaultdict

import h3
import numpy as np
import pandas as pd
import osmium

BASE = '/Users/shantanukumar/Downloads/curbai'
RAW = os.path.join(BASE, 'data', 'raw', 'osm')
OUT = os.path.join(BASE, 'data', 'osmped_us_h3.parquet')
OSMIUM_BIN = '/opt/homebrew/bin/osmium'
GEOFABRIK = 'https://download.geofabrik.de/north-america/us-%s-latest.osm.pbf'

REGIONS = [            # (name, expected bytes as of 2026-07)
    ('pacific', 170259654),
    ('northeast', 1782792205),
    ('midwest', 2464453189),
    ('west', 3364754757),
    ('south', 4081072910),
]

H3_RES = 9
SAMPLE_M = 50.0
ROUGH = frozenset(('bad', 'very_bad', 'horrible', 'very_horrible', 'impassable'))
DELETE_RAW = os.environ.get('KEEP_RAW', '0') != '1'

geo_to_h3 = h3.geo_to_h3
RAD = math.pi / 180.0
R_EARTH = 6371008.8


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def seg_len_m(lat1, lon1, lat2, lon2):
    """Equirectangular approx; plenty accurate for <1km OSM segments."""
    x = (lon2 - lon1) * RAD * math.cos((lat1 + lat2) * 0.5 * RAD)
    y = (lat2 - lat1) * RAD
    return math.hypot(x, y) * R_EARTH


# ----------------------------------------------------------------------------
# accumulator state (global across regions; IDs deduped)
# ----------------------------------------------------------------------------
(K_RAISED, K_LOWERED, K_FLUSH, TACTILE, C_SIG, C_MARK, C_UNMARK,
 SW_SEGS, CW_COUNT, ROUGH_SEGS) = range(10)
NMETRIC = 10

agg_counts = defaultdict(lambda: [0] * NMETRIC)          # h3 -> int counters
agg_len = defaultdict(float)                             # h3 -> sidewalk len m
surface_votes = defaultdict(lambda: defaultdict(int))    # h3 -> {surface: n}
seen_nodes = set()
seen_ways = set()
stats = defaultdict(int)


class PedHandler(osmium.SimpleHandler):

    def node(self, n):
        tags = n.tags
        if not tags:                       # untagged nodes referenced by ways
            return
        kerb = tags.get('kerb')
        tact = tags.get('tactile_paving')
        is_cross = tags.get('highway') == 'crossing'
        if kerb is None and tact != 'yes' and not is_cross:
            return
        nid = n.id
        if nid in seen_nodes:
            return
        seen_nodes.add(nid)
        loc = n.location
        hx = geo_to_h3(loc.lat, loc.lon, H3_RES)
        row = agg_counts[hx]
        if kerb is not None:
            stats['nodes_kerb_any'] += 1
            if kerb == 'raised':
                row[K_RAISED] += 1
            elif kerb == 'lowered':
                row[K_LOWERED] += 1
            elif kerb == 'flush':
                row[K_FLUSH] += 1
            else:
                stats['nodes_kerb_other'] += 1     # no/rolled/yes/unknown ...
        if tact == 'yes':
            row[TACTILE] += 1
        if is_cross:
            stats['nodes_crossing'] += 1
            cr = tags.get('crossing')
            if cr == 'traffic_signals' or tags.get('crossing:signals') == 'yes':
                row[C_SIG] += 1
            elif cr in ('marked', 'zebra'):
                row[C_MARK] += 1
            elif cr in ('unmarked', 'uncontrolled'):
                row[C_UNMARK] += 1
            else:
                stats['nodes_crossing_unclassified'] += 1

    def way(self, w):
        tags = w.tags
        fw = tags.get('footway')
        if fw != 'sidewalk' and fw != 'crossing':
            return
        wid = w.id
        if wid in seen_ways:
            return
        seen_ways.add(wid)
        coords = []
        for nd in w.nodes:
            loc = nd.location
            if loc.valid():
                coords.append((loc.lat, loc.lon))
        if not coords:
            stats['ways_no_coords'] += 1
            return
        is_sw = fw == 'sidewalk'
        hexes = set()
        if len(coords) == 1:
            hexes.add(geo_to_h3(coords[0][0], coords[0][1], H3_RES))
        else:
            lat1, lon1 = coords[0]
            for lat2, lon2 in coords[1:]:
                d = seg_len_m(lat1, lon1, lat2, lon2)
                nsmp = int(d // SAMPLE_M) + 1          # samples every <=50m
                inv = 1.0 / nsmp
                dl = d * inv
                for i in range(nsmp):
                    t = (i + 0.5) * inv
                    hx = geo_to_h3(lat1 + (lat2 - lat1) * t,
                                   lon1 + (lon2 - lon1) * t, H3_RES)
                    hexes.add(hx)
                    if is_sw:
                        agg_len[hx] += dl
                lat1, lon1 = lat2, lon2
        if is_sw:
            stats['ways_sidewalk'] += 1
            for hx in hexes:
                agg_counts[hx][SW_SEGS] += 1
        else:
            stats['ways_crossing'] += 1
            for hx in hexes:
                agg_counts[hx][CW_COUNT] += 1
        surf = tags.get('surface')
        if surf:
            for hx in hexes:
                surface_votes[hx][surf] += 1
        if tags.get('smoothness') in ROUGH:
            for hx in hexes:
                agg_counts[hx][ROUGH_SEGS] += 1


# ----------------------------------------------------------------------------
# download / filter orchestration
# ----------------------------------------------------------------------------

def head_size(url):
    """Return (final_http_code, content_length) following redirects."""
    try:
        out = subprocess.run(['curl', '-sIL', '--max-time', '25', url],
                             capture_output=True, text=True,
                             timeout=40).stdout
    except Exception:
        return None, None
    code = clen = None
    for line in out.splitlines():
        l = line.strip()
        if l.upper().startswith('HTTP/'):
            try:
                code = int(l.split()[1])
                clen = None            # new response resets length
            except (IndexError, ValueError):
                pass
        elif l.lower().startswith('content-length:'):
            try:
                clen = int(l.split(':', 1)[1])
            except ValueError:
                pass
    return code, clen


def downloader_alive():
    pidf = os.path.join(RAW, 'download.pid')
    try:
        pid = int(open(pidf).read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def ensure_pbf(name, size):
    f = os.path.join(RAW, f'us-{name}-latest.osm.pbf')
    url = GEOFABRIK % name
    deadline = time.time() + 90 * 60
    waiting_logged = False
    while time.time() < deadline:
        if os.path.exists(f) and os.path.getsize(f) >= size * 0.98:
            return f
        if not downloader_alive():
            break
        if not waiting_logged:
            log(f'waiting for downloader: us-{name}')
            waiting_logged = True
        time.sleep(10)
    if os.path.exists(f) and os.path.getsize(f) >= size * 0.98:
        return f

    # Fallback: fetch directly. Robust against Geofabrik 5xx outages (their
    # daily update window) and against "latest" rolling over to a new build
    # mid-download (which would make byte-resume corrupt).
    log(f'downloader gone; fetching {url} directly')
    part = f + '.part'
    expected = size
    for attempt in range(1, 46):      # patient: survives ~45min 5xx outage
        code, clen = head_size(url)
        if code == 200 and clen:
            if clen != expected:
                log(f'us-{name}: server size {clen:,} != expected '
                    f'{expected:,} (daily rollover); restarting download')
                expected = clen
                if os.path.exists(part):
                    os.remove(part)
        elif code is None or code >= 500:
            log(f'us-{name}: geofabrik unavailable (HTTP {code}), '
                f'attempt {attempt}, waiting 60s')
            time.sleep(60)
            continue
        rc = subprocess.run(['curl', '-fL', '-sS', '-C', '-', '--retry', '5',
                             '--retry-delay', '15', '--retry-all-errors',
                             '-o', part, url]).returncode
        # accept only a fully completed transfer (curl -f rc==0)
        if rc == 0 and os.path.exists(part):
            got = os.path.getsize(part)
            if got >= expected * 0.98:
                os.replace(part, f)
                return f
            log(f'us-{name}: curl rc=0 but size {got:,} < expected '
                f'{expected:,}; restarting')
            os.remove(part)
        else:
            log(f'us-{name}: curl rc={rc}; will retry (attempt {attempt})')
            time.sleep(30)
    raise RuntimeError(f'download failed for {name} after retries')


def filter_pbf(name, size):
    ped = os.path.join(RAW, f'us-{name}-ped.osm.pbf')
    done = ped + '.done'
    if os.path.exists(done) and os.path.exists(ped):
        log(f'{os.path.basename(ped)}: cached '
            f'({os.path.getsize(ped) / 1e6:.1f} MB)')
        return ped
    raw = ensure_pbf(name, size)
    t0 = time.time()
    cmd = [OSMIUM_BIN, 'tags-filter', '--overwrite', '-o', ped, raw,
           'n/kerb', 'n/tactile_paving=yes', 'n/highway=crossing',
           'w/footway=sidewalk,crossing']
    log(f'tags-filter us-{name} ({os.path.getsize(raw) / 1e9:.2f} GB) ...')
    subprocess.run(cmd, check=True)
    with open(done, 'w') as fh:
        fh.write('ok\n')
    log(f'tags-filter us-{name} done in {time.time() - t0:.0f}s -> '
        f'{os.path.getsize(ped) / 1e6:.1f} MB')
    if DELETE_RAW:
        os.remove(raw)
        log(f'deleted raw us-{name} (disk headroom)')
    return ped


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    t_start = time.time()
    os.makedirs(RAW, exist_ok=True)
    for name, size in REGIONS:
        ped = filter_pbf(name, size)
        t0 = time.time()
        handler = PedHandler()
        handler.apply_file(ped, locations=True, idx='flex_mem')
        log(f'processed us-{name} in {time.time() - t0:.0f}s | '
            f'hexes={len(agg_counts):,} sidewalk_ways={stats["ways_sidewalk"]:,} '
            f'crossing_ways={stats["ways_crossing"]:,} '
            f'crossing_nodes={stats["nodes_crossing"]:,}')

    # ---- finalize ---------------------------------------------------------
    log('building dataframe ...')
    keys = list(agg_counts.keys())
    n = len(keys)
    mat = np.zeros((n, NMETRIC), dtype=np.int32)
    lens = np.zeros(n, dtype=np.float64)
    surfs = [None] * n
    get_len = agg_len.get
    get_surf = surface_votes.get
    for i, k in enumerate(keys):
        mat[i] = agg_counts[k]
        L = get_len(k)
        if L:
            lens[i] = L
        sv = get_surf(k)
        if sv:
            surfs[i] = min(sv.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    df = pd.DataFrame({'h3_index': pd.array(keys, dtype='string')})
    df['osm_kerb_raised'] = mat[:, K_RAISED]
    df['osm_kerb_lowered'] = mat[:, K_LOWERED]
    df['osm_kerb_flush'] = mat[:, K_FLUSH]
    df['osm_tactile'] = mat[:, TACTILE]
    df['osm_cross_signalized'] = mat[:, C_SIG]
    df['osm_cross_marked'] = mat[:, C_MARK]
    df['osm_cross_unmarked'] = mat[:, C_UNMARK]
    df['osm_sidewalk_segs'] = mat[:, SW_SEGS]
    df['osm_sidewalk_len_m'] = np.round(lens, 1)
    df['osm_crossing_ways'] = mat[:, CW_COUNT]
    df['osm_surface_top'] = pd.array(surfs, dtype='string')
    df['osm_rough_segs'] = mat[:, ROUGH_SEGS]
    df = df.sort_values('h3_index', ignore_index=True)
    df.to_parquet(OUT, index=False)
    log(f'wrote {OUT} rows={n:,} size={os.path.getsize(OUT) / 1e6:.1f} MB')

    # ---- verification -----------------------------------------------------
    print('\n=== VERIFY: per-column totals ===', flush=True)
    numcols = [c for c in df.columns if c not in ('h3_index', 'osm_surface_top')]
    for c in numcols:
        v = df[c].sum()
        print(f'  {c:24s} {v:,.1f}' if c == 'osm_sidewalk_len_m'
              else f'  {c:24s} {int(v):,}')
    print(f'  hexes with any pedestrian data: {n:,}')
    print(f'  unique sidewalk ways: {stats["ways_sidewalk"]:,} | '
          f'crossing ways: {stats["ways_crossing"]:,} | '
          f'crossing nodes: {stats["nodes_crossing"]:,} '
          f'(unclassified {stats["nodes_crossing_unclassified"]:,}) | '
          f'kerb nodes: {stats["nodes_kerb_any"]:,} '
          f'(other-class {stats["nodes_kerb_other"]:,})')
    print('  top surfaces:', dict(df['osm_surface_top'].value_counts().head(5)))

    print('\n=== VERIFY: spot checks ===', flush=True)
    dfx = df.set_index('h3_index')
    for city, (lat, lon) in {'Manhattan (40.758,-73.985)': (40.758, -73.985),
                             'Seattle   (47.61,-122.33)': (47.61, -122.33)}.items():
        hx = geo_to_h3(lat, lon, H3_RES)
        print(f'  {city} hex={hx}')
        if hx in dfx.index:
            r = dfx.loc[hx]
            print(f'    exact: sig={int(r.osm_cross_signalized)} marked={int(r.osm_cross_marked)} '
                  f'unmarked={int(r.osm_cross_unmarked)} sidewalk_segs={int(r.osm_sidewalk_segs)} '
                  f'sidewalk_len_m={float(r.osm_sidewalk_len_m):.0f} kerbL={int(r.osm_kerb_lowered)} '
                  f'tactile={int(r.osm_tactile)} surface={r.osm_surface_top}')
        else:
            print('    exact hex: NO ROW')
        ring = [h for h in h3.k_ring(hx, 2) if h in dfx.index]
        if ring:
            sub = dfx.loc[ring]
            print(f'    k2 ring ({len(ring)}/19 hexes): sig={int(sub.osm_cross_signalized.sum())} '
                  f'marked={int(sub.osm_cross_marked.sum())} sidewalk_len_m={float(sub.osm_sidewalk_len_m.sum()):.0f} '
                  f'segs={int(sub.osm_sidewalk_segs.sum())}')

    log(f'ALL DONE in {(time.time() - t_start) / 60:.1f} min')


if __name__ == '__main__':
    sys.exit(main())
