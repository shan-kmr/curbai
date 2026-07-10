#!/usr/bin/env python3
"""Sidewalk WIDTH / INCLINE / SMOOTHNESS attributes from OpenStreetMap for the
US, aggregated to H3 res-9 hexes.

Second pass over the tags-filtered pedestrian extracts cached by
build_osmped.py (data/raw/osm/us-<region>-ped.osm.pbf). Those files keep every
way tagged footway=sidewalk|crossing WITH its full tag set and its referenced
nodes, so geometry resolves with the same pyosmium flex_mem location index --
no new downloads.

From ways with footway=sidewalk (covers highway=footway + footway=sidewalk):
  width=*      -> meters ('2', '2.5', '2 m', '2.5m', "8'", "4'8\"", '250 cm');
                  keep 0.3..15 m
  est_width=*  -> same parse, tracked separately (lower confidence)
  incline=*    -> percent ('5%', '-5%', bare number, '5deg'->tan, '1:12');
                  'up'/'down'/'yes'/'steep' count only; '<=5%' bound counts
                  only; 'no'/'0%' -> 0; keep |pct| <= 60
  smoothness=* -> ordinal 0..7 (excellent..impassable)
  surface=*    -> already in osmped_us_h3.parquet; skipped here.

Way geometry sampled every ~50 m -> res-9 hexes (same as build_osmped);
way IDs deduped across regions (extracts overlap at borders).

Output: data/osmwidth_us_h3.parquet keyed h3_index --
  osmw_width_med_m, osmw_width_n, osmw_estwidth_med_m,
  osmw_incline_pct_med, osmw_incline_n, osmw_smooth_med, osmw_smooth_n
"""
import math
import os
import re
import sys
import time
from collections import defaultdict

import h3
import numpy as np
import pandas as pd
import osmium

BASE = '/Users/shantanukumar/Downloads/curbai'
RAW = os.path.join(BASE, 'data', 'raw', 'osm')
OUT = os.path.join(BASE, 'data', 'osmwidth_us_h3.parquet')

REGIONS = ['pacific', 'northeast', 'midwest', 'west', 'south']

H3_RES = 9
SAMPLE_M = 50.0
WIDTH_MIN, WIDTH_MAX = 0.3, 15.0          # meters; outside = nonsense
INCLINE_MAX = 60.0                        # |%|; beyond = nonsense for sidewalk
FT = 0.3048

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
# value parsers
# ----------------------------------------------------------------------------
_FT_IN = re.compile(r"^(\d+(?:\.\d+)?)\s*'\s*(?:(\d+(?:\.\d+)?)\s*\"?)?$")
_NUM_UNIT = re.compile(r'^(-?\d+(?:\.\d+)?)\s*([a-z"\']+)\.?$')
_RATIO = re.compile(r'^1\s*:\s*(\d+(?:\.\d+)?)$')

DIR_ONLY = frozenset(('up', 'down', 'up/down', 'both', 'yes', 'steep'))


def parse_length_m(raw):
    """OSM length value -> meters, or None if unparseable."""
    v = raw.strip().lower().replace('’', "'").replace('”', '"')
    if not v:
        return None
    if ',' in v and v.count(',') == 1 and '.' not in v:
        v = v.replace(',', '.')            # decimal comma
    m = _FT_IN.match(v)                    # 8'  /  4'8"  /  5' 0"
    if m:
        ft = float(m.group(1))
        if m.group(2):
            ft += float(m.group(2)) / 12.0
        return ft * FT
    try:
        return float(v)                    # bare number = meters
    except ValueError:
        pass
    m = _NUM_UNIT.match(v)
    if not m:
        return None
    x, u = float(m.group(1)), m.group(2)
    if u in ('m', 'meter', 'meters', 'metre', 'metres'):
        return x
    if u == 'cm':
        return x / 100.0
    if u == 'mm':
        return x / 1000.0
    if u in ("'", 'ft', 'feet', 'foot'):
        return x * FT
    if u in ('"', 'in', 'inch', 'inches'):
        return x * 0.0254
    return None


# incline parse outcomes
INC_DIR = 'dir'                            # tagged, direction/bound only


def parse_incline(raw):
    """OSM incline -> abs percent (float), INC_DIR (count-only), or None."""
    v = raw.strip().lower().replace('°', 'deg')
    if not v:
        return None
    if v in DIR_ONLY:
        return INC_DIR
    if v == 'no':
        return 0.0
    if v[0] in '<>':                       # bounds like <=5%, >10% : no value
        return INC_DIR
    if v.startswith('~'):                  # approximate: use the value
        v = v[1:].strip()
    if v.endswith('%'):
        try:
            return abs(float(v[:-1].strip().replace(',', '.')))
        except ValueError:
            return None
    m = _RATIO.match(v)                    # ADA-style 1:12
    if m:
        d = float(m.group(1))
        return 100.0 / d if d else None
    for suf in ('degrees', 'deg'):
        if v.endswith(suf):
            try:
                deg = float(v[:-len(suf)].strip())
            except ValueError:
                return None
            return abs(math.tan(deg * RAD) * 100.0)
    try:
        return abs(float(v))               # bare number = percent
    except ValueError:
        return None


SMOOTH = {'excellent': 0, 'good': 1, 'intermediate': 2, 'bad': 3,
          'very_bad': 4, 'horrible': 5, 'very_horrible': 6, 'impassable': 7}

# parser self-test (fail fast on regressions)
assert abs(parse_length_m("8'") - 2.4384) < 1e-9
assert abs(parse_length_m('4\'8"') - (4 + 8 / 12) * FT) < 1e-9
assert parse_length_m('2 m') == 2.0 and parse_length_m('2.5m') == 2.5
assert parse_length_m('2,5') == 2.5 and parse_length_m('250 cm') == 2.5
assert parse_length_m('3.0480') == 3.048 and parse_length_m('wide') is None
assert parse_incline('5%') == 5.0 and parse_incline('-5%') == 5.0
assert parse_incline('up') is INC_DIR and parse_incline('<=5%') is INC_DIR
assert parse_incline('no') == 0.0 and abs(parse_incline('1:12') - 8.3333) < 1e-3
assert abs(parse_incline('5deg') - 8.7489) < 1e-3 and parse_incline('x') is None


# ----------------------------------------------------------------------------
# accumulator state (global across regions; way IDs deduped)
# ----------------------------------------------------------------------------
W, E, I, S = range(4)                      # width, est_width, incline, smooth
hex_vals = defaultdict(lambda: ([], [], [], []))   # h3 -> 4 value lists
hex_incline_n = defaultdict(int)           # h3 -> ways w/ ANY incline tag
seen_ways = set()
stats = defaultdict(int)
nat_width, nat_estw, nat_incline, nat_smooth = [], [], [], []


class WidthHandler(osmium.SimpleHandler):

    def way(self, w):
        tags = w.tags
        if tags.get('footway') != 'sidewalk':
            return
        wid = w.id
        if wid in seen_ways:
            return
        seen_ways.add(wid)
        stats['ways_sidewalk'] += 1

        # ---- parse first; skip geometry for the ~99% untagged -------------
        width = estw = incline = None
        smooth = None
        v = tags.get('width')
        if v is not None:
            stats['ways_width_tagged'] += 1
            width = parse_length_m(v)
            if width is None:
                stats['ways_width_unparsed'] += 1
            elif not (WIDTH_MIN <= width <= WIDTH_MAX):
                stats['ways_width_range_discard'] += 1
                width = None
        v = tags.get('est_width')
        if v is not None:
            stats['ways_estwidth_tagged'] += 1
            estw = parse_length_m(v)
            if estw is None:
                stats['ways_estwidth_unparsed'] += 1
            elif not (WIDTH_MIN <= estw <= WIDTH_MAX):
                stats['ways_estwidth_range_discard'] += 1
                estw = None
        v = tags.get('incline')
        incline_any = False
        if v is not None:
            stats['ways_incline_tagged'] += 1
            incline = parse_incline(v)
            if incline is INC_DIR:
                stats['ways_incline_dironly'] += 1
                incline, incline_any = None, True
            elif incline is None:
                stats['ways_incline_unparsed'] += 1
            elif incline > INCLINE_MAX:
                stats['ways_incline_range_discard'] += 1
                incline = None
            else:
                stats['ways_incline_numeric'] += 1
                incline_any = True
        v = tags.get('smoothness')
        if v is not None:
            stats['ways_smooth_tagged'] += 1
            smooth = SMOOTH.get(v.strip().lower())
            if smooth is None:
                stats['ways_smooth_unmapped'] += 1

        if width is None and estw is None and smooth is None \
                and not incline_any:
            return

        # ---- geometry -> res-9 hexes (sample every <=50m) ------------------
        coords = []
        for nd in w.nodes:
            loc = nd.location
            if loc.valid():
                coords.append((loc.lat, loc.lon))
        if not coords:
            stats['ways_tagged_no_coords'] += 1
            return
        hexes = set()
        if len(coords) == 1:
            hexes.add(geo_to_h3(coords[0][0], coords[0][1], H3_RES))
        else:
            lat1, lon1 = coords[0]
            for lat2, lon2 in coords[1:]:
                d = seg_len_m(lat1, lon1, lat2, lon2)
                nsmp = int(d // SAMPLE_M) + 1
                inv = 1.0 / nsmp
                for i in range(nsmp):
                    t = (i + 0.5) * inv
                    hexes.add(geo_to_h3(lat1 + (lat2 - lat1) * t,
                                        lon1 + (lon2 - lon1) * t, H3_RES))
                lat1, lon1 = lat2, lon2

        for hx in hexes:
            row = hex_vals[hx]
            if width is not None:
                row[W].append(width)
            if estw is not None:
                row[E].append(estw)
            if incline is not None:
                row[I].append(incline)
            if incline_any:
                hex_incline_n[hx] += 1
            if smooth is not None:
                row[S].append(smooth)
        if width is not None:
            nat_width.append(width)
        if estw is not None:
            nat_estw.append(estw)
        if incline is not None:
            nat_incline.append(incline)
        if smooth is not None:
            nat_smooth.append(smooth)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    t_start = time.time()
    for name in REGIONS:
        ped = os.path.join(RAW, f'us-{name}-ped.osm.pbf')
        if not os.path.exists(ped):
            raise SystemExit(f'missing cached extract: {ped}')
        t0 = time.time()
        WidthHandler().apply_file(ped, locations=True, idx='flex_mem')
        log(f'processed us-{name} ({os.path.getsize(ped) / 1e6:.1f} MB) '
            f'in {time.time() - t0:.0f}s | sidewalk_ways={stats["ways_sidewalk"]:,} '
            f'width={stats["ways_width_tagged"]:,} est={stats["ways_estwidth_tagged"]:,} '
            f'incline={stats["ways_incline_tagged"]:,} smooth={stats["ways_smooth_tagged"]:,} '
            f'hexes={len(hex_vals):,}')

    # ---- finalize -----------------------------------------------------------
    log('building dataframe ...')
    keys = sorted(set(hex_vals) | set(hex_incline_n))
    n = len(keys)
    med = np.median
    wm = np.full(n, np.nan, np.float32); wn = np.zeros(n, np.int32)
    em = np.full(n, np.nan, np.float32)
    im = np.full(n, np.nan, np.float32); inn = np.zeros(n, np.int32)
    sm = np.full(n, np.nan, np.float32); sn = np.zeros(n, np.int32)
    for j, k in enumerate(keys):
        vw, ve, vi, vs = hex_vals.get(k, ((), (), (), ()))
        if vw:
            wm[j] = med(vw); wn[j] = len(vw)
        if ve:
            em[j] = med(ve)
        if vi:
            im[j] = med(vi)
        inn[j] = hex_incline_n.get(k, 0)
        if vs:
            sm[j] = med(vs); sn[j] = len(vs)

    df = pd.DataFrame({'h3_index': pd.array(keys, dtype='string'),
                       'osmw_width_med_m': np.round(wm, 2),
                       'osmw_width_n': wn,
                       'osmw_estwidth_med_m': np.round(em, 2),
                       'osmw_incline_pct_med': np.round(im, 2),
                       'osmw_incline_n': inn,
                       'osmw_smooth_med': np.round(sm, 1),
                       'osmw_smooth_n': sn})
    df.to_parquet(OUT, index=False)
    log(f'wrote {OUT} rows={n:,} size={os.path.getsize(OUT) / 1e3:.0f} KB')

    # ---- verification -------------------------------------------------------
    total_sw = stats['ways_sidewalk']
    print('\n=== VERIFY: national way counts (deduped) ===', flush=True)
    print(f'  sidewalk ways total        {total_sw:,}')
    for key, label in (('ways_width_tagged', 'width tagged'),
                       ('ways_estwidth_tagged', 'est_width tagged'),
                       ('ways_incline_tagged', 'incline tagged'),
                       ('ways_smooth_tagged', 'smoothness tagged')):
        c = stats[key]
        print(f'  {label:26s} {c:>8,}  ({100.0 * c / total_sw:.2f}% of sidewalk ways)')
    print(f'  width usable {len(nat_width):,} (unparsed {stats["ways_width_unparsed"]:,}, '
          f'out-of-range {stats["ways_width_range_discard"]:,}) | '
          f'est usable {len(nat_estw):,} | '
          f'incline numeric {stats["ways_incline_numeric"]:,} / dir-only '
          f'{stats["ways_incline_dironly"]:,} / unparsed {stats["ways_incline_unparsed"]:,} '
          f'/ out-of-range {stats["ways_incline_range_discard"]:,} | '
          f'smooth usable {len(nat_smooth):,} (unmapped {stats["ways_smooth_unmapped"]:,}) | '
          f'tagged-but-no-coords {stats["ways_tagged_no_coords"]:,}')

    print('\n=== VERIFY: national value distributions (way-level) ===', flush=True)
    def q(a, p):
        return float(np.percentile(a, p))
    if nat_width:
        print(f'  width_m   n={len(nat_width):,}  p10={q(nat_width,10):.2f} '
              f'med={q(nat_width,50):.2f} p90={q(nat_width,90):.2f}')
    if nat_estw:
        print(f'  est_w_m   n={len(nat_estw):,}  med={q(nat_estw,50):.2f}')
    if nat_incline:
        print(f'  incline%  n={len(nat_incline):,}  med={q(nat_incline,50):.2f} '
              f'p90={q(nat_incline,90):.2f}')
    if nat_smooth:
        print(f'  smooth    n={len(nat_smooth):,}  med={q(nat_smooth,50):.1f}')

    print('\n=== VERIFY: hex coverage per column ===', flush=True)
    print(f'  rows (hexes with any attr)     {n:,}')
    print(f'  osmw_width_med_m / width_n>0   {int((df.osmw_width_n > 0).sum()):,}')
    print(f'  osmw_estwidth_med_m notna      {int(df.osmw_estwidth_med_m.notna().sum()):,}')
    print(f'  osmw_incline_n > 0             {int((df.osmw_incline_n > 0).sum()):,}')
    print(f'  osmw_incline_pct_med notna     {int(df.osmw_incline_pct_med.notna().sum()):,}')
    print(f'  osmw_smooth_n > 0              {int((df.osmw_smooth_n > 0).sum()):,}')

    print('\n=== VERIFY: top-5 metros by width-tagged hex count (res-4 parents) ===',
          flush=True)
    wdf = df[df.osmw_width_n > 0]
    parents = defaultdict(list)
    for hx in wdf.h3_index:
        parents[h3.h3_to_parent(hx, 4)].append(hx)
    top = sorted(parents.items(), key=lambda kv: -len(kv[1]))[:5]
    for p, members in top:
        plat, plon = h3.h3_to_geo(p)
        slat, slon = h3.h3_to_geo(members[0])
        print(f'  res4 {p}: {len(members):,} width hexes | center '
              f'({plat:.3f},{plon:.3f}) | sample hex ({slat:.4f},{slon:.4f})')

    print('\n=== VERIFY: Seattle spot check (47.61,-122.33) k-ring 2 ===', flush=True)
    dfx = df.set_index('h3_index')
    hx = geo_to_h3(47.61, -122.33, H3_RES)
    ring = [h for h in h3.k_ring(hx, 2) if h in dfx.index]
    print(f'  center hex {hx} | {len(ring)}/19 ring hexes have data')
    if ring:
        sub = dfx.loc[ring]
        wsub = sub[sub.osmw_width_n > 0]
        print(f'  width: {int(sub.osmw_width_n.sum())} tagged ways across '
              f'{len(wsub)} hexes'
              + (f', med-of-meds {float(wsub.osmw_width_med_m.median()):.2f} m'
                 if len(wsub) else ''))
        print(f'  incline: n={int(sub.osmw_incline_n.sum())}, hexes w/ numeric med '
              f'{int(sub.osmw_incline_pct_med.notna().sum())} | smooth: '
              f'n={int(sub.osmw_smooth_n.sum())} across '
              f'{int((sub.osmw_smooth_n > 0).sum())} hexes')

    log(f'ALL DONE in {(time.time() - t_start) / 60:.1f} min')


if __name__ == '__main__':
    sys.exit(main())
