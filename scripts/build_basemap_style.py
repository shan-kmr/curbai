"""
The Janus basemap style — premium white, no ink.

Takes OpenFreeMap's positron (keyless, global, openmaptiles schema) and
recolors every paint property into the house ground palette: paper land,
muted cool-gray water, whisper roads, low-contrast gray labels. Explicit
design rule (2026-07-10): NO cobalt in the cartography — the map recedes,
data speaks, cobalt stays in the UI chrome only.

Writes: curbai/janusmap/frontend/basemap-style.json  (served with the
component; glyphs/sprites stay on OpenFreeMap's CDN, keyless).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

SRC = "https://tiles.openfreemap.org/styles/positron"
CACHE = Path("/tmp/ofm_positron.json")
OUT = Path(__file__).resolve().parents[1] / "curbai" / "janusmap" / "frontend" / "basemap-style.json"

# ---- the ground palette (neutral; NOT cobalt) -------------------------------
PAPER = "#FFFFFF"        # land — premium white (Graphite & Signal, 2026-07-10)
PAPER2 = "#FAFAF9"       # residential / landuse wash
GREEN = "#F0F2EE"        # parks — barely-there
WATER = "#EBEDEF"        # very light cool gray
ROAD = "#FFFFFF"         # road fill
ROAD_CASE = "#E9EAEC"    # road casings — hairlines
ROAD_MINOR = "#FCFCFC"
RAIL = "#E7E8EA"
BOUNDARY = "#DCDDE0"
BUILDING = "#F4F4F3"     # building fill
BUILDING_LINE = "#E9EAEC"
LABEL = "#7A7F85"        # muted gray text
LABEL_BIG = "#4A4E54"    # cities
HALO = PAPER


def paint_all(layer: dict, updates: dict) -> None:
    layer.setdefault("paint", {}).update(updates)


def main() -> None:
    if not CACHE.exists():
        subprocess.run(["curl", "-fsSL", SRC, "-o", str(CACHE)], check=True)
    style = json.loads(CACHE.read_text())
    style["name"] = "janus-paper"

    for lyr in style["layers"]:
        lid, ltype = lyr["id"], lyr["type"]
        paint = lyr.get("paint", {})

        if ltype == "background":
            paint_all(lyr, {"background-color": PAPER})

        elif ltype == "fill":
            if "water" in lid:
                paint_all(lyr, {"fill-color": WATER})
            elif "park" in lid or "wood" in lid or "grass" in lid:
                paint_all(lyr, {"fill-color": GREEN, "fill-opacity": 0.8})
            elif "residential" in lid or "landuse" in lid:
                paint_all(lyr, {"fill-color": PAPER2})
            elif "ice" in lid or "glacier" in lid:
                paint_all(lyr, {"fill-color": PAPER2})
            elif "building" in lid:
                paint_all(lyr, {"fill-color": BUILDING,
                                "fill-outline-color": BUILDING_LINE})
            elif "aeroway" in lid or "pier" in lid:
                paint_all(lyr, {"fill-color": ROAD_MINOR})
            else:
                paint_all(lyr, {"fill-color": PAPER2})

        elif ltype == "line":
            if "casing" in lid:
                paint_all(lyr, {"line-color": ROAD_CASE})
            elif "motorway" in lid or "trunk" in lid or "primary" in lid:
                paint_all(lyr, {"line-color": ROAD})
            elif "rail" in lid or "transit" in lid:
                paint_all(lyr, {"line-color": RAIL})
            elif "boundary" in lid or "admin" in lid:
                paint_all(lyr, {"line-color": BOUNDARY})
            elif "waterway" in lid or "water" in lid:
                paint_all(lyr, {"line-color": WATER})
            elif "bridge" in lid or "tunnel" in lid or "highway" in lid \
                    or "road" in lid or "street" in lid or "path" in lid \
                    or "aeroway" in lid or "pier" in lid:
                paint_all(lyr, {"line-color": ROAD_MINOR
                                if "path" in lid or "minor" in lid else ROAD})
            else:
                paint_all(lyr, {"line-color": ROAD_CASE})

        elif ltype == "symbol":
            big = ("place_label" in lid or "city" in lid or "state" in lid
                   or "country" in lid)
            paint_all(lyr, {"text-color": LABEL_BIG if big else LABEL,
                            "text-halo-color": HALO, "text-halo-width": 1.1})

    # the ne2 shaded-relief raster fights the paper look — drop it
    style["layers"] = [l for l in style["layers"] if l.get("source") != "ne2_shaded"]
    style["sources"].pop("ne2_shaded", None)

    OUT.write_text(json.dumps(style, separators=(",", ":")))
    print(f"[style] {len(style['layers'])} layers -> {OUT.relative_to(OUT.parents[3])} "
          f"({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
