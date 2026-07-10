"""JanusMap — zoom-reactive tiled hex map as a Streamlit custom component.

Hand-rolled (no npm build): frontend/index.html speaks the Streamlit
component protocol directly and renders deck.gl. The map keeps its own
camera across reruns (the iframe persists); Python streams it LOD chunks
on demand and receives {need|select} events back.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_frontend = Path(__file__).parent / "frontend"
_component = components.declare_component("janusmap", path=str(_frontend))


def janusmap(*, r5, chunks, layer, focus=None, height=580, key="janusmap"):
    """Render the tiled map.

    r5      columnar dict for the national res-5 grid:
            {h3: [...], lat: [...], lon: [...], val: [...] | None}
    chunks  {res(str): {res5_id: {h3:[...], val:[...],
             bc:[...], fl:[...], mh:[...](res 9 only)}}}
    layer   {col, label, fmt, vmax: {res: float}} — col None = structure mode
    focus   selected res-9 hex id (accent + card)
    Returns the last event from the frontend:
            {t:'need', res:int, parents:[res5,...]} | {t:'select', h3} | None
    """
    return _component(r5=r5, chunks=chunks, layer=layer, focus=focus,
                      height=height, key=key, default=None)
