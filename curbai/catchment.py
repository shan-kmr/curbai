"""
Walk-time catchment — open-data primitive for "who can get here on foot?"

Runs a single-source Dijkstra on the OSM road network from the selected cell's
nearest node, bounded by a walking-time cutoff (default 15 min at 80 m/min =
4.8 km/h). Each reachable graph node is binned back to its H3 cell; the cell's
walk-time is the minimum time across any node inside it.

The first-party upgrade is actual origin-destination trip flows — we're
estimating who *could* walk here; a platform with device signals would know
who *does* walk here.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import pandas as pd


WALK_SPEED_M_PER_MIN: float = 80.0  # ~4.8 km/h, standard pedestrian speed
H3_RES: int = 9


def build_road_graph_from_gpkg(gpkg_path: Path | str) -> nx.Graph:
    """Build an undirected NetworkX graph from the OSMnx-exported gpkg.

    Walking is symmetric, so we collapse the OSMnx MultiDiGraph to an
    undirected simple graph keeping the minimum edge length between any pair
    of nodes. Nodes carry `y` (lat) and `x` (lon) attributes so we can snap
    lat/lon to the nearest node at query time.
    """
    nodes_gdf = gpd.read_file(gpkg_path, layer="nodes")
    edges_gdf = gpd.read_file(gpkg_path, layer="edges")

    G: nx.Graph = nx.Graph()

    for _, row in nodes_gdf.iterrows():
        node_id = int(row["osmid"])
        G.add_node(node_id, y=float(row["y"]), x=float(row["x"]))

    for u, v, length in zip(
        edges_gdf["u"].values,
        edges_gdf["v"].values,
        edges_gdf["length"].values,
    ):
        u_i = int(u)
        v_i = int(v)
        w = float(length)
        if np.isnan(w) or w <= 0:
            continue
        if G.has_edge(u_i, v_i):
            if w < G[u_i][v_i]["length"]:
                G[u_i][v_i]["length"] = w
        else:
            G.add_edge(u_i, v_i, length=w)

    return G


def _node_coords(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (node_ids, lats, lons) arrays for vectorized nearest-node lookup."""
    if "_coord_cache" in G.graph:
        return G.graph["_coord_cache"]
    ids = np.array(list(G.nodes), dtype=np.int64)
    lats = np.fromiter((G.nodes[n]["y"] for n in ids), dtype=np.float64, count=len(ids))
    lons = np.fromiter((G.nodes[n]["x"] for n in ids), dtype=np.float64, count=len(ids))
    G.graph["_coord_cache"] = (ids, lats, lons)
    return ids, lats, lons


def nearest_node(G: nx.Graph, lat: float, lon: float) -> int:
    """Pick the graph node closest to (lat, lon) using squared degree distance.

    Squared degree distance is fine for a city-sized bbox — local enough that
    we don't need haversine for selecting the nearest node.
    """
    ids, lats, lons = _node_coords(G)
    dlat = lats - lat
    dlon = lons - lon
    idx = int(np.argmin(dlat * dlat + dlon * dlon))
    return int(ids[idx])


def walk_time_cells(
    h3_id: str,
    G: nx.Graph,
    cells_df: pd.DataFrame,
    max_minutes: int = 15,
    speed_m_per_min: float = WALK_SPEED_M_PER_MIN,
) -> dict[str, float]:
    """Return {h3_index: walk_time_minutes} for every H3 cell reachable from
    the selected cell's center within `max_minutes` on foot.

    The selected cell itself always appears with walk_time == 0.
    """
    row = cells_df[cells_df["h3_index"] == h3_id]
    if len(row) == 0:
        return {}
    lat = float(row["center_lat"].iloc[0])
    lon = float(row["center_lon"].iloc[0])

    src = nearest_node(G, lat, lon)
    cutoff_m = max_minutes * speed_m_per_min

    lengths: dict[int, float] = nx.single_source_dijkstra_path_length(
        G, src, cutoff=cutoff_m, weight="length"
    )

    ids, lats, lons = _node_coords(G)
    id_to_idx = {int(nid): i for i, nid in enumerate(ids)}

    best: dict[str, float] = {h3_id: 0.0}
    for node_id, meters in lengths.items():
        i = id_to_idx.get(int(node_id))
        if i is None:
            continue
        cell = h3.geo_to_h3(float(lats[i]), float(lons[i]), H3_RES)
        minutes = meters / speed_m_per_min
        prev = best.get(cell)
        if prev is None or minutes < prev:
            best[cell] = minutes
    return best


def catchment_summary(
    walk_times: dict[str, float],
    cells_df: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Aggregate walk-time results into 5/10/15-min bucket summaries.

    Returns a dict of bucket -> {"n_cells": int, "n_pois": int}.
    """
    out: dict[str, dict[str, int]] = {}
    cells_indexed = cells_df.set_index("h3_index")
    for cutoff in (5, 10, 15):
        within = [h for h, t in walk_times.items() if t <= cutoff]
        sub = cells_indexed.reindex([h for h in within if h in cells_indexed.index])
        out[f"{cutoff}_min"] = {
            "n_cells": int(len(sub)),
            "n_pois": int(sub["poi_count"].fillna(0).sum()) if "poi_count" in sub.columns else 0,
        }
    return out


def pickle_graph(G: nx.Graph, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "_coord_cache" in G.graph:
        del G.graph["_coord_cache"]
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickled_graph(path: Path | str) -> nx.Graph:
    with Path(path).open("rb") as f:
        return pickle.load(f)
