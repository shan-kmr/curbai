"""FAISS-backed nearest-neighbor search over z-scored cell features."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SimilarityIndex:
    h3_index: np.ndarray  # shape (n,), dtype object (strings)
    feature_cols: list[str]
    scaler: StandardScaler
    index: faiss.Index

    def query(self, h3: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k neighbors excluding the cell itself. Distances are L2."""
        mask = self.h3_index == h3
        if not mask.any():
            return []
        row_idx = int(np.where(mask)[0][0])
        vec = self.scaler.transform(self._unscaled_for_row(row_idx))
        d, i = self.index.search(vec.astype(np.float32), k + 1)
        out = []
        for dist, idx in zip(d[0], i[0]):
            if idx == row_idx or idx < 0:
                continue
            out.append((str(self.h3_index[idx]), float(dist)))
            if len(out) == k:
                break
        return out

    _raw_matrix: np.ndarray | None = None

    def _unscaled_for_row(self, row_idx: int) -> np.ndarray:
        assert self._raw_matrix is not None
        return self._raw_matrix[row_idx : row_idx + 1]


def build_similarity(df: pd.DataFrame, feature_cols: list[str]) -> SimilarityIndex:
    raw = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw).astype(np.float32)

    index = faiss.IndexFlatL2(scaled.shape[1])
    index.add(scaled)

    sim = SimilarityIndex(
        h3_index=df["h3_index"].to_numpy(),
        feature_cols=feature_cols,
        scaler=scaler,
        index=index,
    )
    sim._raw_matrix = raw
    return sim
