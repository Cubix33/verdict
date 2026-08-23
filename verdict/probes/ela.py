"""Backwards-compatible shim. The real implementation lives in compression.py."""
from __future__ import annotations

import numpy as np

from .compression import ela_map, robust_region_z  # noqa: F401


def ela_residual(img_bgr, bbox=None, qualities=(90, 75, 50)) -> dict:
    """Legacy single-region ELA.

    Retained so older callers keep working, but note that a region compared
    against whole-page statistics is not a meaningful anomaly score - use
    ela_map plus robust_region_z with a text-region reference population.
    """
    m, stats = ela_map(img_bgr, qualities=qualities)
    if m is None:
        return {"ok": False, "reason": "encode_failed"}
    if bbox is None:
        return {"ok": True, "doc_mean": stats["mean"], "doc_std": stats["std"],
                "crop_mean": stats["mean"], "crop_std": stats["std"], "z_score": 0.0}
    x1, y1, x2, y2 = (int(v) for v in bbox)
    r = m[y1:y2, x1:x2]
    if r.size == 0:
        return {"ok": False, "reason": "empty_region"}
    z = (float(r.mean()) - stats["mean"]) / max(stats["std"], 1e-9)
    return {"ok": True, "doc_mean": stats["mean"], "doc_std": stats["std"],
            "crop_mean": float(r.mean()), "crop_std": float(r.std()),
            "z_score": float(z)}
