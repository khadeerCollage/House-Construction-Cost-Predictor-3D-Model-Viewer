"""
Layout quantities
=================
The parts of a house's cost that depend on *how the rooms are arranged* rather
than on its plinth area — what changes when an owner moves, swaps or resizes
rooms inside the same envelope:

  * internal partition walls (running metres, 115 mm brick)
  * internal doors (count)
  * window glazing (m², 1.2 m standard opening height)
  * plumbing & drainage run (m): a minimum spanning tree over the wet rooms
    (kitchen, utility, toilets) plus the outfall from the nearest wet room to the
    road-side sewer — wet rooms clustered on shared walls give short runs

CostEngine prices the difference between a plan and the design it was edited
from (`layout_state['ref_quantities']`), so moving the kitchen away from the
bathrooms shows up as extra plumbing in the estimate.
"""

import math
from typing import Dict, List, Tuple

WET = ("BATHROOM", "KITCHEN", "UTILITY")
WINDOW_HEIGHT = 1.2


def _wet_points(plan) -> List[Tuple[float, float]]:
    pts = []
    for r in plan.rooms:
        if r.category in WET and "External" not in r.name:
            pts.append((r.x + r.w / 2, r.y + r.h / 2))
    return pts


def _mst_length(pts: List[Tuple[float, float]]) -> float:
    """Prim's MST over Manhattan distances (pipes run along walls)."""
    if len(pts) < 2:
        return 0.0
    inside = {0}
    best = {i: abs(pts[i][0] - pts[0][0]) + abs(pts[i][1] - pts[0][1]) for i in range(1, len(pts))}
    total = 0.0
    while best:
        j = min(best, key=best.get)
        total += best.pop(j)
        inside.add(j)
        for i in best:
            d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
            if d < best[i]:
                best[i] = d
    return total


def layout_quantities(plan) -> Dict[str, float]:
    """Arrangement-dependent quantities of a GeneratedFloorPlan (metres / counts / m²)."""
    partition = sum(w.length for w in plan.walls
                    if not w.is_external and getattr(w, "wall_type", "") != "compound")
    doors = sum(1 for d in plan.doors if not d.is_main_entrance)
    glazing = 0.0
    for w in plan.windows:
        along = w.width if w.wall_side in ("N", "S") else max(w.width, w.height)
        glazing += along * WINDOW_HEIGHT
    pts = _wet_points(plan)
    fp = getattr(plan, "building_footprint", None) or (0.0, 0.0, plan.plot_width, plan.plot_height)
    road_y = fp[1] + fp[3]
    outfall = min((road_y - y for _, y in pts), default=0.0)
    return {"partition_m": round(partition, 2), "doors": float(doors), "glazing_m2": round(glazing, 2),
            "plumbing_m": round(_mst_length(pts) + max(0.0, outfall), 2)}
