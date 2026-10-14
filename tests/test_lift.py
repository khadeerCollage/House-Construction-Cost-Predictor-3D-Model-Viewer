"""
Where the lift stands, and the working lift in 3D.

Covers: 'inside' stays the default and plans without a choice are unchanged; a tower beside the main
door (front yard, its door facing the entrance landing, 230 mm walls, no window behind it); a tower
against a side wall with a side door into a living space only (never a bedroom, bath or pooja) or a
verandah to the main door; what fits is decided before Generate (site_feasibility.lift_options) and
the engine falls back (side → beside the main door → inside) with a note; apartments keep the common
core; a re-plan keeps the lift where it is; and the 3D scene's lift — stops on every floor (and the
stilt), the entrance porch and the upper-floor landings reach its door, and you can walk to it.
"""

import base64
import os
import sys
import unittest
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_generator.site_feasibility import lift_options  # noqa: E402
from floorplan_generator.space_program import lift_car  # noqa: E402
from floorplan_renderer import walkthrough as wt  # noqa: E402

APT = "Apartment / Multi-Storey Building"
NO_ENTRY = {"BEDROOM", "BATHROOM", "POOJA", "STORE", "STAIRS", "LIFT"}
_CACHE = {}


def _plan(w, h, bhk, facing, floors, pos=None, comps=None, btype=None):
    key = (w, h, bhk, facing, floors, pos, str(comps), btype)
    if key not in _CACHE:
        kw = dict(num_floors=floors, has_lift=True, additional_components=comps or {})
        if pos:
            kw["lift_position"] = pos
        if btype:
            kw["building_type"] = btype
        _CACHE[key] = LayoutEngine(w, h, bhk, True, facing, **kw).generate()
    return _CACHE[key]


def _tower(plan):
    return next((r for r in plan.rooms if r.category == "LIFT" and "External" in r.name), None)


def _grid(nav):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(nav["bits"]), dtype=np.uint8))[: nav["nx"] * nav["ny"]]
    return bits.reshape(nav["ny"], nav["nx"]).astype(bool)


def _cell(nav, x, y):
    return int((x - nav["x0"]) / nav["cell"]), int((y - nav["y0"]) / nav["cell"])


def _near_free(nav, g, x, y, r=0.6):
    i0, j0 = _cell(nav, x, y)
    k = int(r / nav["cell"])
    best = None
    for dj in range(-k, k + 1):
        for di in range(-k, k + 1):
            i, j = i0 + di, j0 + dj
            if 0 <= i < nav["nx"] and 0 <= j < nav["ny"] and g[j, i]:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, (i, j))
    return best[1] if best else None


def _reach(nav, a, b) -> bool:
    """Can the walker go from plan point a to plan point b on this grid?"""
    g = _grid(nav)
    s, t = _near_free(nav, g, *a), _near_free(nav, g, *b)
    if s is None or t is None:
        return False
    seen, q = {s}, deque([s])
    while q:
        c = q.popleft()
        if c == t:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + di, c[1] + dj)
            if 0 <= n[0] < nav["nx"] and 0 <= n[1] < nav["ny"] and g[n[1], n[0]] and n not in seen:
                seen.add(n)
                q.append(n)
    return False


def _blocked(nav, x, y) -> bool:
    i, j = _cell(nav, x, y)
    return not _grid(nav)[j, i]


class TestLift(unittest.TestCase):

    def test_01_inside_is_the_default(self):
        a = _plan(12.0, 16.0, "2BHK", "N", 3)
        b = _plan(12.0, 16.0, "2BHK", "N", 3, pos="inside")
        self.assertNotIn("lift_position", a.generation_params)
        self.assertEqual([(r.name, r.x, r.y, r.w, r.h) for r in a.rooms], [(r.name, r.x, r.y, r.w, r.h) for r in b.rooms])
        self.assertEqual(a.engineering["lift"]["position"], "lobby")          # a home on each floor: in the stair lobby
        fx, fy, fw, fd = a.building_footprint
        lift = next(r for r in a.rooms if r.category == "LIFT")
        self.assertTrue(fx - 0.01 <= lift.x and lift.right <= fx + fw + 0.01 and fy - 0.01 <= lift.y and lift.top <= fy + fd + 0.01)

    def test_02_tower_beside_the_main_door(self):
        plan = _plan(12.19, 18.29, "2BHK", "N", 3, pos="entrance")
        info = plan.engineering["lift"]
        self.assertEqual((info["position"], info["entry"]), ("entrance", "main door"))
        t = _tower(plan)
        fx, fy, fw, fd = plan.building_footprint
        top = fy + fd
        self.assertGreater(t.y, top)                                   # in the front yard
        main = next(d for d in plan.doors if d.is_main_entrance)
        ld = next(d for d in plan.doors if d.room_name == t.name)
        self.assertIn(ld.wall_side, ("E", "W"))                        # it faces the entrance landing …
        face_toward_door = (ld.wall_side == "W") == (main.x + main.width / 2 < t.x)
        self.assertTrue(face_toward_door)
        lx0, ly0, lx1, ly1 = info["landing"]                           # … and the landing joins both doors
        self.assertTrue(lx0 <= main.x + main.width / 2 <= lx1 and ly0 <= top + 0.01)
        self.assertTrue(abs(lx1 - info["tower"][0]) < 0.02 or abs(lx0 - info["tower"][2]) < 0.02)
        self.assertGreater(ly1, ld.y + ld.width)
        ring = [w for w in plan.walls if w.wall_type == "external" and
                t.x - 0.01 <= (w.x1 + w.x2) / 2 <= t.right + 0.01 and t.y - 0.01 <= (w.y1 + w.y2) / 2 <= t.top + 0.01]
        self.assertGreaterEqual(len(ring), 4)
        self.assertTrue(all(abs(w.thickness - 0.23) < 1e-6 for w in ring))
        ox0, oy0, ox1, oy1 = info["tower"]
        for wd in plan.windows:                                        # no window looks into the shaft
            if wd.wall_side == "N" and abs(wd.y - top) < 0.05:
                self.assertFalse(wd.x + wd.width / 2 > ox0 and wd.x - wd.width / 2 < ox1, wd)
        inner = (fw - 0.23) * (fd - 0.23)
        self.assertLessEqual(plan.total_carpet_area, inner + 0.01)     # the tower is not carpet
        self.assertTrue(any("Lift tower beside the main door" in n for n in plan.design_notes))

    def test_03_tower_at_the_side(self):
        plan = _plan(12.0, 16.0, "3BHK", "E", 2, pos="side")
        info = plan.engineering["lift"]
        self.assertEqual(info["position"], "side")
        t = _tower(plan)
        fx, fy, fw, fd = plan.building_footprint
        self.assertTrue(t.right < fx or t.x > fx + fw)                 # beside the house, in a side yard
        self.assertTrue(fy - 0.01 <= t.y and t.top <= fy + fd + 0.01)
        by_name = {r.name: r for r in plan.rooms}
        if info["entry"] == "side door":
            wall_x = fx if t.right < fx else fx + fw
            entry = [d for d in plan.doors if d.wall_side in ("E", "W") and abs(d.x - wall_x) < 0.02
                     and by_name[d.room_name].category not in ("LIFT",)
                     and info["landing"][1] - 0.01 <= d.y <= info["landing"][3]]
            self.assertTrue(entry)
            self.assertNotIn(by_name[entry[0].room_name].category, NO_ENTRY)
            self.assertNotIn("Dress", entry[0].room_name)
        # a side wall with no living space on it: a verandah along the front to the main door instead
        v = _plan(12.19, 18.29, "2BHK", "N", 3, pos="side")
        vi = v.engineering["lift"]
        self.assertEqual(vi["position"], "side")
        if vi["layout"] == "verandah":
            main = next(d for d in v.doors if d.is_main_entrance)
            self.assertTrue(vi["landing"][0] <= main.x + main.width / 2 <= vi["landing"][2])
        for d in v.doors:                                               # never a door from the landing into a bedroom
            r = next((q for q in v.rooms if q.name == d.room_name), None)
            if r is not None and vi["landing"][1] - 0.05 <= d.y <= vi["landing"][3] + 0.05 and d.wall_side in ("E", "W") \
                    and (abs(d.x - v.building_footprint[0]) < 0.02 or abs(d.x - sum(v.building_footprint[::2])) < 0.02):
                self.assertNotIn(r.category, NO_ENTRY - {"LIFT"})

    def test_04_what_fits_and_fallbacks(self):
        small = lift_options(9.55, 12.71, "N", "2BHK", num_floors=2)
        self.assertEqual(small["inside"].status, "ok")
        self.assertEqual(small["side"].status, "no")                   # a 3-cent plot has no 2 m side yard
        self.assertNotEqual(small["entrance"].status, "no")
        self.assertIn("side yard", small["side"].reason)
        plan = _plan(9.55, 12.71, "2BHK", "N", 2, pos="side")
        self.assertEqual(plan.engineering["lift"]["position"], "entrance")
        self.assertTrue(any("No side yard is wide enough" in n for n in plan.design_notes))
        big = lift_options(15.0, 20.0, "W", "3BHK", num_floors=3)
        self.assertTrue(all(big[k].status != "no" for k in ("entrance", "side")), {k: v.reason for k, v in big.items()})
        apt = _plan(18.0, 24.0, "2BHK", "E", 5, pos="side", btype=APT)
        self.assertNotIn("lift_position", apt.generation_params)
        self.assertIsNone(_tower(apt))                                  # flats share the core's lift
        self.assertEqual(lift_car(4)["fits"]["wheelchair"], False)
        self.assertEqual(lift_car(6)["fits"]["wheelchair"], True)
        self.assertEqual(lift_car(8)["fits"]["helper"], True)

    def test_04b_small_plot_keeps_stair_and_lift(self):
        # a 3-cent G+2: the stair and the lift used to squeeze the house into the no-stair fallback layout;
        # the front band (stair + lobby + lift) keeps both, and the home behind it is a real solved plan
        plan = _plan(9.55, 12.71, "2BHK", "N", 3)
        self.assertIn(plan.engineering["lift"]["position"], ("lobby", "entrance"))
        self.assertTrue(plan.layout_state)                              # a real (solved) layout
        self.assertTrue(any(r.category == "STAIRS" for r in plan.rooms))
        self.assertTrue(any(r.category == "LIFT" for r in plan.rooms))
        self.assertNotIn("lift_position", plan.generation_params)       # the owner's choice is still 'inside'

    def test_05_replan_keeps_the_lift_outside(self):
        plan = _plan(12.19, 18.29, "2BHK", "N", 3, pos="entrance")
        self.assertEqual(plan.generation_params.get("lift_position"), "entrance")
        again = LayoutEngine(**plan.generation_params).generate()
        self.assertEqual((_tower(again).x, _tower(again).y), (_tower(plan).x, _tower(plan).y))
        opts = LayoutEngine(**plan.generation_params, design_intents=[{"type": "size", "room": "living", "factor": 1.15}],
                            base_layout=plan.layout_state).generate_alternatives(1)
        self.assertEqual(opts[0].engineering["lift"]["position"], "entrance")

    def test_06_the_3d_lift(self):
        # beside the main door: stops on every floor, a porch at floor level to its door, a landing above
        plan = _plan(12.19, 18.29, "2BHK", "N", 3, pos="entrance")
        sc = wt.build_scene(plan)
        L = sc["lift"]
        self.assertTrue(L["tower"] and L["movable"])
        self.assertEqual([s["L"] for s in L["stops"]], [0, 1, 2])
        self.assertEqual([s["z"] for s in L["stops"]], [lv["z"] for lv in sc["levels_list"][:3]])
        self.assertGreater(L["head"], sc["shell"]["roof_z"])
        mx, my = L["door"]["mid"]
        nx, ny = L["door"]["n"]
        front = (mx - nx * 0.6, my - ny * 0.6)                           # standing in front of the lift door
        plat = sc["steps"][0]
        self.assertTrue(plat[0] <= front[0] <= plat[2] and plat[1] <= front[1] <= plat[3])
        self.assertAlmostEqual(plat[5], sc["levels"]["ffl"])            # the porch is at floor level
        main = next(d for d in sc["doors"] if d["kind"] == "main")
        out = (main["mid"][0] - main["n"][0] * 0.6, main["mid"][1] - main["n"][1] * 0.6)
        inside = (main["mid"][0] + main["n"][0] * 0.8, main["mid"][1] + main["n"][1] * 0.8)
        self.assertTrue(_reach(sc["nav"], inside, front))                # ground: from the hall to the lift
        self.assertEqual(main["balcony"], [round(v, 3) for v in plan.engineering["lift"]["landing"]])
        self.assertTrue(_reach(sc["nav_typ"], inside, front))            # upper floor: front door ↔ lift landing
        ox0, oy0, ox1, oy1 = L["outer"]
        for nav in (sc["nav"], sc["nav_typ"]):
            self.assertTrue(_blocked(nav, (ox0 + ox1) / 2, (oy0 + oy1) / 2))
        # at the side, with a side door: from the room it opens into to the lift, on every floor
        sp = _plan(12.0, 16.0, "3BHK", "E", 2, pos="side")
        ss = wt.build_scene(sp)
        sl = ss["lift"]
        if sp.engineering["lift"]["entry"] == "side door":
            ent = next(d for d in ss["doors"] if d.get("balcony") and d["kind"] == "service"
                       and d["balcony"] == [round(v, 3) for v in sp.engineering["lift"]["landing"]])
            room_pt = (ent["mid"][0] + ent["n"][0] * 0.8, ent["mid"][1] + ent["n"][1] * 0.8)
            sfront = (sl["door"]["mid"][0] - sl["door"]["n"][0] * 0.6, sl["door"]["mid"][1] - sl["door"]["n"][1] * 0.6)
            self.assertTrue(_reach(ss["nav"], room_pt, sfront))
            self.assertTrue(_reach(ss["nav_typ"] or ss["nav1"], room_pt, sfront))
        # stilt parking: the tower stops at the yard too
        st = _plan(9.55, 12.71, "2BHK", "N", 2, pos="entrance",
                   comps={'car_parking': {'enabled': True, 'slots': 1, 'mode': 'stilt'}})
        stl = wt.build_scene(st)["lift"]
        self.assertEqual(stl["stops"][0]["label"], "P")
        self.assertEqual(stl["stops"][0]["z"], 0.0)
        # an apartment's core lift: every floor, not movable; a house's inside lift: movable
        ap = wt.build_scene(_plan(18.0, 24.0, "2BHK", "E", 5, btype=APT))["lift"]
        self.assertEqual((ap["position"], ap["movable"], len(ap["stops"])), ("core", False, 5))
        self.assertEqual(ap["speed"], 1.0)
        ins = wt.build_scene(_plan(12.0, 16.0, "2BHK", "N", 3))["lift"]
        self.assertEqual((ins["position"], ins["tower"], ins["movable"]), ("lobby", False, True))

    def test_07_viewer_runs_the_lift(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_folder", "walkthrough3d"))
        js = open(os.path.join(root, "walk.js"), encoding="utf-8").read()
        html = open(os.path.join(root, "index.html"), encoding="utf-8").read()
        for name in ("function liftCall", "function boardLift", "function stepOut", "function setXray", "function liftTrip",
                     "function buildGhosts"):
            self.assertIn(name, js)
        for el in ('id="cop"', 'id="btnLift"', 'id="liftMove"'):
            self.assertIn(el, html)


if __name__ == "__main__":
    unittest.main()
