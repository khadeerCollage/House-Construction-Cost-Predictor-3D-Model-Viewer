"""
Gates sized for what passes through them (floorplan_generator/gates.py).

People only → a small gate, two-wheelers → a medium one, a car → a wide one found by simulating the design
car turning in from the road (a narrower road never asks for a narrower gate), 6+ cars → two lanes, a building
15 m or taller → 4.5 m for the fire engine. Leaves open inward and clear of the parked car and the porch (else a
sliding / telescopic / bi-fold gate); stilt cars park only where a car can drive in clear of the stair, lift,
lobby and columns; the drawing, the schedule and the 3D (with the drive-in path) all show the same gates.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine, _is_site_room  # noqa: E402
from floorplan_generator.family_analyzer import FamilyProfile  # noqa: E402
from floorplan_generator import gates as gt  # noqa: E402
from floorplan_renderer import walkthrough as wt  # noqa: E402
from floorplan_renderer.svg_renderer import SVGRenderer  # noqa: E402

FAM = FamilyProfile(adults=2, children=2, elderly=0, needs_pooja_room=True)
APT = "Apartment / Multi-Storey Building"
CAR = {"car_parking": {"enabled": True, "slots": 1}}
BIKES = {"bike_parking": {"enabled": True, "slots": 2}}
_CACHE = {}


def _gen(w, h, bhk, facing, floors=1, btype="Independent House", comps=None, **kw):
    key = (w, h, bhk, facing, floors, btype, repr(comps), tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = LayoutEngine(w, h, bhk, True, facing, kw.pop("quality", "Standard"), num_floors=floors,
                                   building_type=btype, family_profile=FAM, additional_components=comps or {},
                                   **kw).generate()
    return _CACHE[key]


def _gates(plan):
    return plan.engineering.get("gates") or []


def _kind(plan, *kinds):
    return [g for g in _gates(plan) if g["kind"] in kinds]


def _gaps(plan):
    """x-ranges of the gaps in the compound wall along the road."""
    H, W = plan.plot_height, plan.plot_width
    cov = sorted((min(w.x1, w.x2), max(w.x1, w.x2)) for w in plan.walls
                 if getattr(w, "wall_type", "") == "compound" and abs(w.y1 - H) < 0.03 and abs(w.y2 - H) < 0.03)
    gaps, pos = [], 0.0
    for lo, hi in cov:
        if lo - pos > 0.3:
            gaps.append((round(pos, 2), round(lo, 2)))
        pos = max(pos, hi)
    if W - pos > 0.3:
        gaps.append((round(pos, 2), round(W, 2)))
    return gaps


def _overlap(a, b):
    return a[0] < b[2] - 1e-6 and b[0] < a[2] - 1e-6 and a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6


class TestGates(unittest.TestCase):

    def test_01_size_follows_what_passes(self):
        """People only < two-wheelers < a car, on the same plot and facing; a block of homes gets a wider walk gate."""
        for facing in "NESW":
            walk = _kind(_gen(9.55, 12.71, "2BHK", facing), "walk")
            bike = _kind(_gen(9.55, 12.71, "2BHK", facing, comps=BIKES), "two_wheeler")
            car = _gen(12.33, 16.4, "2BHK", facing, comps=dict(CAR, **BIKES))
            veh = _kind(car, "vehicle", "vehicle_wicket")
            self.assertEqual(len(walk), 1, facing)
            self.assertAlmostEqual(walk[0]["width"], 1.05, places=2)
            self.assertEqual(walk[0]["feet"], "3½ ft")
            self.assertEqual(len(bike), 1, facing)
            self.assertAlmostEqual(bike[0]["width"], 1.5, places=2)
            self.assertEqual(len(veh), 1, facing)
            self.assertGreaterEqual(veh[0]["width"], 3.04, facing)
            self.assertTrue(walk[0]["width"] < bike[0]["width"] < veh[0]["width"])
            # with a car gate, two-wheelers use it and the people's gate stays small
            people = _kind(car, "walk")
            if people:
                self.assertAlmostEqual(people[0]["width"], 1.05, places=2)
            # the compound wall is cut exactly at the gates
            for plan in (_gen(9.55, 12.71, "2BHK", facing), car):
                self.assertEqual(sorted((round(g["x0"], 2), round(g["x1"], 2)) for g in _gates(plan)), _gaps(plan), facing)
            self.assertIn("Gates on the 30 ft road", " ".join(car.design_notes))
        apt = _gen(12.33, 16.4, "2BHK", "N", floors=3, btype=APT)
        self.assertAlmostEqual(_kind(apt, "walk")[0]["width"], 1.2, places=2)

    def test_02_narrow_road_needs_a_wider_gate(self):
        """The swept-path simulation: a narrower road never asks for a narrower gate, for every design car."""
        for car in gt.CARS:
            prev = 0.0
            for road in (18.0, 12.0, 9.0, 7.5, 6.0, 4.5):
                r = gt._need(car, max(2.6, road - 2 * gt.DRAIN), 0.2 + gt.TURN_IN, -1.3 - gt.SWING_OUT, 1.3 + gt.SWING_OUT)
                ok = [v for v in r.values() if v]
                self.assertTrue(ok, (car, road))
                half = max(max(v[1], -v[0]) for v in ok)
                self.assertGreaterEqual(half + 1e-6, prev, (car, road))
                prev = half
            self.assertGreaterEqual(prev * 2, gt.CARS[car]["W"])
        # the same house: a 20 ft road gets a wider car gate than a 30 ft one, a 60 ft road no wider
        wide = _kind(_gen(12.33, 16.4, "2BHK", "N", comps=CAR, road_width=18.0), "vehicle")[0]["width"]
        mid = _kind(_gen(12.33, 16.4, "2BHK", "N", comps=CAR), "vehicle")[0]["width"]
        narrow = _kind(_gen(12.33, 16.4, "2BHK", "N", comps=CAR, road_width=6.0), "vehicle")[0]["width"]
        self.assertLessEqual(wide, mid)
        self.assertLess(mid, narrow)
        plan = _gen(12.33, 16.4, "2BHK", "N", comps=CAR, road_width=6.0)
        self.assertEqual(plan.generation_params.get("road_width"), 6.0)
        self.assertNotIn("road_width", _gen(12.33, 16.4, "2BHK", "N", comps=CAR).generation_params)
        again = LayoutEngine(**plan.generation_params).generate()
        self.assertEqual(_gates(again), _gates(plan))

    def test_03_fire_engine_and_two_lanes(self):
        """A building 15 m or taller keeps a 4.5 m gate (NBC 2016 Part 4) even with no car; 6 cars need two lanes."""
        tall = _gen(12.33, 16.4, "3BHK", "N", floors=5, btype=APT, has_lift=True)
        veh = _kind(tall, "vehicle", "vehicle_wicket")
        self.assertEqual(len(veh), 1)
        self.assertGreaterEqual(veh[0]["width"], gt.FIRE_W)
        self.assertIn("fire engine", " ".join(veh[0]["why"]))
        low = _gen(12.33, 16.4, "3BHK", "N", floors=3, btype=APT, has_lift=True)
        self.assertFalse(_kind(low, "vehicle", "vehicle_wicket"))
        d = gt.GateDesigner(20.0, 25.0, "N", True, 9.0, "Standard", "", 2, cars=6)
        walk = d.walk_gate(15.0, 16.0, True)
        g, _, _ = d.vehicle_gate([{"axis": 4.0, "throat": 0.2, "lo": 2.7, "hi": 5.3, "park_y": 22.0, "stop_y": 24.8}],
                                 walk, [], [], (2.0, 2.0, 18.0, 18.0), "front")
        self.assertGreaterEqual(g.width, 2 * gt.CARS["suv"]["W"] + 3 * gt.PIER - 1e-6)

    def test_04_leaves_open_inward_clear_of_things(self):
        """Swing leaves swing inside the plot and miss the parked car and the house; a sliding leaf runs behind
        the wall within the plot and never across another opening."""
        cases = [_gen(12.33, 16.4, "2BHK", f, comps=dict(CAR, **BIKES)) for f in "NESW"]
        cases += [_gen(12.33, 16.4, "2BHK", "N", comps=CAR, road_width=6.0),
                  _gen(15, 20, "3BHK", "W", comps={"car_parking": {"enabled": True, "slots": 2}}, quality="Premium")]
        for plan in cases:
            H, W = plan.plot_height, plan.plot_width
            fx, fy, fw, fd = plan.building_footprint
            house = (fx - 0.12, fy - 0.12, fx + fw + 0.12, fy + fd + 0.12)
            parked = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in plan.rooms if r.category == "PARKING"]
            self.assertTrue(parked, "the car the owner asked for is parked")
            gates = _gates(plan)
            for g in gates:
                self.assertGreaterEqual(g["x0"], 0.0)
                self.assertLessEqual(g["x1"], W)
                for lf in g["leaves"]:
                    if lf["type"] in ("swing", "fold"):
                        L = lf["len"] * (2 if lf["type"] == "fold" else 1)
                        x0 = lf["hinge"] if lf["dir"] > 0 else lf["hinge"] - lf["len"]
                        arc = (x0, H - lf["len"], x0 + lf["len"], H)
                        self.assertGreater(H, arc[1])            # inward, never over the road
                        if g["kind"].startswith("vehicle") and lf["type"] == "swing" and not lf.get("wicket"):
                            for p in parked + [house]:
                                self.assertFalse(_overlap(arc, p), (g["tag"], lf))
                        self.assertGreater(L, 0)
                if g["runback"]:
                    a, b = g["runback"]
                    self.assertGreaterEqual(a, 0.0)
                    self.assertLessEqual(b, W)
                    for o in gates:
                        if o is not g:
                            self.assertFalse(o["x0"] < b and a < o["x1"], "run-back crosses another gate")
                    self.assertIn(g["op"], ("sliding", "telescopic"))

    def test_05_stilt_cars_clear_of_the_core(self):
        """Stilt bays never sit in the stair, lift, duct, lobby or on a column; the 3D parks the cars in them."""
        for facing in "NE":
            plan = _gen(12.33, 16.4, "3BHK", facing, floors=4, btype=APT, has_lift=True, quality="Premium",
                        comps={"car_parking": {"enabled": True, "slots": 3, "mode": "stilt"}})
            st = plan.engineering.get("stilt")
            self.assertIsNotNone(st)
            self.assertGreaterEqual(st["fitted"], 1)
            shafts, lobby, cols = gt.stilt_context(plan.rooms, plan.building_footprint, _is_site_room)
            car = gt.CARS[st["car"]]
            for b in st["bays"]:
                cx = (b[0] + b[2]) / 2
                body = (cx - car["W"] / 2, b[3] - 0.15 - car["L"], cx + car["W"] / 2, b[3] - 0.15)
                for s in shafts + cols:
                    self.assertFalse(_overlap(body, s), (facing, b, s))
                for lb in lobby:
                    self.assertFalse(_overlap(tuple(b[:4]), lb), (facing, b, lb))
            if st["fitted"] < st["wanted"]:
                self.assertIn("fit under the building", " ".join(plan.design_notes))
            veh = _kind(plan, "vehicle", "vehicle_wicket")
            self.assertEqual(len(veh), 1)
            self.assertGreaterEqual(veh[0]["width"], gt.FIRE_W)           # 4 floors on a stilt ≥ 15 m
            sc = wt.build_scene(plan)
            stilt_cars = [c for c in sc["cars"] if c.get("stilt")]
            self.assertEqual(len(stilt_cars), st["fitted"])
            for c in stilt_cars:
                for lb in lobby + shafts:
                    self.assertFalse(lb[0] < c["c"][0] < lb[2] and lb[1] < c["c"][1] < lb[3], "car in the core")

    def test_06_the_car_really_drives_in(self):
        """The drive-in path starts on the road, crosses the wall inside its gate and ends in the car's bay."""
        cases = [_gen(12.33, 16.4, "2BHK", f, comps=CAR) for f in "NESW"]
        cases.append(_gen(12.33, 16.4, "2BHK", "N", comps=CAR, road_width=6.0))
        cases.append(_gen(12.33, 16.4, "3BHK", "N", floors=4, btype=APT, has_lift=True, quality="Premium",
                          comps={"car_parking": {"enabled": True, "slots": 3, "mode": "stilt"}}))
        for plan in cases:
            dv = plan.engineering.get("drive")
            self.assertIsNotNone(dv)
            H = plan.plot_height
            path = dv["path"]
            self.assertGreater(path[0][1], H + 1.0, "starts on the road")
            self.assertLess(math.hypot(path[-1][0] - dv["park"][0], path[-1][1] - dv["park"][1]), 0.05)
            g = _kind(plan, "vehicle", "vehicle_wicket")[0]
            L, Wc = dv["dims"]["L"], dv["dims"]["W"]
            for x, y, h in path:                     # every body corner crossing the wall line is inside the gate
                pts = [(x + su * L / 2 * math.cos(h) - sv * Wc / 2 * math.sin(h),
                        y + su * L / 2 * math.sin(h) + sv * Wc / 2 * math.cos(h)) for su in (1, -1) for sv in (1, -1)]
                ys = [p[1] for p in pts]
                if min(ys) < H < max(ys):
                    for i in range(4):
                        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % 4]
                        if (y1 - H) * (y2 - H) < 0:
                            xc = x1 + (H - y1) / (y2 - y1) * (x2 - x1)
                            self.assertTrue(g["x0"] - 0.05 <= xc <= g["x1"] + 0.05, (plan.orientation, xc, g))

    def test_07_drawing_and_3d_show_the_same_gates(self):
        plan = _gen(12.33, 16.4, "2BHK", "N", comps=dict(CAR, **BIKES), road_width=6.0)
        svg = SVGRenderer().render_to_string(plan, mode="blueprint")
        self.assertIn("SCHEDULE OF DOORS, WINDOWS &amp; GATES", svg)
        self.assertIn("20 FT (6 M)", svg)
        for g in _gates(plan):
            self.assertIn(f">{g['tag']}<", svg)
        sc = wt.build_scene(plan)
        self.assertEqual([(g["x0"], g["x1"]) for g in sc["gates"]],
                         [(round(g["x0"], 3), round(g["x1"], 3)) for g in _gates(plan)])
        self.assertTrue(all(g["leaves"] for g in sc["gates"]))
        self.assertEqual(sc["road"]["w"], 6.0)
        self.assertIsNotNone(sc["drive"])
        self.assertEqual(sum(1 for c in sc["cars"] if c.get("drive")), 1)
        tour = sc["tour"]
        walk = _kind(plan, "walk")[0]
        self.assertTrue(walk["x0"] - 1.0 <= tour[0]["path"][0][0] <= walk["x1"] + 1.0, "the tour starts at the walking gate")
        js = open(os.path.join(os.path.dirname(wt.__file__), "..", "model_folder", "walkthrough3d", "walk.js"),
                  encoding="utf-8").read()
        for word in ("function startDrive", "gateLeaf", '"slide"', '"fold"'):
            self.assertIn(word, js)
        # Vastu: every gate's pada on the plot boundary is reported
        self.assertTrue(all(g["pada"] and g["pada"]["id"] for g in _gates(plan)))


if __name__ == "__main__":
    unittest.main()
