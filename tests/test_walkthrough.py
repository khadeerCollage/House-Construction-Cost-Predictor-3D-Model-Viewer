"""
3D walk-through (floorplan_renderer/walkthrough.py + model_folder/walkthrough3d).

Covers: the scene is built from the plan for every facing (walls split around windows, lintels
over doors and arches, door leaves keep the drawing's hinge and swing), heights follow the
engineering report (storey = risers × riser, G+1 shell, stilt parking), the 3D furniture is
exactly the drawing's furniture, every room can be reached on foot and the guided tour visits
it, the sun inputs (north, latitude), wishes from the 3D view map onto Smart Re-Plan intents,
and the viewer works offline (no URL outside its own folder).
"""

import base64
import json
import math
import os
import re
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_renderer.drafting import build_drafting  # noqa: E402
from floorplan_renderer._common import north_angle  # noqa: E402
from floorplan_renderer import walkthrough as wt  # noqa: E402

COMPS = {'car_parking': {'enabled': True, 'slots': 1}, 'garden': {'enabled': True, 'pct': 10},
         'water_tank': {'enabled': True}, 'bike_parking': {'enabled': True, 'slots': 2}}
_CACHE = {}


def _plan(facing="N", w=9.55, h=12.71, bhk="2BHK", comps=None, floors=1):
    key = (facing, w, h, bhk, json.dumps(comps or {}, sort_keys=True), floors)
    if key not in _CACHE:
        plan = LayoutEngine(w, h, bhk, True, facing, additional_components=comps or {}, num_floors=floors).generate()
        _CACHE[key] = (plan, wt.build_scene(plan, state="Andhra Pradesh"))
    return _CACHE[key]


def _free(scene):
    nav = scene["nav"]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(nav["bits"]), dtype=np.uint8))[: nav["nx"] * nav["ny"]]
    grid = bits.reshape(nav["ny"], nav["nx"]).astype(bool)

    def ok(x, y):
        i, j = int((x - nav["x0"]) / nav["cell"]), int((y - nav["y0"]) / nav["cell"])
        return 0 <= i < nav["nx"] and 0 <= j < nav["ny"] and bool(grid[j, i])
    return ok


def _overlap(a0, a1, b0, b1, eps=0.01):
    return a0 < b1 - eps and b0 < a1 - eps


class TestWalkthrough(unittest.TestCase):

    def test_01_scene_for_every_facing(self):
        for facing in "NESW":
            plan, sc = _plan(facing)
            json.dumps(sc)                                   # JSON-safe
            self.assertTrue(sc["walls"]["ext"] and sc["walls"]["int"], facing)
            self.assertEqual(len(sc["doors"]), len(plan.doors), facing)
            self.assertEqual(len(sc["windows"]), len(plan.windows), facing)
            ffl = sc["levels"]["ffl"]
            for d in sc["doors"]:
                turn = ((d["open"] - d["closed"] + 540) % 360) - 180
                self.assertEqual(abs(turn), 90, (facing, d))     # leaves swing a quarter turn
                nx, ny = d["n"]                                   # …into the room that owns the door
                ox, oy = math.cos(math.radians(d["open"])), math.sin(math.radians(d["open"]))
                self.assertGreater(nx * ox + ny * oy, 0.99, (facing, d["room"]))
            for w in sc["windows"]:
                self.assertTrue(ffl + 0.85 <= w["z0"] < w["z1"] <= ffl + 2.3, (facing, w))

    def test_02_openings_are_open(self):
        _, sc = _plan("E", 16.0, 22.0, "3BHK", COMPS)
        ffl = sc["levels"]["ffl"]
        boxes = sc["walls"]["ext"] + sc["walls"]["int"]
        for w in sc["windows"]:                  # no wall between the sill and the head
            horiz = abs(w["a"][1] - w["b"][1]) < 1e-6
            a, b = (w["a"][0], w["b"][0]) if horiz else (w["a"][1], w["b"][1])
            line = w["a"][1] if horiz else w["a"][0]
            for bx in boxes:
                lo, hi = (bx[0], bx[2]) if horiz else (bx[1], bx[3])
                c0, c1 = (bx[1], bx[3]) if horiz else (bx[0], bx[2])
                if c0 <= line <= c1 and _overlap(lo, hi, a, b):
                    self.assertFalse(_overlap(bx[4], bx[5], w["z0"], w["z1"]), (w, bx))
        for d in sc["doors"]:                    # walk-through below 2.1 m, lintel above
            ca = math.radians(d["closed"])
            a = d["hinge"][0 if d["horiz"] else 1]
            b = a + (math.cos(ca) if d["horiz"] else math.sin(ca)) * d["w"]
            a, b = sorted((a, b))
            line = d["hinge"][1] if d["horiz"] else d["hinge"][0]
            over = []
            for bx in boxes:
                lo, hi = (bx[0], bx[2]) if d["horiz"] else (bx[1], bx[3])
                c0, c1 = (bx[1], bx[3]) if d["horiz"] else (bx[0], bx[2])
                if c0 <= line <= c1 and _overlap(lo, hi, a, b):
                    over.append(bx)
                    self.assertGreaterEqual(bx[4], ffl + 2.1 - 1e-6, (d["room"], bx))
            self.assertTrue(over, f"no lintel over the {d['room']} door")

    def test_03_heights_follow_the_engineering_report(self):
        plan, sc = _plan("N", 12.0, 16.0, "2BHK", None, 2)
        st = plan.engineering["stair"]
        lv = sc["levels"]
        self.assertAlmostEqual(lv["storey"], st["risers"] * st["riser_mm"] / 1000, places=2)
        self.assertAlmostEqual(lv["ffl"], 0.45)
        self.assertEqual(sc["shell"]["floors"], 2)
        self.assertAlmostEqual(sc["shell"]["roof_z"], lv["ffl"] + 2 * lv["storey"], places=2)
        inner = [s for s in sc["stairs"] if not s["external"]]
        self.assertTrue(inner)
        top = max(b[5] for b in inner[0]["boxes"])
        self.assertAlmostEqual(top, lv["ffl"] + lv["storey"], delta=0.02)          # stairs reach the next floor
        self.assertTrue(sc["shell"]["mumty"] is not None)
        # stilt parking: the house sits one storey up, columns and the car below
        plan, sc = _plan("N", 9.55, 12.71, "2BHK", {'car_parking': {'enabled': True, 'slots': 1, 'mode': 'stilt'}}, 2)
        self.assertEqual(plan.engineering["car_parking"], "stilt")
        self.assertGreater(sc["levels"]["ffl"], 2.9)
        self.assertTrue(sc["stilt"]["columns"])
        self.assertTrue(any(c.get("stilt") for c in sc["cars"]))
        self.assertFalse(sc["steps"])

    def test_04_furniture_is_the_drawings_furniture(self):
        plan, sc = _plan("W", 16.0, 22.0, "3BHK", COMPS)
        items = build_drafting(plan).items
        self.assertEqual([i["kind"] for i in items], [f["kind"] for f in sc["furniture"]])
        rooms = {r.name: r for r in plan.rooms}
        for f in sc["furniture"]:
            self.assertIn(f["room"], rooms)
            r = rooms[f["room"]]
            x, y = (f["x"], f["y"]) if "x" in f else f["o"]
            self.assertTrue(r.x - 0.01 <= x <= r.right + 0.01 and r.y - 0.01 <= y <= r.top + 0.01, f)
        beds = [f for f in sc["furniture"] if f["kind"] == "bed"]
        self.assertEqual(len(beds), 3)                       # one per bedroom of a 3BHK

    def test_05_every_room_is_walkable_and_toured(self):
        for args in (("N",), ("E",), ("S", 12.0, 15.0, "2BHK", None, 2), ("W", 16.0, 22.0, "3BHK", COMPS),
                     ("E", 18.0, 24.0, "4BHK", {'swimming_pool': {'enabled': True}})):
            plan, sc = _plan(*args)
            ok = _free(sc)
            tour = sc["tour"]
            self.assertEqual(tour[0]["kind"], "start")
            self.assertGreater(tour[0]["path"][0][1], plan.plot_height, "starts on the road")
            self.assertTrue(any(s["kind"] == "door" for s in tour), "walks up to the main door")
            self.assertEqual(tour[-1]["kind"], "end")
            visited = {s.get("room") for s in tour}
            for r in sc["rooms"]:
                if r["cat"] in ("STAIRS", "LIFT") or "Duct" in r["name"]:
                    continue
                self.assertIn(r["name"], visited, f"{args}: the tour misses {r['name']}")
            for s in tour:
                for x, y in s["path"]:
                    self.assertTrue(ok(x, y), (args, s["title"], x, y))
                self.assertTrue(s["say"])
            living = next(s for s in tour if s.get("room") and "Living" in s["room"])
            self.assertRegex(living["say"], r"\d+'\d+\"")      # sizes in feet and inches
            self.assertIn("square feet", living["say"])

    def test_06_sun_inputs(self):
        for facing in "NESW":
            plan, sc = _plan(facing)
            a = math.radians(north_angle(plan))
            self.assertAlmostEqual(sc["north"][0], math.sin(a), places=3)
            self.assertAlmostEqual(sc["north"][1], math.cos(a), places=3)
        self.assertAlmostEqual(sc["lat"], wt.STATE_LAT["Andhra Pradesh"])
        plan, _ = _plan("N")
        self.assertAlmostEqual(wt.build_scene(plan, latitude=28.6)["lat"], 28.6)
        self.assertAlmostEqual(wt.build_scene(plan)["lat"], wt.DEFAULT_LAT)

    def test_07_wishes_from_3d_map_to_intents(self):
        from model_folder import walkthrough_ui as ui
        plan, sc = _plan("N")
        room = next(r for r in sc["rooms"] if r["key"] and r["cat"] == "BEDROOM")
        it, why = ui._wish_from_event(plan, {"kind": "bigger", "room": room["key"], "name": room["name"]})
        self.assertEqual(it, {"type": "size", "room": room["key"], "factor": 1.2})
        it, _ = ui._wish_from_event(plan, {"kind": "smaller", "room": room["key"]})
        self.assertEqual(it["factor"], 0.85)
        it, _ = ui._wish_from_event(plan, {"kind": "lock", "room": room["key"]})
        self.assertEqual(it, {"type": "lock", "room": room["key"]})
        it, why = ui._wish_from_event(plan, {"kind": "opposite", "room": room["key"], "name": room["name"]})
        self.assertTrue(it or why)
        it, why = ui._wish_from_event(plan, {"kind": "bigger", "room": "NOT_A_ROOM", "name": "Garden"})
        self.assertIsNone(it)
        self.assertIn("Garden", why)

    def test_08_viewer_is_offline(self):
        d = os.path.join(ROOT, "model_folder", "walkthrough3d")
        for f in ("index.html", "walk.js", "vendor/three.module.min.js", "vendor/three.core.min.js",
                  "vendor/LICENSE-three.txt"):
            self.assertTrue(os.path.isfile(os.path.join(d, f)), f)
        for f in ("index.html", "walk.js"):
            src = open(os.path.join(d, f), encoding="utf-8").read()
            self.assertIsNone(re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}/", src, re.I), f"{f} loads from the network")
        js = open(os.path.join(d, "walk.js"), encoding="utf-8").read()
        self.assertIn('from "./vendor/three.module.min.js"', js)
        mod = open(os.path.join(d, "vendor", "three.module.min.js"), encoding="utf-8").read()
        specs = set(re.findall(r'from\s*"([^"]+)"', mod))       # every module it pulls in sits next to it
        self.assertEqual(specs, {"./three.core.min.js"})


if __name__ == "__main__":
    unittest.main()
