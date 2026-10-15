"""
Every floor reached on foot, and the guided walk of a building of floors.

The stair of every building type (1 / 2 / 4-flat apartments on every facing, a narrow plate, a home on every
floor, a stilt house, a duplex, a one-storey house) is walked the way walk.js walks it — the walking grid of each
level, a level change only where a flight's lower half meets its upper half — from the road to every floor and up
to the terrace. Each flight starts on its floor and lands on the next; at the top you step off into the lobby /
hall (never through a wall into a flat). Apartments and houses of floors get the journey tour: road → gate →
parking → stair and lift → the owner's choice (lift or stairs, which floor) → that home's front door and rooms →
the terrace.
"""

import base64
import os
import sys
import unittest
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_generator.family_analyzer import FamilyProfile  # noqa: E402
from floorplan_renderer import walkthrough as wt  # noqa: E402

FAM = FamilyProfile(adults=2, children=2, elderly=0, needs_pooja_room=True)
APT = "Apartment / Multi-Storey Building"
STILT = {"car_parking": {"enabled": True, "slots": 2, "mode": "stilt"}}
_CACHE = {}


def _scene(w, h, bhk, facing, floors, btype, lift=False, comps=None, flats=None):
    key = (w, h, bhk, facing, floors, btype, lift, repr(comps), flats)
    if key not in _CACHE:
        kw = dict(num_floors=floors, building_type=btype, has_lift=lift, family_profile=FAM,
                  additional_components=comps or {})
        if flats:
            kw["flats_per_floor"] = flats
        plan = LayoutEngine(w, h, bhk, True, facing, "Standard", **kw).generate()
        _CACHE[key] = (plan, wt.build_scene(plan))
    return _CACHE[key]


def _grid(nav):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(nav["bits"]), dtype=np.uint8))[: nav["nx"] * nav["ny"]]
    return bits.reshape(nav["ny"], nav["nx"]).astype(bool)


class _Walk:
    """walk.js's walking rules: a grid per kind of level, stair halves, a level change between the halves."""

    def __init__(self, sc):
        n = sc["nav"]
        self.x0, self.y0, self.c, self.nx, self.ny = n["x0"], n["y0"], n["cell"], n["nx"], n["ny"]
        self.g0 = _grid(n)
        self.typ = _grid(sc["nav_typ"]) if sc.get("nav_typ") else self.g0
        self.top_g = _grid(sc["nav1"])
        self.TOP = sc["top"]
        self.seg = np.full((self.ny, self.nx), -1, dtype=np.int8)
        for st in sc["stairs"]:
            for b in st["boxes"]:
                for j in range(max(0, int((b[1] - self.y0) / self.c)), min(self.ny, int((b[3] - self.y0) / self.c) + 1)):
                    for i in range(max(0, int((b[0] - self.x0) / self.c)), min(self.nx, int((b[2] - self.x0) / self.c) + 1)):
                        cx, cy = self.x0 + (i + .5) * self.c, self.y0 + (j + .5) * self.c
                        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                            self.seg[j, i] = b[6]

    def grid(self, L):
        return self.g0 if L <= 0 else (self.top_g if L >= self.TOP else self.typ)

    def cell(self, x, y):
        return int((x - self.x0) / self.c), int((y - self.y0) / self.c)

    def reach(self, x, y, L=0):
        i, j = self.cell(x, y)
        seen = {(L, i, j)}
        dq = deque([(L, i, j)])
        while dq:
            L, i, j = dq.popleft()
            g = self.grid(L)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (not di and not dj) or not (0 <= a < self.nx and 0 <= b < self.ny) or not g[b, a]:
                        continue
                    if di and dj and not (g[j, a] and g[b, i]):
                        continue
                    L2 = L
                    if self.seg[j, i] == 0 and self.seg[b, a] == 1:
                        L2 = L + 1
                    elif self.seg[j, i] == 1 and self.seg[b, a] == 0:
                        L2 = L - 1
                    if not (0 <= L2 <= self.TOP) or not self.grid(L2)[b, a] or (L2, a, b) in seen:
                        continue
                    seen.add((L2, a, b))
                    dq.append((L2, a, b))
        return seen

    def can_stand(self, seen, L, x, y, r=0.4):
        i, j = self.cell(x, y)
        k = int(r / self.c)
        return any((L, i + di, j + dj) in seen for di in range(-k, k + 1) for dj in range(-k, k + 1))


CASES = [
    ("apt1 stilt N", (12.33, 16.4, "3BHK", "N", 4, APT, True, STILT, 1)),
    ("apt1 E", (12.33, 16.4, "3BHK", "E", 3, APT, True, None, 1)),
    ("apt1 S no lift", (12.33, 16.4, "2BHK", "S", 3, APT, False, None, 1)),
    ("apt1 W", (12.33, 16.4, "2BHK", "W", 3, APT, True, None, 1)),
    ("apt1 narrow plate", (6.2, 22.0, "1BHK", "S", 3, APT, True, None, 1)),
    ("apt2", (18.0, 16.0, "2BHK", "W", 3, APT, True, None, 2)),
    ("apt4 stilt", (20.0, 24.0, "2BHK", "N", 4, APT, True, STILT, 4)),
    ("house of floors", (12.33, 16.4, "3BHK", "N", 3, "Independent House", False, None, None)),
    ("stilt house", (12.33, 16.4, "2BHK", "W", 2, "Independent House", False,
                     {"car_parking": {"enabled": True, "slots": 1, "mode": "stilt"}}, None)),
    ("duplex", (12.33, 16.4, "3BHK", "E", 2, "Duplex Home", False, None, None)),
    ("one storey", (12.33, 16.4, "2BHK", "N", 1, "Independent House", False, None, None)),
]


class TestJourney(unittest.TestCase):

    def test_01_every_floor_and_the_terrace_on_foot(self):
        for name, args in CASES:
            plan, sc = _scene(*args)
            w = _Walk(sc)
            start = sc["tour"][0]["path"][0]
            seen = w.reach(*start, L=0)
            fx, fy, fw, fd = sc["footprint"]
            for lv in sc["levels_list"]:
                inside = sum(1 for (L, i, j) in seen if L == lv["i"] and fx < w.x0 + (i + .5) * w.c < fx + fw
                             and fy < w.y0 + (j + .5) * w.c < fy + fd)
                self.assertGreater(inside, 1000, f"{name}: {lv['label']} cannot be reached on foot")
            ap = sc.get("apartment") or {}
            if ap.get("lobby"):                                    # every floor's lobby, by the stairs
                lob = next(r for r in sc["rooms"] if r["name"] == ap["lobby"])["r"]
                for lv in sc["levels_list"]:
                    if lv["kind"] == "rooms":
                        self.assertTrue(w.can_stand(seen, lv["i"], (lob[0] + lob[2]) / 2, (lob[1] + lob[3]) / 2),
                                        f"{name}: the lobby of {lv['label']}")

    def test_02_flights_meet_the_floors_and_land_in_the_hall(self):
        for name, args in CASES:
            plan, sc = _scene(*args)
            lv = sc["levels"]
            for st in sc["stairs"]:
                if st["external"]:
                    continue
                low = [b for b in st["boxes"] if b[6] == 0]
                first = min(low, key=lambda b: b[5])
                self.assertAlmostEqual(first[4], st["low"], places=2, msg=f"{name}: the first step stands on the floor")
                self.assertLess(first[5] - st["low"], 0.2, f"{name}: the first riser")
                self.assertAlmostEqual(st["top"], st["low"] + lv["storey"], delta=0.02, msg=f"{name}: lands on the next floor")
                tops = sorted(b[5] for b in st["boxes"])
                self.assertLess(st["top"] - tops[-1], 0.2, f"{name}: the last riser")
                e = st["exit"]
                cx, cy = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
                host = [r for r in sc["rooms"] if r["r"][0] < cx < r["r"][2] and r["r"][1] < cy < r["r"][3]]
                ap = sc.get("apartment") or {}
                if ap.get("lobby"):
                    self.assertTrue(host and host[0]["name"] == ap["lobby"],
                                    f"{name}: you step off the stair into {host[0]['name'] if host else 'nothing'}")

    def test_03_the_journey_tour(self):
        for name, args in CASES:
            plan, sc = _scene(*args)
            tour = sc["tour"]
            kinds = [s["kind"] for s in tour]
            journey = "choice" in kinds
            if sc.get("upper") or sc["top"] < 2:
                self.assertFalse(journey, name)                   # a duplex / one storey: the home's own walk
                continue
            self.assertTrue(journey, name)
            self.assertEqual(sc["tour_level"], 0)
            self.assertEqual(kinds[0], "start")
            self.assertGreater(tour[0]["path"][0][1], plan.plot_height, f"{name}: starts on the road")
            self.assertEqual(kinds[1], "gate")
            if any(c.get("stilt") for c in sc["cars"]) or any(r["cat"] == "PARKING" for r in sc["site"]):
                self.assertLess(kinds.index("parking"), kinds.index("choice"), name)
            ch = tour[kinds.index("choice")]
            homes = [l["i"] for l in sc["levels_list"] if l["kind"] == "rooms"]
            self.assertEqual([o["L"] for o in ch["options"]], homes)
            self.assertEqual(ch["via"], ["lift", "stairs"] if sc["lift"] else ["stairs"])
            self.assertIn(ch["default"]["L"], homes)
            after = tour[kinds.index("choice") + 1:]
            self.assertEqual(after[0]["kind"], "ride")
            self.assertEqual(set(after[0]["say_by"]), {"lift", "stairs"})
            rooms_seen = [s for s in after if s["kind"] == "room"]
            self.assertTrue(rooms_seen and all(s["level"] == "chosen" for s in rooms_seen), name)
            self.assertTrue(any(s["kind"] == "door" and "{home}" in s["title"] for s in after), name)
            ter = [s for s in after if s["kind"] == "terrace"]
            self.assertEqual(len(ter), 1, name)
            self.assertEqual(ter[0]["level"], sc["top"])
            self.assertEqual(kinds[-1], "end")
            # every target can be stood on, on its own floor's grid
            w = _Walk(sc)
            for s in tour:
                if s.get("target"):
                    L = (homes[1] if len(homes) > 1 else homes[0]) if s["level"] == "chosen" else s["level"]
                    i, j = w.cell(*s["target"])
                    self.assertTrue(w.grid(L)[j, i], f"{name}: {s['title']}")
            ap = sc.get("apartment") or {}
            first = (ap.get("labels") or [""])[0]
            if first:
                self.assertTrue(all(s["room"].startswith(first + " · ") for s in rooms_seen), name)

    def test_05_stilt_parking_is_shown_even_when_no_car_fits(self):
        plan, sc = _scene(12.33, 16.4, "2BHK", "N", 4, APT, True,
                          {"car_parking": {"enabled": True, "slots": 3, "mode": "stilt"}}, 1)
        st = plan.engineering["stilt"]
        park = [s for s in sc["tour"] if s["kind"] == "parking"]
        self.assertEqual(len(park), 1)
        if st["fitted"] == 0:                           # the walk says why, with the measured opening
            self.assertIn(f"{st['front_gap']:.2f}", park[0]["say"])
            self.assertTrue(any("No car fits under the building" in n for n in plan.design_notes))
        for b in st["bays"]:                            # a flat's foyer upstairs never blocks a bay on the stilt
            self.assertGreaterEqual(b[6] - b[5], st["needs"] - 1e-6)

    def test_04_viewer_takes_the_lift_or_the_stairs(self):
        js = open(os.path.join(os.path.dirname(wt.__file__), "..", "model_folder", "walkthrough3d", "walk.js"),
                  encoding="utf-8").read()
        for word in ("function resolveStop", "function showChoice", "liftTrip(st.level, true)",
                     "lift.auto && lift.auto.tour", 'tour.phase === "ride"'):
            self.assertIn(word, js)
        html = open(os.path.join(os.path.dirname(wt.__file__), "..", "model_folder", "walkthrough3d", "index.html"),
                    encoding="utf-8").read()
        self.assertIn('id="capC"', html)


if __name__ == "__main__":
    unittest.main()
