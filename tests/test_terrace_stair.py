"""
2026-10-16: the owner chooses the side of the outside terrace stair (single-storey houses) —
front / back / left / right, judged first for every side (fits?, Vastu in simple words, house size,
yard, daily use, safety), applied by Smart Re-Plan; the stair climbs clockwise (rising towards the
South / West) wherever its top can land on the roof.
"""

import base64
import os
import sys
import unittest
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floorplan_generator import intent_engine as ie
from floorplan_generator.family_analyzer import FamilyProfile
from floorplan_generator.land_units import LandUnitsEngine
from floorplan_generator.layout_engine import LayoutEngine
from floorplan_generator.terrace_stair import stair_sides
from floorplan_generator.vastu_engine import plan_sides_for_facing
from floorplan_generator.wish_parser import understand
from floorplan_renderer.elevation import front_elevation_svg
from floorplan_renderer.svg_renderer import SVGRenderer
from floorplan_renderer.walkthrough import build_scene

FAM = FamilyProfile(adults=2, children=2)
CAR = {"car_parking": {"enabled": True, "slots": 1}}
PLAN_SIDE = {"front": "top", "back": "bottom", "left": "left", "right": "right"}
_CACHE = {}


def _house(cents, bhk, facing, comps=None, **kw):
    key = (cents, bhk, facing, str(comps), str(sorted(kw.items())))
    if key not in _CACHE:
        p = LandUnitsEngine.solve_parcel(cents, "Rectangle", 1.33)
        _CACHE[key] = LayoutEngine(p.width_m, p.length_m, bhk, True, facing, "Standard", family_profile=FAM,
                                   additional_components=comps or {}, **kw).generate()
    return _CACHE[key]


def _side(plan, side):
    key = (id(plan), side)
    if key not in _CACHE:
        _CACHE[key] = LayoutEngine(**plan.generation_params, terrace_stair=side).generate()
    return _CACHE[key]


def _grid(nav):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(nav["bits"]), dtype=np.uint8))[: nav["nx"] * nav["ny"]]
    return bits.reshape(nav["ny"], nav["nx"]).astype(bool)


def _cell(nav, x, y):
    return int((x - nav["x0"]) / nav["cell"]), int((y - nav["y0"]) / nav["cell"])


def _near(g, c, k=15):
    best = None
    for dj in range(-k, k + 1):
        for di in range(-k, k + 1):
            i, j = c[0] + di, c[1] + dj
            if 0 <= j < g.shape[0] and 0 <= i < g.shape[1] and g[j, i]:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, (i, j))
    return best[1] if best else None


def _reach(g, a):
    seen, dq = {a}, deque([a])
    while dq:
        i, j = dq.popleft()
        for n in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= n[1] < g.shape[0] and 0 <= n[0] < g.shape[1] and g[n[1], n[0]] and n not in seen:
                seen.add(n)
                dq.append(n)
    return seen


class TestStairSide(unittest.TestCase):
    def test_01_default_climbs_clockwise(self):
        """Automatic plans: the stair still stands on a South / West side away from the road, it rises
        towards the South / West where its top lands on the roof, and the 2D arrow and the 3D treads agree."""
        for cents, bhk, facing in ((3, "2BHK", "N"), (5, "3BHK", "E"), (4, "2BHK", "S"), (5, "2BHK", "W")):
            plan = _house(cents, bhk, facing, CAR)
            info = plan.engineering["terrace_stair"]
            self.assertNotEqual(info["word"], "front", (cents, facing))
            self.assertIsNone(info["want"])
            self.assertNotIn("terrace_stair", plan.generation_params)          # old plans rebuild unchanged
            self.assertTrue(info["climb_ok"], (cents, facing, info))
            self.assertEqual(plan.layout_state["terrace_stair"]["word"], info["word"])
            sc = next(s for s in plan.staircases if "External" in s.name)
            st = next(s for s in build_scene(plan)["stairs"] if s["external"])
            first, last = min(st["boxes"], key=lambda b: b[5]), max(st["boxes"], key=lambda b: b[5])
            axis = 1 if sc.direction in "NS" else 0
            rising = (last[axis] + last[axis + 2]) / 2 - (first[axis] + first[axis + 2]) / 2
            self.assertGreater(rising * (1 if sc.direction in "NE" else -1), 0, (cents, facing, sc.direction))
            self.assertTrue(st["bridge"])
        self.assertIn(">UP<", SVGRenderer().render_to_string(plan, mode="clean"))

    def test_02_every_side_on_request(self):
        """front / back / left / right put the stair on that side (compass by facing), the plan rebuilds bit
        for bit from generation_params, and its foot can be walked to from the gate and climbed to the roof."""
        for cents, bhk, facing in ((8, "3BHK", "N"), (5, "3BHK", "E"), (5, "2BHK", "W")):
            base = _house(cents, bhk, facing, CAR if facing != "W" else None)
            sides = plan_sides_for_facing(facing)
            for word in ("front", "back", "left", "right"):
                plan = _side(base, word)
                info = plan.engineering["terrace_stair"]
                self.assertEqual(info["word"], word, (cents, facing, word, info.get("refused")))
                self.assertEqual(info["compass"], sides[PLAN_SIDE[word]])
                self.assertEqual(plan.generation_params["terrace_stair"], word)
                self.assertTrue(plan.layout_state)                         # a real layout, re-plannable
                again = LayoutEngine(**plan.generation_params).generate()
                self.assertEqual([(r.name, r.x, r.y, r.w, r.h) for r in again.rooms],
                                 [(r.name, r.x, r.y, r.w, r.h) for r in plan.rooms])
                sc = build_scene(plan)
                st = next(s for s in sc["stairs"] if s["external"])
                self.assertTrue(st["bridge"], (facing, word))
                g0, nav = _grid(sc["nav"]), sc["nav"]
                gate = next(g for g in sc["gates"] if g["kind"] == "walk")
                start = _near(g0, _cell(nav, (gate["x0"] + gate["x1"]) / 2, sc["plot"]["h"] - 0.6))
                low = min(st["boxes"], key=lambda b: b[5])
                foot = _near(g0, _cell(nav, (low[0] + low[2]) / 2, (low[1] + low[3]) / 2), 6)
                self.assertIn(foot, _reach(g0, start), f"{facing} {word}: the first step cannot be reached")

    def test_03_front_stair(self):
        """On the road side: alongside the house face at one end, the main door on the rest, the car still
        parked, and the front elevation shows the flight."""
        base = _house(8, "3BHK", "N", CAR)
        plan = _side(base, "front")
        r = next(x for x in plan.rooms if "External" in x.name)
        fx, fy, fw, fd = plan.building_footprint
        self.assertGreater(r.y, fy + fd)                                     # in the front yard
        self.assertGreaterEqual(r.x, fx - 0.12)
        self.assertLessEqual(r.right, fx + fw + 0.12)                         # alongside the house face
        door = next(d for d in plan.doors if d.is_main_entrance)
        self.assertFalse(door.x < r.right and door.x + door.width > r.x, "the main door is behind the stair")
        self.assertTrue(any(x.category == "PARKING" for x in plan.rooms))
        self.assertIn("TERRACE STAIR", front_elevation_svg(plan))
        self.assertNotIn("TERRACE STAIR", front_elevation_svg(base))         # at the back it is hidden

    def test_04_refused_with_a_reason(self):
        """A side that cannot take the stair says why (measured) and the stair stays where it fits."""
        base = _house(3, "2BHK", "N", CAR)
        plan = _side(base, "front")
        info = plan.engineering["terrace_stair"]
        self.assertNotEqual(info["word"], "front")
        self.assertIn("front of the house is only", info["refused"])
        self.assertTrue(any("cannot take it" in n for n in plan.design_notes))
        card = next(c for c in stair_sides(base)["options"] if c["side"] == "front")
        self.assertEqual(card["status"], "no")
        self.assertIn("main door", card["reason"])

    def test_05_the_advisor(self):
        """Four cards, one recommended, Vastu in simple words, measured lines; houses of floors have none."""
        import time
        base = _house(8, "3BHK", "N", CAR)
        t = time.time()
        res = stair_sides(base)
        self.assertLess(time.time() - t, 2.0)
        self.assertTrue(res["available"])
        cards = {c["side"]: c for c in res["options"]}
        self.assertEqual(set(cards), {"front", "back", "left", "right"})
        self.assertEqual(sum(1 for c in cards.values() if c["recommended"]), 1)
        self.assertEqual(cards[base.engineering["terrace_stair"]["word"]]["status"], "now")
        rec = next(c for c in cards.values() if c["recommended"])
        self.assertTrue(rec["why"].startswith("Recommended"))
        for c in cards.values():
            if c["status"] == "no":
                continue
            v = c["vastu"]
            self.assertIn(v["word"], ("good", "fine", "not good", "avoid"))
            self.assertTrue(v["line"])
            if v["zone"] in ("S", "W", "SW"):
                self.assertEqual(v["word"], "good")
            self.assertTrue(c["lines"]["house"] and c["lines"]["yard"] and c["lines"]["use"])
            self.assertTrue(any("walk from the main door" in s for s in c["lines"]["use"]))
        self.assertTrue(any("road" in s for s in cards["front"]["lines"]["care"]))     # security at the front
        floors = _house(8, "3BHK", "N", None, num_floors=2)
        self.assertFalse(stair_sides(floors)["available"])

    def test_06_wishes_and_replan(self):
        """Typed wishes become terrace-stair intents; a re-plan meets them with the reason when it cannot."""
        base = _house(8, "3BHK", "N", CAR)
        ls = base.layout_state
        now = ls["terrace_stair"]["word"]
        got = understand("move the terrace stairs to the right side", ls)[0]["intents"]
        self.assertEqual(got, [{"type": "terrace_stair", "side": "right"}])
        self.assertEqual(understand("outside steps east side", ls)[0]["intents"],
                         [{"type": "terrace_stair", "side": "right"}])        # N-facing: East is on the right
        self.assertEqual(understand("mettlu mundu", ls)[0]["intents"], [{"type": "terrace_stair", "side": "front"}])
        self.assertTrue(understand(f"stairs {now} side", ls)[0]["note"].startswith("The terrace stair is already"))
        self.assertTrue(understand("move the stairs", ls)[0]["suggestions"])
        self.assertEqual(ie.describe_intent({"type": "terrace_stair", "side": "right"}, ls),
                         "Terrace stair → right side (East)")
        self.assertIsNone(ie.validate_intent({"type": "terrace_stair", "side": "right"}, ls))
        floors = _house(8, "3BHK", "N", None, num_floors=2)
        self.assertIn("no outside terrace stair",
                      ie.validate_intent({"type": "terrace_stair", "side": "back"}, floors.layout_state or {}))
        opts = LayoutEngine(**base.generation_params, base_layout=ls,
                            design_intents=[{"type": "terrace_stair", "side": "right"},
                                            {"type": "lock", "room": "kitchen"}]).generate_alternatives(2)
        for o in opts:
            rep = o.edit_report
            line = next(d for d in rep["intents"] if d["text"].startswith("Terrace stair"))
            self.assertTrue(line["satisfied"], line)
            self.assertEqual(rep["satisfied"], rep["total"])
            self.assertEqual(o.engineering["terrace_stair"]["word"], "right")
            self.assertEqual(o.generation_params["terrace_stair"], "right")
        # refused in a re-plan: the report says why
        small = _house(3, "2BHK", "N", CAR)
        o = LayoutEngine(**small.generation_params, base_layout=small.layout_state,
                         design_intents=[{"type": "terrace_stair", "side": "front"}]).generate_alternatives(1)[0]
        line = next(d for d in o.edit_report["intents"] if d["text"].startswith("Terrace stair"))
        self.assertFalse(line["satisfied"])
        self.assertIn("front of the house", line["blocked_by"])

    def test_08_a_wish_never_takes_away_another(self):
        """The owner's case (3 cents, N, 3BHK Luxury, a car parked at the front): no side but the back keeps the
        car, so the others are locked with that reason; asked anyway, the stair stays and the car stays."""
        fam = FamilyProfile(adults=2, children=0, needs_home_office=True)
        comps = {"car_parking": {"enabled": True, "slots": 1}, "bike_parking": {"enabled": True, "slots": 2}}
        p = LandUnitsEngine.solve_parcel(3, "Rectangle", 1.33)
        base = LayoutEngine(p.width_m, p.length_m, "3BHK", True, "N", "Luxury", family_profile=fam,
                            additional_components=comps).generate()
        self.assertTrue(any(r.category == "PARKING" for r in base.rooms))
        cards = {c["side"]: c for c in stair_sides(base)["options"]}
        for side in ("left", "right", "front"):
            self.assertEqual(cards[side]["status"], "no", side)
            self.assertIn("car parking", cards[side]["reason"])
            plan = LayoutEngine(**base.generation_params, terrace_stair=side).generate()
            self.assertTrue(any(r.category == "PARKING" for r in plan.rooms), side)          # the car stays
            self.assertTrue(any(g["kind"].startswith("vehicle") for g in plan.engineering["gates"]), side)
            self.assertEqual(plan.engineering["terrace_stair"]["word"], "back")
            self.assertIn("car parking would no longer fit", plan.engineering["terrace_stair"]["refused"])
        # the Design form offers only what keeps the car; without the car the sides open up
        from floorplan_generator.terrace_stair import form_sides
        kw = dict(plot_width=p.width_m, plot_height=p.length_m, bhk_config="3BHK", vastu_enabled=True,
                  orientation="N", quality_level="Luxury", family_profile=fam)
        r = form_sides(**kw, additional_components=comps)
        self.assertEqual([o["side"] for o in r["options"] if o["ok"]], ["back"])
        r2 = form_sides(**kw, additional_components={"bike_parking": {"enabled": True, "slots": 2}})
        self.assertGreaterEqual(sum(o["ok"] for o in r2["options"]), 2)
        # and a side chosen in the form is built and kept in the plan
        side = next(o["side"] for o in r2["options"] if o["ok"] and not o["auto"])
        plan = LayoutEngine(**kw, additional_components={"bike_parking": {"enabled": True, "slots": 2}},
                            terrace_stair=side).generate()
        self.assertEqual(plan.engineering["terrace_stair"]["word"], side)
        self.assertEqual(plan.generation_params["terrace_stair"], side)

    def test_09_the_stair_stays_on_the_plot(self):
        for cents, bhk, facing in ((3, "3BHK", "N"), (3, "2BHK", "E"), (4, "3BHK", "S"), (5, "2BHK", "W")):
            p = LandUnitsEngine.solve_parcel(cents, "Rectangle", 1.33)
            for fam in (FamilyProfile(adults=2, children=0, needs_home_office=True), FAM):
                plan = LayoutEngine(p.width_m, p.length_m, bhk, True, facing, "Luxury", family_profile=fam,
                                    additional_components=CAR).generate()
                r = next(x for x in plan.rooms if "External" in x.name)
                self.assertTrue(r.x >= 0 and r.y >= 0 and r.right <= plan.plot_width + 1e-6
                                and r.top <= plan.plot_height + 1e-6, (cents, facing, r))

    def test_07_history_names_the_move(self):
        from model_folder.intent_studio import version_changes
        base = _house(8, "3BHK", "N", CAR)
        moved = _side(base, "right")
        self.assertTrue(any("Terrace stair" in c for c in version_changes(base, moved)))


if __name__ == "__main__":
    unittest.main()
