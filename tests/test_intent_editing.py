"""
Intent-based plan editing (floorplan_generator/intent_engine.py + LayoutEngine.generate_alternatives).

Covers: layout_state / generation_params recorded on every plan, natural-language
parsing, each intent type being honoured by the re-plan, refusal of wishes the
topology cannot honour (with a reason), honest "blocked by" explanations,
determinism, and that plain generation is unaffected by the editing hooks.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_generator.family_analyzer import FamilyProfile  # noqa: E402
from floorplan_generator import intent_engine as ie  # noqa: E402


def _base_2bhk():
    return LayoutEngine(plot_width=9.55, plot_height=12.71, bhk_config="2BHK", vastu_enabled=True,
                        orientation="N", family_profile=FamilyProfile(adults=2, children=2,
                                                                      needs_pooja_room=True)).generate()


def _replan(base, text, k=3):
    intents, msgs = ie.parse_intents(text, base.layout_state)
    assert intents, f"no intents parsed from {text!r}: {msgs}"
    return LayoutEngine(**base.generation_params, design_intents=intents,
                        base_layout=base.layout_state).generate_alternatives(k)


class TestIntentEditing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base = _base_2bhk()
        cls.ls = cls.base.layout_state

    def test_01_plan_records_editing_state(self):
        p = self.base
        self.assertTrue(p.generation_params)
        self.assertIn("rooms", self.ls)
        self.assertIn(self.ls["mode"], ("std", "deep", "corridor"))
        keys = {r.key for r in p.rooms if r.key}
        self.assertTrue({"kitchen", "bed1", "living"} <= keys)
        # rebuilding from the recorded params reproduces the plan exactly
        again = LayoutEngine(**p.generation_params).generate()
        self.assertEqual([(r.name, r.x, r.y, r.w, r.h) for r in again.rooms],
                         [(r.name, r.x, r.y, r.w, r.h) for r in p.rooms])
        # foyer fillers / dining spine / dress areas are not offered as movable blocks
        names = {v["name"] for _, v in ie.editable_rooms(self.ls)}
        self.assertNotIn("Foyer", names)
        self.assertFalse(any("Dress" in n for n in names))

    def test_02_parser(self):
        cases = {
            "move kitchen to north west": ("zone", "kitchen"),
            "kitchen opposite": ("zone", "kitchen"),
            "attached bathroom on the other side": ("flip", "bed1_bath"),
            "swap master bedroom and children's bedroom": ("swap", "bed1"),
            "pooja next to the living room": ("adjacent", "pooja"),
            "make the living room bigger": ("size", "living"),
            "keep the master bedroom where it is": ("lock", "bed1"),
        }
        for text, (typ, key) in cases.items():
            intents, msgs = ie.parse_intents(text, self.ls)
            self.assertEqual(len(intents), 1, f"{text}: {intents} {msgs}")
            it = intents[0]
            self.assertEqual(it["type"], typ, text)
            self.assertEqual(it.get("room") or it.get("a"), key, text)
        mirror, _ = ie.parse_intents("mirror the plan", self.ls)
        self.assertEqual(mirror, [{"type": "mirror"}])
        # several wishes in one sentence
        multi, _ = ie.parse_intents("keep master bedroom. move kitchen to the left", self.ls)
        self.assertEqual([i["type"] for i in multi], ["lock", "zone"])
        self.assertEqual(set(multi[1]["zones"]), {"NW", "W", "SW"})   # plan-left of a N-facing plot

    def test_03_unreachable_wishes_are_refused_with_reason(self):
        intents, msgs = ie.parse_intents("put the living room at the back", self.ls)
        self.assertEqual(intents, [])
        self.assertTrue(any("entrance" in m for m in msgs), msgs)
        intents, msgs = ie.parse_intents("swap the study and the kitchen", self.ls)   # no study here
        self.assertEqual(intents, [])
        self.assertTrue(msgs)

    def test_04_zone_move_is_met_and_reported(self):
        plans = _replan(self.base, "move kitchen to north west")
        self.assertGreaterEqual(len(plans), 2)
        for p in plans:
            rep = p.edit_report
            self.assertEqual(rep["satisfied"], rep["total"], rep)
            self.assertEqual(ie.current_zone(p.layout_state, "kitchen"), "NW")
            # complete plan: walls, doors to every room, windows, a main entrance
            self.assertTrue(p.walls and p.doors and p.windows)
            self.assertEqual(sum(d.is_main_entrance for d in p.doors), 1)
        sigs = {tuple(sorted((k, ie.current_zone(p.layout_state, k)) for k, _ in ie.editable_rooms(p.layout_state)))
                for p in plans}
        self.assertEqual(len(sigs), len(plans), "options must be distinct layouts")
        self.assertEqual(plans[0].edit_report["label"], "Balanced — recommended")

    def test_05_swap_flip_lock_size(self):
        p = _replan(self.base, "swap master bedroom and children's bedroom", k=1)[0]
        self.assertEqual(p.edit_report["satisfied"], 1, p.edit_report)
        self.assertEqual(ie.current_zone(p.layout_state, "bed1"), ie.current_zone(self.ls, "bed2"))

        p = _replan(self.base, "attached bathroom on the other side", k=1)[0]
        self.assertEqual(p.edit_report["satisfied"], 1, p.edit_report)
        b0 = self.ls["rooms"]["bed1_bath"]["cx"] - self.ls["rooms"]["bed1"]["cx"]
        b1 = p.layout_state["rooms"]["bed1_bath"]["cx"] - p.layout_state["rooms"]["bed1"]["cx"]
        self.assertLess(b0 * b1, 0, "bath must change sides of the bedroom")

        p = _replan(self.base, "keep master bedroom where it is; move kitchen to north west", k=1)[0]
        self.assertEqual(p.edit_report["satisfied"], 2, p.edit_report)
        self.assertEqual(ie.current_zone(p.layout_state, "bed1"), ie.current_zone(self.ls, "bed1"))

        p = _replan(self.base, "make the living room bigger", k=1)[0]
        self.assertEqual(p.edit_report["satisfied"], 1, p.edit_report)
        self.assertGreater(p.layout_state["rooms"]["living"]["area"], self.ls["rooms"]["living"]["area"] * 1.05)
        self.assertAlmostEqual(p.total_built_up_area, self.base.total_built_up_area, places=2)  # same envelope

    def test_06_mirror(self):
        p = _replan(self.base, "mirror the plan", k=1)[0]
        self.assertEqual(p.edit_report["satisfied"], 1, p.edit_report)
        mid = self.ls["footprint"][0] + self.ls["footprint"][2] / 2
        for k in ("kitchen", "bed1", "bed2"):
            before = self.ls["rooms"][k]["cx"] - mid
            after = p.layout_state["rooms"][k]["cx"] - mid
            self.assertLess(before * after, 0.01, k)

    def test_07_unmet_wish_is_explained(self):
        # an over-packed 4BHK: moving the kitchen into the taboo NE corner is not worth it
        base = LayoutEngine(plot_width=15, plot_height=20, bhk_config="4BHK", vastu_enabled=True,
                            orientation="W", num_floors=3, has_lift=True,
                            family_profile=FamilyProfile(adults=3, children=2, elderly=1, needs_home_office=True,
                                                         needs_pooja_room=True, needs_store_room=True),
                            additional_components={"car_parking": {"enabled": True, "slots": 1}}).generate()
        self.assertGreater(base.layout_state.get("relax_level", 0), 0)   # base needed relaxation
        p = _replan(base, "move kitchen to north east", k=1)[0]
        d = p.edit_report["intents"][0]
        if not d["satisfied"]:
            self.assertTrue(d.get("blocked_by"), d)

    def test_08_deterministic(self):
        a = _replan(self.base, "move kitchen to north west", k=1)[0]
        b = _replan(self.base, "move kitchen to north west", k=1)[0]
        self.assertEqual([(r.name, r.x, r.y, r.w, r.h) for r in a.rooms],
                         [(r.name, r.x, r.y, r.w, r.h) for r in b.rooms])

    def test_10_everyday_language(self):
        """Owners' own words: Indian English, spelling mistakes, complaints, Hindi/Malayalam, negation."""
        from floorplan_generator.wish_parser import understand
        cases = {
            "please make hall is big": ("size", "living", 1.2),
            "hall too small": ("size", "living", 1.2),
            "kitchen is too big": ("size", "kitchen", 0.85),
            "kichen to north west": ("zone", "kitchen", ["NW"]),
            "kitchen should be in SE": ("zone", "kitchen", ["SE"]),
            "master room attached bathroom other side": ("flip", "bed1_bath", None),
            "dont move master bedroom": ("lock", "bed1", None),
            "keep kitchen in north west": ("zone", "kitchen", ["NW"]),
            "adukkala vadakku padinjaru": ("zone", "kitchen", ["NW"]),
            "रसोई बड़ा करो": ("size", "kitchen", 1.2),
            "അടുക്കള വലിയ": ("size", "kitchen", 1.2),
            "masterbedroom to southwest": ("zone", "bed1", ["SW"]),
        }
        for text, (typ, key, val) in cases.items():
            res = understand(text, self.ls)
            self.assertEqual(len(res), 1, text)
            its = res[0]["intents"]
            self.assertEqual(len(its), 1, f"{text}: {res}")
            it = its[0]
            self.assertEqual(it["type"], typ, text)
            self.assertEqual(it.get("room") or it.get("a"), key, text)
            if typ == "size":
                self.assertAlmostEqual(it["factor"], val, msg=text)
            if typ == "zone":
                self.assertEqual(it["zones"], val, text)
        r = understand("kichen to north west", self.ls)[0]
        self.assertEqual(r["corrections"], [("kichen", "kitchen")])
        self.assertEqual(r["confidence"], "medium")
        # two wishes in one sentence, split on "and"; "swap a and b" is NOT split
        self.assertEqual(len(understand("make hall big and kitchen small", self.ls)), 2)
        self.assertEqual(len(understand("swap kitchen and master bedroom", self.ls)), 1)
        # negation → anywhere reachable except the named zone
        neg = understand("no kitchen in north east", self.ls)[0]["intents"][0]
        self.assertNotIn("NE", neg["zones"])
        self.assertTrue(neg["zones"])
        # lower-case Hindi "se" (= from) is not South-East; unsupported wishes get a helpful note
        self.assertFalse(any(i.get("zones") == ["SE"] for r in understand("kitchen se hall tak", self.ls)
                             for i in r["intents"]))
        wall = understand("remove the wall between kitchen and hall", self.ls)[0]
        self.assertFalse(wall["intents"])
        self.assertIn("not possible", wall["note"])
        # "no toilet near pooja" is a keep-apart wish, never turned into "next to"
        apart = understand("no toilet near pooja", self.ls)[0]
        self.assertFalse(apart["intents"])
        self.assertIn("Already taken care of", apart["note"])
        # unclear → suggestions, never a silent guess
        vague = understand("kitchen", self.ls)[0]
        self.assertFalse(vague["intents"])
        self.assertTrue(vague["suggestions"])

    def test_11_edits_are_priced(self):
        from model_folder.cost_engine import CostEngine
        e0 = CostEngine.estimate_for_plan(self.base, quality_level="Standard", country="India")
        self.assertEqual(e0["layout_delta"]["items"], [])
        self.assertNotIn(CostEngine.LAYOUT_LINE, e0["extras"])
        moved = _replan(self.base, "move kitchen to north west", k=3)
        priced = [CostEngine.estimate_for_plan(p, quality_level="Standard", country="India") for p in moved]
        for p, e in zip(moved, priced):
            self.assertEqual(p.layout_state["ref_quantities"], self.ls["ref_quantities"])   # carried forward
            items = e["layout_delta"]["items"]
            self.assertAlmostEqual(sum(i["amount"] for i in items), e["layout_delta"]["total"], delta=2)
            if items:
                self.assertNotEqual(round(e["total_cost"]), round(e0["total_cost"]))

    def test_12_parking_on_narrow_plot(self):
        """3-cent plot (9.55 m frontage): the car bay used to be silently dropped."""
        for facing in "NESW":
            p = LayoutEngine(plot_width=9.55, plot_height=12.71, bhk_config="2BHK", vastu_enabled=True,
                             orientation=facing,
                             additional_components={"car_parking": {"enabled": True, "slots": 1},
                                                    "garden": {"enabled": True, "pct": 10}}).generate()
            cars = [r for r in p.rooms if r.category == "PARKING"]
            self.assertEqual(len(cars), 1, facing)
            car = cars[0]
            main = next(d for d in p.doors if d.is_main_entrance)
            self.assertTrue(main.x + main.width <= car.x - 0.3 or main.x >= car.x + car.w + 0.3,
                            f"{facing}: main door behind the car bay")

    def test_09_grid_helpers(self):
        # E-facing: road (East) at the top, North on the plan's left
        self.assertEqual(ie.cell_zone(0, 1, "E"), "E")
        self.assertEqual(ie.cell_zone(-1, 1, "E"), "NE")
        self.assertEqual(ie.plan_grid("N")[0], ["NW", "N", "NE"])
        self.assertEqual(ie.zone_cell("SW", "N"), (-1, -1))
        self.assertEqual(set(ie.feasible_zones(self.ls, "living")), {"NW", "N", "NE"})
        score, taboo = ie.zone_fit(self.ls, "kitchen", "NE")
        self.assertTrue(taboo)


if __name__ == "__main__":
    unittest.main()
