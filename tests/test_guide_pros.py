"""
2026-10-16: the guide shows a home the way a good broker does — only its good points, measured from the plan,
the strongest first and never the same words twice in a row — and a bedroom's own bathroom (and dressing room)
is shown right after that bedroom, with its own warm words, in English, Guntur Telugu and Hindi.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floorplan_generator.family_analyzer import FamilyProfile
from floorplan_generator.layout_engine import LayoutEngine
from floorplan_renderer import narration as nar
from floorplan_renderer import walkthrough as wt

CASES = {
    "villa4": ((16.0, 22.0, "4BHK", True, "N", "Premium"),
               dict(family_profile=FamilyProfile(adults=4, children=2, elderly=2),
                    additional_components={"car_parking": {"enabled": True, "slots": 1}, "garden": {"enabled": True, "pct": 10}})),
    "villa3": ((12.0, 15.0, "3BHK", True, "E", "Standard"), dict(family_profile=FamilyProfile(adults=2, children=2, elderly=1))),
    "apartment": ((14.0, 20.0, "2BHK", True, "N", "Standard"),
                  dict(family_profile=FamilyProfile(adults=2, children=2, elderly=1), building_type="Apartment",
                       num_floors=4, has_lift=True)),
    "duplex": ((12.0, 16.0, "3BHK", True, "S", "Standard"),
               dict(family_profile=FamilyProfile(adults=2, children=2, elderly=1), building_type="Duplex", num_floors=2)),
}
_CACHE = {}


def _scene(name):
    if name not in _CACHE:
        args, kw = CASES[name]
        plan = LayoutEngine(*args, **kw).generate()
        _CACHE[name] = (plan, wt.build_scene(plan))
    return _CACHE[name]


def _rooms(sc):
    return [s for s in sc["tour"] if s["kind"] == "room"]


class TestBathroomAfterItsBedroom(unittest.TestCase):
    def test_01_order(self):
        """Every attached bath / dressing room comes right after its bedroom (dress first), never before it."""
        for name in CASES:
            plan, sc = _scene(name)
            rooms = _rooms(sc)
            n_baths = 0
            for i, st in enumerate(rooms):
                f = st["facts"]
                if f["kind"] not in ("ATTACHED_BATH", "DRESS") or not f.get("parent"):
                    continue
                n_baths += f["kind"] == "ATTACHED_BATH"
                j = i - 1
                while j >= 0 and rooms[j]["facts"]["kind"] in ("DRESS", "ATTACHED_BATH") and rooms[j]["facts"].get("parent"):
                    j -= 1
                self.assertGreaterEqual(j, 0, (name, st["title"]))
                self.assertEqual(rooms[j]["facts"]["kind"], f["parent"], (name, st["title"], rooms[j]["title"]))
                self.assertLessEqual(i - j, 2, (name, st["title"]))
            self.assertGreaterEqual(n_baths, 1, name)

    def test_02_the_bedroom_leads_you_in_and_the_bath_has_its_words(self):
        plan, sc = _scene("villa4")
        rooms = _rooms(sc)
        for i, st in enumerate(rooms):
            f = st["facts"]
            if f["kind"] == "ATTACHED_BATH" and f.get("parent"):
                bedroom = next(r for r in reversed(rooms[:i]) if r["facts"]["kind"] == f["parent"])
                lead = bedroom["talk"]["en"]["speech"][-1]
                self.assertIn("show you", lead["t"])
                self.assertEqual(lead.get("c"), "bath")
                self.assertTrue(lead.get("at"))                          # the camera turns to the bathroom door
                en, te, hi = (st["talk"][l]["say"] for l in ("en", "te", "hi"))
                par = nar.ROOM[f["parent"]]
                self.assertIn(f"{par[0]}'s own bathroom", en)
                self.assertIn(f"{par[1]} కి అటాచ్డ్ బాత్రూమ్ అండీ", te)
                self.assertIn("का अपना बाथरूम", hi)
                self.assertGreaterEqual(len(st["talk"]["te"]["speech"]), 3)    # not a bare name: its good points


class TestOnlyTheGood(unittest.TestCase):
    def test_03_highlights_are_measured(self):
        """A highlight is only said when the plan makes it true."""
        for name in CASES:
            plan, sc = _scene(name)
            for st in _rooms(sc):
                f = st["facts"]
                ids = [pid for pid, _ in f.get("_hl") or []]
                if "serve" in ids:
                    self.assertTrue({"DINING", "LIVING", "FAMILY"} & set(f["links"]), (name, st["title"]))
                if "quiet" in ids:
                    self.assertGreaterEqual(f["depth"], 0.6)
                if "largest" in ids:
                    self.assertTrue(f["largest"])
                if "east" in ids:
                    self.assertEqual(f["wins"], ["E"])
                if "v_special" in ids or "v_best" in ids or "v_good" in ids:
                    self.assertGreaterEqual(f["score"], 3)
                if "pipes" in ids:
                    self.assertTrue(f["wet_share"])
                self.assertLessEqual(len(ids), 4)

    def test_04_never_the_same_words_twice_in_a_row(self):
        for name in CASES:
            plan, sc = _scene(name)
            prev = set()
            for st in _rooms(sc):
                lines = {ln["t"] for ln in st["talk"]["te"]["speech"][1:]}
                self.assertFalse(lines & prev, (name, st["title"], lines & prev))
                prev = lines

    def test_05_the_home_in_one_line(self):
        plan, sc = _scene("villa4")
        end = sc["tour"][-1]["talk"]
        self.assertIn("What makes this home special", end["en"]["say"])
        self.assertIn("ఈ ఇంటి స్పెషల్ ఏంటంటే", end["te"]["say"])
        self.assertIn("इस घर की ख़ासियत", end["hi"]["say"])


if __name__ == "__main__":
    unittest.main()
