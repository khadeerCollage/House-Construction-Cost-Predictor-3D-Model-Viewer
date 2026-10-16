"""
2026-10-16: the guided walk as a real visit — the camera looks at what the guide is talking about, sentence by
sentence (narration cues → line 'at'); a house is shown from the gate round the yard to every room and up its
stair onto the terrace for the view; an apartment shows one home on one floor and then asks: the next floor
(the same home, said in fewer words), or straight up to the terrace (by the lift), or finish.
"""

import base64
import os
import re
import sys
import unittest
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floorplan_generator.family_analyzer import FamilyProfile
from floorplan_generator.layout_engine import LayoutEngine
from floorplan_renderer import narration as nar
from floorplan_renderer import walkthrough as wt

FAM = FamilyProfile(adults=2, children=2, elderly=1)
YARD = {"car_parking": {"enabled": True, "slots": 1}, "garden": {"enabled": True, "pct": 10},
        "water_tank": {"enabled": True}}
CASES = {
    "villa": ((12.0, 15.0, "3BHK", True, "E", "Standard"), dict(family_profile=FAM, additional_components=YARD)),
    "villa_N": ((12.33, 16.4, "3BHK", True, "N", "Standard"), dict(family_profile=FAM,
                                                                    additional_components={"car_parking": {"enabled": True, "slots": 1}})),
    "apartment": ((14.0, 20.0, "2BHK", True, "N", "Standard"),
                  dict(family_profile=FAM, building_type="Apartment", num_floors=4, has_lift=True)),
    "floors": ((12.0, 18.0, "2BHK", True, "S", "Standard"), dict(family_profile=FAM, num_floors=3)),
    "duplex": ((12.0, 16.0, "3BHK", True, "S", "Standard"),
               dict(family_profile=FAM, building_type="Duplex", num_floors=2)),
}
_CACHE = {}


def _scene(name):
    if name not in _CACHE:
        args, kw = CASES[name]
        plan = LayoutEngine(*args, **kw).generate()
        _CACHE[name] = (plan, wt.build_scene(plan))
    return _CACHE[name]


def _grid(nav):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(nav["bits"]), dtype=np.uint8))[: nav["nx"] * nav["ny"]]
    return bits.reshape(nav["ny"], nav["nx"]).astype(bool)


def _ok(nav, x, y):
    g = _grid(nav)
    i, j = int((x - nav["x0"]) / nav["cell"]), int((y - nav["y0"]) / nav["cell"])
    return 0 <= j < g.shape[0] and 0 <= i < g.shape[1] and bool(g[j, i])


def _lines(stop, lang="en"):
    tk = stop["talk"][lang]
    out = list(tk.get("speech") or [])
    for v in (tk.get("by") or {}).values():
        out += v.get("speech") or []
    out += (tk.get("again") or {}).get("speech") or []
    return out


class TestCameraFollowsTheWords(unittest.TestCase):
    def test_01_sentences_carry_where_to_look(self):
        """Rooms look at their bed, sofa, counter, windows by compass side, the bath door; outside at the gate,
        the car, the garden, the sump; the roof at the tank corner, the South edge, the East, the view."""
        plan, sc = _scene("villa")
        self.assertIsNone(sc.get("talk_error"))
        tour = sc["tour"]
        by_cue = {}
        for st in tour:
            for lang in nar.LANGS:
                for ln in _lines(st, lang):
                    self.assertEqual(ln["k"], nar.key(lang, ln["t"]))          # the cue never changes the voice clip
                    if "at" in ln:
                        x, y, z, g = ln["at"]
                        self.assertIn(g, (0, 1))
                        self.assertTrue(-2 <= x <= plan.plot_width + 45 and -45 <= y <= plan.plot_height + 45, ln)
                        by_cue.setdefault(ln.get("c"), set()).add(st["kind"])
        for cue, kind in (("bed", "room"), ("counter", "room"), ("sofa", "room"), ("gate_car", "gate"),
                          ("car", "parking"), ("garden", "yard"), ("sump", "yard"), ("door", "door"),
                          ("steps", "door"), ("flight", "stair"), ("top", "stair"), ("tank", "terrace"),
                          ("solar", "terrace"), ("east", "terrace"), ("garden", "terrace")):
            self.assertIn(kind, by_cue.get(cue, set()), cue)
        self.assertTrue(any(c.startswith("win:") for c in by_cue))
        # a bedroom's bed sentence points at the drawn bed of that room
        bedroom = next(s for s in tour if s["kind"] == "room" and "Master" in s["title"])
        bed = next(ln for ln in _lines(bedroom) if ln.get("c") == "bed")
        r = next(rm for rm in sc["rooms"] if rm["name"] == bedroom["room"])["r"]
        self.assertTrue(r[0] <= bed["at"][0] <= r[2] and r[1] <= bed["at"][1] <= r[3], (bed, r))
        # captions sentence by sentence, beside the spoken lines (walk.js lights the one being said)
        for st in tour:
            tk = st["talk"]["en"]
            if tk.get("speech"):
                self.assertEqual(len(tk["lines"]), len(tk["speech"]), st["kind"])
        # the windows by compass side: a 'win:E' cue sits on a window that faces East
        comp = sc["compass"]
        for st in tour:
            for ln in _lines(st):
                if ln.get("c", "").startswith("win:") and ln["c"] != "win:any" and "at" in ln:
                    w = [w for w in sc["windows"] if w["room"] == st.get("room") and comp[w["side"]] == ln["c"][4:]]
                    self.assertTrue(w, (st.get("room"), ln["c"]))

    def test_02_a_house_is_shown_inside_and_out(self):
        """Gate → parking → round the yard → main door → every room → out to the terrace stair → the terrace."""
        for name in ("villa", "villa_N"):
            plan, sc = _scene(name)
            kinds = [s["kind"] for s in sc["tour"]]
            self.assertEqual(kinds[0], "start")
            self.assertEqual(kinds[1], "gate", name)
            self.assertIn("parking", kinds)
            if name == "villa":
                self.assertIn("yard", kinds)
                self.assertLess(kinds.index("yard"), kinds.index("door"))
            self.assertLess(kinds.index("gate"), kinds.index("door"))
            last_room = max(i for i, k in enumerate(kinds) if k == "room")
            self.assertEqual(kinds[last_room + 1:], ["stair", "terrace", "end"], name)
            stair, ter = sc["tour"][last_room + 1], sc["tour"][last_room + 2]
            self.assertTrue(stair.get("external"))
            # the outside stair is said with its side, its Vastu and its clockwise climb
            say = stair["talk"]["en"]["say"]
            self.assertIn("outside the house", say)
            info = plan.engineering["terrace_stair"]
            if info.get("climb_ok"):
                self.assertIn("clockwise", say)
            self.assertEqual(ter["level"], 1)
            self.assertTrue(_ok(sc["nav1"], *ter["target"]), name)               # stood on the roof
            o = sc["shell"]["outer"]
            x, y = ter["target"]
            edge = min(x - o[0], o[2] - x, y - o[1], o[3] - y)
            self.assertLess(edge, 1.3, f"{name}: by the parapet, for the view")
            for s in sc["tour"]:
                for p in s.get("path") or []:
                    self.assertTrue(_ok(sc["nav"], *p), (name, s["title"], p))
            # nobody is sent out on the road to look at the main door
            door = next(s for s in sc["tour"] if s["kind"] == "door")
            self.assertLess(door["path"][-1][1], plan.plot_height)

    def test_03_apartment_floor_by_floor(self):
        """One home on the lowest floor first, then 'next floor or the terrace?' — with buttons per floor."""
        for name in ("apartment", "floors"):
            plan, sc = _scene(name)
            tour = sc["tour"]
            kinds = [s["kind"] for s in tour]
            ch = tour[kinds.index("choice")]
            homes = [o["L"] for o in ch["options"]]
            self.assertEqual(ch["default"]["L"], homes[0])
            mi = kinds.index("more")
            self.assertEqual(kinds[mi - 1], "room")
            self.assertEqual(kinds[mi + 1], "terrace")
            for lang in nar.LANGS:
                by = tour[mi]["talk"][lang]["by"]
                self.assertEqual(set(by), {str(L) for L in homes})
                for L in homes:
                    acts = [o["act"] for o in by[str(L)]["options"]]
                    self.assertEqual(acts.count("terrace"), 1)
                    self.assertIn("end", acts)
                    if L != homes[-1]:
                        nxt = [o for o in by[str(L)]["options"] if o["act"] == "next"]
                        self.assertEqual({o["L"] for o in nxt}, {homes[homes.index(L) + 1]})
                        self.assertEqual({o["via"] for o in nxt}, set(ch["via"]))
                    else:
                        self.assertNotIn("next", acts)
            # the same rooms a floor up: fewer words, still in three languages, still looking at things
            rooms = [s for s in tour if s["kind"] == "room"]
            for s in rooms:
                for lang in nar.LANGS:
                    ag = s["talk"][lang]["again"]
                    self.assertTrue(ag["speech"])
                    self.assertLess(len(ag["speech"]), len(s["talk"][lang]["speech"]) + 1)
            ter = tour[mi + 1]
            o = sc["shell"]["outer"]
            self.assertGreater(ter["target"][1], (o[1] + o[3]) / 2, f"{name}: the street side of the roof")

    def test_04_duplex_goes_up_to_the_terrace(self):
        plan, sc = _scene("duplex")
        kinds = [s["kind"] for s in sc["tour"]]
        self.assertEqual(kinds[-2:], ["terrace", "end"])
        self.assertEqual(sc["tour"][-2]["level"], sc["top"])

    def test_05_new_words_in_three_languages(self):
        """Yard, outside stair, terrace view, next-floor: Telugu and Hindi speech in their own script."""
        latin = re.compile(r"[A-Za-z0-9]")
        for name in ("villa", "apartment"):
            plan, sc = _scene(name)
            for st in sc["tour"]:
                if st["kind"] not in ("yard", "stair", "terrace", "more", "room"):
                    continue
                for lang in ("te", "hi"):
                    for ln in _lines(st, lang):
                        self.assertIsNone(latin.search(ln["t"]), (name, st["kind"], lang, ln["t"]))
                        self.assertNotRegex(ln["t"], "[—·×²]")

    def test_07_never_stands_in_a_door_swing(self):
        """Doors close behind the guide (walk.js): he never stands where a leaf swings; a room shown from its
        doorway names the door that stays open."""
        for name in ("villa", "apartment", "duplex"):
            plan, sc = _scene(name)
            for st in sc["tour"]:
                if st["kind"] != "room":
                    continue
                at = st.get("target") or (st["path"][-1] if st.get("path") else None)
                if not at or st.get("peek"):
                    continue
                doors = (sc.get("upper") or {}).get("doors") if (isinstance(st.get("level"), int) and sc.get("upper")
                                                                   and st["level"] > sc["r0"]) else sc["doors"]
                for d in doors or []:
                    if d.get("room") != st["room"]:
                        continue
                    hx, hy = d["hinge"]
                    inside = (at[0] - d["mid"][0]) * d["n"][0] + (at[1] - d["mid"][1]) * d["n"][1] > -0.05
                    self.assertFalse(inside and ((at[0] - hx) ** 2 + (at[1] - hy) ** 2) ** 0.5 < d["w"] + 0.3,
                                     (name, st["title"], at, d["mid"]))

    def test_06_viewer_hooks(self):
        js = open(os.path.join(os.path.dirname(wt.__file__), "..", "model_folder", "walkthrough3d", "walk.js"),
                  encoding="utf-8").read()
        for word in ("function aimFor", "function onLine", "function steer", "director.lean", "function showMore",
                     "function moreGo", "tour.loop", "tk.again", 'st.kind === "terrace" && tour.choice',
                     "function silentNext", "tour.i === -1", "function routeCrosses", "function inOpening",
                     "tour.cur.peek"):
            self.assertIn(word, js)
        html = open(os.path.join(os.path.dirname(wt.__file__), "..", "model_folder", "walkthrough3d", "index.html"),
                    encoding="utf-8").read()
        self.assertIn("span.ln.now", html)


if __name__ == "__main__":
    unittest.main()
