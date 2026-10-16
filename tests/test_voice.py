"""
The guide's words and voice (2026-10-15): the guided walk talks like a broker in English, Telugu or Hindi
(floorplan_renderer/narration.py) and is spoken by a local neural voice (model_folder/voice_engine.py →
voice_worker.py in .venv-voice) or the browser's voice.
"""

import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floorplan_generator.family_analyzer import FamilyProfile
from floorplan_generator.layout_engine import LayoutEngine
from floorplan_renderer import narration as nar
from floorplan_renderer import walkthrough as wt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAM = FamilyProfile(adults=2, children=2, elderly=1)
CASES = {
    "house_E": ((12.0, 15.0, "3BHK", True, "E", "Standard"),
                dict(family_profile=FAM, additional_components={"car_parking": {"enabled": True, "count": 1}})),
    "house_W": ((11.0, 14.0, "2BHK", True, "W", "Standard"), dict(family_profile=FAM)),
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


def _all_speech(stop, lang):
    tk = stop["talk"][lang]
    out = list(tk.get("speech") or [])
    for v in (tk.get("by") or {}).values():
        out += v.get("speech") or []
    out += (tk.get("again") or {}).get("speech") or []          # the same room, a floor up
    return out


class TestNumbers(unittest.TestCase):
    def test_01_numbers_in_words(self):
        self.assertEqual(nar.en_words(1938), "one thousand nine hundred and thirty eight")
        self.assertEqual(nar.te_words(215), "రెండు వందల పదిహేను")
        self.assertEqual(nar.te_words(100), "వంద")
        self.assertEqual(nar.te_words(123), "నూట ఇరవై మూడు")
        self.assertEqual(nar.te_words(1, attr=True), "ఒక")
        self.assertEqual(nar.te_words(2000), "రెండు వేలు")
        self.assertEqual(nar.hi_words(87), "सत्तासी")
        self.assertEqual(nar.hi_words(1938), "एक हज़ार नौ सौ अड़तीस")
        self.assertEqual(len(nar._HI), 100)
        sp = {l: nar.T(l, True) for l in nar.LANGS}
        self.assertEqual(sp["te"].h(4.5), "నాలుగున్నర")
        self.assertEqual(sp["hi"].h(3.5), "साढ़े तीन")
        self.assertEqual(sp["hi"].h(2.5), "ढाई")
        self.assertEqual(sp["en"].h(4.5), "four and a half")
        self.assertEqual(sp["te"].digits("301"), "త్రీ జీరో వన్")
        self.assertEqual(nar.T("en", False).h(4.5), "4.5")          # captions keep digits


class TestWords(unittest.TestCase):
    def test_02_every_stop_talks_in_three_languages(self):
        latin = re.compile(r"[A-Za-z0-9]")
        for name in CASES:
            plan, sc = _scene(name)
            self.assertIsNone(sc.get("talk_error"), name)
            self.assertEqual(sc["langs"], ["en", "te", "hi"])
            self.assertTrue(sc["tour"], name)
            for st in sc["tour"]:
                self.assertIn("talk", st, (name, st["kind"]))
                for lang in nar.LANGS:
                    tk = st["talk"][lang]
                    self.assertTrue(tk["title"] or tk.get("by"), (name, st["kind"], lang))
                    sp = _all_speech(st, lang)
                    self.assertTrue(sp, (name, st["kind"], lang))
                    for s in sp:
                        self.assertEqual(s["k"], nar.key(lang, s["t"]))
                        self.assertNotRegex(s["t"], r"\d", (name, lang, s["t"]))        # numbers are spoken as words
                        self.assertNotRegex(s["t"], "[—·×²]", (name, lang, s["t"]))
                        if lang in ("te", "hi"):                                       # the Indic voices read their script
                            self.assertIsNone(latin.search(s["t"]), (name, lang, s["t"]))
                    if lang == "te":
                        self.assertRegex(" ".join(x["t"] for x in sp), "[ఀ-౿]")
                    if lang == "hi":
                        self.assertRegex(" ".join(x["t"] for x in sp), "[ऀ-ॿ]")

    def test_03_words_are_measured_from_the_plan(self):
        plan, sc = _scene("house_E")
        rooms = {r["name"]: r for r in sc["rooms"]}
        n_rooms = 0
        for st in sc["tour"]:
            if st["kind"] != "room":
                continue
            f = st["facts"]
            self.assertEqual(f["w"], rooms[st["room"]]["facts"]["w"])
            say = st["talk"]["en"]["say"]
            if f["kind"] not in ("POOJA", "BATHROOM", "ATTACHED_BATH", "STORE", "UTILITY", "FOYER", "BALCONY", "SERVANT",
                                 "CORRIDOR", "LOBBY"):
                self.assertIn(f"{round(f['w'] / 0.3048)} by {round(f['h'] / 0.3048)} feet", say, st["room"])
                n_rooms += 1
            s = f.get("score")
            if s is not None and s < 0:                                    # a weak Vastu place is simply not talked about
                self.assertNotIn("Vastu", say.replace("Vastu-wise too", ""), st["room"])
        self.assertGreaterEqual(n_rooms, 4)
        start = sc["tour"][0]["talk"]
        cents = plan.plot_width * plan.plot_height / nar.M2_PER_CENT
        self.assertIn(f"{round(cents * 2) / 2:g} cent plot", start["en"]["say"])
        self.assertIn("గజాలు", start["te"]["say"])                        # Telugu buyers hear gajalu too
        self.assertIn("गज", start["hi"]["say"])                             # Hindi buyers hear gaj
        end = sc["tour"][-1]["talk"]["en"]["say"]
        if plan.vastu_score >= 70:
            self.assertIn(f"Vastu score {round(plan.vastu_score)} out of hundred", end)

    def test_03b_only_the_good(self):
        """The guide shows a home in its best light: no weak point, no 'honestly', no 'too small' in any language."""
        bad = {"en": ("Honestly", "honest", "small compared", "weak", "tight", "not a good", "can be improved",
                      "worry", "warm, a curtain"),
               "te": ("నిజం చెప్పాలంటే", "నిజం చెప్తా", "అంత మంచిది కాదు", "ఇరుకు —", "చిన్నవి"),
               "hi": ("सच कहूँ", "सच बताऊँ", "ठीक नहीं", "कमज़ोर", "तंग है", "छोटे हैं")}
        for name in CASES:
            plan, sc = _scene(name)
            for st in sc["tour"]:
                for lang in nar.LANGS:
                    for ln in _all_speech(st, lang):
                        for w in bad[lang]:
                            self.assertNotIn(w, ln["t"], (name, st["kind"], lang))

    def test_04_floor_choice_in_every_language(self):
        plan, sc = _scene("apartment")
        tour = sc["tour"]
        ch = next(s for s in tour if s["kind"] == "choice")
        levels = [o["L"] for o in ch["options"]]
        for lang in nar.LANGS:
            tk = ch["talk"][lang]
            self.assertEqual([o["L"] for o in tk["options"]], levels)
            self.assertEqual(set(tk["via"]), set(ch["via"]))
            ride = next(s for s in tour if s["kind"] == "ride")["talk"][lang]["by"]
            self.assertEqual(set(ride), {f"{L}|{v}" for L in levels for v in ch["via"]})
            front = next(s for s in tour if s["kind"] == "door" and s["facts"].get("door") == "front")["talk"][lang]["by"]
            self.assertEqual(set(front), {str(L) for L in levels})
        te = next(s for s in tour if s["kind"] == "door" and s["facts"].get("door") == "front")["talk"]["te"]["by"]
        self.assertIn("ఫ్లాట్ 301", te["3"]["title"])
        self.assertIn("త్రీ జీరో వన్", " ".join(x["t"] for x in te["3"]["speech"]))
        self.assertIn("apartment", tour[0]["talk"]["en"]["say"])
        floors = _scene("floors")[1]["tour"][0]["talk"]
        self.assertIn("rent", floors["en"]["say"])                         # a home on every floor: the rent pitch
        self.assertIn("అద్దె", floors["te"]["say"])

    def test_05_speech_lines_for_the_voice(self):
        plan, sc = _scene("apartment")
        for lang in nar.LANGS:
            lines = nar.speech_lines(sc["tour"], lang)
            keys = [l["k"] for l in lines]
            self.assertEqual(len(keys), len(set(keys)))
            every = {s["k"] for st in sc["tour"] for s in _all_speech(st, lang)}
            self.assertEqual(set(keys), every)
            self.assertEqual(lines[0]["k"], sc["tour"][0]["talk"][lang]["speech"][0]["k"])   # walking order


class TestVoiceEngine(unittest.TestCase):
    def test_06_voices_recordings_and_jobs(self):
        from model_folder import voice_engine as ve
        old = (ve.REFS, ve.CLIPS, ve.DATA, ve.VENV_PY)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ve.REFS, ve.CLIPS, ve.DATA = os.path.join(tmp, "refs"), os.path.join(tmp, "clips"), tmp
                # the owner's choice: Telugu Venkatesh, Hindi Rohan, English Ryan
                self.assertEqual([v["id"] for v in ve.voices("en")], ["piper-ryan", "browser"])
                self.assertEqual([v["id"] for v in ve.voices("te")], ["piper-venkatesh", "browser"])
                self.assertEqual(ve.voices("hi")[0]["id"], "piper-rohan")
                self.assertTrue(all(v["model"].startswith(("te/te_IN/", "hi/hi_IN/", "en/en_US/"))
                                    for l in ("en", "te", "hi") for v in ve.voices(l) if v["engine"] == "piper"))
                self.assertEqual(ve.OWN_LANGS, ())                                # own voice: GPU machines only
                ve.OWN_LANGS = ("te", "hi")
                vid = ve.save_recording("te", b"RIFF0000WAVEfmt ", ve.REF_SENTENCE["te"], "Amma")
                own = ve.voices("te")[0]
                self.assertEqual((own["id"], own["engine"], own["label"]), (vid, "f5", "Amma"))
                self.assertEqual(own["ref_text"], ve.REF_SENTENCE["te"])
                self.assertEqual(ve.voices("hi")[0]["id"], "piper-rohan")          # a Telugu recording is not Hindi
                ve.VENV_PY = os.path.join(tmp, "missing", "python.exe")
                self.assertIn("setup_voice.bat", ve.ready(own))
                self.assertIsNone(ve.ready(ve.voices("te")[-1]))                  # the browser voice always works
                items = [{"k": "a" * 16, "t": "x"}, {"k": "b" * 16, "t": "y"}]
                pr = ve.ensure(own, items)                                        # nothing installed: nothing starts
                self.assertEqual((pr["have"], pr["total"], pr["running"]), (0, 2, False))
                os.makedirs(ve.clip_dir(vid))
                open(os.path.join(ve.clip_dir(vid), "a" * 16 + ".wav"), "wb").close()
                self.assertEqual(ve.progress(own, items)["have"], 1)
                ve.delete_recording(vid)
                self.assertEqual(ve.voices("te")[0]["id"], "piper-venkatesh")
                self.assertFalse(os.path.exists(ve.clip_dir(vid)))
            finally:
                ve.REFS, ve.CLIPS, ve.DATA, ve.VENV_PY = old
                ve.OWN_LANGS = ()

    def test_07_viewer_plays_the_voice_offline(self):
        js = open(os.path.join(ROOT, "model_folder", "walkthrough3d", "walk.js"), encoding="utf-8").read()
        html = open(os.path.join(ROOT, "model_folder", "walkthrough3d", "index.html"), encoding="utf-8").read()
        for hook in ("function speakLines", "function words(", "function setLang", "`${v.url}/${line.k}.wav`",
                     "_status.json", "browserVoice.te", 'action: "lang"', "a.voices", "UI.walking[state.lang]"):
            self.assertIn(hook, js)
        self.assertIn('id="startLang"', html)
        self.assertIn('id="capLang"', html)
        self.assertNotRegex(js, r"https?://(?!www\.w3\.org)")                    # no network
        ui = open(os.path.join(ROOT, "model_folder", "walkthrough_ui.py"), encoding="utf-8").read()
        self.assertIn("voices=talk[\"voices\"]", ui)
        gi = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
        for p in ("model_folder/walkthrough3d/voice/", "model_folder/voice_data/"):
            self.assertIn(p, gi)                                                  # recordings never committed


if __name__ == "__main__":
    unittest.main()
