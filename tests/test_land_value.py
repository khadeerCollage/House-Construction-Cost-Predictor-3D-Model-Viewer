"""
Land value & property worth (floorplan_generator/land_value.py + CostEngine location index).

Covers: the street-position factors reproduce real Chilakaluripet market rates (inner ₹6–8 L/cent,
main road ₹10–11 L, outskirts ₹3–4 L) from its official guideline value, every AP and Telangana
district carries official town data, registration charges follow each state's rules (on the higher
of market and guideline value, rural split, women's concession), the owner's own rate wins,
building cost follows the plot's location, and the money report adds up.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator import land_value as lv  # noqa: E402

PLOT_3C = 3.0 * 40.4686            # m²
LAKH = 1e5


def _per_cent_lakh(rate_sqft):
    return lv.per_unit(rate_sqft, "per cent") / LAKH


class TestLandValue(unittest.TestCase):

    def test_01_chilakaluripet_matches_local_market(self):
        expect = {"inner": (6.0, 8.0), "main_road": (10.0, 11.0), "outskirts": (3.0, 4.0)}
        for area, (lo, hi) in expect.items():
            e = lv.estimate_land(PLOT_3C, "Andhra Pradesh", "Palnadu", "Chilakaluripet", area)
            self.assertEqual(e.confidence, "official value", area)
            mid = _per_cent_lakh(e.rate_sqft[1])
            self.assertTrue(lo - 0.5 <= mid <= hi + 0.5, (area, mid))
        inner = lv.estimate_land(PLOT_3C, "Andhra Pradesh", "Palnadu", "Chilakaluripet", "inner")
        main = lv.estimate_land(PLOT_3C, "Andhra Pradesh", "Palnadu", "Chilakaluripet", "main_road")
        self.assertGreater(main.value[1], inner.value[1] * 1.3)

    def test_02_official_data_covers_every_district(self):
        ap, ts = lv.districts("Andhra Pradesh"), lv.districts("Telangana")
        self.assertGreaterEqual(len(ap), 20, "AP has 26-28 districts")
        self.assertGreaterEqual(len(ts), 10)
        for st, ds in (("Andhra Pradesh", ap), ("Telangana", ts)):
            for d in ds:
                towns = lv.towns(st, d)
                self.assertTrue(towns, f"{st}/{d} has no towns")
                e = lv.estimate_land(PLOT_3C, st, d, towns[0], "inner")
                self.assertGreater(e.value[1], 0, f"{st}/{d}")

    def test_03_registration_rules(self):
        ap = lv.estimate_land(PLOT_3C, "Andhra Pradesh", "Palnadu", "Chilakaluripet", "inner")
        r = lv.registration_charges(ap)
        self.assertAlmostEqual(r["pct"], 7.5)                    # 5 stamp + 1.5 transfer + 1 registration
        guide = ap.guideline_sqft * ap.plot_sqft
        self.assertAlmostEqual(r["base"], max(ap.value[1], guide), delta=1)
        ts_town = lv.estimate_land(PLOT_3C, "Telangana", "", "", "inner", "town")
        ts_village = lv.estimate_land(PLOT_3C, "Telangana", "", "", "village", "village")
        self.assertAlmostEqual(lv.registration_charges(ts_town)["pct"], 6.0)
        self.assertAlmostEqual(lv.registration_charges(ts_village)["pct"], 7.5)
        up = lv.estimate_land(PLOT_3C, "Uttar Pradesh", "", "", "inner", "town")
        self.assertGreater(lv.registration_charges(up)["female_saves"], 0)
        self.assertLess(lv.registration_charges(up, buyer_female=True)["total"],
                        lv.registration_charges(up)["total"])
        for name, st in lv.STATES.items():
            self.assertTrue(st["portal"][1].startswith("https://"), name)
            self.assertTrue(3 <= st["stamp"][0] + st["reg"] + st.get("other", 0) <= 14, name)

    def test_04_own_rate_wins(self):
        e = lv.estimate_land(PLOT_3C, "Andhra Pradesh", "Palnadu", "Chilakaluripet", "inner",
                             known_rate=700000, known_unit="per cent")
        self.assertEqual(e.confidence, "your rate")
        self.assertAlmostEqual(e.value[1], 21.0 * LAKH, delta=0.05 * LAKH)
        sq = lv.estimate_land(PLOT_3C, "Kerala", known_rate=1500, known_unit="per sq ft")
        self.assertAlmostEqual(sq.rate_sqft[1], 1500)

    def test_05_building_cost_follows_the_plot(self):
        from floorplan_generator.layout_engine import LayoutEngine
        from model_folder.cost_engine import CostEngine
        plan = LayoutEngine(9.55, 12.71, "2BHK", True, "N").generate()
        town = lv.estimate_land(PLOT_3C, "Andhra Pradesh", "Palnadu", "Chilakaluripet", "inner")
        base = CostEngine.estimate_for_plan(plan, country="India")
        here = CostEngine.estimate_for_plan(plan, country="India",
                                            location={"name": town.place, "index": town.construction_index})
        self.assertLess(here["effective_rate"], base["effective_rate"])
        self.assertTrue(1800 <= here["effective_rate"] <= 2300, here["effective_rate"])   # Guntur-area band
        self.assertIn("Chilakaluripet", here["region"])
        kochi = CostEngine.estimate_for_plan(plan, country="India",
                                             location={"name": "Kochi", "index": lv.CITIES["Kochi / Ernakulam"][2]})
        self.assertGreater(kochi["total_cost"], here["total_cost"])

        pw = lv.property_worth(town, here)
        self.assertAlmostEqual(pw["investment"], town.value[1] + pw["registration"]["total"] + here["total_cost"],
                               delta=2)
        self.assertAlmostEqual(pw["worth"][1], town.value[1] + here["construction_cost"], delta=2)
        own = lv.property_worth(town, here, already_own=True)
        self.assertAlmostEqual(own["investment"], here["total_cost"], delta=1)


if __name__ == "__main__":
    unittest.main()
