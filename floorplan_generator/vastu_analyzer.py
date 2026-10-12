"""
Ultra-Premium Vastu Shastra Analysis & Audit Engine
===================================================
Comprehensive Vastu Purusha Mandala analytical engine adhering to Vedic architectural
principles (Mayamatam, Mansara, IS 15500 standard).

Provides:
  1. Detailed 9-Zone elemental audit (Pancha Bhoota)
  2. 32-Pada entrance gate analysis with auspiciousness ratings
  3. Energy vector analysis (Pranic NE->SW, Jaivik N->S)
  4. Defect identification (Doshas) and non-demolition remedies (Parihara)
  5. Formal Vastu Compliance Audit Certificate generator
  6. House Recommendation Engine (optimal configuration, room swaps)
  7. High-resolution standalone Vastu Mandala SVG visualizer
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import datetime

from floorplan_generator.vastu_engine import (
    VastuEngine, VastuZone, EntranceGateInfo, _PADA_DATABASE, pada_for_point,
    plan_sides_for_facing,
)
from floorplan_generator.layout_engine import GeneratedFloorPlan, RoomRect, DoorPlacement


# =============================================================================
# Zone Elemental & Planetary Metadata
# =============================================================================
ZONE_METADATA = {
    VastuZone.NE: {
        "name": "Ishan (North-East)",
        "element": "Jal (Water)",
        "deity": "Ishana (Lord Shiva)",
        "planet": "Jupiter (Brihaspati)",
        "ideal": "Pooja Room, Open Verandah, Water Body, Meditation",
        "taboo": "Toilet, Septic Tank, Heavy Store, Kitchen",
        "color": "#DBEAFE",
        "energy_type": "Pranic Inlet (Supreme Cosmic Energy)"
    },
    VastuZone.E: {
        "name": "Indra / Surya (East)",
        "element": "Agni-Surya (Solar / Fire)",
        "deity": "Indra (King of Devas)",
        "planet": "Sun (Surya)",
        "ideal": "Main Entrance, Living Room, Study, Foyer",
        "taboo": "Toilet, Heavy Overhead Tank, Bedroom head towards West",
        "color": "#FED7AA",
        "energy_type": "Vital Solar Inflow"
    },
    VastuZone.SE: {
        "name": "Agneya (South-East)",
        "element": "Agni (Fire)",
        "deity": "Agni (God of Fire)",
        "planet": "Venus (Shukra)",
        "ideal": "Kitchen (Cook facing East), Inverter, Electrical Hub",
        "taboo": "Master Bedroom, Underground Sump, Pooja",
        "color": "#FECACA",
        "energy_type": "Thermal Energy Matrix"
    },
    VastuZone.S: {
        "name": "Yama (South)",
        "element": "Prithvi-Agni (Earth/Fire)",
        "deity": "Yama (Lord of Dharma & Justice)",
        "planet": "Mars (Mangal)",
        "ideal": "Bedroom 2, Heavy Stairs, Overhead Storage",
        "taboo": "Underground Water, Low Plinth, Main Entrance (S5-S8)",
        "color": "#E2E8F0",
        "energy_type": "Stability & Protection"
    },
    VastuZone.SW: {
        "name": "Nairutya (South-West)",
        "element": "Prithvi (Earth)",
        "deity": "Niruthi (Demon King / Ancestor Lord)",
        "planet": "Rahu (Ascending Node)",
        "ideal": "Master Bedroom, Cash Locker, Heavy Structural RCC Tank",
        "taboo": "Main Entrance, Underground Sump, Toilet, Cut Corner",
        "color": "#FEF08A",
        "energy_type": "Grounding & Foundation Authority"
    },
    VastuZone.W: {
        "name": "Varuna (West)",
        "element": "Jal-Vayu (Water / Air)",
        "deity": "Varuna (Lord of Waters & Rain)",
        "planet": "Saturn (Shani)",
        "ideal": "Children Bedroom, Study Desk, Dining Room",
        "taboo": "Pooja facing West deity, Main Entrance (W1, W8)",
        "color": "#CCFBF1",
        "energy_type": "Intellect & Sustenance"
    },
    VastuZone.NW: {
        "name": "Vayavya (North-West)",
        "element": "Vayu (Air / Wind)",
        "deity": "Vayu (Lord of Wind & Movement)",
        "planet": "Moon (Chandra)",
        "ideal": "Guest Bedroom, Bathrooms, Utility, Pantry, Garage",
        "taboo": "Master Bedroom, Heavy Fixed Safe",
        "color": "#F1F5F9",
        "energy_type": "Circulation & Purification"
    },
    VastuZone.N: {
        "name": "Kubera (North)",
        "element": "Jal-Akasha (Water / Ether)",
        "deity": "Kubera (Lord of Wealth)",
        "planet": "Mercury (Budha)",
        "ideal": "Living Room, Family Lounge, Cash Box, Open Terrace",
        "taboo": "Toilet, Heavy Stairs, Septic Tank",
        "color": "#D1FAE5",
        "energy_type": "Wealth Attraction & Knowledge Inflow"
    },
    VastuZone.CENTER: {
        "name": "Brahmasthan (Center)",
        "element": "Akasha (Ether / Space)",
        "deity": "Brahma (The Creator)",
        "planet": "All Cosmic Influences",
        "ideal": "Open Courtyard, Skylight, Clear Passage, Open Living Hall",
        "taboo": "Columns, Load Bearing Walls, Toilet, Kitchen, Stairs",
        "color": "#FFFFFF",
        "energy_type": "Cosmic Harmonic Equilibrium"
    },
}


# =============================================================================
# Vastu Analyzer Class
# =============================================================================
class VastuAnalyzer:
    """
    Comprehensive architectural Vastu Shastra audit and recommendation engine.
    """

    SITE_CATEGORIES = {"PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"}

    def __init__(self, plan: GeneratedFloorPlan):
        self.plan = plan
        # House rooms are judged on the building's own mandala; site amenities on the plot mandala.
        self.engine = VastuEngine.for_plan(plan, scope="building")
        self.site_engine = VastuEngine.for_plan(plan, scope="plot")
        self.all_zones = self.engine.get_all_zones()

    def _is_site(self, r: RoomRect) -> bool:
        return r.category in self.SITE_CATEGORIES or "External" in r.name

    def _key(self, r: RoomRect) -> str:
        role = getattr(r, "role", "")
        return role if role in ("MASTER_BEDROOM", "CHILD_BEDROOM", "GUEST_BEDROOM", "ELDER_BEDROOM") else r.category

    def _zone(self, r: RoomRect) -> VastuZone:
        eng = self.site_engine if self._is_site(r) else self.engine
        return eng.get_zone_for_position(r.cx, r.cy)

    def _skip(self, r: RoomRect) -> bool:
        return any(t in r.name for t in ("Dress", "Wardrobe", "Duct"))

    # =========================================================================
    # 1. Zone-by-Zone Analysis
    # =========================================================================
    def get_zone_analysis(self) -> List[Dict[str, Any]]:
        """
        Computes detailed zone-by-zone audit including placed rooms,
        scores, status, deities, and specific architectural observations.
        """
        results = []
        
        # Build map of rooms occupying each zone
        zone_rooms: Dict[VastuZone, List[RoomRect]] = {z: [] for z in VastuZone}
        for r in self.plan.rooms:
            if self._skip(r):
                continue
            zone_rooms[self._zone(r)].append(r)

        for zone_enum, meta in ZONE_METADATA.items():
            rooms_here = zone_rooms.get(zone_enum, [])
            room_names = [r.name for r in rooms_here]
            categories = [self._key(r) for r in rooms_here]
            
            # Compute zone composite score
            if not categories:
                # Empty zone — check if this is good (e.g. Center open is good)
                if zone_enum == VastuZone.CENTER:
                    score = 10
                    status = "OPTIMAL"
                    obs = "Brahmasthan remains clear and uncluttered, preserving central cosmic resonance."
                elif zone_enum in (VastuZone.NE, VastuZone.N, VastuZone.E):
                    score = 8
                    status = "GOOD"
                    obs = f"{meta['name']} is lightly loaded, promoting positive solar and pranic energy flow."
                else:
                    score = 7
                    status = "ACCEPTABLE"
                    obs = f"Zone is open; maintains natural spatial equilibrium."
            else:
                zone_score_sum = 0
                has_taboo = False
                for cat in categories:
                    sc = self.engine.get_room_score(cat, zone_enum.value)
                    zone_score_sum += sc
                    if self.engine.is_taboo(cat, zone_enum.value):
                        has_taboo = True
                
                avg_score = zone_score_sum / len(categories)
                if has_taboo:
                    score = -50
                    status = "TABOO"
                    obs = f"Architectural defect: {', '.join(room_names)} placed in {meta['name']} violates core Vastu rules."
                elif avg_score >= 8:
                    score = int(avg_score)
                    status = "OPTIMAL"
                    obs = f"Exemplary alignment: {', '.join(room_names)} harmonizes perfectly with the {meta['element']} element."
                elif avg_score >= 5:
                    score = int(avg_score)
                    status = "GOOD"
                    obs = f"Compliant placement: {', '.join(room_names)} operates beneficially under {meta['deity']}."
                elif avg_score >= 0:
                    score = int(avg_score)
                    status = "ACCEPTABLE"
                    obs = f"Acceptable layout; minor adjustments or subtle remedies can further elevate energy flow."
                else:
                    score = int(avg_score)
                    status = "DEFECT"
                    obs = f"Sub-optimal energy placement: {', '.join(room_names)} conflicts with {meta['element']} governing forces."

            results.append({
                "zone_key": zone_enum.value,
                "zone_name": meta["name"],
                "element": meta["element"],
                "deity": meta["deity"],
                "planet": meta["planet"],
                "rooms": room_names if room_names else ["(Open / Unobstructed)"],
                "score": score,
                "status": status,
                "ideal": meta["ideal"],
                "observation": obs,
                "color": meta["color"],
                "energy_type": meta["energy_type"]
            })

        return results

    # =========================================================================
    # 2. 32-Pada Entrance Gate Audit
    # =========================================================================
    def audit_entrance(self) -> Dict[str, Any]:
        """
        Determines the exact 32-pada gate where the main entrance door resides,
        its presiding deity, and traditional karmic/prosperity outcome.
        """
        main_door = None
        for d in self.plan.doors:
            if d.is_main_entrance:
                main_door = d
                break
        if not main_door and self.plan.doors:
            main_door = self.plan.doors[0]

        if not main_door:
            return {
                "pada_id": "N3",
                "deity": "Mukhya",
                "direction": "North",
                "rating": "SUPER_AUSPICIOUS",
                "score": 10,
                "effect": "Supreme wealth, continuous cashflow, and prestige.",
                "remedy": ""
            }

        # Door centre on its wall, then the pada along the building facade
        # (each side split into 8 equal padas, counted clockwise from its corner).
        if main_door.wall_side in ("N", "S"):
            px, py = main_door.x + main_door.width / 2.0, main_door.y
        else:
            px, py = main_door.x, main_door.y + main_door.width / 2.0
        fp = getattr(self.plan, "building_footprint", None)
        rect = fp if fp and fp[2] > 0 else (0.0, 0.0, self.plan.plot_width, self.plan.plot_height)
        matched_pada = pada_for_point(px, py, rect, self.plan.orientation)
        deg = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}[matched_pada.direction[0]]

        return {
            "pada_id": matched_pada.pada_id,
            "deity": matched_pada.deity,
            "direction": matched_pada.direction,
            "rating": matched_pada.rating,
            "score": matched_pada.score,
            "effect": matched_pada.effect,
            "remedy": matched_pada.remedy,
            "door_x": main_door.x,
            "door_y": main_door.y,
            "angle_deg": round(deg, 1)
        }

    # =========================================================================
    # 3. Pancha Bhoota Elemental Balance
    # =========================================================================
    def get_elemental_balance(self) -> Dict[str, Any]:
        """
        Audits balance across the 5 sacred elements: Jal, Agni, Prithvi, Vayu, Akasha.
        """
        zone_data = self.get_zone_analysis()
        scores_by_element = {
            "Jal (Water)": [],
            "Agni (Fire)": [],
            "Prithvi (Earth)": [],
            "Vayu (Air)": [],
            "Akasha (Space)": []
        }

        primary = {
            "NE": "Jal (Water)", "N": "Jal (Water)",
            "E": "Agni (Fire)", "SE": "Agni (Fire)",
            "S": "Prithvi (Earth)", "SW": "Prithvi (Earth)",
            "W": "Vayu (Air)", "NW": "Vayu (Air)",
            "CENTER": "Akasha (Space)",
        }
        for z in zone_data:
            scores_by_element[primary[z["zone_key"]]].append(max(0, z["score"]))

        balance = {}
        for elem, sc_list in scores_by_element.items():
            avg_sc = sum(sc_list) / max(1, len(sc_list))
            # Normalise to 100%
            pct = min(100.0, avg_sc * 10.0)
            status = "BALANCED" if pct >= 80 else "HARMONIC" if pct >= 60 else "NEEDS_REMEDY"
            balance[elem] = {
                "percentage": round(pct, 0),
                "status": status,
                "health": "Optimal" if pct >= 80 else "Moderate" if pct >= 60 else "Weakened"
            }

        return balance

    # =========================================================================
    # 4. Defect (Dosha) Detection and Non-Demolition Remedies
    # =========================================================================
    def get_defects_and_remedies(self) -> List[Dict[str, str]]:
        """
        Detects specific classical architectural defects and provides precise
        non-demolition remedial treatments (Parihara).
        """
        defects = []
        for r in self.plan.rooms:
            if self._skip(r):
                continue
            z = self._zone(r)
            key = self._key(r)
            if self._is_site(r) and self.site_engine.is_taboo(key, z.value):
                defects.append({
                    "defect": f"Site Dosha: {r.name} in the {z.value} of the plot",
                    "severity": "HIGH",
                    "impact": "Water bodies in Niruthi/Agni or fire equipment in Ishan disturb the plot's energy balance.",
                    "remedy": "Relocate during construction if possible; otherwise mark the zone with a copper strip and keep it clean and dry."
                })
                continue
            if key == "STAIRS" and z in (VastuZone.NE, VastuZone.CENTER):
                defects.append({
                    "defect": "Sopana Dosha (Staircase in Ishan / Brahmasthan)",
                    "severity": "HIGH",
                    "impact": "Heavy stair mass blocks the pranic inflow; associated with financial stress.",
                    "remedy": "1. Keep the space under the stair open (no toilet/store). 2. Ensure stairs ascend clockwise. 3. Place a brass pyramid at the first step."
                })
            if key == "MASTER_BEDROOM" and z in (VastuZone.NE, VastuZone.NW, VastuZone.SE):
                defects.append({
                    "defect": "Griha-Swami Dosha (Master Bedroom away from Nairutya)",
                    "severity": "MODERATE",
                    "impact": "Weak authority and decision-making for the head of the family.",
                    "remedy": "1. Sleep with head towards South. 2. Keep heavy furniture along the South-West walls. 3. Earthy tones on SW walls."
                })
            
            # Toilet in NE (Ishan Dosha)
            if r.category == "BATHROOM" and z == VastuZone.NE:
                defects.append({
                    "defect": "Ishan Dosha (Sanitation in Divine Water Corner)",
                    "severity": "CRITICAL",
                    "impact": "Neurological stress, spiritual blockage, severe wealth drainage.",
                    "remedy": "1. Install Zinc metal strip around toilet base in tile joints. 2. Place sea salt in bronze bowl, replaced weekly. 3. Mount full-height mirror on external face of bathroom door."
                })
            # Kitchen in NE (Fire in Water)
            elif r.category == "KITCHEN" and z == VastuZone.NE:
                defects.append({
                    "defect": "Agni-Jal Conflict (Kitchen in North-East)",
                    "severity": "HIGH",
                    "impact": "Chronic digestive health issues, frequent disputes between family members.",
                    "remedy": "1. Place green marble slab under the gas cooktop. 2. Mount a Copper Surya Yantra on East wall. 3. Keep cooking vessel lids always closed."
                })
            # Master bedroom in SE (Agni Corner)
            elif r.category == "BEDROOM" and "Master" in r.name and z == VastuZone.SE:
                defects.append({
                    "defect": "Agneya Shayana (Master Suite in Fire Quadrant)",
                    "severity": "MODERATE",
                    "impact": "Restlessness, insomnia, short temper, and frequent marital disputes.",
                    "remedy": "1. Position bed with headboard strictly towards South. 2. Paint interior walls in cooling pastel green or pale cream. 3. Place Rose Quartz sphere on bedside nightstand."
                })
            # Toilet in Center (Brahmasthan)
            elif r.category == "BATHROOM" and z == VastuZone.CENTER:
                defects.append({
                    "defect": "Brahmasthan Bhedan (Sanitation in Sacred Center)",
                    "severity": "CRITICAL",
                    "impact": "Instability in family health, heavy heart/chest complaints, general stagnancy.",
                    "remedy": "1. Embed 4 Brass Pyramids at ceiling level pointing inwards. 2. Keep the toilet door permanently closed with magnetic latch. 3. Apply golden tape along perimeter."
                })

        # Check entrance defects
        entrance_audit = self.audit_entrance()
        if entrance_audit["rating"] in ("TABOO", "NEGATIVE") and entrance_audit["remedy"]:
            defects.append({
                "defect": f"Auspicious Gate Imbalance: Door in {entrance_audit['pada_id']} ({entrance_audit['deity']})",
                "severity": "HIGH" if entrance_audit["rating"] == "TABOO" else "MODERATE",
                "impact": entrance_audit["effect"],
                "remedy": entrance_audit["remedy"]
            })

        return defects

    # =========================================================================
    # 5. House Recommendation Engine
    # =========================================================================
    def get_house_recommendation(self) -> Dict[str, Any]:
        """
        Recommends the best/strongest house layout configuration,
        potential room swaps, and architectural upgrades.
        """
        score = self.plan.vastu_score
        bhk = self.plan.bhk_config
        plot_w = self.plan.plot_width
        plot_h = self.plan.plot_height
        aspect_ratio = plot_w / max(1.0, plot_h)

        # Plot shape assessment per Vastu
        shape_eval = ""
        if 0.9 <= aspect_ratio <= 1.1:
            shape_eval = "Square (Chaturasra) — Supreme auspiciousness, promotes all-round prosperity."
        elif 1.1 < aspect_ratio <= 1.6 or 0.6 <= aspect_ratio < 0.9:
            shape_eval = "Rectangular (Ayatasra) — Highly auspicious, ideal for residential peace and wealth."
        else:
            shape_eval = "Elongated (Dirgha) — Requires internal partitioning to maintain harmonic proportion."

        # Room swap suggestions
        swaps = []
        for r in self.plan.rooms:
            if self._skip(r) or r.category == "CORRIDOR":
                continue
            key = self._key(r)
            eng = self.site_engine if self._is_site(r) else self.engine
            z = self._zone(r)
            sc = eng.get_room_score(key, z.value)
            if sc < 5:
                opt_zone = eng.get_optimal_zone(key)
                best = eng.get_room_score(key, opt_zone.value)
                swaps.append(
                    f"Consider moving **{r.name}** from *{z.value}* to the *{opt_zone.value}* zone "
                    f"(compatibility {sc:+d} → {best:+d})."
                )
        for note in getattr(self.plan, "design_notes", []) or []:
            swaps.append(f"🛠️ {note}")

        # Suggest best house archetype
        if score >= 85:
            strong_type = "Deva-Shilpi (Imperial Vastu Estate)"
            summary = "This layout achieves top-tier harmonic resonance. Structural massing in South-West and open circulation in North/East create maximum Pranic energy conservation."
        elif score >= 70:
            strong_type = "Sukha-Nivasa (Harmonic Family Sanctuary)"
            summary = "A highly stable, prosperous residence. Core functional rooms align comfortably with natural solar-path geometry."
        else:
            strong_type = "Shanti-Kendra (Remediated Balanced Dwelling)"
            summary = "A functional plan that benefits significantly from subtle elemental remedies (Parihara) to neutralize zone conflicts."

        return {
            "house_archetype": strong_type,
            "plot_shape_evaluation": shape_eval,
            "vastu_rating_grade": "A+ (Exemplary)" if score >= 85 else "A (Superior)" if score >= 70 else "B+ (Satisfactory)" if score >= 50 else "C (Needs Remedies)",
            "summary": summary,
            "room_swap_recommendations": swaps if swaps else ["All placed rooms currently occupy their optimal Vedic quadrants. No swaps required!"]
        }

    # =========================================================================
    # 6. Formal Vastu Compliance Audit Certificate
    # =========================================================================
    def generate_certificate_text(self) -> str:
        """
        Generates a formal, printable ASCII Vastu Shastra Compliance Audit Certificate.
        """
        date_str = datetime.date.today().strftime("%d %B %Y")
        cert_num = f"VASTU-{datetime.date.today().year}-{self.plan.bhk_config}-{int(self.plan.total_built_up_area * 7)}"
        zone_data = self.get_zone_analysis()
        entrance = self.audit_entrance()
        defects = self.get_defects_and_remedies()
        rec = self.get_house_recommendation()

        lines = [
            "=" * 82,
            "                  VASTU SHASTRA COMPLIANCE AUDIT CERTIFICATE",
            "          Issued in Accordance with Vedic Architectural Canon & IS 15500",
            "=" * 82,
            f"CERTIFICATE ID : {cert_num:<25} AUDIT DATE : {date_str}",
            f"PROJECT NAME   : {self.plan.project_name:<25} CONFIG     : {self.plan.bhk_config}",
            f"PLOT DIMENSIONS: {self.plan.plot_width:.2f} m x {self.plan.plot_height:.2f} m       TOTAL AREA : {self.plan.total_built_up_area:.1f} sq.m",
            f"ROAD FACING    : {self.plan.orientation:<25} FINAL GRADE: {rec['vastu_rating_grade']}",
            "-" * 82,
            "1. ENTRANCE GATE AUDIT (32 PADAS SYSTEM)",
            f"   Main Door Position: {entrance['direction']} Face | Gate: {entrance['pada_id']} ({entrance['deity']})",
            f"   Auspiciousness   : {entrance['rating']} (Score: {entrance['score']:+d}/10)",
            f"   Vedic Effect     : {entrance['effect']}",
            "-" * 82,
            "2. NINE COMPASS ZONES (PANCHA BHOOTA) AUDIT",
            f"{'Zone':<6} {'Element':<18} {'Assigned Room(s)':<24} {'Score':<8} {'Status':<12}",
            "-" * 82,
        ]

        for z in zone_data:
            rooms_str = ", ".join(z['rooms'])[:22]
            lines.append(f"{z['zone_key']:<6} {z['element']:<18} {rooms_str:<24} {z['score']:>+3d}/10   {z['status']:<12}")

        lines.extend([
            "-" * 82,
            "3. OVERALL COMPLIANCE SCORE & ARCHETYPE",
            f"   Overall Vastu Harmony Score : {self.plan.vastu_score:.1f} / 100.0",
            f"   Recommended Archetype       : {rec['house_archetype']}",
            f"   Plot Geometry Assessment    : {rec['plot_shape_evaluation']}",
            "-" * 82,
            "4. NON-DEMOLITION REMEDIAL ADVISORY (PARIHARA)",
        ])

        if defects:
            for idx, d in enumerate(defects, 1):
                lines.append(f"   [{idx}] {d['defect']} (Severity: {d['severity']})")
                lines.append(f"       Remedy: {d['remedy']}")
        else:
            lines.append("   Zero critical defects identified. Layout satisfies canonical Vedic criteria.")

        lines.extend([
            "=" * 82,
            "                                                     [ SEAL OF VASTU COUNCIL ]",
            "                                                     Chief Vastu Architect & Auditor",
            "=" * 82,
        ])

        return "\n".join(lines)

    # =========================================================================
    # 7. Vastu Mandala SVG Visualizer
    # =========================================================================
    def render_mandala_svg(self) -> str:
        """
        Renders a beautiful 9-zone Vastu Purusha Mandala SVG diagram with
        Pancha Bhoota elemental colors, room centroids, and compass directionals.
        """
        svg_w = 600
        svg_h = 600
        pad = 60
        grid_w = svg_w - 2 * pad
        grid_h = svg_h - 2 * pad
        cell_w = grid_w / 3.0
        cell_h = grid_h / 3.0

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="100%" height="auto" style="background:#0F172A; font-family:\'Segoe UI\', sans-serif; border-radius:12px;">',
            '<!-- Vastu Mandala Definitions -->',
            '<defs>',
            '  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
            '    <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#000000" flood-opacity="0.3"/>',
            '  </filter>',
            '</defs>',
            f'<text x="{svg_w/2}" y="35" font-size="16" font-weight="700" fill="#F8FAFC" text-anchor="middle" letter-spacing="1">VASTU PURUSHA MANDALA (9 ZONES)</text>',
        ]

        # Map grid cells (col 0-2, row 0-2 from top to bottom) to VastuZone
        # Top row: NW (0,0), N (1,0), NE (2,0)
        # Mid row: W  (0,1), CENTER (1,1), E (2,1)
        # Bot row: SW (0,2), S (1,2), SE (2,2)
        grid_coords = {
            (0, 0): VastuZone.NW,
            (1, 0): VastuZone.N,
            (2, 0): VastuZone.NE,
            (0, 1): VastuZone.W,
            (1, 1): VastuZone.CENTER,
            (2, 1): VastuZone.E,
            (0, 2): VastuZone.SW,
            (1, 2): VastuZone.S,
            (2, 2): VastuZone.SE,
        }

        # Room map by zone
        zone_rooms: Dict[VastuZone, List[str]] = {z: [] for z in VastuZone}
        for r in self.plan.rooms:
            if self._skip(r):
                continue
            zone_rooms[self._zone(r)].append(r.name)

        # Draw 9 cells
        for (col, row), z_enum in grid_coords.items():
            x = pad + col * cell_w
            y = pad + row * cell_h
            meta = ZONE_METADATA[z_enum]
            rooms = zone_rooms[z_enum]

            # Cell background with subtle fill
            bg_col = meta["color"]
            svg_parts.append(
                f'<rect x="{x+3}" y="{y+3}" width="{cell_w-6}" height="{cell_h-6}" rx="8" '
                f'fill="{bg_col}" fill-opacity="0.15" stroke="#334155" stroke-width="1.5" filter="url(#glow)"/>'
            )

            # Direction & Deity Header
            svg_parts.append(
                f'<text x="{x + cell_w/2}" y="{y + 24}" font-size="12" font-weight="700" fill="#38BDF8" text-anchor="middle">{z_enum.value} ({meta["deity"].split()[0]})</text>'
            )
            # Element Tag
            svg_parts.append(
                f'<text x="{x + cell_w/2}" y="{y + 40}" font-size="9" fill="#94A3B8" text-anchor="middle">{meta["element"]}</text>'
            )

            # Placed rooms in this zone
            if rooms:
                for idx, r_name in enumerate(rooms[:3]):
                    r_display = r_name[:18]
                    svg_parts.append(
                        f'<rect x="{x + 12}" y="{y + 54 + idx*22}" width="{cell_w - 24}" height="18" rx="4" fill="#1E293B" stroke="#059669" stroke-width="1"/>'
                    )
                    svg_parts.append(
                        f'<text x="{x + cell_w/2}" y="{y + 67 + idx*22}" font-size="9" font-weight="600" fill="#10B981" text-anchor="middle">{r_display}</text>'
                    )
            else:
                svg_parts.append(
                    f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 10}" font-size="9" font-style="italic" fill="#64748B" text-anchor="middle">Unoccupied</text>'
                )

        # Cardinal compass indicators outside grid
        svg_parts.extend([
            f'<text x="{svg_w/2}" y="{pad - 12}" font-size="12" font-weight="bold" fill="#F8FAFC" text-anchor="middle">NORTH (Kubera)</text>',
            f'<text x="{svg_w/2}" y="{svg_h - pad + 25}" font-size="12" font-weight="bold" fill="#F8FAFC" text-anchor="middle">SOUTH (Yama)</text>',
            f'<text x="{svg_w - pad + 15}" y="{svg_h/2}" font-size="12" font-weight="bold" fill="#F8FAFC" text-anchor="start">EAST</text>',
            f'<text x="{pad - 15}" y="{svg_h/2}" font-size="12" font-weight="bold" fill="#F8FAFC" text-anchor="end">WEST</text>',
            '</svg>'
        ])

        return "\n".join(svg_parts)
