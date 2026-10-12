"""
House report — the four result panels, measured from the drawing
================================================================
Every number in these panels is traced back to the generated plan (and the same engines
that produced it), so the report explains *this* house rather than repeating generic text:

  room_schedule(plan)      rooms in small groups with inside (wall-to-wall) sizes, ft-in,
                           and an NBC 2016 Part 3 check per room (min area / width)
  vastu_ledger(plan)       exact decomposition of plan.vastu_score: how many of the 100
                           points each room, the entrance pada and each penalty costs
                           (same formula as layout_engine.audit_vastu)
  vastu_checks(plan)       Vastu rules verified with measurements (zones, pada, setbacks,
                           Brahmasthan, plot shape …) + what can only be checked on site
  vastu_plan_svg(plan)     the plan drawn to scale under its own 3×3 mandala, rooms
                           coloured by how well they sit, door pada, Brahmasthan, north
  health_card(plan, est)   six measured scores (space, Vastu, light & air, efficiency,
                           open space, build economy) → one House Health score used to
                           compare designs the engine can make for the same plot

Pure Python (no Streamlit).
"""

from html import escape
from typing import Dict, List, Optional, Tuple
import math

from floorplan_generator.vastu_engine import (
    VastuEngine, ROOM_IMPORTANCE, _VASTU_COMPATIBILITY, north_edge_for_facing, plan_sides_for_facing,
)

SQFT = 10.7639
SITE = {"PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"}
EXT_WALL, INT_WALL = 0.23, 0.115
ZONE_WORDS = {"N": "North", "NE": "North-East", "E": "East", "SE": "South-East", "S": "South",
              "SW": "South-West", "W": "West", "NW": "North-West", "CENTER": "Centre"}
ICON = {"LIVING": "🛋️", "DINING": "🍽️", "BEDROOM": "🛏️", "KITCHEN": "🍳", "BATHROOM": "🚿", "POOJA": "🪔",
        "STUDY": "📚", "UTILITY": "🧺", "STORE": "📦", "SERVANT": "🧹", "CORRIDOR": "🚶", "FOYER": "🚪",
        "STAIRS": "🪜", "LIFT": "🛗", "PARKING": "🚗", "BIKE_PARKING": "🏍️", "GARDEN": "🌿", "POOL": "🏊",
        "WATER_TANK": "💧", "GENERATOR": "⚡", "SECURITY": "🛡️"}
GROUPS = [   # (key, title) in display order
    ("living", "Living spaces"), ("bed", "Bedrooms"), ("kitchen", "Kitchen & utility"),
    ("bath", "Toilets"), ("extra", "Pooja, study & others"), ("move", "Passages & stairs"),
    ("site", "Outside the house"),
]


def is_site(r) -> bool:
    return r.category in SITE or "External" in r.name


def ft_in(m: float) -> str:
    """3.45 → 11'4\"."""
    inches = round(m / 0.0254)
    return f"{inches // 12}'{inches % 12}\""


def _group(r) -> str:
    if is_site(r):
        return "site"
    if any(t in r.name for t in ("Dress", "Wardrobe")):
        return "bed"
    return {"LIVING": "living", "DINING": "living", "BEDROOM": "bed", "KITCHEN": "kitchen", "UTILITY": "kitchen",
            "STORE": "kitchen", "BATHROOM": "bath", "POOJA": "extra", "STUDY": "extra", "SERVANT": "extra",
            "CORRIDOR": "move", "FOYER": "move", "STAIRS": "move", "LIFT": "move"}.get(r.category, "extra")


def _clear_dims(r, fp) -> Tuple[float, float]:
    """Inside (wall-to-wall) width and length, matching how the plan computes clear areas."""
    if is_site(r) or not fp:
        return r.w, r.h
    fx, fy, fw, fd = fp
    tl = EXT_WALL if abs(r.x - fx) < 0.03 else INT_WALL
    tr = EXT_WALL if abs(r.x + r.w - (fx + fw)) < 0.03 else INT_WALL
    tb = EXT_WALL if abs(r.y - fy) < 0.03 else INT_WALL
    tt = EXT_WALL if abs(r.y + r.h - (fy + fd)) < 0.03 else INT_WALL
    return max(0.0, r.w - (tl + tr) / 2), max(0.0, r.h - (tb + tt) / 2)


def _nbc_rule(r, is_first_habitable: bool) -> Optional[Tuple[float, float, str]]:
    """(min area m², min width m, clause) from NBC 2016 Part 3 §8, or None when NBC sets no size."""
    name = r.name.lower()
    if r.category in ("LIVING", "DINING", "BEDROOM", "STUDY", "SERVANT") and "dress" not in name \
            and "wardrobe" not in name:
        return (9.5 if is_first_habitable else 7.5, 2.4, "habitable room (§8.1)")
    if r.category == "KITCHEN":
        return (5.0, 1.8, "kitchen (§8.4)")
    if r.category == "BATHROOM":
        if "toilet" in name and "servant" in name:
            return (1.1, 0.9, "water closet (§8.5)")
        return (2.8, 1.2, "bath + WC combined (§8.5)")
    if r.category == "CORRIDOR" and "passage" in name:
        return (0.0, 1.0, "passage width (§8.6)")
    return None


def room_schedule(plan) -> Dict:
    """Rooms grouped for small cards, inside sizes, ft-in, and an NBC check per room."""
    fp = getattr(plan, "building_footprint", None)
    eng = VastuEngine.for_plan(plan, scope="building")
    site_eng = VastuEngine.for_plan(plan, scope="plot")
    habitable = sorted((r for r in plan.rooms if not is_site(r) and r.category in ("LIVING", "BEDROOM")),
                       key=lambda r: -r.area)
    first = habitable[0].name if habitable else None
    rows = []
    for r in plan.rooms:
        cw, ch = _clear_dims(r, fp)
        area = r.area or cw * ch
        rule = _nbc_rule(r, r.name == first)
        nbc = {"status": "na", "text": "No NBC size rule"}
        if rule:
            amin, wmin, clause = rule
            short = min(cw, ch)
            if area + 1e-6 >= amin and short + 1e-6 >= wmin:
                nbc = {"status": "ok", "text": f"Meets NBC {clause}: ≥ {amin:g} m², ≥ {wmin:g} m wide"
                                               if amin else f"Meets NBC {clause}: ≥ {wmin:g} m wide"}
            else:
                gaps = []
                if area + 1e-6 < amin:
                    gaps.append(f"{amin - area:.2f} m² under {amin:g} m²")
                if short + 1e-6 < wmin:
                    gaps.append(f"{(wmin - short) * 100:.0f} cm narrower than {wmin:g} m")
                nbc = {"status": "low", "text": f"NBC {clause}: " + ", ".join(gaps)}
        zone = (site_eng if is_site(r) else eng).get_zone_for_position(r.cx, r.cy).value
        rows.append({
            "name": r.name, "category": r.category, "icon": ICON.get(r.category, "▫️"), "group": _group(r),
            "w": round(cw, 2), "h": round(ch, 2), "w_ft": ft_in(cw), "h_ft": ft_in(ch),
            "area": round(area, 2), "sqft": round(area * SQFT), "zone": zone, "nbc": nbc,
            "cl_w": round(r.w, 2), "cl_h": round(r.h, 2),
        })
    building = [x for x in rows if x["group"] != "site"]
    move = sum(x["area"] for x in building if x["group"] == "move" and x["category"] != "STAIRS")
    carpet = getattr(plan, "total_carpet_area", 0) or sum(x["area"] for x in building)
    built = getattr(plan, "total_built_up_area", 0) or carpet
    return {
        "rows": rows,
        "groups": [(k, t, [x for x in rows if x["group"] == k]) for k, t in GROUPS
                   if any(x["group"] == k for x in rows)],
        "totals": {"carpet": round(carpet, 2), "built_up": round(built, 2),
                   "efficiency": round(100 * carpet / max(1e-6, built), 1),
                   "circulation_pct": round(100 * move / max(1e-6, carpet), 1),
                   "rooms": len(building), "outdoor_m2": round(sum(x["area"] for x in rows if x["group"] == "site"), 1),
                   "nbc_low": [x["name"] for x in rows if x["nbc"]["status"] == "low"]},
    }


# =============================================================================
# Vastu engineering
# =============================================================================

def good_zones(key: str, min_score: int = 8) -> List[str]:
    table = _VASTU_COMPATIBILITY.get(key.upper(), {})
    return [z for z, s in sorted(table.items(), key=lambda kv: -kv[1]) if s >= min_score]


def _plain_key(key: str) -> str:
    return {"MASTER_BEDROOM": "master bedroom", "CHILD_BEDROOM": "children's bedroom",
            "GUEST_BEDROOM": "guest bedroom", "ELDER_BEDROOM": "elders' bedroom"}.get(key, key.lower().replace("_", " "))


def vastu_ledger(plan) -> Dict:
    """
    Where the Vastu points go — the exact terms of audit_vastu: every placement earns
    weight × (score + 50) / 60 of its share, the main door's pada likewise; each strict
    taboo −8, each Brahmasthan violation −8, N/E open space smaller than S/W −4.
    """
    bd = getattr(plan, "vastu_breakdown", None) or {}
    rows = bd.get("rooms") or []
    ent = bd.get("entrance_score")
    total_w = sum(ROOM_IMPORTANCE.get(r["key"], 1.0) for r in rows) + (ROOM_IMPORTANCE["ENTRANCE"] if ent is not None else 0)
    items = []
    for r in rows:
        w = ROOM_IMPORTANCE.get(r["key"], 1.0)
        lost = 100.0 * w * (10 - r["score"]) / 60.0 / max(total_w, 1e-6)
        items.append({"item": r["room"], "key": r["key"], "zone": r["zone"], "score": r["score"],
                      "weight": w, "lost": round(lost, 2), "_lost": lost, "taboo": bool(r.get("taboo")),
                      "best": good_zones(r["key"]) or good_zones(r["key"], 5)})
    if ent is not None:
        es = max(-50.0, min(10.0, ent))
        lost = 100.0 * ROOM_IMPORTANCE["ENTRANCE"] * (10 - es) / 60.0 / max(total_w, 1e-6)
        items.append({"item": f"Main door (pada {bd.get('entrance_pada')})", "key": "ENTRANCE", "zone": bd.get("entrance_pada"),
                      "score": ent, "weight": ROOM_IMPORTANCE["ENTRANCE"], "lost": round(lost, 2), "_lost": lost,
                      "taboo": False, "best": []})
    penalties = []
    taboos = [i for i in items if i["taboo"]]
    for t in taboos:
        penalties.append({"item": f"Strict taboo: {t['item']} in {ZONE_WORDS.get(t['zone'], t['zone'])}", "lost": 8.0})
    for name in bd.get("brahmasthan_violations") or []:
        penalties.append({"item": f"{name} on the Brahmasthan (centre)", "lost": 8.0})
    ne, sw = bd.get("open_space_ne_m", 0), bd.get("open_space_sw_m", 0)
    if getattr(plan, "vastu_enabled", True) and (ne or sw) and ne + 0.05 < sw:
        penalties.append({"item": f"Less open space on North + East ({ne:.2f} m) than South + West ({sw:.2f} m)",
                          "lost": 4.0})
    raw = 100.0 - sum(i["_lost"] for i in items) - sum(p["lost"] for p in penalties)
    for i in items:
        i.pop("_lost")
    return {"items": sorted(items, key=lambda i: -i["lost"]), "penalties": penalties,
            "score": round(max(0.0, min(100.0, raw)), 1), "reported": getattr(plan, "vastu_score", None)}


def _room_by_key(plan, key: str):
    for r in plan.rooms:
        if (getattr(r, "role", "") or r.category) == key or r.category == key:
            yield r


def vastu_checks(plan) -> Dict:
    """Vastu rules verified with measurements from the drawing (+ what needs a site visit)."""
    eng = VastuEngine.for_plan(plan, scope="building")
    site_eng = VastuEngine.for_plan(plan, scope="plot")
    bd = getattr(plan, "vastu_breakdown", None) or {}
    checks: List[Dict] = []

    def zone_of(r):
        return (site_eng if is_site(r) else eng).get_zone_for_position(r.cx, r.cy).value

    def add(rule, status, evidence, why, fix=""):
        checks.append({"rule": rule, "status": status, "evidence": evidence, "why": why, "fix": fix})

    # 1 Entrance
    from floorplan_generator.vastu_analyzer import VastuAnalyzer
    ent = VastuAnalyzer(plan).audit_entrance()
    st = "pass" if ent["score"] >= 5 else "warn" if ent["score"] >= 0 else "fail"
    add("Main door on an auspicious pada", st,
        f"Door in pada {ent['pada_id']} ({ent['deity']}) on the {ent['direction']} side — {ent['score']:+d}/10",
        ent["effect"], ent.get("remedy", "") if st != "pass" else "")
    # 2 Brahmasthan
    br = bd.get("brahmasthan_violations") or []
    add("Brahmasthan (centre) kept light", "pass" if not br else "fail",
        "Nothing heavy or wet on the centre point" if not br else "On the centre: " + ", ".join(br),
        "The centre of the house should stay open — no toilet, kitchen, stairs or store on it.",
        "" if not br else "Use Smart Re-Plan: tap the room → 📍 Move it away from the centre.")
    # 3 Master bedroom
    for r in [x for x in plan.rooms if getattr(x, "role", "") == "MASTER_BEDROOM"][:1]:
        z = zone_of(r)
        st = "pass" if z == "SW" else "warn" if z in ("S", "W") else "fail"
        add("Master bedroom in the South-West", st, f"{r.name} sits in the {ZONE_WORDS[z]}",
            "South-West (Nairutya) gives the head of the family stability.",
            "" if st == "pass" else "Smart Re-Plan: tap Master Bedroom → 📍 Move it → SW.")
    # 4 Kitchen
    for r in [x for x in plan.rooms if x.category == "KITCHEN"][:1]:
        z = zone_of(r)
        st = "pass" if z == "SE" else "warn" if z in ("NW", "E", "S") else "fail"
        add("Kitchen in the South-East (Agni)", st, f"Kitchen sits in the {ZONE_WORDS[z]}",
            "Fire belongs to the South-East; North-West is the accepted second choice.",
            "" if st == "pass" else "Smart Re-Plan: tap Kitchen → 📍 Move it → SE.")
    # 5 Pooja
    for r in [x for x in plan.rooms if x.category == "POOJA"][:1]:
        z = zone_of(r)
        st = "pass" if z == "NE" else "warn" if z in ("N", "E") else "fail"
        add("Pooja room in the North-East (Ishan)", st, f"Pooja room sits in the {ZONE_WORDS[z]}",
            "The North-East is the corner of the divine; North or East also work.",
            "" if st == "pass" else "Smart Re-Plan: tap Pooja Room → 📍 Move it → NE.")
    # 6 Toilets
    baths = [x for x in plan.rooms if x.category == "BATHROOM" and not is_site(x)]
    if baths:
        bad = [f"{x.name} ({zone_of(x)})" for x in baths if zone_of(x) in ("NE", "CENTER")]
        soft = [f"{x.name} ({zone_of(x)})" for x in baths if zone_of(x) == "SW"]
        add("Toilets away from North-East, centre and South-West",
            "fail" if bad else "warn" if soft else "pass",
            "All toilets in " + ", ".join(sorted({ZONE_WORDS[zone_of(x)] for x in baths}))
            if not bad and not soft else "Check: " + ", ".join(bad + soft),
            "Toilets in the North-East or centre are the strongest Vastu defects.",
            "" if not (bad or soft) else "Smart Re-Plan: tap the toilet → 📍 Move it → NW or W.")
    # 7 Staircase
    for r in [x for x in plan.rooms if x.category == "STAIRS"][:1]:
        z = zone_of(r)
        st = "fail" if z in ("NE", "CENTER") else "pass" if z in ("S", "SW", "W") else "warn"
        add("Staircase in the South / West", st, f"{r.name} in the {ZONE_WORDS[z]}"
            + (" of the plot" if is_site(r) else ""),
            "Stairs are heavy; they belong to the South or West, never the North-East or centre.",
            "" if st != "fail" else "Smart Re-Plan: tap the staircase → 📍 Move it → SW.")
    # 8 Open space / mass
    ne, sw = bd.get("open_space_ne_m", 0), bd.get("open_space_sw_m", 0)
    if ne or sw:
        add("More open space on North + East than South + West", "pass" if ne + 0.05 >= sw else "warn",
            f"Setbacks N + E = {ne:.2f} m, S + W = {sw:.2f} m",
            "Open North / East lets in morning light; the building's weight sits to the South-West.",
            "" if ne + 0.05 >= sw else "Keep the N/E yards free of sheds or tanks; plant only low shrubs there.")
    # 9 Plot shape
    ratio = max(plan.plot_width, plan.plot_height) / max(0.1, min(plan.plot_width, plan.plot_height))
    add("Plot shape square or rectangle up to 1 : 2",
        "pass" if ratio <= 1.6 else "warn" if ratio <= 2.0 else "fail",
        f"{plan.plot_width:.2f} m × {plan.plot_height:.2f} m — ratio 1 : {ratio:.2f}",
        "Square (Chaturasra) and moderate rectangles (Ayatasra) are the auspicious shapes.")
    # 10-12 Site elements
    for cat, rule, ok, soso, why in (
            ("WATER_TANK", "Underground water in the North-East", ("NE", "N", "E"), (),
             "Water belongs to the North-East / North / East."),
            ("GENERATOR", "Generator / fire equipment in the South-East", ("SE", "S"), ("NW",),
             "Electrical fire sits with Agni in the South-East."),
            ("PARKING", "Vehicles in the North-West or South-East", ("NW", "SE"), ("N", "E", "W", "S"),
             "Moving things belong to Vayu (NW); the SE is the second choice.")):
        for r in [x for x in plan.rooms if x.category == cat][:1]:
            z = zone_of(r)
            add(rule, "pass" if z in ok else "warn" if z in soso else "fail",
                f"{r.name} in the {ZONE_WORDS[z]} of the plot", why)
    site_only = [
        "Plot slope — the ground should fall towards North / East (water flows to Ishan).",
        "Road hits (Veedhi Shoola) — no road should end straight at the plot.",
        "Big trees, poles or water tanks of neighbours in the North-East.",
        "Bed head towards South (master) or East (children); cook facing East.",
    ]
    score = {"pass": 0, "warn": 0, "fail": 0}
    for c in checks:
        score[c["status"]] += 1
    return {"checks": checks, "site_only": site_only, "counts": score}


def vastu_plan_svg(plan, width_px: int = 560) -> str:
    """The plan to scale under its own 3×3 mandala, rooms coloured by their Vastu fit."""
    W, H = plan.plot_width, plan.plot_height
    pad = 34
    s = (width_px - 2 * pad) / W
    h_px = H * s + 2 * pad + 18
    X = lambda x: pad + x * s                       # noqa: E731
    Y = lambda y: pad + 18 + (H - y) * s            # noqa: E731  plan y-up → SVG y-down, road at the top
    bd = getattr(plan, "vastu_breakdown", None) or {}
    by_room = {r["room"]: r for r in bd.get("rooms") or []}
    fp = getattr(plan, "building_footprint", None) or (0, 0, W, H)
    fx, fy, fw, fd = fp
    eng = VastuEngine.for_plan(plan, scope="building")

    def colour(sc, taboo):
        if taboo or sc < 0:
            return "#EF4444"
        return "#22C55E" if sc >= 8 else "#84CC16" if sc >= 5 else "#F59E0B"

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {h_px:.0f}" width="100%" '
         f'style="max-width:{width_px}px;font-family:Segoe UI,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{width_px}" height="{h_px:.0f}" rx="12" fill="#0B1220"/>',
         f'<text x="{width_px / 2}" y="20" fill="#94A3B8" font-size="11" text-anchor="middle" '
         f'letter-spacing="2">ROAD · {plan_sides_for_facing(plan.orientation)["top"]}-FACING</text>',
         f'<rect x="{X(0):.1f}" y="{Y(H):.1f}" width="{W * s:.1f}" height="{H * s:.1f}" fill="none" '
         f'stroke="#334155" stroke-dasharray="5 4"/>']
    # site items, faint
    for r in plan.rooms:
        if is_site(r):
            p.append(f'<rect x="{X(r.x):.1f}" y="{Y(r.y + r.h):.1f}" width="{r.w * s:.1f}" height="{r.h * s:.1f}" '
                     f'fill="#1E293B" stroke="#475569" stroke-width="0.8"/>')
    # rooms coloured by fit
    for r in plan.rooms:
        if is_site(r):
            continue
        info = by_room.get(r.name)
        fill = "#334155" if info is None else colour(info["score"], info.get("taboo"))
        op = 0.18 if info is None else 0.42
        p.append(f'<rect x="{X(r.x):.1f}" y="{Y(r.y + r.h):.1f}" width="{r.w * s:.1f}" height="{r.h * s:.1f}" '
                 f'fill="{fill}" fill-opacity="{op}" stroke="#CBD5E1" stroke-width="1.1"/>')
        if r.w * s > 34 and r.h * s > 16:
            label = r.name.replace("Bedroom", "Bed").replace("Bathroom", "Bath").replace("Attached", "Att.")
            label = label if len(label) <= max(6, int(r.w * s / 6.2)) else label[:max(5, int(r.w * s / 6.2)) - 1] + "…"
            p.append(f'<text x="{X(r.cx):.1f}" y="{Y(r.cy) + 3:.1f}" fill="#F1F5F9" font-size="9.5" '
                     f'text-anchor="middle">{escape(label)}</text>')
    # mandala grid on the building
    for i in (1, 2):
        xx, yy = fx + fw * i / 3, fy + fd * i / 3
        p.append(f'<line x1="{X(xx):.1f}" y1="{Y(fy):.1f}" x2="{X(xx):.1f}" y2="{Y(fy + fd):.1f}" '
                 f'stroke="#38BDF8" stroke-width="1.2" stroke-dasharray="6 4" opacity="0.8"/>')
        p.append(f'<line x1="{X(fx):.1f}" y1="{Y(yy):.1f}" x2="{X(fx + fw):.1f}" y2="{Y(yy):.1f}" '
                 f'stroke="#38BDF8" stroke-width="1.2" stroke-dasharray="6 4" opacity="0.8"/>')
    for i in range(3):
        for j in range(3):
            cx, cy = fx + fw * (i + 0.5) / 3, fy + fd * (j + 0.5) / 3
            z = eng.get_zone_for_position(cx, cy).value
            p.append(f'<text x="{X(fx + fw * i / 3) + 4:.1f}" y="{Y(fy + fd * (j + 1) / 3) + 12:.1f}" fill="#38BDF8" '
                     f'font-size="10" font-weight="700">{"C" if z == "CENTER" else z}</text>')
    p.append(f'<rect x="{X(fx):.1f}" y="{Y(fy + fd):.1f}" width="{fw * s:.1f}" height="{fd * s:.1f}" fill="none" '
             f'stroke="#E2E8F0" stroke-width="2.2"/>')
    # Brahmasthan
    rb = min(fw, fd) / 9 * s
    p.append(f'<circle cx="{X(fx + fw / 2):.1f}" cy="{Y(fy + fd / 2):.1f}" r="{rb:.1f}" fill="none" '
             f'stroke="#FBBF24" stroke-width="1.5" stroke-dasharray="3 3"/>')
    # main door
    main = next((d for d in plan.doors if d.is_main_entrance), None)
    if main is not None:
        if main.wall_side in ("N", "S"):
            x0, x1, y0 = main.x, main.x + main.width, main.y
            p.append(f'<line x1="{X(x0):.1f}" y1="{Y(y0):.1f}" x2="{X(x1):.1f}" y2="{Y(y0):.1f}" '
                     f'stroke="#FBBF24" stroke-width="5" stroke-linecap="round"/>')
            tx, ty = X((x0 + x1) / 2), Y(y0) - 8
        else:
            y1 = main.y + main.width
            p.append(f'<line x1="{X(main.x):.1f}" y1="{Y(main.y):.1f}" x2="{X(main.x):.1f}" y2="{Y(y1):.1f}" '
                     f'stroke="#FBBF24" stroke-width="5" stroke-linecap="round"/>')
            tx, ty = X(main.x) + 10, Y((main.y + y1) / 2)
        p.append(f'<text x="{tx:.1f}" y="{ty:.1f}" fill="#FBBF24" font-size="10" font-weight="700" '
                 f'text-anchor="middle">Door {bd.get("entrance_pada") or ""}</text>')
    # north arrow
    ne = north_edge_for_facing(plan.orientation)
    ang = {"N": 0, "E": 90, "S": 180, "W": 270}[ne]
    ax, ay = width_px - pad + 6, pad + 30
    p.append(f'<g transform="translate({ax},{ay}) rotate({ang})"><polygon points="0,-16 6,6 0,2 -6,6" '
             f'fill="#F8FAFC"/><text x="0" y="-19" fill="#F8FAFC" font-size="10" font-weight="700" '
             f'text-anchor="middle" transform="rotate({-ang},0,-22)">N</text></g>')
    # legend
    ly = h_px - 10
    for i, (c, t) in enumerate((("#22C55E", "ideal"), ("#84CC16", "good"), ("#F59E0B", "neutral"),
                                ("#EF4444", "avoid"), ("#FBBF24", "door / Brahmasthan"))):
        lx = pad + i * 92
        p.append(f'<rect x="{lx}" y="{ly - 8}" width="10" height="10" rx="2" fill="{c}"/>'
                 f'<text x="{lx + 14}" y="{ly + 1}" fill="#94A3B8" font-size="10">{t}</text>')
    p.append("</svg>")
    return "".join(p)


# =============================================================================
# House health card
# =============================================================================

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _family(plan):
    gp = getattr(plan, "generation_params", None) or {}
    return gp.get("family_profile")


def health_card(plan, est: Optional[Dict] = None) -> Dict:
    """Six measured scores (0-100) with their evidence, and the weighted House Health score."""
    fam = _family(plan)
    members = max(1, (getattr(fam, "adults", 2) + getattr(fam, "children", 0) + getattr(fam, "elderly", 0))
                  if fam is not None else 4)
    rooms = [r for r in plan.rooms if not is_site(r)]
    carpet = getattr(plan, "total_carpet_area", 0) or sum(r.area for r in rooms)
    built = getattr(plan, "total_built_up_area", 0) or carpet
    dims: List[Dict] = []

    # 1 Space for the family
    pp = carpet / members
    beds = sum(1 for r in rooms if r.category == "BEDROOM")
    need = None
    if fam is not None:
        try:
            from floorplan_generator.family_analyzer import FamilyAnalyzer
            need = FamilyAnalyzer().analyze(fam).bedrooms_needed
        except Exception:
            need = None
    sc = 100 * _clamp01((pp - 7.5) / 10.0)
    ev = f"{carpet:.0f} m² for {members} people = {pp:.1f} m² each (comfortable ≥ 15)"
    if need and beds < need:
        sc *= 0.75
        ev += f"; {beds} bedrooms for a family that needs {need}"
    elif need:
        ev += f"; {beds} bedrooms (family needs {need})"
    dims.append({"key": "space", "label": "Space for your family", "icon": "👨‍👩‍👧", "score": round(sc), "evidence": ev})

    # 2 Vastu
    vs = float(getattr(plan, "vastu_score", 0) or 0)
    dims.append({"key": "vastu", "label": "Vastu", "icon": "🕉️", "score": round(vs),
                 "evidence": f"{vs:.0f}/100 on the house's own mandala, door pada and site"})

    # 3 Light & air (NBC Part 3 §8.2: openings ≥ 1/10 of floor area for habitable rooms & kitchens)
    hab = [r for r in rooms if r.category in ("LIVING", "DINING", "BEDROOM", "STUDY", "KITCHEN")
           and "Dress" not in r.name and "Wardrobe" not in r.name]
    win: Dict[str, float] = {}
    for w in plan.windows:
        along = w.width if w.wall_side in ("N", "S") else max(w.width, w.height)
        win[w.room_name] = win.get(w.room_name, 0.0) + along * 1.2
    ratios = [(r.name, win.get(r.name, 0.0) / max(r.area, 0.1)) for r in hab]
    dark = [n for n, x in ratios if x <= 0]
    low = [n for n, x in ratios if 0 < x < 0.10]
    sc = 100 * (sum(_clamp01(x / 0.10) for _, x in ratios) / max(1, len(ratios)))
    lit = len(ratios) - len(dark)
    ev = f"{lit} of {len(ratios)} main rooms (living, bedrooms, kitchen) have windows"
    if ratios:
        avg = sum(x for _, x in ratios) / len(ratios)
        ev += f"; window area {avg * 100:.0f}% of floor on average (NBC asks ≥ 10%)"
    if dark:
        ev += "; no window: " + ", ".join(dark)
    elif low:
        ev += "; below 10%: " + ", ".join(low)
    dims.append({"key": "light", "label": "Light & fresh air", "icon": "☀️", "score": round(sc), "evidence": ev})

    # 4 Space efficiency
    move = sum(r.area for r in rooms if r.category in ("CORRIDOR", "FOYER") and "Dress" not in r.name
               and "Wardrobe" not in r.name)
    circ = move / max(carpet, 1e-6)
    eff = carpet / max(built, 1e-6)
    sc = 100 * (0.5 * _clamp01((0.25 - circ) / 0.17) + 0.5 * _clamp01((eff - 0.72) / 0.13))
    dims.append({"key": "efficiency", "label": "Space efficiency", "icon": "📐", "score": round(sc),
                 "evidence": f"passages & foyer {circ * 100:.0f}% of floor (good ≤ 10%); "
                             f"carpet is {eff * 100:.0f}% of built-up (good ≥ 80%)"})

    # 5 Open space on the plot
    eng = getattr(plan, "engineering", None) or {}
    cov = eng.get("ground_coverage_pct")
    if cov is None:
        cov = 100 * built / max(1e-6, plan.plot_width * plan.plot_height)
    open_pct = 100 - cov
    sc = 100 * _clamp01((open_pct - 15) / 25.0)
    dims.append({"key": "open", "label": "Open space around", "icon": "🌳", "score": round(sc),
                 "evidence": f"house covers {cov:.0f}% of the plot, {open_pct:.0f}% stays open for light, "
                             "air, parking and garden"})

    # 6 Build economy — less wall per m² of floor is cheaper to build
    wall = sum(w.length for w in plan.walls if getattr(w, "wall_type", "") != "compound")
    wpm = wall / max(carpet, 1e-6)
    sc = 100 * _clamp01((1.6 - wpm) / 0.7)
    ev = f"{wall:.0f} m of wall for {carpet:.0f} m² of floor = {wpm:.2f} m per m² (compact ≤ 0.9)"
    if est and est.get("cost_per_unit"):
        ev += f"; {est.get('display', {}).get('rate', '')}"
    dims.append({"key": "economy", "label": "Build economy", "icon": "🧱", "score": round(sc), "evidence": ev})

    weights = {"space": 0.20, "vastu": 0.20, "light": 0.20, "efficiency": 0.15, "open": 0.10, "economy": 0.15}
    overall = sum(d["score"] * weights[d["key"]] for d in dims)
    grade = "Excellent" if overall >= 85 else "Very good" if overall >= 75 else "Good" if overall >= 65 \
        else "Fair" if overall >= 50 else "Needs work"
    return {"dims": dims, "overall": round(overall), "grade": grade, "weights": weights}
