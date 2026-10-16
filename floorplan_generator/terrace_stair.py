"""
Terrace-stair side advisor
==========================
A single-storey house climbs to its terrace by a straight stair outside the walls. The engine
puts it on the South / West side by default (never the road side); the owner may want it
anywhere — at the back, on a side, even at the front. Before they choose, every side of *their*
house is judged with the planner itself (LayoutEngine.site_preview: the same setbacks, footprint
and yard placement a re-plan builds, in milliseconds) and explained in plain words:

  · does it fit at all (and if not, the measured reason)
  · Vastu, simply: the zone it stands in (South / West best, North / East weak, never North-East)
    and whether it climbs clockwise, rising towards the South / West
  · what it costs the house: the footprint grows or shrinks when a yard must widen for it
  · what it changes in the yard: the car bay, the garden, the walkway left beside it
  · daily life: the walk from the main door to the stair's foot, the rooms it runs past (a
    kitchen / utility next to it means wet clothes go up to dry in a minute; a bedroom window
    behind it means less light and privacy), and the front's security and looks

Each side gets a weighted worth (Vastu + comfort − loss) and the best is marked as recommended.
The owner's tap becomes a {"type": "terrace_stair", "side": …} wish for Smart Re-Plan.
"""

from typing import Dict, List, Optional, Tuple

from floorplan_generator.intent_engine import ZONE_NAMES, stair_side_name
from floorplan_generator.vastu_engine import VastuEngine, plan_sides_for_facing

SQFT = 10.7639
SIDES = ("front", "left", "right", "back")
PLAN_SIDE = {"front": "top", "back": "bottom", "left": "left", "right": "right"}
WALL_SIDE = {"top": "N", "bottom": "S", "left": "W", "right": "E"}      # plan-relative wall sides
COMPOUND = 0.075                       # half the 150 mm compound wall at the plot edge
WALKWAY_OK, WALKWAY_MIN = 0.9, 0.6     # a comfortable side passage / the least you can walk through
SITE_LABEL = {"PARKING": "car parking", "BIKE_PARKING": "two-wheeler parking", "GARDEN": "garden",
              "POOL": "swimming pool", "WATER_TANK": "water sump", "GENERATOR": "generator",
              "SECURITY": "security cabin"}


def _vastu_words(zone: str, score: int, taboo: bool) -> Tuple[str, str, str]:
    """(word, emoji, one simple sentence) for a stair standing in `zone`."""
    name = ZONE_NAMES.get(zone, zone)
    if taboo:
        return "avoid", "🔴", f"Not in the {name} — that corner must stay open and light (Vastu's most sacred corner)."
    if score >= 8:
        return "good", "🟢", f"{name} is the best place for a heavy stair in Vastu."
    if score >= 5:
        return "good", "🟢", f"{name} is a good place for a stair in Vastu."
    if score >= 0:
        return "fine", "🟡", f"{name} is acceptable for a stair."
    return "not good", "🟠", (f"{name} should stay open and light in Vastu — a stair there blocks the "
                             "morning sun and fresh air.")


def _climb_words(climb: Optional[str]) -> Tuple[bool, str]:
    if climb in ("S", "W"):
        return True, f"Climbs clockwise, rising towards the {ZONE_NAMES[climb]} ✓"
    if climb:
        return False, (f"Rises towards the {ZONE_NAMES[climb]} — Vastu prefers rising towards the South or "
                       "West (the house is too short on this side to turn it round).")
    return False, ""


def _ring_pos(px: float, py: float, ring: Tuple[float, float, float, float]) -> float:
    """Distance along a rectangle's outline (anticlockwise from its bottom-left) of the nearest point."""
    x0, y0, x1, y1 = ring
    w, h = x1 - x0, y1 - y0
    cx, cy = min(max(px, x0), x1), min(max(py, y0), y1)
    d = {abs(cy - y0): ("b", cx - x0), abs(cx - x1): ("r", w + cy - y0),
         abs(cy - y1): ("t", w + h + x1 - cx), abs(cx - x0): ("l", 2 * w + h + y1 - cy)}
    return d[min(d)][1]


def walk_around(a: Tuple[float, float], b: Tuple[float, float], footprint) -> float:
    """Walking distance between two points of the yard, going round the house (not through it)."""
    fx, fy, fw, fd = footprint
    off = 0.115 + 0.45
    ring = (fx - off, fy - off, fx + fw + off, fy + fd + off)
    per = 2 * ((ring[2] - ring[0]) + (ring[3] - ring[1]))
    t1, t2 = _ring_pos(*a, ring), _ring_pos(*b, ring)
    along = min(abs(t1 - t2), per - abs(t1 - t2))
    return along + 2 * 0.45


def _foot(rect, climb: Optional[str]) -> Tuple[float, float]:
    """Centre of the bottom step: the end opposite the one the stair arrives at."""
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    return {"N": (cx, y), "S": (cx, y + h), "E": (x, cy), "W": (x + w, cy)}.get(climb, (cx, cy))


def _span(rect, side: str) -> Tuple[float, float]:
    x, y, w, h = rect
    return (x, x + w) if side in ("top", "bottom") else (y, y + h)


def _map_span(span, side: str, fp_from, fp_to) -> Tuple[float, float]:
    """A span along a face of footprint `fp_from`, carried proportionally onto `fp_to`."""
    a0, a1 = (fp_from[0], fp_from[0] + fp_from[2]) if side in ("top", "bottom") else (fp_from[1], fp_from[1] + fp_from[3])
    b0, b1 = (fp_to[0], fp_to[0] + fp_to[2]) if side in ("top", "bottom") else (fp_to[1], fp_to[1] + fp_to[3])
    k = (b1 - b0) / max(1e-6, a1 - a0)
    return tuple(b0 + (v - a0) * k for v in span)


def _site_items(plan) -> Dict[str, float]:
    from floorplan_generator.layout_engine import _is_site_room
    out: Dict[str, float] = {}
    for r in plan.rooms:
        if _is_site_room(r) and r.category in SITE_LABEL:
            out[r.category] = out.get(r.category, 0.0) + r.w * r.h
    return out


def _passage(rect, side: str, W: float, H: float) -> float:
    x, y, w, h = rect
    return {"bottom": y, "top": H - (y + h), "left": x, "right": W - (x + w)}[side] - COMPOUND


def stair_sides(plan) -> Dict:
    """
    Every side of this house judged for the outside terrace stair. Returns
    {"available": bool, "why": str, "now": word, "facing": …, "options": [card, …]} where each card is
    {side, title, compass, status 'now'|'ok'|'tight'|'no', reason, vastu{…}, lines{house, yard, use,
    care}, worth, recommended, intent}.
    """
    from floorplan_generator.layout_engine import LayoutEngine
    info = (getattr(plan, "engineering", None) or {}).get("terrace_stair")
    params = dict(getattr(plan, "generation_params", None) or {})
    if not info or not params:
        return {"available": False, "options": [],
                "why": "This house has no outside terrace stair — houses of two or more floors climb by the "
                       "staircase inside."}
    facing = plan.orientation
    sides = plan_sides_for_facing(facing)
    W, H = plan.plot_width, plan.plot_height
    fp_now = plan.building_footprint
    area_now = fp_now[2] * fp_now[3]
    door = next((d for d in plan.doors if d.is_main_entrance), None)
    door_x = (door.x, door.x + door.width) if door else None
    door_pt = ((door.x + door.width / 2) if door else fp_now[0] + fp_now[2] / 2, fp_now[1] + fp_now[3])
    items_now = _site_items(plan)
    eng = VastuEngine.for_facing(W, H, facing)
    from floorplan_generator.layout_engine import _is_site_room
    rooms_now = [r for r in plan.rooms if not _is_site_room(r)]
    params.pop("terrace_stair", None)
    cards = []
    for word in SIDES:
        ps = PLAN_SIDE[word]
        compass = sides[ps]
        card = {"side": word, "title": (lambda t: t[:1].upper() + t[1:])(stair_side_name(word, facing)), "compass": compass,
                "plan_side": ps, "intent": {"type": "terrace_stair", "side": word},
                "lines": {"house": [], "yard": [], "use": [], "care": []}, "recommended": False}
        cards.append(card)
        if word == info.get("word"):
            prev = {"site": None}
            rect, climb, zone, fp = info["rect"], info.get("climb"), info["zone"], fp_now
            card["status"] = "now"
            site_items = items_now
            passage = _passage(rect, ps, W, H)
        else:
            prev = LayoutEngine(**params, terrace_stair=word).site_preview(door_x)
            site = prev.get("site")
            st = next((it for it in (site.items if site else []) if it.key == "ext_stair"), None)
            if site is None or site.ext_stair_side != ps or st is None:
                card["status"] = "no"
                card["reason"] = (site.ext_stair_refused if site is not None and site.ext_stair_refused
                                  else "the stair could not be fitted clear of the gate path and the parking")
                card["worth"] = -999.0
                continue
            rect, fp = st.rect, site.footprint
            from floorplan_generator.site_planner import SitePlanner
            sp = SitePlanner(W, H, facing, plan.vastu_enabled)
            climb = sides[{"N": "top", "S": "bottom", "E": "right", "W": "left"}[sp.ext_stair_climb(rect, fp)]]
            zone = eng.get_zone_for_position(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2).value
            site_items = {}
            for it in site.items:
                if it.category in SITE_LABEL:
                    site_items[it.category] = site_items.get(it.category, 0.0) + it.rect[2] * it.rect[3]
            passage = _passage(rect, ps, W, H)
            card["status"] = "ok"
        # ── Vastu, simply ─────────────────────────────────────────────────
        score = eng.get_room_score("STAIRS", zone)
        taboo = eng.is_taboo("STAIRS", zone)
        vword, emoji, vline = _vastu_words(zone, score, taboo)
        climb_ok, cline = _climb_words(climb)
        card["vastu"] = {"zone": zone, "zone_name": ZONE_NAMES.get(zone, zone), "score": score, "word": vword,
                         "emoji": emoji, "line": vline, "climb": climb, "climb_ok": climb_ok, "climb_line": cline,
                         "taboo": taboo}
        # ── the house ─────────────────────────────────────────────────────
        d_area = fp[2] * fp[3] - area_now
        L = card["lines"]
        if abs(d_area) < 0.3:
            L["house"].append("The house keeps its size.")
        elif d_area < 0:
            L["house"].append(f"The house gets {abs(d_area) * SQFT:.0f} sq ft ({abs(d_area):.1f} m²) smaller — "
                              "this yard must widen for the stair, so the rooms get a little tighter.")
            if abs(d_area) / max(area_now, 1e-6) > 0.03:
                card["status"] = "tight" if card["status"] == "ok" else card["status"]
        else:
            L["house"].append(f"The house gets {d_area * SQFT:.0f} sq ft ({d_area:.1f} m²) bigger — "
                              "the yard it frees is given to the rooms.")
        # ── the yard ──────────────────────────────────────────────────────
        for cat, a0 in items_now.items():
            a1 = site_items.get(cat, 0.0)
            nm = SITE_LABEL[cat]
            if a1 <= 0.01:                      # a side never takes away something the owner asked for
                card["status"] = "no"
                card["reason"] = f"the {nm} would no longer fit"
            elif cat == "GARDEN" and a0 - a1 > 1.0:
                L["yard"].append(f"Garden {a0 - a1:.0f} m² smaller.")
        if passage < WALKWAY_MIN:
            L["yard"].append("No walkway left on this side: the stair fills the gap to the compound wall, so you "
                             "cannot walk round the house here.")
        elif passage < WALKWAY_OK:
            L["yard"].append(f"Narrow walkway beside the stair ({passage:.2f} m) — one person at a time.")
        else:
            L["yard"].append(f"A {passage:.1f} m walkway stays clear beside the stair.")
        # ── daily life ────────────────────────────────────────────────────
        foot = _foot(rect, {v: k for k, v in {"N": sides["top"], "S": sides["bottom"], "E": sides["right"],
                                                "W": sides["left"]}.items()}.get(climb))
        walk = walk_around(door_pt, foot, fp)
        L["use"].append(f"About {walk:.0f} m walk from the main door to the first step.")
        span = _map_span(_span(rect, ps), ps, fp, fp_now) if fp != fp_now else _span(rect, ps)
        past, windows = [], []
        for r in rooms_now:
            edge = {"top": abs(r.top - (fp_now[1] + fp_now[3])), "bottom": abs(r.y - fp_now[1]),
                    "left": abs(r.x - fp_now[0]), "right": abs(r.right - (fp_now[0] + fp_now[2]))}[ps]
            lo, hi = (r.x, r.right) if ps in ("top", "bottom") else (r.y, r.top)
            if edge < 0.05 and min(hi, span[1]) - max(lo, span[0]) > 0.6:
                past.append(r)
        for wd in plan.windows:
            if wd.wall_side != WALL_SIDE[ps]:
                continue
            c = wd.x if ps in ("top", "bottom") else wd.y
            if c + wd.width / 2 > span[0] + 0.1 and c - wd.width / 2 < span[1] - 0.1:
                windows.append(wd.room_name)
        wet = [r.name for r in past if r.category in ("KITCHEN", "UTILITY")]
        if wet:
            L["use"].append(f"Right outside the {' and '.join(dict.fromkeys(wet))} — handy for drying clothes "
                            "on the terrace.")
        beds = list(dict.fromkeys(n for n in windows if any(x.name == n and x.category == "BEDROOM" for x in rooms_now)))
        if beds:
            L["use"].append(f"Runs past the {' and '.join(beds)} window{'s' if len(beds) > 1 else ''} — less light and privacy there "
                            "(frosted glass or a higher window helps).")
        others = [n for n in dict.fromkeys(windows) if n not in beds and n not in wet]
        if others:
            L["use"].append(f"Runs past the {' and '.join(others)} window{'s' if len(others) > 1 else ''}.")
        # ── care: security and looks ──────────────────────────────────────
        if word == "front":
            L["care"].append("Seen from the road: anyone at the gate can reach your terrace — fit a locked grill "
                             "gate at the first step.")
            L["care"].append("It becomes part of the front of the house (see the front elevation).")
        elif word == "back":
            L["care"].append("Out of sight at the back — keep the terrace door locked at night.")
        # ── worth (for the recommendation) ───────────────────────────────
        worth = score + (3.0 if climb_ok else 0.0) - 1.5 * max(0.0, -d_area) + 0.5 * max(0.0, d_area)
        worth -= 6.0 if word == "front" else 0.0
        worth -= 4.0 if passage < WALKWAY_MIN else (1.5 if passage < WALKWAY_OK else 0.0)
        worth += 2.0 if wet else 0.0
        worth -= 1.5 * len(beds)
        worth -= 8.0 * sum(1 for cat in items_now if site_items.get(cat, 0.0) <= 0.01)
        card["worth"] = round(worth, 2) if card["status"] != "no" else -999.0
        good = []
        if score >= 5 and not taboo:
            good.append("good Vastu" + (", climbs clockwise" if climb_ok else ""))
        if wet:
            good.append("steps from the kitchen")
        if abs(d_area) < 0.3:
            good.append("the house keeps its size")
        elif d_area > 0:
            good.append("a slightly bigger house")
        if passage >= WALKWAY_OK:
            good.append("a clear walkway beside it")
        if word == "back":
            good.append("out of sight")
        card["good"] = good
    ok = [c for c in cards if c["status"] != "no"]
    if ok:
        best = max(ok, key=lambda c: (c["worth"], c["status"] == "now"))
        best["recommended"] = True
        best["why"] = ("Recommended: " + ", ".join(best["good"][:3]) + ".") if best["good"] else "Recommended."
    return {"available": True, "now": info.get("word"), "facing": facing, "options": cards, "why": ""}


def form_sides(**engine_kwargs) -> Dict:
    """
    Before Generate (the Design form): every side of the house for the outside terrace stair, on this plot with
    the amenities switched on — planned the way Generate plans it (LayoutEngine.site_preview). A side is offered
    only when the stair fits there AND everything else asked for still fits (car, two-wheelers, garden, pool,
    sump …); otherwise it is locked with the reason. The automatic side (Vastu's South / West, never the road
    side) is the default.
    Returns {"available": bool, "auto": word, "options": [{side, title, ok, reason, auto, vastu{…}, d_sqft}]}.
    """
    from floorplan_generator.layout_engine import LayoutEngine
    kw = dict(engine_kwargs)
    kw.pop("terrace_stair", None)
    base = LayoutEngine(**kw).site_preview()
    if not base or base.get("site") is None:
        return {"available": False, "auto": None, "options": []}
    site0 = base["site"]
    W, H = float(kw["plot_width"]), float(kw["plot_height"])
    facing = (kw.get("orientation") or "N").upper()[0]
    eng = VastuEngine.for_facing(W, H, facing)
    word_of = {v: k for k, v in PLAN_SIDE.items()}
    auto = word_of.get(site0.ext_stair_side)

    def items(site):
        out = {}
        for it in site.items:
            if it.category in SITE_LABEL:
                out[it.category] = out.get(it.category, 0.0) + it.rect[2] * it.rect[3]
        return out
    have = items(site0)
    area0 = site0.footprint[2] * site0.footprint[3]
    from floorplan_generator.site_planner import SitePlanner
    sp = SitePlanner(W, H, facing, bool(kw.get("vastu_enabled", True)))
    sides = plan_sides_for_facing(facing)
    options = []
    for word in ("back", "left", "right", "front"):
        prev = base if word == auto else LayoutEngine(**kw, terrace_stair=word).site_preview()
        site = prev["site"]
        st = next((it for it in site.items if it.key == "ext_stair"), None)
        opt = {"side": word, "title": (lambda t: t[:1].upper() + t[1:])(stair_side_name(word, facing)),
               "auto": word == auto, "ok": False, "reason": "", "vastu": None, "d_sqft": 0}
        options.append(opt)
        if st is None or site.ext_stair_side != PLAN_SIDE[word]:
            opt["reason"] = site.ext_stair_refused or "the stair cannot be fitted on that side"
            continue
        lost = [SITE_LABEL[c] for c in have if items(site).get(c, 0.0) <= 0.01]
        if lost:
            opt["reason"] = f"the {' and the '.join(lost)} would no longer fit"
            continue
        zone = eng.get_zone_for_position(st.rect[0] + st.rect[2] / 2, st.rect[1] + st.rect[3] / 2).value
        score = eng.get_room_score("STAIRS", zone)
        word_v, emoji, line = _vastu_words(zone, score, eng.is_taboo("STAIRS", zone))
        climb = sides[{"N": "top", "S": "bottom", "E": "right", "W": "left"}[sp.ext_stair_climb(st.rect, site.footprint)]]
        ok_climb, climb_line = _climb_words(climb)
        opt.update(ok=True, vastu={"zone": zone, "score": score, "word": word_v, "emoji": emoji, "line": line,
                                   "climb_ok": ok_climb, "climb_line": climb_line},
                   d_sqft=round((site.footprint[2] * site.footprint[3] - area0) * SQFT))
    return {"available": True, "auto": auto, "options": options}
