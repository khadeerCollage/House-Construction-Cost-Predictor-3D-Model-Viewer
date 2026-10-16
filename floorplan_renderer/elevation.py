"""
🏠 Front elevation — the house as seen from the road
====================================================
The drawing an owner asks an architect for first: the street face of the house, every floor, with its
windows, doors, balconies, slabs, parapet and the stair head-room on the terrace, the stilt columns when
the cars park under the house, and the compound wall with its gates in front.

Nothing is drawn that is not in the design: every shape comes from the 3D scene (walkthrough.build_scene),
which is itself measured from the plan — the same walls split round the same windows (sill / head from the
door & window schedule), the same doors and what each becomes on the floors above (a balcony, a closed wall,
an open parapet), the same storey height, slab, parapet and compound wall. So the elevation, the 3D walk and
the BOQ always agree.

Standing on the road and looking at the house, the plan's left and right swap: elevation u = plot width − x.
Heights z are metres above the ground (GL ±0.000).

    front_elevation_svg(plan, scene=None, mode="clean") -> str      # an SVG document
"""

import math
from typing import Dict, List, Optional, Tuple

PPM = 50                     # px per metre (the plan SVG uses the same)
PAD_L, PAD_R, PAD_T, PAD_B = 1.2, 4.6, 1.6, 3.4     # metres of paper round the drawing (right: level marks)

PALETTE = {
    "clean": dict(bg="#FFFFFF", line="#1F2937", thin="#6B7280", wall="#F3EDE2", wall_line="#374151",
                  void="#3F4A5A", recess="#DCD2C2", glass="#BFDBFE", frame="#334155", door="#9A6B43", slab="#CBD5E1",
                  plinth="#D6D3D1", hatch="#A8A29E", cmp="#E7E5E4", gate="#374151", ground="#78716C",
                  text="#111827", dim="#4B5563", rail="#475569", behind="#E5E7EB", col="#94A3B8",
                  title="#111827"),
    "blueprint": dict(bg="#0A2342", line="#00FFFF", thin="#6B8DAF", wall="#123456", wall_line="#00DDFF",
                      void="#061629", recess="#0D2B4A", glass="#0E4A73", frame="#00BFFF", door="#1B5E6E", slab="#1B3A5C",
                      plinth="#1E4D6E", hatch="#2E6E9C", cmp="#16304F", gate="#00FF88", ground="#6B8DAF",
                      text="#E0F0FF", dim="#8EB8E5", rail="#00DDDD", behind="#0F2D4D", col="#00E5FF",
                      title="#FFFFFF"),
}
FONT = "Arial, Helvetica, sans-serif"


def _num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-6 else f"{int(round(v))}"


class _Canvas:
    """A tiny SVG writer in elevation metres (u right, z up)."""

    def __init__(self, width_m: float, height_m: float, z_top: float, colors: Dict):
        self.w, self.h, self.zt, self.c = width_m, height_m, z_top, colors
        self.parts: List[str] = []

    def X(self, u):
        return (u + PAD_L) * PPM

    def Y(self, z):
        return (self.zt - z + PAD_T) * PPM

    def rect(self, u0, z0, u1, z1, fill="none", stroke=None, sw=0.8, opacity=None, dash=None):
        if u1 - u0 <= 1e-6 or z1 - z0 <= 1e-6:
            return
        a = [f'x="{self.X(u0):.2f}" y="{self.Y(z1):.2f}" width="{(u1 - u0) * PPM:.2f}" height="{(z1 - z0) * PPM:.2f}"',
             f'fill="{fill}"', f'stroke="{stroke or "none"}"', f'stroke-width="{sw}"']
        if opacity is not None:
            a.append(f'fill-opacity="{opacity}"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        self.parts.append(f"<rect {' '.join(a)}/>")

    def line(self, u0, z0, u1, z1, stroke, sw=0.8, dash=None, opacity=None):
        a = f'x1="{self.X(u0):.2f}" y1="{self.Y(z0):.2f}" x2="{self.X(u1):.2f}" y2="{self.Y(z1):.2f}" ' \
            f'stroke="{stroke}" stroke-width="{sw}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            a += f' stroke-opacity="{opacity}"'
        self.parts.append(f"<line {a}/>")

    def poly(self, pts, fill="none", stroke=None, sw=0.8, opacity=None):
        p = " ".join(f"{self.X(u):.2f},{self.Y(z):.2f}" for u, z in pts)
        extra = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"{extra}/>')

    def text(self, u, z, s, size=10, fill=None, anchor="start", bold=False, px=False):
        x, y = (u, z) if px else (self.X(u), self.Y(z))
        w = ' font-weight="bold"' if bold else ""
        s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill or self.c["text"]}" '
                          f'font-family="{FONT}" text-anchor="{anchor}"{w}>{s}</text>')

    def svg(self) -> str:
        W, H = (self.w + PAD_L + PAD_R) * PPM, (self.h + PAD_T + PAD_B) * PPM
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
                f'viewBox="0 0 {W:.0f} {H:.0f}">')
        defs = (f'<defs><pattern id="stone" width="8" height="8" patternUnits="userSpaceOnUse">'
                f'<path d="M0 8 L8 0" stroke="{self.c["hatch"]}" stroke-width="0.6"/></pattern>'
                f'<pattern id="earth" width="10" height="10" patternUnits="userSpaceOnUse">'
                f'<path d="M0 10 L10 0 M-2 2 L2 -2 M8 12 L12 8" stroke="{self.c["ground"]}" stroke-width="0.6"/>'
                f'</pattern></defs>')
        bg = f'<rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="{self.c["bg"]}"/>'
        return head + defs + bg + "".join(self.parts) + "</svg>"


def _front(box, yf, tol=0.06) -> bool:
    """A horizontal wall box lying on the front line y = yf."""
    x0, y0, x1, y1 = box[:4]
    return (x1 - x0) >= (y1 - y0) and y0 - tol <= yf <= y1 + tol


def front_elevation_svg(plan, scene: Optional[Dict] = None, mode: str = "clean") -> str:
    """The front (road-side) elevation of the design as an SVG document."""
    if scene is None:
        from floorplan_renderer.walkthrough import build_scene
        scene = build_scene(plan)
    c = PALETTE.get(mode, PALETTE["clean"])
    W = float(scene["plot"]["w"])
    lv = scene["levels"]
    fx, fy, fw, fd = scene["footprint"]
    yf = fy + fd                                               # the building's front wall line (centre)
    outer = scene["shell"]["outer"]
    shell = scene["shell"]
    levels = scene.get("levels_list") or []
    rooms_levels = [l for l in levels if l["kind"] == "rooms"]
    ext = scene["walls"].get("ext") or []
    base = min((w[4] for w in ext), default=lv["ffl"])         # the floor template's level (its walls' foot)
    roof = float(shell["roof_z"])
    parapet_top = roof + lv.get("parapet", 1.0)
    head = parapet_top
    if shell.get("mumty_roof"):
        head = max(head, shell["mumty_roof"][5])
    lift = scene.get("lift") or {}
    if lift.get("head"):
        head = max(head, float(lift["head"]))
    z_top = head + 0.3
    cv = _Canvas(W, z_top, z_top, c)
    U = lambda x: W - x                                        # noqa: E731 — the road-side view mirrors the plan

    def urange(x0, x1):
        a, b = U(x1), U(x0)
        return (a, b) if a <= b else (b, a)

    stilt = bool(lv.get("stilt"))
    first = rooms_levels[0]["z"] if rooms_levels else lv["ffl"]
    slab = float(lv.get("slab", 0.125))
    bu0, bu1 = urange(outer[0], outer[2])

    # ── behind the parapet: the stair head-room (mumty) and the lift's overrun
    if shell.get("mumty_roof"):
        m = shell["mumty_roof"]
        a, b = urange(m[0], m[2])
        cv.rect(a + 0.2, roof, b - 0.2, m[4], fill=c["behind"], stroke=c["thin"], sw=0.8)
        cv.rect(a, m[4], b, m[5], fill=c["slab"], stroke=c["line"], sw=0.9)
    if lift.get("room") and lift.get("head") and not lift.get("tower"):
        r = lift["room"]
        a, b = urange(r[0], r[2])
        if float(lift["head"]) > parapet_top:
            cv.rect(a, roof, b, float(lift["head"]), fill=c["behind"], stroke=c["thin"], sw=0.8)
            cv.text((a + b) / 2, float(lift["head"]) - 0.45, "LIFT", 8, c["dim"], "middle")

    # ── the building's front face, floor by floor: an opening shows as shadow, then the wall panels on top
    for L in rooms_levels:
        dz = L["z"] - base
        z0 = L["z"]
        # a duplex's first floor has its own plan (scene['upper']: its windows, doors, sit-outs); it is built at
        # the same template level as the ground floor
        own = scene["upper"] if (L.get("plan") == 1 and scene.get("upper")) else scene
        src = dict(scene, windows=own.get("windows") or [], doors=own.get("doors") or [], rooms=own.get("rooms") or [],
                   walls=dict(scene["walls"], parapet=(own.get("walls") or {}).get("parapet") or []))
        ceil = z0 + lv["storey"] - slab
        # one plastered face per floor; an open sit-out / gallery on the front shows as shadow over its parapet
        cv.rect(bu0, z0, bu1, ceil, fill=c["wall"], stroke=c["wall_line"], sw=0.6)
        open_front = [w for w in src["walls"].get("parapet") or [] if _front(w, yf) and w[4] < roof - 0.5]
        ground_floor = L is rooms_levels[0] or own is not scene     # a floor with its own plan: its own doors
        for w in open_front:
            # through the open front you see the wall behind the sit-out, set back, with its own windows and doors
            a, b = urange(w[0], w[2])
            cv.rect(a, z0, b, ceil, fill=c["recess"], stroke=c["wall_line"], sw=0.4)
            back = _room_behind(src, w, yf)
            if back is not None:
                _openings_on(cv, src, back, w[0], w[2], dz, z0, U, ground_floor, lv)
            cv.rect(a, z0, b, w[5] + dz, fill=c["wall"], stroke=c["wall_line"], sw=0.5)     # the parapet
            cv.line(a - 0.05, w[5] + dz, b + 0.05, w[5] + dz, c["line"], 1.6)
        # windows (the schedule's sill and head), each with its frame, glass and a centre mullion
        for wd in src.get("windows") or []:
            if (wd.get("out") or [0, 0])[1] < 0.5 or abs(wd["a"][1] - yf) > 0.3:
                continue
            a, b = urange(wd["a"][0], wd["b"][0])
            s0, s1 = wd["z0"] + dz, wd["z1"] + dz
            _window(cv, a, s0, b, s1, bool(wd.get("vent")))
        # doors on the front: the ground floor's own; above, what the scene makes of them
        for d in src.get("doors") or []:
            if not d.get("horiz") or abs(d["mid"][1] - yf) > 0.25 or d["kind"] in ("lift", "internal", "bath"):
                continue
            a, b = urange(d["mid"][0] - d["w"] / 2, d["mid"][0] + d["w"] / 2)
            if ground_floor:
                _door(cv, a, z0, b, z0 + d.get("h", lv["door"]))
                continue
            up = d.get("upper")
            if up == "balcony":
                _door(cv, a, z0, b, z0 + d.get("h", lv["door"]))
                bal = d.get("balcony")
                ba, bb = urange(bal[0], bal[2]) if bal else (a - 0.3, b + 0.3)
                _railing(cv, ba, z0, bb, z0 + 1.0)
            elif up == "parapet":
                _railing(cv, a, z0, b, z0 + 1.0)
            else:                                              # closed: the wall carries on, with a window
                cv.rect(a, z0, b, z0 + lv["storey"] - slab, fill=c["wall"], stroke=c["wall_line"], sw=0.5)
                _window(cv, a + 0.1, z0 + 0.9, b - 0.1, z0 + lv["door"], False)
        # the slab edge of this floor (the one above it) as a band
        cv.rect(bu0 - 0.05, z0 + lv["storey"] - slab, bu1 + 0.05, z0 + lv["storey"], fill=c["slab"],
                stroke=c["line"], sw=1.0)

    # ── the roof: front parapet with its coping
    cv.rect(bu0, roof, bu1, parapet_top, fill=c["wall"], stroke=c["wall_line"], sw=0.6)
    cv.line(bu0 - 0.08, parapet_top, bu1 + 0.08, parapet_top, c["line"], 2.0)

    # ── below the first floor: the stilt (columns over open parking) or the plinth
    if stilt:
        cv.rect(bu0, 0, bu1, first - slab, fill=c["void"], opacity=0.55)
        cols = (scene.get("stilt") or {}).get("columns") or []
        for col in cols:
            if col[3] < yf - 0.5:
                continue
            a, b = urange(col[0], col[2])
            cv.rect(a, 0, b, first - slab, fill=c["col"], stroke=c["line"], sw=0.8)
        cv.rect(bu0 - 0.05, first - slab - 0.45, bu1 + 0.05, first - slab, fill=c["slab"], stroke=c["line"], sw=1.0)
        cv.text((bu0 + bu1) / 2, (first - slab) / 2, "STILT PARKING", 11, c["bg"] if mode == "clean" else c["text"],
                "middle", bold=True)
    else:
        cv.rect(bu0 - 0.08, 0, bu1 + 0.08, lv["ffl"], fill="url(#stone)", stroke=c["line"], sw=1.0)
    # steps up to the main door
    for s in scene.get("steps") or []:
        a, b = urange(s[0], s[2])
        cv.rect(a, s[4], b, s[5], fill=c["plinth"], stroke=c["line"], sw=0.7)

    # ── a lift tower standing outside the house: full height, in front of the wall it serves
    if lift.get("tower") and lift.get("outer"):
        o = lift["outer"]
        a, b = urange(o[0], o[2])
        cv.rect(a, 0, b, float(lift.get("head") or parapet_top), fill=c["wall"], stroke=c["line"], sw=1.0)
        cv.text((a + b) / 2, float(lift.get("head") or parapet_top) - 0.6, "LIFT", 9, c["dim"], "middle", bold=True)

    # ── the outside terrace stair, where the road sees it: along the front (its stepped flight over the face)
    # or on a side (end-on, beside the house); one at the back is hidden by the house
    for stp in scene.get("stairs") or []:
        if not stp.get("external"):
            continue
        boxes = [bx for bx in stp.get("boxes") or []
                 if bx[3] > yf + 0.05 or not (urange(bx[0], bx[2])[0] >= bu0 - 0.01 and urange(bx[0], bx[2])[1] <= bu1 + 0.01)]
        if not boxes:
            continue
        for bx in sorted(boxes, key=lambda q: q[3]):           # far first: the nearer treads drawn over them
            a, b = urange(bx[0], bx[2])
            cv.rect(a, bx[4], b, bx[5], fill=c["plinth"], stroke=c["line"], sw=0.6)
        for rail in stp.get("rails") or []:
            pts = [(U(q[0]), q[2]) for q in rail]
            for (u0, z0), (u1, z1) in zip(pts, pts[1:]):
                cv.line(u0, z0, u1, z1, c["line"], 1.2)
            for u, z in pts[::3]:                               # balusters
                cv.line(u, z - 0.9, u, z, c["line"], 0.5)
        ua = min(urange(bx[0], bx[2])[0] for bx in boxes)
        ub = max(urange(bx[0], bx[2])[1] for bx in boxes)
        ztop = max(bx[5] for bx in boxes)
        cv.text((ua + ub) / 2, ztop + 1.25, "TERRACE STAIR", 8, c["dim"], "middle", bold=True)

    # ── in front: the compound wall, its pillars and gates
    cmp_h = lv.get("compound", 1.5)
    H = float(scene["plot"]["h"])
    for w in scene["walls"].get("cmp") or []:
        if _front(w, H, 0.3):
            a, b = urange(w[0], w[2])
            cv.rect(a, 0, b, w[5] if len(w) > 5 else cmp_h, fill=c["cmp"], stroke=c["line"], sw=0.9, opacity=0.88)
    for p in scene["walls"].get("pillar") or []:
        if p[3] > H - 0.6:
            a, b = urange(p[0], p[2])
            cv.rect(a, 0, b, p[5] if len(p) > 5 else cmp_h + 0.3, fill=c["cmp"], stroke=c["line"], sw=1.0)
    for g in scene.get("gates") or []:
        a, b = urange(g["x0"], g["x1"])
        _gate(cv, a, b, float(g.get("h", cmp_h)), g)

    # ── ground, road and annotations
    cv.rect(-0.8, -0.6, W + 0.8, 0, fill="url(#earth)")
    cv.line(-0.8, 0, W + 0.8, 0, c["line"], 2.0)
    _levels(cv, W, [(0.0, "GL ±0.000"), *([] if stilt else [(lv["ffl"], f"FFL +{lv['ffl']:.3f}")])]
            + [(L["z"], f"+{L['z']:.3f}  {L['label'].upper()}") for L in rooms_levels
               if stilt or L["z"] > lv["ffl"] + 0.05]
            + [(roof, f"+{roof:.3f}  TERRACE"), (parapet_top, f"+{parapet_top:.3f}  PARAPET")])
    _dims(cv, bu0, bu1, W, parapet_top)
    eng = getattr(plan, "engineering", None) or {}
    road = eng.get("road_m")
    face = {"N": "North", "E": "East", "S": "South", "W": "West"}.get(getattr(plan, "orientation", "N"), "")
    sub = (f"Seen from the {round(road / 0.3048)} ft road" if road else "Seen from the road") + \
          (f" · {face}-facing" if face else "") + f" · {len(rooms_levels)} floor{'s' if len(rooms_levels) != 1 else ''}"
    cv.text(-0.8, -1.45, "FRONT ELEVATION", 16, c["title"], bold=True)
    cv.text(-0.8, -2.05, sub, 10, c["dim"])
    _scale_bar(cv, W - 5.0, -1.6)
    return cv.svg()


# ─────────────────────────────────────────────────────────── parts
def _room_behind(scene: Dict, w, yf: float) -> Optional[float]:
    """The y of the wall seen through an open front (the far side of the sit-out / gallery behind it)."""
    mid = (w[0] + w[2]) / 2
    for r in scene.get("rooms") or []:
        x0, y0, x1, y1 = r["r"]
        if x0 - 0.05 <= mid <= x1 + 0.05 and abs(y1 - yf) < 0.1:
            return y0
    return None


def _openings_on(cv: _Canvas, scene: Dict, y_back: float, x0: float, x1: float, dz: float, z0: float, U,
                 ground_floor: bool, lv: Dict):
    """Windows and doors on the set-back wall at y_back between x0 and x1 (seen through an open front)."""
    lo, hi = min(x0, x1) - 0.05, max(x0, x1) + 0.05
    for wd in scene.get("windows") or []:
        if (wd.get("out") or [0, 0])[1] < 0.5 or abs(wd["a"][1] - y_back) > 0.3:
            continue
        if not (lo <= min(wd["a"][0], wd["b"][0]) and max(wd["a"][0], wd["b"][0]) <= hi):
            continue
        a, b = sorted((U(wd["a"][0]), U(wd["b"][0])))
        _window(cv, a, wd["z0"] + dz, b, wd["z1"] + dz, bool(wd.get("vent")))
    for d in scene.get("doors") or []:
        if not d.get("horiz") or abs(d["mid"][1] - y_back) > 0.25 or d["kind"] in ("lift", "internal", "bath"):
            continue
        if not (lo <= d["mid"][0] <= hi):
            continue
        a, b = sorted((U(d["mid"][0] - d["w"] / 2), U(d["mid"][0] + d["w"] / 2)))
        _door(cv, a, z0, b, z0 + d.get("h", lv["door"]))


def _window(cv: _Canvas, u0, z0, u1, z1, vent: bool):
    c = cv.c
    cv.rect(u0 - 0.05, z0 - 0.06, u1 + 0.05, z0, fill=c["slab"], stroke=c["line"], sw=0.6)       # the sill
    cv.rect(u0, z0, u1, z1, fill=c["glass"], stroke=c["frame"], sw=1.1)
    if vent:
        n = max(2, int((z1 - z0) / 0.1))
        for i in range(1, n):
            z = z0 + i * (z1 - z0) / n
            cv.line(u0 + 0.04, z, u1 - 0.04, z, c["frame"], 0.5)
        return
    if u1 - u0 > 0.8:
        cv.line((u0 + u1) / 2, z0, (u0 + u1) / 2, z1, c["frame"], 0.9)
    cv.line(u0, z0 + (z1 - z0) * 0.72, u1, z0 + (z1 - z0) * 0.72, c["frame"], 0.5)


def _door(cv: _Canvas, u0, z0, u1, z1):
    c = cv.c
    cv.rect(u0, z0, u1, z1, fill=c["door"], stroke=c["line"], sw=1.0)
    n = 2 if u1 - u0 > 1.1 else 1
    for i in range(n):
        a = u0 + (u1 - u0) * i / n
        b = u0 + (u1 - u0) * (i + 1) / n
        cv.rect(a + 0.1, z0 + 0.15, b - 0.1, z0 + (z1 - z0) * 0.45, stroke=c["line"], sw=0.5)
        cv.rect(a + 0.1, z0 + (z1 - z0) * 0.52, b - 0.1, z1 - 0.15, stroke=c["line"], sw=0.5)


def _railing(cv: _Canvas, u0, z0, u1, z1):
    c = cv.c
    cv.rect(u0, z0, u1, z0 + 0.12, fill=c["slab"], stroke=c["line"], sw=0.8)
    cv.line(u0, z1, u1, z1, c["rail"], 1.6)
    k = max(2, int((u1 - u0) / 0.12))
    for i in range(k + 1):
        u = u0 + i * (u1 - u0) / k
        cv.line(u, z0 + 0.12, u, z1, c["rail"], 0.6)


def _gate(cv: _Canvas, u0, u1, h, g: Dict):
    c = cv.c
    cv.rect(u0, 0.05, u1, h, fill=c["bg"], opacity=0.35)
    cv.line(u0, h, u1, h, c["gate"], 1.6)
    cv.line(u0, 0.1, u1, 0.1, c["gate"], 1.2)
    cv.line(u0, h * 0.55, u1, h * 0.55, c["gate"], 0.9)
    k = max(3, int((u1 - u0) / 0.11))
    for i in range(k + 1):
        u = u0 + i * (u1 - u0) / k
        cv.line(u, 0.1, u, h, c["gate"], 0.7)
    if len(g.get("leaves") or []) == 2 and g.get("op") != "sliding":
        cv.line((u0 + u1) / 2, 0.1, (u0 + u1) / 2, h, c["gate"], 1.6)
    cv.text((u0 + u1) / 2, h + 0.18, f"{g.get('tag', '')} · {g.get('feet', '')}".strip(" ·"), 7.5, c["dim"], "middle")


def _levels(cv: _Canvas, W: float, marks: List[Tuple[float, str]]):
    c = cv.c
    u = W + 0.4
    seen: List[float] = []
    for z, label in sorted(marks):
        if any(abs(z - s) < 0.25 for s in seen):
            continue
        seen.append(z)
        cv.line(u - 0.2, z, u + 0.6, z, c["dim"], 0.8)
        cv.poly([(u + 0.15, z), (u + 0.3, z + 0.18), (u, z + 0.18)], fill=c["dim"])
        cv.text(u + 0.7, z - 0.07, label, 8.5, c["dim"], bold=True)


def _dims(cv: _Canvas, bu0, bu1, W, top):
    c = cv.c
    for (a, b, z, text) in ((bu0, bu1, -0.95, f"{bu1 - bu0:.2f} m  ({(bu1 - bu0) / 0.3048:.0f} ft) building"),
                            (0.0, W, -1.0 - 0.0, "")):
        if not text:
            continue
        cv.line(a, z, b, z, c["dim"], 0.8)
        for u in (a, b):
            cv.line(u, z - 0.12, u, z + 0.12, c["dim"], 0.8)
            cv.line(u - 0.1, z - 0.1, u + 0.1, z + 0.1, c["dim"], 1.0)
        cv.text((a + b) / 2, z + 0.12, text, 8.5, c["dim"], "middle")
    u = -0.55                                                  # total height on the left
    cv.line(u, 0, u, top, c["dim"], 0.8)
    for z in (0, top):
        cv.line(u - 0.12, z, u + 0.12, z, c["dim"], 0.8)
    cv.text(cv.X(u) - 4, cv.Y(top / 2), f"{top:.2f} m", 8.5, c["dim"], "end", px=True)


def _scale_bar(cv: _Canvas, u, z):
    c = cv.c
    for i in range(5):
        cv.rect(u + i, z, u + i + 1, z + 0.15, fill=c["line"] if i % 2 == 0 else c["bg"], stroke=c["line"], sw=0.6)
    cv.text(u, z + 0.3, "0", 7.5, c["dim"], "middle")
    cv.text(u + 5, z + 0.3, "5 m", 7.5, c["dim"], "middle")
