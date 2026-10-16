"""
📦 Builder & bank pack — one PDF to hand to the engineer, the mason and the home-loan officer
==========================================================================================
Everything the owner's people ask for, from the design itself (no second formula anywhere):

  1  Cover          plot, building, areas, Vastu score, estimated cost and build time, the front elevation
  2  Floor plan     the same drawing as on screen (SVGRenderer, clean CAD style) — a duplex's first floor too
  3  Elevation      the front (road-side) elevation (floorplan_renderer.elevation)
  4  Schedules      every room's inside size with its NBC 2016 check; doors, windows and gates
  5  Cost           CostEngine.estimate_for_plan (the app's one cost) + the measured BOQ (key materials and the
                    stage-by-stage payment plan — what a bank releases loan instalments against)
  6  Vastu          score, where the points went, the rule checks, the plan on the mandala
  7  Structure      RCC frame, slab, beams, stair, lift, plumbing, parking and gate notes from the engine

Drawings go in as vectors (svglib turns the SVGs into ReportLab drawings), so they print sharp at any size.
The text says "indicative": a licensed structural engineer designs the frame and the local body approves.

    build_pack(plan, cost_data, quality="Standard (Medium)") -> bytes   # the PDF
"""

import datetime as _dt
import io
import os
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

PAGE = landscape(A4)
MARGIN = 12 * mm
INK = colors.HexColor("#111827")
DIM = colors.HexColor("#4B5563")
ACCENT = colors.HexColor("#0E7490")
RULE = colors.HexColor("#CBD5E1")
ZEBRA = colors.HexColor("#F1F5F9")
OK = colors.HexColor("#047857")
BAD = colors.HexColor("#B91C1C")
WARN = colors.HexColor("#B45309")


# ─────────────────────────────────────────────────────────── fonts (₹ and ≥ need a real TrueType font)
def _font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for name, path, bold in (("Arial", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
                             ("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if os.path.exists(path) and os.path.exists(bold):
            try:
                if name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(name, path))
                    pdfmetrics.registerFont(TTFont(name + "-Bold", bold))
                    from reportlab.lib.fonts import addMapping
                    addMapping(name, 0, 0, name)
                    addMapping(name, 1, 0, name + "-Bold")
                return name
            except Exception:  # noqa: BLE001 — a broken font file: fall back to Helvetica
                continue
    return "Helvetica"


def _clean(s: Any, font: str) -> str:
    """Text for the PDF: XML-escaped; on the built-in Helvetica, symbols it lacks spelled out."""
    s = str(s if s is not None else "")
    if font == "Helvetica":
        for a, b in (("₹", "Rs. "), ("≥", ">="), ("≤", "<="), ("→", "->"), ("✓", "ok"), ("✗", "x"), ("≈", "~")):
            s = s.replace(a, b)
        s = s.encode("cp1252", "replace").decode("cp1252")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _styles(font: str) -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    bold = font + "-Bold" if font != "Helvetica" else "Helvetica-Bold"
    mk = lambda n, **k: ParagraphStyle(n, parent=base, fontName=k.pop("fn", font), textColor=k.pop("c", INK), **k)  # noqa: E731
    return {
        "title": mk("title", fontSize=24, leading=28, fn=bold),
        "h1": mk("h1", fontSize=15, leading=19, fn=bold, c=ACCENT, spaceAfter=4),
        "h2": mk("h2", fontSize=11, leading=14, fn=bold, spaceBefore=6, spaceAfter=3),
        "body": mk("body", fontSize=9, leading=12),
        "small": mk("small", fontSize=7.5, leading=9.5, c=DIM),
        "cell": mk("cell", fontSize=8, leading=10),
        "cellb": mk("cellb", fontSize=8, leading=10, fn=bold),
        "big": mk("big", fontSize=13, leading=16, fn=bold),
        "bold": bold,
    }


# ─────────────────────────────────────────────────────────── helpers
def _svg_drawing(svg: str, max_w: float, max_h: float):
    """An SVG string → a ReportLab drawing scaled to fit max_w × max_h (points); None if it cannot."""
    try:
        from svglib.svglib import svg2rlg
    except ImportError:
        return None
    import re
    # the PDF's own Helvetica for every label: svglib otherwise searches the system for "Arial" on every text
    # (seconds per drawing); the drawings only use Latin text, which Helvetica covers
    svg = re.sub(r'font-family="[^"]*"', 'font-family="Helvetica"', svg)
    svg = re.sub(r"font-family:[^;\"']*", "font-family:Helvetica", svg)
    try:
        d = svg2rlg(io.BytesIO(svg.encode("utf-8")))
    except Exception:  # noqa: BLE001 — a drawing that cannot be converted is left out, the pack still builds
        return None
    if d is None or not d.width or not d.height:
        return None
    k = min(max_w / d.width, max_h / d.height)
    d.width, d.height = d.width * k, d.height * k
    d.scale(k, k)
    return d


def _money(v: float, country: str = "India") -> str:
    try:
        from model_folder.cost_engine import CostEngine
        return CostEngine.format_money(v, country)
    except Exception:  # noqa: BLE001
        return f"{v:,.0f}"


def _table(rows: List[List[Any]], widths: List[float], st: Dict, head: bool = True, font: str = "Helvetica",
           colors_by_cell: Optional[Dict] = None) -> Table:
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(_clean(v, font), st["cellb"] if (head and i == 0) else st["cell"]) for v in r])
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2FE")), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    for i in range(1 if head else 0, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    for (r, cidx), col in (colors_by_cell or {}).items():
        style.append(("TEXTCOLOR", (cidx, r), (cidx, r), col))
    t.setStyle(TableStyle(style))
    return t


# ─────────────────────────────────────────────────────────── the pack
def build_pack(plan, cost_data: Dict[str, Any], quality: str = "Standard (Medium)", land=None,
               owner: str = "") -> bytes:
    from floorplan_generator import house_report as hr
    from floorplan_renderer.drafting import build_drafting
    from floorplan_renderer.elevation import front_elevation_svg
    from floorplan_renderer.svg_renderer import SVGRenderer
    from floorplan_renderer.walkthrough import build_scene
    from model_folder.cost_engine import CostEngine

    font = _font()
    st = _styles(font)
    P = lambda text, style="body": Paragraph(_clean(text, font), st[style])  # noqa: E731
    country = cost_data.get("country", "India")
    eng = getattr(plan, "engineering", None) or {}
    ap = eng.get("apartment") or {}
    W, H = float(plan.plot_width), float(plan.plot_height)
    cents = W * H / 40.4686
    face = {"N": "North", "E": "East", "S": "South", "W": "West"}.get(getattr(plan, "orientation", "N"), "")
    floors = max(1, int(getattr(plan, "num_floors", 1) or 1))
    usable_w = PAGE[0] - 2 * MARGIN
    usable_h = PAGE[1] - 2 * MARGIN - 14 * mm
    draw_h = usable_h - 16 * mm                      # a drawing page: room for its heading
    try:
        scene = build_scene(plan)
    except Exception:  # noqa: BLE001 — the elevation is then left out
        scene = None
    elev_svg = front_elevation_svg(plan, scene) if scene is not None else None
    try:
        boq = CostEngine.boq_for_plan(plan, cost_data, quality)
    except Exception:  # noqa: BLE001
        boq = {"available": False, "reason": "The bill of quantities could not be measured for this design."}
    project = getattr(plan, "project_name", "") or "My house"
    today = _dt.date.today().strftime("%d %b %Y")
    story: List = []

    # 1 ─ cover
    kind = ("Apartment building" if ap.get("kind") == "apartment" else "Home on every floor" if ap.get("kind") == "house"
            else "Duplex house" if "duplex" in (getattr(plan, "building_type", "") or "").lower()
            else getattr(plan, "building_type", "House"))
    road = eng.get("road_m")
    facts = [
        ["Plot", f"{cents:.2f} cents · {W:.2f} × {H:.2f} m ({W / 0.3048:.0f} × {H / 0.3048:.0f} ft) · {face}-facing"
                 + (f" · {road / 0.3048:.0f} ft road" if road else "")],
        ["Building", f"{kind} · {plan.bhk_config} · {floors} floor{'s' if floors > 1 else ''}"
                     + (f" · {ap.get('flats_per_floor')} flat(s) per floor, {ap.get('flats_total')} homes"
                        if ap.get("kind") == "apartment" else "")
                     + (" · stilt parking" if eng.get("car_parking") == "stilt" or eng.get("stilt") else "")],
        ["Areas", f"carpet {plan.total_carpet_area:.1f} m² ({plan.total_carpet_area * 10.764:.0f} sq ft)"
                  f" · built-up {eng.get('built_up_m2_per_floor', plan.total_built_up_area):.1f} m² a floor,"
                  f" {eng.get('built_up_m2_total', plan.total_built_up_area * floors):.1f} m² in all"],
        ["Vastu", f"{plan.vastu_score:.0f} / 100" if getattr(plan, "vastu_enabled", True) else "not applied"],
        ["Estimated cost", f"{cost_data['display']['total']} ({cost_data['display']['rate']}) — "
                           f"{cost_data.get('region', '')}, {cost_data.get('quality_level', quality)}; excludes land"],
    ]
    if boq.get("available"):
        facts.append(["Build time", f"about {boq['weeks']} weeks (≈ {boq['weeks'] / 4.33:.0f} months)"])
    if getattr(plan, "family_description", ""):
        facts.append(["Family", plan.family_description])
    left = [P("House Plan & Cost Pack", "title"), Spacer(1, 2 * mm), P(project + (f" — for {owner}" if owner else ""), "big"),
            P(f"Prepared {today} · for the engineer, the mason and the bank — every figure measured from the drawing",
              "small"), Spacer(1, 5 * mm),
            _table(facts, [32 * mm, 118 * mm], st, head=False, font=font), Spacer(1, 6 * mm),
            P("In this pack", "h2"),
            P("1 Floor plan · 2 Front elevation · 3 Room, door, window & gate schedules · 4 Cost estimate, materials and "
              "stage payments · 5 Vastu report · 6 Structure & services notes"),
            Spacer(1, 6 * mm),
            P("Indicative design. The RCC frame must be designed and signed by a licensed structural engineer, and the "
              "plan approved by the local body (setbacks, FAR and height by the local building rules), before work "
              "starts. Costs are today's local estimates; get contractor quotes.", "small")]
    right = []
    if elev_svg:
        d = _svg_drawing(elev_svg, 118 * mm, 120 * mm)
        if d is not None:
            right = [d]
    story.append(Table([[left, right]], colWidths=[usable_w * 0.56, usable_w * 0.44],
                       style=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(PageBreak())

    # 2 ─ floor plan(s)
    sheets = [("Floor plan" + (" — ground floor" if getattr(plan, "upper_plan", None) is not None else
                               (" — typical floor (every floor)" if floors > 1 else "")), plan)]
    if getattr(plan, "upper_plan", None) is not None:
        sheets.append(("Floor plan — first floor", plan.upper_plan))
    for title, p in sheets:
        story.append(P(title, "h1"))
        d = _svg_drawing(SVGRenderer().render_to_string(p, mode="clean"), usable_w, draw_h)
        story.append(d if d is not None else P("(The drawing could not be embedded — see the SVG download.)"))
        story.append(PageBreak())

    # 3 ─ front elevation
    if elev_svg:
        story.append(P("Front elevation — as seen from the road", "h1"))
        d = _svg_drawing(elev_svg, usable_w, draw_h)
        if d is not None:
            story.append(d)
        story.append(PageBreak())

    # 4 ─ schedules
    story.append(P("Room schedule — inside sizes, checked against NBC 2016 Part 3", "h1"))
    rs = hr.room_schedule(plan)
    rows = [["Room", "Inside size", "Feet", "Area", "Zone", "NBC 2016 check"]]
    marks = {}
    for i, r in enumerate(rs["rows"], 1):
        if r.get("group") == "site":
            continue
        rows.append([r["name"], f"{r['w']:.2f} × {r['h']:.2f} m", f"{r['w_ft']} × {r['h_ft']}",
                     f"{r['area']:.1f} m² ({r['sqft']} sq ft)", r.get("zone", ""), r["nbc"]["text"]])
        if r["nbc"]["status"] == "low":
            marks[(len(rows) - 1, 5)] = BAD
    story.append(_table(rows, [48 * mm, 30 * mm, 30 * mm, 36 * mm, 14 * mm, usable_w - 158 * mm], st, font=font,
                        colors_by_cell=marks))
    tot = rs["totals"]
    story.append(Spacer(1, 2 * mm))
    story.append(P(f"Carpet {tot['carpet']:.1f} m² · built-up {tot['built_up']:.1f} m² · efficiency {tot['efficiency']:.0f}% "
                   f"· circulation {tot['circulation_pct']:.0f}%", "small"))
    try:
        df = build_drafting(plan)
        sched = df.schedule
    except Exception:  # noqa: BLE001
        sched = []
    if sched:
        story.append(Spacer(1, 4 * mm))
        rows = [["Tag", "Size (mm)", "Sill", "Nos", "Specification"]] + \
               [[s["tag"], s["size"], s["sill"], s["nos"], s["spec"]] for s in sched]
        story.append(KeepTogether([P("Doors, windows & gates", "h2"),
                                   _table(rows, [16 * mm, 30 * mm, 18 * mm, 12 * mm, usable_w - 76 * mm], st, font=font)]))
    story.append(PageBreak())

    # 5 ─ cost
    story.append(P("Cost estimate — construction, materials and stage payments", "h1"))
    soft = cost_data.get("soft_costs", {}) or {}
    rows = [["Item", "Amount"], ["Construction (hard) cost", _money(cost_data.get("construction_cost", 0), country)]]
    for k, lab in (("fees", "Architect / engineer fees"), ("approvals", "Approvals & permits"),
                   ("contingency", "Contingency"), ("tax", "Taxes (GST)")):
        if soft.get(k):
            rows.append([lab, _money(soft[k], country)])
    rows.append(["Total project cost (excluding land)", cost_data["display"]["total"]])
    story.append(_table(rows, [90 * mm, 50 * mm], st, font=font))
    story.append(P(f"{cost_data.get('region', '')}, {country} · {cost_data.get('quality_level', '')} · "
                   f"{cost_data['display']['rate']}. {cost_data.get('description', '')}", "small"))
    if boq.get("available"):
        story.append(Spacer(1, 3 * mm))
        rows = [["Main material (measured from the drawing)", "Quantity", "Rate", "Amount"]] + \
               [[m["label"], f"{m['qty']:,} {m['unit']}", _money(m["rate"], country), _money(m["amount"], country)]
                for m in boq["materials"]]
        story.append(KeepTogether([P("Key materials", "h2"), _table(rows, [80 * mm, 34 * mm, 28 * mm, 36 * mm], st, font=font)]))
        f = boq["finishes"]
        story.append(P(f"Also: concrete {f['concrete_m3']:.1f} m³ · brickwork {f['brickwork_m3']:.1f} m³ · floor tiles "
                       f"{f['floor_tiles_m2']:.0f} m² · paint ≈ {f['paint_litres']:.0f} L. Quantities include normal wastage.",
                       "small"))
        story.append(Spacer(1, 3 * mm))
        rows = [["#", "Stage", "What happens", "Weeks", "Cost", "Paid so far", "Share"]] + \
               [[i + 1, s["label"], s["what"], f"{s['start_week'] + 1}–{s['start_week'] + s['weeks']}",
                 _money(s["amount"], country), _money(s["cumulative"], country), f"{s['pct']:.0f}%"]
                for i, s in enumerate(boq["stages"])]
        story.append(KeepTogether([P("Stage-by-stage payment plan (what a bank releases loan instalments against)", "h2"),
                                   _table(rows, [8 * mm, 46 * mm, usable_w - 160 * mm, 18 * mm, 30 * mm, 32 * mm, 14 * mm],
                                          st, font=font)]))
    else:
        story.append(P(boq.get("reason", ""), "small"))
    story.append(PageBreak())

    # 6 ─ Vastu
    if getattr(plan, "vastu_enabled", True):
        story.append(P(f"Vastu report — {plan.vastu_score:.0f} / 100", "h1"))
        led = hr.vastu_ledger(plan)
        items = sorted(led["items"], key=lambda r: -r["lost"])
        rows = [["Room / door", "Zone", "Score (−50…+10)", "Points lost", "Best zones"]] + \
               [[r["item"], r["zone"], r["score"], f"{r['lost']:.1f}", ", ".join(r.get("best") or [])] for r in items[:10]]
        marks = {(i + 1, 2): (BAD if r["score"] < -10 else WARN if r["score"] < 3 else OK) for i, r in enumerate(items[:10])}
        vc = hr.vastu_checks(plan)
        crow = [["Rule", "Result", "Evidence", "Remedy"]] + \
               [[c["rule"], c["status"].upper(), c["evidence"], c.get("fix", "")] for c in vc["checks"]]
        cmarks = {(i + 1, 1): (OK if c["status"] == "pass" else WARN if c["status"] == "warn" else BAD)
                  for i, c in enumerate(vc["checks"])}
        left = [P("Where the points went (the 10 that cost the most)", "h2"), _table(rows, [44 * mm, 14 * mm, 24 * mm, 20 * mm, 30 * mm], st, font=font,
                                                          colors_by_cell=marks)]
        right = []
        try:
            d = _svg_drawing(hr.vastu_plan_svg(plan), usable_w - 140 * mm, 95 * mm)
            if d is not None:
                right = [P("The plan on the Vastu mandala", "h2"), d]
        except Exception:  # noqa: BLE001
            pass
        story.append(Table([[left, right]], colWidths=[136 * mm, usable_w - 136 * mm],
                           style=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(Spacer(1, 3 * mm))
        story.append(P("Rule checks", "h2"))
        story.append(_table(crow, [56 * mm, 16 * mm, usable_w - 136 * mm, 64 * mm], st, font=font, colors_by_cell=cmarks))
        story.append(PageBreak())

    # 7 ─ structure & services
    story.append(P("Structure & services — notes from the design engine", "h1"))
    grid = eng.get("grid") or {}
    rows = [["Item", "Design"],
            ["RCC frame", f"{eng.get('columns', '–')} columns (230 × 300 mm) on a {len(grid.get('x', []))} × "
                          f"{len(grid.get('y', []))} grid, largest bay {grid.get('max_bay_m', 0):.2f} m"],
            ["Slab & beams", f"{eng.get('slab_mm', '–')} mm two-way slab · beams {eng.get('beam_mm', ['–', '–'])[0]} × "
                             f"{eng.get('beam_mm', ['–', '–'])[1]} mm · M20 concrete, Fe500 steel"],
            ["Footprint", f"{eng.get('footprint_m', ['–', '–'])[0]} × {eng.get('footprint_m', ['–', '–'])[1]} m · ground "
                          f"coverage {eng.get('ground_coverage_pct', '–')}% · FAR {eng.get('far', '–')}"]]
    if eng.get("stair"):
        s = eng["stair"]
        rows.append(["Staircase", ", ".join(f"{k}: {v}" for k, v in s.items() if not isinstance(v, (list, dict)))])
    if eng.get("lift"):
        l = eng["lift"]
        rows.append(["Lift", f"{l.get('persons', '')} persons · {l.get('load_kg', '')} kg · {l.get('position', '')}"])
    if eng.get("stilt"):
        s = eng["stilt"]
        rows.append(["Stilt parking", f"{s.get('fitted', 0)} of {s.get('wanted', 0)} cars fit "
                                      f"({'columns set out between the car bays' if eng.get('grid_x') else 'between the columns'})"])
    for g in eng.get("gates") or []:
        rows.append([f"Gate {g.get('tag', '')}", f"{g.get('feet', '')} {g.get('kind', '').replace('_', ' ')} gate · "
                                                 f"{g.get('op', '')}"])
    story.append(_table(rows, [36 * mm, usable_w - 36 * mm], st, font=font))
    story.append(Spacer(1, 3 * mm))
    notes = [n for n in (getattr(plan, "design_notes", None) or [])
             if any(k in n for k in ("RCC", "Plumbing", "exhaust", "Daylight", "Column lines", "stack", "Setbacks",
                                     "Staircase", "IS 14665", "lift", "Lift", "stilt", "Stilt", "flat", "home"))]
    if notes:
        story.append(P("Engineering notes", "h2"))
        for n in notes[:18]:
            story.append(P("• " + n, "body"))
    story.append(Spacer(1, 4 * mm))
    story.append(P("Next steps: (1) a licensed structural engineer designs the footings and frame from a soil test; "
                   "(2) the plan goes to the local body for approval; (3) the contractor quotes against this BOQ and "
                   "stage plan; (4) the bank releases the loan stage by stage.", "small"))

    buf = io.BytesIO()

    def frame(c, doc):
        c.saveState()
        c.setFont(font, 7.5)
        c.setFillColor(DIM)
        c.drawString(MARGIN, 7 * mm, _clean(f"{project} · {plan.bhk_config} · {cents:.2f} cents · {face}-facing", font)
                     .replace("&amp;", "&"))
        c.drawRightString(PAGE[0] - MARGIN, 7 * mm, f"Page {doc.page}")
        c.setStrokeColor(RULE)
        c.line(MARGIN, 10 * mm, PAGE[0] - MARGIN, 10 * mm)
        c.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                            bottomMargin=14 * mm, title=f"{project} — house plan & cost pack", author="AI Architect")
    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    return buf.getvalue()
