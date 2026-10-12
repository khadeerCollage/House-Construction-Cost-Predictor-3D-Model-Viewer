"""
Material take-off measured from the drawing
===========================================
A quantity surveyor's take-off for an Indian RCC-framed, brick-walled house, measured
from the generated plan itself instead of a per-sq-ft thumb rule:

  structure   isolated footings + PCC under every column of the plan's column grid,
              column stubs and columns, plinth beams and floor beams under every wall,
              slabs over the footprint, lintels over every door and window, stairs
  masonry     230 mm outer walls / 115 mm partitions (net of door and window openings,
              masonry above doors added back), roof parapet
  finishes    internal / external / ceiling plaster, floor tiles, wall dado, skirting,
              putty + primer + emulsion, waterproofing (toilets + terrace)
  openings    doors by type, windows (m²), lintels
  services    electrical points by room, sanitary fixtures, pipe runs (wet-room MST)

Every line carries how it was measured. Materials are then split out with standard
mix proportions (IS 456 / IS 2212 / CPWD analysis of rates):

  RCC M20 1:1.5:3     8.06 bags cement, 0.42 m³ sand, 0.84 m³ 20 mm aggregate per m³
  PCC M7.5 1:4:8      3.40 bags cement, 0.47 m³ sand, 0.95 m³ 40 mm aggregate per m³
  brickwork           500 bricks + 0.25 m³ wet CM 1:6 mortar per m³
  plaster             12 mm CM 1:6 inside, 20 mm CM 1:4 outside, 6 mm CM 1:3 ceilings
  dry volume          wet × 1.33 (bulking + wastage)
  steel (kg/m³)       footing 70 · column 170 · plinth beam 110 · beam 125 · slab 85 ·
                      lintel 60 · stair 95

Pure geometry — prices live in CostEngine.boq_for_plan, which reconciles the priced
materials with the project's construction cost (one cost formula, not two).
"""

from typing import Dict, List, Optional
import math

FLOOR_TO_FLOOR = 3.0
PLINTH_H = 0.45
FOUNDATION_DEPTH = 1.5
PARAPET_H = 1.0
DOOR_H = 2.1
WINDOW_H = 1.2
M3_TO_CFT = 35.3147
CEMENT_BAG_PER_M3 = 1440 / 50.0          # 1 m³ of loose cement = 28.8 bags
DRY = 1.33

MIX = {   # cement m³ fraction, sand fraction, aggregate fraction of the dry volume
    "RCC": (1 / 5.5, 1.5 / 5.5, 3 / 5.5, 1.54),    # dry-volume factor for concrete
    "PCC": (1 / 13, 4 / 13, 8 / 13, 1.54),
}
STEEL_KG_PER_M3 = {"footing": 70, "column": 170, "plinth_beam": 110, "beam": 125, "slab": 85,
                   "lintel": 60, "stair": 95}
ELECTRICAL_POINTS = {"BEDROOM": 9, "LIVING": 12, "DINING": 6, "KITCHEN": 10, "BATHROOM": 4,
                     "POOJA": 3, "STUDY": 8, "UTILITY": 4, "STORE": 2, "SERVANT": 5,
                     "CORRIDOR": 2, "FOYER": 3, "STAIRS": 3, "LIFT": 2}
SITE = {"PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"}

# Construction stages in build order: materials of each line go to one stage
STAGES = [
    ("foundation", "Foundation", "Digging, PCC bed, footings, column stubs, plinth beams"),
    ("plinth", "Plinth & ground floor bed", "Plinth filling and the PCC base under the floor"),
    ("frame", "Columns, beams & roof slab", "RCC frame, slabs, lintels and stairs, floor by floor"),
    ("masonry", "Brick walls", "Outer walls, partitions and the roof parapet"),
    ("rough_in", "Wiring & pipes (concealed)", "Conduits, wiring and pipes before plastering"),
    ("plaster", "Plastering & waterproofing", "Wall and ceiling plaster, toilet and terrace waterproofing"),
    ("flooring", "Flooring & tiling", "Floor tiles, dado, skirting and their mortar bed"),
    ("openings", "Doors & windows", "Frames, shutters, windows and hardware"),
    ("finishing", "Painting, fixtures & handover", "Putty, paint, sanitary and electrical fittings"),
]


def _building_rooms(plan) -> List:
    return [r for r in plan.rooms if r.category not in SITE and "External" not in r.name]


def _along(w) -> float:
    return w.width if w.wall_side in ("N", "S") else max(w.width, w.height)


def _mix(volume_m3: float, kind: str) -> Dict[str, float]:
    c, s, a, dry = MIX[kind]
    d = volume_m3 * dry
    return {"cement_bags": d * c * CEMENT_BAG_PER_M3, "sand_m3": d * s, "aggregate_m3": d * a}


def _mortar(wet_m3: float, cement_parts: float, sand_parts: float) -> Dict[str, float]:
    d = wet_m3 * DRY
    tot = cement_parts + sand_parts
    return {"cement_bags": d * cement_parts / tot * CEMENT_BAG_PER_M3, "sand_m3": d * sand_parts / tot}


def material_takeoff(plan, floors: Optional[int] = None, design_floors: Optional[int] = None) -> Dict:
    """
    Quantities of one house (all floors). `design_floors` sizes the foundation for a
    future storey (columns and footings carry it) without building it now.

    Returns {"lines": [...], "totals": {...}, "basis": {...}} where each line is
    {group, stage, item, qty, unit, how, cement_bags, sand_m3, aggregate_m3, steel_kg, bricks}.
    """
    n = max(1, int(floors or getattr(plan, "num_floors", 1) or 1))
    nf = max(n, int(design_floors or n))
    eng = getattr(plan, "engineering", None) or {}
    fp = getattr(plan, "building_footprint", None) or (0, 0, plan.plot_width, plan.plot_height)
    fw, fd = fp[2], fp[3]
    rooms = _building_rooms(plan)
    carpet = getattr(plan, "total_carpet_area", 0) or sum(r.area for r in rooms)
    slab_t = (eng.get("slab_mm") or 125) / 1000.0
    beam_d = ((eng.get("beam_mm") or [230, 300])[1]) / 1000.0
    columns = int(eng.get("columns") or max(4, round(fw * fd / 9.0)))
    h_mas = FLOOR_TO_FLOOR - beam_d                      # masonry height between floor and beam soffit

    ext_len = sum(w.length for w in plan.walls if w.is_external and getattr(w, "wall_type", "") != "compound")
    int_len = sum(w.length for w in plan.walls if not w.is_external and getattr(w, "wall_type", "") != "compound")
    comp_len = sum(w.length for w in plan.walls if getattr(w, "wall_type", "") == "compound")
    doors = list(plan.doors)
    main_doors = [d for d in doors if d.is_main_entrance]
    bath_names = {r.name for r in rooms if r.category == "BATHROOM"}
    bath_doors = [d for d in doors if not d.is_main_entrance and d.room_name in bath_names]
    room_doors = [d for d in doors if not d.is_main_entrance and d.room_name not in bath_names]
    door_w_ext = sum(d.width for d in main_doors)
    door_w_int = sum(d.width for d in doors if not d.is_main_entrance)
    win_area = sum(_along(w) * WINDOW_H for w in plan.windows)
    win_len = sum(_along(w) for w in plan.windows)
    perimeter = 2 * (fw + fd)
    slab_area = (fw + 0.23) * (fd + 0.23)

    lines: List[Dict] = []

    def add(group, stage, item, qty, unit, how, per_floor=False, **mat):
        k = n if per_floor else 1
        line = {"group": group, "stage": stage, "item": item, "qty": round(qty * k, 2), "unit": unit,
                "how": how + (f" × {n} floors" if per_floor and n > 1 else "")}
        for key in ("cement_bags", "sand_m3", "aggregate_m3", "steel_kg", "bricks"):
            line[key] = round(mat.get(key, 0.0) * k, 3)
        lines.append(line)

    # ── Foundation ──────────────────────────────────────────────────────
    side = 1.0 + 0.25 * nf                                # isolated footing plan size grows with storeys
    pad_t = 0.30 + 0.05 * nf
    footing = columns * side * side * pad_t
    pcc_f = columns * (side + 0.15) ** 2 * 0.075
    excav = columns * (side + 0.6) ** 2 * FOUNDATION_DEPTH
    stub = columns * 0.23 * 0.30 * (FOUNDATION_DEPTH - pad_t + PLINTH_H)
    plinth_beam_len = ext_len + int_len + door_w_ext + door_w_int
    plinth_beam = plinth_beam_len * 0.23 * 0.30
    add("Earthwork", "foundation", "Excavation for footings", excav, "m³",
        f"{columns} pits of {side + 0.6:.2f} × {side + 0.6:.2f} m, {FOUNDATION_DEPTH} m deep")
    add("Concrete", "foundation", "PCC 1:4:8 under footings", pcc_f, "m³",
        f"{columns} beds of {side + 0.15:.2f} m square × 75 mm", **_mix(pcc_f, "PCC"))
    add("Concrete", "foundation", "RCC footings (M20)", footing, "m³",
        f"{columns} footings {side:.2f} × {side:.2f} × {pad_t:.2f} m" + (f" (sized for G+{nf - 1})" if nf > 1 else ""),
        steel_kg=footing * STEEL_KG_PER_M3["footing"], **_mix(footing, "RCC"))
    add("Concrete", "foundation", "RCC column stubs to plinth", stub, "m³",
        f"{columns} columns 230 × 300 mm × {FOUNDATION_DEPTH - pad_t + PLINTH_H:.2f} m",
        steel_kg=stub * STEEL_KG_PER_M3["column"], **_mix(stub, "RCC"))
    add("Concrete", "foundation", "RCC plinth beams", plinth_beam, "m³",
        f"{plinth_beam_len:.1f} m under every wall × 230 × 300 mm",
        steel_kg=plinth_beam * STEEL_KG_PER_M3["plinth_beam"], **_mix(plinth_beam, "RCC"))

    # ── Plinth & ground bed ─────────────────────────────────────────────
    fill = fw * fd * (PLINTH_H - 0.10)
    pcc_floor = carpet * 0.10
    add("Earthwork", "plinth", "Plinth filling (earth / murram)", fill, "m³",
        f"{fw:.2f} × {fd:.2f} m footprint × {PLINTH_H - 0.10:.2f} m")
    add("Concrete", "plinth", "PCC 1:4:8 floor base", pcc_floor, "m³",
        f"{carpet:.1f} m² carpet × 100 mm", **_mix(pcc_floor, "PCC"))

    # ── Frame, slabs, lintels, stairs (per floor) ───────────────────────
    col = columns * 0.23 * 0.30 * FLOOR_TO_FLOOR
    beam_len = ext_len + int_len + door_w_ext + door_w_int
    beam = beam_len * 0.23 * max(0.0, beam_d - slab_t)
    slab = slab_area * slab_t
    n_open = len(doors) + len(plan.windows)
    lintel = sum((d.width + 0.30) * (0.23 if d.is_main_entrance else 0.115) * 0.15 for d in doors) + \
        sum((_along(w) + 0.30) * 0.23 * 0.15 for w in plan.windows)
    add("Concrete", "frame", "RCC columns", col, "m³", f"{columns} columns 230 × 300 mm × {FLOOR_TO_FLOOR} m",
        per_floor=True, steel_kg=col * STEEL_KG_PER_M3["column"], **_mix(col, "RCC"))
    add("Concrete", "frame", "RCC beams", beam, "m³",
        f"{beam_len:.1f} m of beams under every wall × 230 × {(beam_d - slab_t) * 1000:.0f} mm below the slab",
        per_floor=True, steel_kg=beam * STEEL_KG_PER_M3["beam"], **_mix(beam, "RCC"))
    add("Concrete", "frame", f"RCC slab ({slab_t * 1000:.0f} mm)", slab, "m³",
        f"{slab_area:.1f} m² over the footprint × {slab_t * 1000:.0f} mm", per_floor=True,
        steel_kg=slab * STEEL_KG_PER_M3["slab"], **_mix(slab, "RCC"))
    add("Concrete", "frame", "RCC lintels over openings", lintel, "m³",
        f"{n_open} openings, bearing 150 mm each side, 150 mm deep", per_floor=True,
        steel_kg=lintel * STEEL_KG_PER_M3["lintel"], **_mix(lintel, "RCC"))
    stair = 0.0
    if any(r.category == "STAIRS" and "External" not in r.name for r in plan.rooms):
        stair = 1.6 * max(1, n - 1) + 0.6        # waist slab + steps per flight set, + headroom slab
        how = f"dog-leg flights for {max(1, n - 1)} storey height(s) + landing"
    elif any("External" in r.name and r.category == "STAIRS" for r in plan.rooms):
        stair = 1.2
        how = "outside straight stair to the terrace (waist slab + steps)"
    if stair:
        add("Concrete", "frame", "RCC staircase", stair, "m³", how,
            steel_kg=stair * STEEL_KG_PER_M3["stair"], **_mix(stair, "RCC"))

    # ── Masonry ─────────────────────────────────────────────────────────
    ext_area = ext_len * h_mas - win_area + door_w_ext * (h_mas - DOOR_H)
    int_area = int_len * h_mas + door_w_int * max(0.0, h_mas - DOOR_H)
    ext_bw = ext_area * 0.23
    int_bw = int_area * 0.115
    parapet = perimeter * PARAPET_H * 0.115
    for item, vol, how, per in (
            ("Brick walls 230 mm (outer)", ext_bw,
             f"{ext_len:.1f} m × {h_mas:.2f} m − {win_area:.1f} m² windows − doors, × 0.23 m", True),
            ("Brick walls 115 mm (partitions)", int_bw,
             f"{int_len:.1f} m × {h_mas:.2f} m (door openings left out), × 0.115 m", True),
            ("Roof parapet 115 mm", parapet, f"{perimeter:.1f} m perimeter × {PARAPET_H} m × 0.115 m", False)):
        add("Masonry", "masonry", item, vol, "m³", how, per_floor=per, bricks=vol * 500,
            **_mortar(vol * 0.25, 1, 6))

    # ── Wiring & pipes ──────────────────────────────────────────────────
    points = sum(ELECTRICAL_POINTS.get(r.category, 2) for r in rooms) + 2   # + DB and meter board
    from floorplan_generator.quantities import layout_quantities
    plumbing_m = layout_quantities(plan)["plumbing_m"]
    wet_rooms = [r for r in rooms if r.category in ("BATHROOM", "KITCHEN", "UTILITY")]
    pipes = 2 * plumbing_m + 6 * len(wet_rooms)          # supply + drain along the wet-room tree, + risers/branches
    add("Electrical", "rough_in", "Electrical points (conduit + wiring)", points, "points",
        f"by room type over {len(rooms)} rooms (bedroom 9, living 12, kitchen 10, toilet 4 …)", per_floor=True)
    add("Plumbing", "rough_in", "Water supply & drainage pipes", pipes, "m",
        f"2 × {plumbing_m:.1f} m wet-room network + 6 m branches per wet room ({len(wet_rooms)})", per_floor=True)

    # ── Plaster & waterproofing ─────────────────────────────────────────
    in_plaster = 2 * int_area + (ext_len * h_mas - win_area)
    out_plaster = ext_len * FLOOR_TO_FLOOR - win_area
    ceiling = carpet
    baths = [r for r in rooms if r.category == "BATHROOM"]
    bath_area = sum(r.area for r in baths)
    wp_area = sum(r.area + 0.3 * 2 * (r.w + r.h) for r in baths)
    add("Plaster", "plaster", "Inside wall plaster 12 mm (1:6)", in_plaster, "m²",
        "both faces of partitions + inner face of outer walls, less windows", per_floor=True,
        **_mortar(in_plaster * 0.012, 1, 6))
    add("Plaster", "plaster", "Outside plaster 20 mm (1:4)", out_plaster, "m²",
        f"{ext_len:.1f} m outer wall × {FLOOR_TO_FLOOR} m, less windows", per_floor=True,
        **_mortar(out_plaster * 0.020, 1, 4))
    add("Plaster", "plaster", "Parapet plaster (both faces)", perimeter * PARAPET_H * 2, "m²",
        f"{perimeter:.1f} m × {PARAPET_H} m × 2 faces", **_mortar(perimeter * PARAPET_H * 2 * 0.015, 1, 4))
    add("Plaster", "plaster", "Ceiling plaster 6 mm (1:3)", ceiling, "m²", f"{carpet:.1f} m² of slab soffit",
        per_floor=True, **_mortar(ceiling * 0.006, 1, 3))
    add("Waterproofing", "plaster", "Toilet waterproofing", wp_area, "m²",
        f"{len(baths)} toilet floors {bath_area:.1f} m² + 300 mm up the walls", per_floor=True)
    add("Waterproofing", "plaster", "Terrace waterproofing", slab_area, "m²", f"{slab_area:.1f} m² roof")

    # ── Flooring & tiling ───────────────────────────────────────────────
    dry_floor = sum(r.area for r in rooms if r.category not in ("BATHROOM", "UTILITY", "STAIRS", "LIFT"))
    wet_floor = sum(r.area for r in rooms if r.category in ("BATHROOM", "UTILITY"))
    dado = sum(2 * (r.w + r.h) * 2.1 for r in baths) + \
        sum((r.w + r.h) * 0.6 for r in rooms if r.category == "KITCHEN")
    skirting = sum(2 * (r.w + r.h) * 0.1 for r in rooms if r.category not in ("BATHROOM", "UTILITY", "STAIRS", "LIFT"))
    add("Flooring", "flooring", "Vitrified floor tiles", dry_floor * 1.05 + skirting, "m²",
        f"{dry_floor:.1f} m² rooms + {skirting:.1f} m² skirting, 5 % cutting waste", per_floor=True,
        **_mortar((dry_floor) * 0.040, 1, 4))
    add("Flooring", "flooring", "Anti-skid tiles (toilets, utility)", wet_floor * 1.05, "m²",
        f"{wet_floor:.1f} m² wet floors, 5 % waste", per_floor=True, **_mortar(wet_floor * 0.040, 1, 4))
    add("Flooring", "flooring", "Wall dado tiles", dado * 1.05, "m²",
        "toilets to 2.1 m, kitchen counter back 0.6 m, 5 % waste", per_floor=True)

    # ── Doors & windows ─────────────────────────────────────────────────
    add("Doors & windows", "openings", "Main door (frame + shutter)", len(main_doors), "nos", "main entrance")
    add("Doors & windows", "openings", "Room doors", len(room_doors), "nos",
        "bedrooms, kitchen, pooja, store …", per_floor=True)
    add("Doors & windows", "openings", "Toilet doors (WPC / FRP)", len(bath_doors), "nos", "one per toilet",
        per_floor=True)
    add("Doors & windows", "openings", "Windows & ventilators", win_area, "m²",
        f"{len(plan.windows)} windows, {win_len:.1f} m along walls × {WINDOW_H} m", per_floor=True)

    # ── Painting & fixtures ─────────────────────────────────────────────
    paint_in = in_plaster + ceiling
    add("Painting", "finishing", "Inside: putty, primer, 2 coats emulsion", paint_in, "m²",
        "all plastered inside walls + ceilings", per_floor=True)
    add("Painting", "finishing", "Outside: primer + 2 coats weatherproof paint",
        out_plaster + perimeter * PARAPET_H * 2 / n, "m²", "outer walls + parapet", per_floor=True)
    add("Fixtures", "finishing", "Toilet sanitary sets", len(baths), "sets",
        "WC, wash basin, shower mixer, 2 taps, health faucet per toilet", per_floor=True)
    kitchens = sum(1 for r in rooms if r.category == "KITCHEN")
    add("Fixtures", "finishing", "Kitchen sink sets", kitchens, "sets", "sink + 2 taps", per_floor=True)

    # ── Compound wall (site) ────────────────────────────────────────────
    if comp_len > 0.5:
        cw = comp_len * 1.5 * 0.15
        add("Site works", "finishing", "Compound wall 150 mm, 1.5 m high", cw, "m³",
            f"{comp_len:.1f} m along the plot edge (gates left out)", bricks=cw * 500, **_mortar(cw * 0.25, 1, 6))

    tot = {k: round(sum(l[k] for l in lines), 2) for k in ("cement_bags", "sand_m3", "aggregate_m3",
                                                            "steel_kg", "bricks")}
    tot["sand_cft"] = round(tot["sand_m3"] * M3_TO_CFT)
    tot["aggregate_cft"] = round(tot["aggregate_m3"] * M3_TO_CFT)
    tot["concrete_m3"] = round(sum(l["qty"] for l in lines if l["group"] == "Concrete"), 2)
    tot["rcc_m3"] = round(sum(l["qty"] for l in lines if l["group"] == "Concrete" and "RCC" in l["item"]), 2)
    tot["brickwork_m3"] = round(sum(l["qty"] for l in lines if l["unit"] == "m³" and l["bricks"]), 2)
    tot["floor_tiles_m2"] = round(sum(l["qty"] for l in lines if l["group"] == "Flooring"), 1)
    tot["paint_m2"] = round(sum(l["qty"] for l in lines if l["group"] == "Painting"), 1)
    tot["paint_litres"] = round(tot["paint_m2"] * 0.20)          # 2 finish coats + primer ≈ 0.2 L/m²
    tot["putty_kg"] = round(sum(l["qty"] for l in lines if l["item"].startswith("Inside:")) * 0.9)
    built_up = (fw + 0.23) * (fd + 0.23) * n
    return {
        "lines": lines,
        "totals": tot,
        "basis": {"floors": n, "design_floors": nf, "columns": columns, "slab_mm": round(slab_t * 1000),
                  "beam_depth_mm": round(beam_d * 1000), "outer_wall_m": round(ext_len, 1),
                  "partition_m": round(int_len, 1), "compound_wall_m": round(comp_len, 1),
                  "built_up_m2": round(built_up, 1), "built_up_sqft": round(built_up * 10.7639),
                  "carpet_m2": round(carpet * n, 1)},
    }
