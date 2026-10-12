# CLAUDE.md — Project Memory (platform-neutral)

> Written 2026-10-08 from a full read of the repo (all `.py`, config, tests, docs) plus spot-checks that were
> actually executed. Goal: any AI assistant or human can start a feature here **without re-reading the code**.
> Tags: **[V]** = verified by running code or reading the exact lines · **[D]** = derived from reading, not run ·
> **[NV]** = not verified. Line numbers are as of this date and drift — grep the symbol if one is off.
> Keep this file current: when you change behaviour, update the relevant section (see §13).
> `GEMINI_MEMORY.md` is an older Gemini-targeted analysis; parts are stale (see §12). Prefer this file.

---

## 1. What this project is

**"House Construction Cost Predictor & 3D Model Viewer" (`vit_project`)** — one Streamlit app with two products:

1. **Analyze an existing floor-plan image** (original product): upload JPG/PNG → detect walls/doors/rooms
   (OpenCV Hough + OWL-ViT zero-shot) → classify rooms → calibrate scale → estimate cost → extrude a 3D `.ply`.
2. **Generative "Design Your Home" studio** (added Sept 2026, being extended feature-by-feature): plot size
   (feet / metres / Indian **cents**) + family profile + Vastu preference → NBC-2016-style 2D plan →
   export **SVG / DXF / PDF / PNG**, Vastu audit, cost + BOQ, and an interactive editor.

Target market: India (INR, NBC 2016, Vastu Shastra, cents/gaj/guntha land units, CPWD-style cost tiers).
Docker image: `khadee/house-3d-cost-design-and-detection`. Git user `khadeeCollage`; branch `main`.

**Working mode right now:** the owner adds one feature at a time to an existing codebase. The generative studio is
the active area; the image-analysis side is mostly frozen.

## 2. Run / test / environment

```bash
# UI (Design tab works standalone; Analyze tab needs Flask on :5000)
streamlit run model_folder/frontend.py          # http://localhost:8501
python model_folder/app.py                      # Flask API, binds 127.0.0.1:5000, debug=True
streamlit run app_streamlit.py                  # standalone cloud variant: in-process detection, NO Flask
start.bat                                       # Windows: creates .venv310, installs, launches both
docker run -d -p 8501:8501 -p 5000:5000 khadee/house-3d-cost-design-and-detection
python model_folder/train_cost_model.py         # regenerates house_cost_model_v2.pkl + enhanced CSV
python -m pytest tests -q                       # see §11 for side-effects before running
```

- **Entry points:** `model_folder/frontend.py` (main UI) · `app.py` (repo root; Hugging Face Spaces shim that
  `exec`s `frontend.py`) · `app_streamlit.py` (self-contained) · `model_folder/app.py` (Flask).
- **Python:** Dockerfile/README target **3.10**. This dev machine's default `python` is **3.14.7** with
  numpy 2.5, pandas 3.0, Flask 3.1, streamlit 1.64, torch 2.14 (CPU), open3d 0.20, sklearn 1.9, ezdxf 1.4.4,
  shapely 2.1, reportlab 5.0 — i.e. the code runs on new libs; `requirements.txt` has **no upper bounds**. [V]
  Other interpreters: 3.10, 3.11, 3.12, 3.13 are installed. `.venv/` in the repo is broken (only `Lib/`); `start.bat`
  uses `.venv310`.
- **Missing on this machine:** `ultralytics` (only used by dead code). `streamlit_drawable_canvas` **fails to import**
  in a bare `python` process (`StreamlitAPIException`), so `floorplan_editor.CANVAS_AVAILABLE == False` there and the
  interactive canvas is disabled; whether it works under `streamlit run` is **[NV]**. [V]
- First detection downloads OWL-ViT `google/owlvit-base-patch32` from the HF hub (already cached on this machine).
- Hard ports/URLs: Flask 5000, Streamlit 8501, and `http://127.0.0.1:5000` is hard-coded ≥7× in `frontend.py`.

## 3. Repo map (status: LIVE / WIP / DEAD)

```
app.py                        LIVE  HF-Spaces shim -> exec(model_folder/frontend.py)
app_streamlit.py              LIVE  standalone Streamlit (detector in-process + design tab). px² areas, no scale/3D
Dockerfile, docker-compose.yml, start.bat, .devcontainer/, deploy-page.html, .dockerignore
requirements.txt              no pins above floors
GEMINI_MEMORY.md, README.md, PROJECT_DOCUMENTATION.docx, vit_project-2d-3d.txt (interview-prep notes)  docs
demo.jpeg, floor-plan-with-furniture-...png   sample inputs

floorplan_generator/          LIVE+WIP  generative engine (pure Python, metres, no CV deps)
  room_specs.py   NBC room specs, wall/door/window specs, 1-4 BHK templates        (778 lines, WIP edits)
  layout_engine.py LayoutEngine + all plan dataclasses; THE core                   (1570, WIP edits)
  vastu_engine.py  3x3 Vastu mandala grid + room/zone score matrix                 (480)
  vastu_analyzer.py post-hoc audit of a finished plan (padas, elements, certificate) (669)
  family_analyzer.py household -> recommended BHK                                  (303)
  land_units.py    cents/gaj/guntha/ground/ankanam conversions + plot solver       (184)
  intent_engine.py IntentSolver (wish-aware re-plan), intent validation/description, zone/grid helpers (2026-10-11)
  wish_parser.py   offline NLU: owners' own words → intents (Indian English, typos, Hindi/Malayalam/Tamil) (2026-10-12)
  quantities.py    arrangement-dependent quantities (partitions, doors, glazing, plumbing MST) for edit pricing
  site_feasibility.py what fits BEFORE Generate: BHK options + every site amenity judged with the planner itself
                  + a comfort rule; drives the locked 🔒 toggles in the Design form (2026-10-12)
  takeoff.py       QS material take-off measured from the plan (footings per grid column, beams under walls,
                  slabs, net brickwork, plaster, tiles, paint, doors/windows, points, pipes) + STAGES (2026-10-12)
  house_report.py  the 4 result tabs' analysis: room_schedule (inside sizes, ft-in, NBC 2016 Part 3 checks),
                  vastu_ledger (exact split of vastu_score), vastu_checks, vastu_plan_svg, health_card
  structure.py     RCC column grid (bays 2.4-4.5 m on wall centrelines); used by engineering report + drafting
floorplan_renderer/           LIVE  plan -> files. SVG(drawsvg) DXF(ezdxf) PDF(reportlab) PNG(matplotlib)
  drafting.py      build_drafting(plan): wall poche, column grid, Vastu-aware furniture, tiles, levels, door/window tags + schedule (SVG only so far)
model_folder/
  frontend.py     Streamlit UI (nav radio: Design vs Analyze)                      (1082)
  design_ui.py    `render_design_tab()` — the whole Design tab, direct imports     (675, WIP)
  house_report_ui.py the 4 result tabs under the blueprint (Room Schedule / Vastu Engineering / Cost & BOQ /
                  Best House Recommendation); 'better design' = engine alternatives, applied via intent_studio.commit
  floorplan_editor.py Fabric.js export (`plan_to_fabric_json`, `extract_edited_rooms`) + `render_editor_section`
                  → Smart Re-Plan only (old Move/Add/Delete/Resize tabs + canvas drag/drop retired 2026-10-12)
  intent_studio.py the editing section: tap-a-room house map, action buttons, own-words box, ideas, options,
                  design & price history, commit/undo (2026-10-11/12)
  app.py          Flask API (11 routes)                                            (943)
  cost_engine.py  CostEngine — canonical INR cost + BOQ                            (200)
  train_cost_model.py, house_cost_model.pkl(3-feat), house_cost_model_v2.pkl(8-feat, 24 MB, untracked), *.csv
  utils.py        detect_rooms_and_walls() bridge detector -> Flask/Streamlit      (308)
  vit_detector.py OWL-ViT singleton wrapper                                        (131)
  scale_calibration.py ScaleCalibrator px<->m                                      (121)
  2d-3d.py        Open3D extrusion CLI (called as SUBPROCESS by Flask)             (434)
  static/ temp/   runtime output (gitignored but some files are tracked anyway)
floorplan_detector/           image analysis
  floor_plan_analyzer.py  LIVE engine (WallDoorDetector, RoomAnalyzer, FloorPlanAnalyzer) (1161)
  smart_detection.py, pipeline.py, wall_detection.py, room_detection.py, preprocessing.py,
  model_3d.py, yolo_room_labeler.py     DEAD (only reachable via pipeline.py or not imported at all) [D]
tests/  test_generative_pipeline.py, test_phase1_upgrades.py, test_land_units.py  (unittest style; 0 tests for detector)
test_output/  tracked sample outputs (rewritten by tests)
```

**Dependency direction:** `room_specs → vastu_engine → layout_engine → {vastu_analyzer, floorplan_renderer/*,
floorplan_editor, design_ui, model_folder/app.py}`. The generator/renderer never import the detector; the detector
never imports the generator. `floor_plan_analyzer.py` imports `model_folder.vit_detector` (inverted dependency; makes
**torch a hard import** for the whole detect path and for Flask start-up). [D]

## 4. The two data flows

### 4a. Generative studio (the active area) — engine rewritten 2026-10-10
```
design_ui.render_design_tab()              (direct imports; NOT via Flask)
  site_feasibility.bhk_options(...) / check_site(...)  (st.cache_data, keyed on the form state + the module's
     mtime) → BHK dropdown lists only BHKs that fit (✅ / ⚠️ small rooms; 🔒 caption 'needs about N cents');
     amenity toggles disabled with reason + unlock hint; car/bike counts, garden %, pool sizes capped to what fits;
     an amenity that fitted before a plot/BHK/floors change is switched off with a note; lift + staircase type
     locked for a single storey (engine ignores both there); apply_fit(components, fit) → what the engine gets
  LandUnitsEngine.solve_parcel(...) -> plot W,H (m)        FamilyAnalyzer().analyze(profile, plot_area) -> bhk
  importlib.reload(space_program, site_planner, layout_solver, layout_engine)   # every Generate click
  LayoutEngine(..., family_profile=profile, has_lift, stair_type, num_floors, building_type, additional_components).generate()
    1 space_program.build_program()  rooms+suites from BHK, family roles (master/child/elder/guest), lifestyle
                                     rooms (pooja/study/store/servant), quality scaling, StairDesign/LiftDesign
    2 site_planner.SitePlanner.plan_footprint()  bylaw setbacks, Vastu N/E ≥ S/W open space (only if house keeps
                                     its target area), coverage cap, footprint sized to need, anchored SW;
                                     parking/garden/pool/terrace-stair reservations; narrow plots that fit no
                                     other bay get a parallel front bay at the corner keeping the best door
                                     pada free, and `SitePlan.door_exclusion` keeps the main door off it
                                     (solver penalty `entrance_blocked`)
                                     car_parking `mode: "stilt"` (set by site_feasibility for multi-level
                                     houses where an open bay would squeeze the rooms) forces the stilt floor
    3 layout_solver.BandSpineSolver.solve()  band-spine zoning (front / [private] / spine / rear bands, modes
                                     std|deep|corridor), randomised feasible construction + iterated local search;
                                     objective = Vastu (building mandala, role-aware, pada, Brahmasthan) +
                                     functional − penalties (access, NBC min, daylight, kitchen↔dining, wet walls,
                                     anchors master→SW kitchen→SE pooja→NE, taboos) + plumbing bonus
       relaxation chain if no access-complete layout; legacy functional subdivision as last resort
    4 doors (attached→parent/dress, others→best circulation), arches (living↔dining, stairs, foyers),
      main door at best 32-pada on the living facade
    5 SitePlanner.place_amenities()  Vastu-ranked, collision-free, clear entrance path, gates; two-wheelers fall back to parallel
                                     parking two abreast (n/2 × 2.0 m by 1.5 m) for 1.65 m-deep small-plot front yards
    6 walls: 230 envelope / merged 115 partitions / 150 compound, openings cut for doors/arches/gates
    7 NBC windows (≥1/10 floor area, N/E preferred) · dimensions (3 tiers at plot edge) · audit_vastu() ·
      RERA carpet + plinth area · engineering report (columns, slab/beam, plumbing stacks, notes)
  CostEngine.estimate_for_plan(plan, city_tier, quality, floors)  -> plinth × floors + lift + site works
  renderers / VastuAnalyzer(plan) / FloorPlanEditor as before

Intent editing loop (2026-10-11, UI reworked 2026-10-12) — the Design tab's "🧠 Smart Re-Plan" expander
(model_folder/intent_studio.py; floorplan_editor.render_editor_section only calls it):
  wishes = JSON dicts (zone | flip | swap | adjacent | lock | size | mirror) added by ① tapping a room on a 3×3
  house map (laid out like the drawing, road at top) + ② an action button (move → Vastu-coloured zone grid,
  swap/next-to → tap a 2nd room, other side/bigger/smaller/don't-move → one tap), by typed text
  (wish_parser.understand → confident wishes added, unclear ones shown as "did you mean" buttons), or by
  "💡 Ideas" (wish_parser.suggest_wishes). No canvas drag & drop (retired: hard for many owners).
  → LayoutEngine(**plan.generation_params, design_intents=…, base_layout=plan.layout_state).generate_alternatives(3)
  → IntentSolver (subclass of BandSpineSolver): objective = base − Σ wish penalties (W=40, graded) − λ·disruption
    of unmentioned rooms − collateral taboos + anchor refund for rooms the user moved; seeds = base state, direct
    edits, their neighbourhoods + fresh constructions; ILS with kicks (6000 evals ≈ 1.5–6 s)
  → options: "Balanced — recommended" (cleanest within 6 pts of best), "Best Vastu", "Least change"/"Alternative";
    each plan.edit_report = {intents[{text,satisfied,detail,blocked_by}], changes, warnings, satisfied/total}
  → "Use this design" = intent_studio.commit(): re-renders SVG/DXF/PDF/PNG into a private temp dir, replaces
    gen_* session keys (main blueprint above updates), pushes the old design to plan_history (Undo, last 10)
    and appends to design_history ("📈 Your design & price history": one row per version with a 👁️ View
    button → `_version_dialog` (st.dialog, large): that version's blueprint, Before → after vs the previous
    version, room-by-room `version_changes`, and "Make this my design again" (`_restore` = commit + st.rerun,
    because a button inside a dialog reruns only the dialog); single-series cost line ≥ 3 versions).
  Option cards: label + why, wishes met/blocked_by, 💰 cost vs now with work-item reasons (CostEngine.layout_delta),
  rooms moved, Vastu/NBC warnings.
```
Flask `/generate-plan` now passes the same profile/multi-storey/amenity args and returns `setbacks`, `engineering`,
`vastu_breakdown`, `design_notes`. Typical solve time 0.3–1.5 s (4BHK + everything ≈ 4 s).

### 4b. Image analysis
```
frontend.py --HTTP--> Flask /detect-walls-rooms --> utils.detect_rooms_and_walls(img, output_dir, plot_width, plot_length)
   resize<=1200px -> FloorPlanAnalyzer.analyze():  OWL-ViT objects (always, first) -> WallDoorDetector (adaptive thr,
   Canny 50/150 + HoughLinesP, orthogonal only (<20° / 70-110°), merge same-band lines 25px) -> doors (gaps 10<g<60px +
   ViT "door"; arc detection only if ViT found nothing) -> RoomAnalyzer.segment_rooms (RETR_TREE contours, effective min
   area ≈200-500 px² NOT 1500, fallback connected-components if <3 rooms) -> classify_rooms (largest->Living/Dining,
   symbols bed/toilet/stove, small->Bathroom, rooms[1:4]->Kitchen, rest Bedroom/Hallway/Storage)
   -> ScaleCalibrator (plot dims, else fallback ppm guess assuming plan fills 65% of canvas) -> matplotlib overlay PNG
frontend.py --HTTP--> /generate-3d --> subprocess `python 2d-3d.py --input --output temp/floorplan_3d_model.ply` (timeout 120 s)
   + detect_rooms_and_walls AGAIN (analysis runs twice, ViT loaded twice) -> floorplan_data.json -> static/
   -> Three.js r128 + PLYLoader in an iframe (base64 PLY) in frontend.py
frontend.py --HTTP--> /estimate-construction-cost --> CostEngine
```
`detect_rooms_and_walls` returns: `wall_count, door_count, room_count, visualization(abs path), rooms[], rooms_by_type,
room_type_counts, room_config, estimated_area (m²; =0.7×canvas if sum is 0), pixels_per_meter`. Each room:
`id,type,type_id,label,icon,area(m²),area_px,bounds[x,y,w,h],aspect_ratio,features(static from ROOM_CONFIG),description,color`.
No wall/door geometry is returned in JSON (only drawn into the PNG).

## 5. Generator core: data model & conventions (floorplan_generator/)

**Units:** metres and m² everywhere. Origin `(0,0)` = plot bottom-left, **+x right, +y up**. Plot top edge = `y = plot_height`.
**Orientation (fixed 2026-10-10):** `orientation` = road/entrance **facing**; the road side is the plan **top**. Compass
mapping via `vastu_engine.plan_sides_for_facing(facing)` / `north_edge_for_facing(facing)` (E-facing ⇒ top=E, left=N).
Never pass `orientation` straight into `VastuEngine(north_direction=...)` — use `VastuEngine.for_facing(...)` or
`VastuEngine.for_plan(plan, scope='building'|'plot')`. Rooms are scored on the **building** mandala (footprint origin),
site amenities on the **plot** mandala. Room rects are wall centre-lines tiling `plan.building_footprint` exactly;
`room.area` = clear carpet area (half-walls removed). Doors: hinge at `(x, y)`, `RIGHT` opens along +x/+y, leaf swings
into `room_name` (`floorplan_renderer/_common.py` holds the shared drawing conventions).
New plan fields: `north_edge, building_footprint, setbacks (by compass), stair_type, openings, design_notes,
engineering, vastu_breakdown`; `RoomRect.role`, `WallSegment.wall_type` ('external'|'internal'|'compound').
Dimension-label strings are millimetres. Snap grid 0.05 m (solver) / 0.15 (legacy).

`layout_engine.py` plain `@dataclass`es (use `field(default_factory=...)` for lists):
| Class | Fields |
|---|---|
| `RoomRect` | name, category(str = `RoomCategory.value`), x, y (bottom-left), w, h, color(hex), area, role, **key** (space-program key, '' for amenities/legacy); props cx, cy, right, top |
| `WallSegment` | x1,y1,x2,y2, thickness (0.23 ext / 0.115 int), is_external — **centrelines** |
| `DoorPlacement` | x,y (low-coord corner of opening in generator; renderers treat as hinge), width, wall_side N/S/E/W, swing_direction, room_name, is_main_entrance |
| `WindowPlacement` | x,y (centre), width, height, wall_side, room_name |
| `StaircaseRect` | name, stair_type, x,y,w,h, direction="N", num_steps=15, tread .25, riser .175, has_landing, landing_depth 1.0 |
| `DimLine` | x1,y1,x2,y2, offset, label(mm str), tier (generator emits 1 and 3 only) |
| `GeneratedFloorPlan` | plot_width, plot_height, orientation, rooms, walls, doors, windows, staircases, dimensions, vastu_score, vastu_enabled, total_carpet_area, total_built_up_area, bhk_config, family_description, quality_level, project_name, has_lift, lift_capacity, building_type, num_floors, additional_components |
| (intent fields) | **generation_params** (exact LayoutEngine kwargs → rebuilds the plan bit-for-bit), **layout_state** (`mode, bands, cells, flip, rooms{key: name,category,role,group,parent,band,kind,bands,cx,cy,w,h,area}, footprint, facing, vastu, relax_level`; empty for the legacy fallback ⇒ no Smart Re-Plan), **edit_report** |

`wall_side` values are **plan-relative** (N = high y), not compass-relative. `GeneratedFloorPlan` has **no** `stair_type`
field (renderers read it from `plan.staircases` by room name; `getattr(plan,'stair_type','dog_leg')` always falls back).

**`LayoutEngine.generate()` order:** `_place_rooms` (→ `_place_rooms_vastu` or `_place_rooms_functional`) →
`_clamp_nbc_dimensions` → `_generate_walls` → `_place_staircases` → `_place_doors` → `_place_windows` →
`_generate_dimensions` → `_place_additional_components` (mutates `rooms`; amenities get no walls/doors/windows/dims) →
Vastu score (centroid→zone→`compute_plan_score`) → areas. Carpet area excludes CORRIDOR, PARKING, BIKE_PARKING, GARDEN,
POOL, WATER_TANK; built-up = carpet + Σ wall length×thickness. `orientation` only rotates the *scoring* grid; the vastu
placement itself is a fixed band template (N band: 2nd bedroom | living | pooja; middle: bath stack | dining&lounge;
S band: master | staircase | kitchen(+utility 1.40 m)); main door always on the plan's top wall.

**room_specs.py:** `RoomCategory(str, Enum)` (22 members incl. WIP STAIRS, LIFT, PARKING, BIKE_PARKING, GARDEN, POOL,
WATER_TANK, GENERATOR, SECURITY) · `NBC_ROOM_SPECS` (25 snake_case keys) · `WallThickness` (ext .230, int .115,
wet .150, plaster .015) · `get_bhk_config("2BHK")` (raises ValueError) · `recommend_bhk_for_plot(m²)`: <55→1, <100→2,
<160→3, else 4. Key NBC minima (m², m): living 9.5/2.4 · living_dining 14/3.0 · kitchen 5.0/1.8 · master 9.5/3.0 ·
bedroom 7.5/2.4 · bathroom 1.8/1.2 · attached_bath 2.8/1.2 · pooja 1.5/1.2 · staircase 3.0/1.0 · car 12.5/2.5.
Doors (w×h): main 1.0×2.1, bedroom .9×2.1, bath .75×2.0, kitchen .8×2.0. (These are re-hard-coded in layout_engine;
the template `door_spec/window_spec`, `WallThickness.internal_wet/plaster`, `RoomSpec.validate_dimensions` are unused.)

**vastu_engine.py:** `VastuZone` NE,E,SE,S,SW,W,NW,N,CENTER · `VastuEngine(plot_w, plot_h, north_direction)` ·
`get_room_score(cat, zone)` from `_VASTU_COMPATIBILITY` (13 categories, scores −50…+10; **missing category scores 0**) ·
`is_taboo` from `_STRICT_TABOOS` (bath-NE, kitchen-NE, kitchen-CENTER, bath-CENTER, bedroom-CENTER, foyer-SW, servant-NE) ·
`compute_plan_score([(category, zone)])` · ratings 85/70/50/30. `VastuAnalyzer(plan)` wraps the engine: `get_zone_analysis`,
`audit_entrance` (32 padas), `get_elemental_balance`, `get_defects_and_remedies`, `get_house_recommendation`,
`generate_certificate_text`, `render_mandala_svg` (grades 90/75 — inconsistent with engine's 85/70).
**family_analyzer.py:** bedrooms = 1 master + (adults−2)//2 + max(1,(children+1)//2) + elderly + guests, cap 4; BHK = bedrooms
(+1 tier if ≥2 extra rooms and BHK<3); then plot-area clamp (downgrade only). `additional_rooms` is **advisory text only**.
**land_units.py:** 1 cent = 435.6 sq ft = 40.4686 m²; `LandUnitsEngine.solve_parcel(cents, shape="Square"|"Rectangle",
aspect_ratio=1.33, frontage_ft=None)` → `LandParcel(width_m = X axis, length_m = Y axis, ...)`; `recommend_bhk_for_cents`:
<1.8→1, <3.2→2, <5.2→3, else 4.

## 6. Renderers & editor (floorplan_renderer/, model_folder/floorplan_editor.py)

All renderers take a `GeneratedFloorPlan` and write a file (`render(plan, path)`); SVG also has
`render_to_string(plan, mode='clean'|'blueprint')`; PNG also has `render_wall_mask(plan, ppm=50) -> np.ndarray`
(centreline strokes, no door cut-outs, intended for a 3D pipeline).
- **SVG** (`svg_renderer.py`, drawsvg, richest): `PPM=50` px/m, pad 3.0 m, canvas `(plot_w+6+5)×(plot_h+6)` m (extra 5 m
  = title block), `origin=(-150,-150)`. Draw order: bg, room fills, amenities, walls (hatched), staircases, windows, doors,
  labels, dimensions, north arrow, title block. Stair types: dog_leg(default), straight, l_shaped, open_well, spiral, zigzag.
  Blueprint palette: bg `#0A2342`, wall `#00FFFF`, door `#00FF88`, dim `#6B8DAF`.
- **DXF**: layers `A-WALL-FULL, A-DOOR-FULL, A-GLAZ-FULL, A-ANNO-DIMS, A-ANNO-TEXT, A-TTLB-BORD` (tests assert these exact names).
- **PDF**: A3 landscape, auto-fit scale (not really 1:50), walls as centrelines only. **PNG**: 300 DPI matplotlib.
- **Drafting layer (2026-10-11):** `drafting.build_drafting(plan)` returns renderer-neutral primitives in plan metres (`poly/line/circle/text` tuples + styles) for wall poché (shapely union, cut at openings), structural grid A,B,…/1,2,… (`floorplan_generator.structure.structural_grid`, restricted to the house footprint — external terrace stairs get no columns), clearance-checked Vastu-aware furniture per room, wet-area tiles, FFL/GL level marks, opening tags (MD, D1…, W1…, V1…) and the door/window schedule. Only `svg_renderer.py` consumes it today; PDF/PNG/DXF/editor do not. Car-parking bays draw a plan-view car per stall.
- Amenity/stair/lift **symbols exist only in SVG and the editor**; PDF/PNG draw them as plain room fills; DXF only a text label.
- **Editor**: `FloorPlanEditor` (static methods) — `plan_to_fabric_json(plan, w, h)` builds Fabric.js JSON (v"4.4.0"),
  `sync_canvas_drag_edits` maps drags back to the plan, `render_editor_section(plan, cost_params)` is the UI
  (tabs: Move, Add, Delete, Resize). Constants `PPM_EDITOR=50`, `PAD_M=3.0` must mirror the SVG renderer. Identity is
  smuggled via `strokeDashArray=[tag,0]` (rooms 1000+i, doors 2000+i, windows 3000+i, amenities 4000+i). "Apply Edits"
  deep-copies the plan, rebuilds walls/dims (calls private `LayoutEngine._generate_walls/_generate_dimensions`) and re-renders
  all four formats. No undo, no disk persistence.
- **Smart Re-Plan (2026-10-12):** the whole editing section. The design-tab expander has `key="editor_expander"`
  so it stays open across reruns. Session keys: `intent_stack, intent_options, intent_feedback, intent_sel,
  intent_pending, plan_history (undo snapshots), design_history (versions), intent_base_plan_id, intent_flash`.
  Note for any future canvas work: the installed streamlit-drawable-canvas has **no `drawing_mode="transform"`**
  (selection is the toolbar's pencil toggle).
- **Tests couple to editor output text:** labels must contain `m²` and `×`; `"SCALE: 1:50"` must exist; a compass `N`.

## 7. App layer

**Flask routes (`model_folder/app.py`, CORS on):**
| Route | Purpose |
|---|---|
| `GET /` | banner |
| `POST /predict` | ML cost (loads `house_cost_model_v2.pkl` else `house_cost_model.pkl`; 8 or 3 features). Not called by the UI |
| `POST /detect-walls-rooms` | multipart `file` (+`plot_width`,`plot_length`) → detection JSON + `/static/detection_<ts>.png` + thumbnail |
| `POST /generate-3d` | subprocess `2d-3d.py` → `/static/floorplan_3d_model.ply` + `floorplan_data` |
| `POST /estimate-construction-cost` | `CostEngine.estimate_cost` (the real one) |
| `POST /update-location-type` | legacy separate formula (1200/m² + per-room add-ons × location factor). Not CostEngine; not called by UI |
| `POST /estimate-cost` | raw `model.predict(DataFrame(json))`. Not called by UI |
| `GET /download-ply/<file>` | **broken on Flask ≥2.2** (`attachment_filename`) → returns 500 JSON [D] |
| `POST /generate-plan` | generative plan + 4 files under `static/generated_plans/<plan_id>/` + cost; returns `svg_content`, `download_urls` |
| `GET /download-plan/<fmt>/<plan_id>` | dxf/svg/pdf/png |
| `GET /static/<path>` | static files. A global `errorhandler(Exception)` returns tracebacks to clients |

**CostEngine (`cost_engine.py`) — canonical cost, multi-market (rewritten 2026-10-10).** All market data lives in the
`MARKETS` dict at the top of the file: India (₹/sqft, default), United States ($/sqft), United Kingdom (£/m²), UAE (AED/sqft),
Canada (C$/sqft), Australia (A$/m²). Each market has quality `rates` (2026 reference-city hard cost), `regions` (location
index, optional per-region `tax`/`approvals` overrides), `materials` (BOQ: key,label,unit,qty/unit-area,price,basis floor|footprint),
`site` (amenity/lift rates), `soft` % (fees, approvals, contingency, tax), trade `breakdown` shares, `sources`.
Math: `eff_rate = base[q] × location × size_factor((ref/area)^0.1, 0.90–1.15) × storey_factor`; construction = area×eff_rate +
wet-area (kitchen/bath × quality factor × loc) + site works + lift; total = construction + fees + approvals + contingency + tax.
API: `estimate_cost(area_sqm, bedrooms, bathrooms, kitchens, living_rooms, city_tier, quality_level, num_floors, country=None, region=None)`
→ old keys (`total_cost, area_sqft, cost_per_sqft, cost_breakdown, materials, comparison` keyed by quality engine keys, `description`)
+ `country, region, currency, area_unit, cost_per_unit, effective_rate, factors, soft_costs, calculation` (step table), `notes`,
`display{total,total_compact,rate,construction}` (Lakhs/Crore for India, K/M style elsewhere). `estimate_for_plan(plan, city_tier,
quality_level, num_floors, country, region)` adds `extras`, `building_cost`, `construction_cost`. `region_matrix_for_plan(plan, country)`
→ every region × Economy/Standard/Premium/Luxury. Helpers: `markets()`, `regions(c)`, `default_region(c)`, `format_money`,
`resolve_location` (legacy Indian tier labels map to India regions "Other Tier-1 Metro" / "Other Tier-2 City (average)" / …).
India Standard ₹2,000/sqft at Tier-2 avg (Mumbai ×1.55, Delhi ×1.30, Kochi ×1.25, Rural ×0.80). Design UI: "Country / Market" +
**Edit pricing (2026-10-12):** `plan.layout_state['ref_quantities']` = quantities of the design the owner started from
(carried through every re-plan). `estimate_for_plan` adds an extras line `CostEngine.LAYOUT_LINE` = Σ(Δqty × rate) for
partitions/doors/glazing/plumbing (`LAYOUT_RATES_INDIA`, scaled to other markets by Standard rate per m², × quality,
× floors; location index + soft costs applied as usual) and returns `layout_delta{items,total}`; unedited plans get no
line, so their estimates are unchanged.
"City / Region" dropdowns; the cost is re-priced on every rerun (no regenerate needed). Streamlit markdown/`st.table` treat `$…$`
as LaTeX and leading `+ ` as bullets — escape `$` (`_md` in design_ui) and keep the `＋` step prefixes.
ML model: `RandomForestRegressor(100, depth 12)` on `[Area_m2, Rooms, Bathrooms, Kitchens, Living_Rooms, City_Tier(0=T3,1=T2,2=T1),
Quality_Level(0..2), Num_Floors]`, trained on **synthetic** data generated from the same formula (so it just mimics CostEngine; R²≈0.99).

**Deploy:** single-stage `python:3.10-slim` + CPU torch; `CMD python model_folder/app.py & streamlit run ...` (no supervisor).
`docker-compose.yml` pulls the prebuilt Docker Hub image (does not build). Flask binds `127.0.0.1`, so port 5000 isn't reachable
from the host even though it's published (Streamlit→Flask inside the container works).

## 8. Detection internals (only if touching the image-analysis side)
- Live classes: `floor_plan_analyzer.py` → `WallDoorDetector` (:158), `RoomAnalyzer` (:483, `min_room_area=1500` is only the
  constructor default), `FloorPlanAnalyzer.analyze(image: np.ndarray) -> dict{walls,doors,rooms,binary,shape}` (:1012).
  `RoomLabel` enum (CubiCasa-style ids 0-14). `VitDetector` queries: bed, door arc/swing, toilet, bathtub, kitchen stove, sofa, dining table.
- `2d-3d.py` (`ImprovedFloorPlan3D`): fixed `scale=0.01` m/px (ignores calibration & plot dims), wall h 2.5 m, thickness 0.12 m,
  ASCII PLY, walls as 8-vertex boxes painted `#F2EBE0`, per-room floor colours, **no door/window cut-outs, no ceiling**.
- The room→name mapping is duplicated in 4 places (`floor_plan_analyzer.py`, `utils.py`, `2d-3d.py`, `app_streamlit.py`).
- Extension points: new symbol → `vit_detector.py` queries + `classify_rooms` rule + `_extract_features`; new JSON field → `utils.detect_rooms_and_walls`
  return dict → Flask route → `frontend.py`; 3D doors/windows → `ImprovedFloorPlan3D.create_wall_mesh` (`process()` already returns doors).

## 9. Git / working-tree state (as of 2026-10-08)
`main`, last commit `03669e6` ("Update floor plan editor dimensions and title block layout for SVGRenderer consistency").
**Uncommitted WIP** (≈+3,000 lines): `layout_engine.py` (+763), `room_specs.py` (+99), `svg_renderer.py` (+355),
`design_ui.py` (+170), `floorplan_editor.py` (≈+1,060 net), `static/floorplan_3d_model.ply`. Theme: **multi-storey + lift + stair
types + site amenities + building type + draggable editor blocks**, plus a rewrite of `_place_rooms_vastu` ("AVZ engine") and
rule-based `_place_doors`. Treat the working tree as the source of truth; don't `git checkout`/`reset` those files.
`.gitignore` lists `*.pkl *.pt *.ply __pycache__ temp/ model_folder/static/*` but many such files are **tracked anyway**
(yolov8n.pt, house_cost_model.pkl, PLYs, `.pyc`, detection PNGs); `house_cost_model_v2.pkl` is untracked-but-present.
Windows CRLF/LF warnings appear on commit.

## 10. KNOWN ISSUES (ranked; fix-or-avoid before building on them)
**Status 2026-10-10 (engine upgrade):** #1, #2, #3, #4 (generative studio + Flask), #6, #7, #8, #9, #10 (largely), #11,
and the editor's Vastu HUD are **FIXED** — see `tests/test_engineering_upgrades.py` (11 tests; full suite 31 pass).
Still open: #5 (3D-flow room_counts case), #12 (reload/duck typing — still don't add isinstance), #13 editor UX items
(reset/undo/annotations), #14, #15, #16, #17 (partly reduced via `_common.py`). New caveats: setback/coverage tables are
*indicative* bylaws; 4BHK + all lifestyle rooms on small footprints can still place a secondary bedroom in NE/SE;
the original list below describes the pre-upgrade engine.

1. **[V] Vastu-mode layouts ignore the BHK.** `_place_rooms_vastu` ends with `if len(bed_entries) >= 3: pass` (~l.725):
   1BHK-4BHK all produce the same 9 rooms with 2 bedrooms (3BHK and 4BHK plans identical; carpet 255.12 m² on 15×18). Foyer, servant, store,
   extra baths and bedrooms 3-4 are silently dropped. The non-vastu path gives correct bedroom counts (2/3/4).
2. **[V] Non-vastu plans score 0.0** (`vastu_score`), vastu plans always ≈100 (see #3).
3. **[V] `VastuEngine.compute_plan_score` saturates:** it computes `(avg+50)/15`, which hits 100 for any average ≥ −35, so the score ≈
   `100 − 15×taboo_count`; −20 placements don't reduce it. Docstring "guarantees 100%" is effectively true for orientation N.
4. **[V] Cost tier/quality strings don't match `CostEngine` keys.** `design_ui.py:165-166` offers quality `Economy/Standard/Premium/Luxury` and
   city tier `Tier-1 Metro/Tier-2 Urban/Tier-3 Semi-Urban/Rural` (Flask `/generate-plan` defaults are `Standard` / `Tier-2 Urban`); CostEngine only knows the
   parenthesised strings (§7), so **every generative-studio estimate silently uses the Standard rate × 1.0 city multiplier** whatever the user picks
   (100 m² Tier-1 Premium: ₹17.2 L returned vs ₹37.0 L with proper keys). Also `Luxury`, `Economy`, `Semi-Urban`, `Rural` have no engine equivalent.
5. **[V] 3D-flow room counts always show 0 bedrooms/bathrooms:** `/generate-3d` returns `room_counts` with display-name keys
   (`"Bedroom": 15`), `frontend.py:773` reads lowercase `"bedroom"`. `/update-location-type` has the same mismatch and looks for
   `<stem>_data.json` while the file is `floorplan_data.json`. The fallback branches in `generate_3d` are effectively dead.
   Real sample: 32 rooms/15 "Bedroom" → detector over-segments noisy plans (no overlap de-dup).
6. **[D] Amenities overlap the building.** The building fills the whole plot interior (no setbacks); amenities are placed at absolute coords
   inside it (security (0.5,0.5), parking y=0.5, pool right, …). Carpet area ≈ 93% of plot [V]. Lift (1.6×1.6) leaves an empty void in the S band.
7. **[D] Orientation:** SVG is **y-down with no flip** (`y*PPM`; drawsvg 2.x), so North (high y) is drawn at the **bottom**, north arrow bottom-right;
   the editor/PDF/PNG are y-up → editor and SVG are vertical mirrors despite "consistency" comments; editor title-block rows are upside-down vs SVG.
   Canvas clamp (w 700-1400, h 500-1000) breaks SVG parity for plots > ~17 m wide / ~14 m tall.
8. **[D] Entrance steps** drawn for wall `'S'` only in DXF/PDF/PNG while the generator puts the main door on `'N'` → no steps in those files for default plans.
   SVG door leaf/swing for N/S walls swings outward; others inward.
9. **[D] Dimension lines:** renderers use `tier_offsets{1:.8,2:1.6,3:2.4}` and ignore the generator's negative offsets → dims sit *inside* the plan edge
   in SVG/PDF/PNG (only DXF honours sign). Editor ignores `plan.dimensions` (draws overall W/H only).
10. **[D] `_clamp_nbc_dimensions` doesn't enforce NBC min width/aspect** (only a 1.2 m floor + grid snap of w/h but not x/y → ≤75 mm gaps/overlaps).
    Module header promises "Stage 7 corridor synthesis" and 3 dimension tiers — not implemented. Small/shallow plots can yield negative band heights.
11. **[D] `attached_to` in BHK templates matches by name string**; "Bedroom 2"/"Bedroom 3" don't exist in the vastu path. `needs_home_office`,
    `needs_store_room`, `needs_servant_quarter`, and `needs_pooja_room` (1-2BHK) never create rooms.
12. **[D] `importlib.reload(layout_engine)` per Generate** → classes get redefined; works only by duck typing (no `isinstance` across the reload). Don't add `isinstance` checks.
13. **[D] Editor:** Reset after Apply resets to the *edited* plan; no undo; freedraw/line/rect annotations not mapped to plan; HUD area math
    (carpet = Σ all rooms; built-up = carpet×1.22) disagrees with the generator; "Add Column" adds a `STRUCTURE` room without rebuilding walls;
    flat-tread/landing-depth inputs are ignored; `id()`-based cache keys; `except: pass` in `_rebuild_walls_and_dims`.
14. **[D] Shared fixed output paths** collide across sessions: `static/generated_plans/session_plan/blueprint.*` (design_ui), `%TEMP%/blueprint_edit.*` (editor).
15. **[D] Flask hygiene:** unsanitised upload filename in `/generate-3d` (`temp/<file.filename>`), fixed output name (concurrent requests collide), `debug=True`,
    traceback leakage, `save_visualization=False` is swallowed by `**kwargs` (PNG still written each call), `/download-ply` TypeError.
16. **[D] `2d-3d.py` open3d guard:** `HAS_OPEN3D=False` but `o3d` is never defined and an `-> o3d.geometry.TriangleMesh` annotation is evaluated at class definition → NameError.
    `floorplan_detector.HAS_3D` is always True (misleading).
17. **[D] Units hygiene:** `10.764`, `10.7639`, `435.6`, `3.28`/`3.28084` repeated across files; amenity/abbreviation/door/window/title-block code duplicated 3-5× across renderers & editor.
18. **[V] Test side-effects:** see §11. Detector has zero tests.

## 11. Tests
**2026-10-12: 56/56 pass** (+ `tests/test_house_report.py`, 6: ledger = score, inside sizes/NBC, Vastu checks + SVG, take-off thumb-rule bands, BOQ stages = construction cost, health card; + `tests/test_site_feasibility.py`, 6: 3-cent 2BHK = bikes not car on N/E/S/W, bigger plot
unlocks the car, stilt parking for G+1, BHK options, garden % = real lawn, apply_fit). Before that 44/44 (`tests/test_intent_editing.py` now 12: + everyday-language parsing, edit pricing, parking
on a 3-cent plot for N/E/S/W). **2026-10-11: 41/41 pass** (+ `tests/test_intent_editing.py`, 9 tests, ~25 s: parser, every intent type, refusals with
reasons, blocked-by explanations, determinism, plan rebuild from generation_params). Earlier: **2026-10-10: 32/32 pass** (`python -m pytest tests -q`, ~90 s; test_12 = cost markets/regions bands). `test_engineering_upgrades.py` covers BHK fidelity, facing
geometry, padas, score non-saturation, anchors/access/tiling for N/E/S/W, setbacks, lifestyle rooms, amenity zones,
stairs, cost normalisation, cut openings, SVG orientation. `test_06` now collects editor texts recursively (room labels
live in Fabric groups). Restore artifacts afterwards: `git checkout -- test_output/`.
`tests/` use `unittest` classes (pytest collects them): `test_generative_pipeline.py` (8: NBC specs, BHK configs, vastu engine,
family analyzer, layout 1-4BHK, 4 renderers, CostEngine, Flask `/generate-plan` + downloads), `test_phase1_upgrades.py` (6: SVG symbols/blueprint colours,
PNG+mask, PDF, DXF layers, VastuAnalyzer, editor Fabric JSON), `test_land_units.py` (6: conversions, square/rect solver, frontage solver,
BHK-from-cents, E2E with "Cents" in SVG title block). **Test status against the current WIP was NOT run during this analysis** [NV].
Caveats: (a) `test_generative_pipeline` imports `model_folder.app` → loads the 24 MB pkl and imports torch/transformers via `utils` (slow);
(b) running them **rewrites tracked files** in `test_output/unit_test/` and creates `model_folder/static/generated_plans/plan_*` dirs — check
`git status` afterwards; (c) tests assert on user-visible strings (SVG `" x "`, `" m2"`, `LANDING|UP`, blueprint colours `#0A2342 #00FFFF`;
editor labels `m²`, `×`, `SCALE: 1:50`; DXF layer names) — keep them when editing renderers; (d) 3BHK/4BHK vastu-mode plans pass today
because assertions only require `len(rooms) > 3` (they don't catch issue #1).

## 12. Stale/wrong claims in older docs (don't trust)
`GEMINI_MEMORY.md` / README / `vit_project-2d-3d.txt` say: "numpy<2 / Python 3.10 mandatory" (no pin; runs on 3.14) · "1500 px² room filter"
(effective ~200-500) · "collinear within 5° & 15 px" (that's the dead `wall_detection.py`; live = orthogonal 20°, 25 px band) · MSER/CLAHE preprocessing
and YOLOv8 are "in the pipeline" (dead code) · "Multi-stage Dockerfile" (single) · DXF layer `A-WALL` (really `A-WALL-FULL`) · "largest central room /
adjacent-to-bedroom bathroom" (no such logic) · line counts (app.py is 943, frontend.py 1082) · README project tree and "Flask inside frontend.py" are outdated ·
GEMINI "Phase 1 COMPLETED" predates the multi-storey/amenity WIP.

## 13. How to add a feature (checklists) and house rules
**General:** keep changes small and per-feature; update this file's §5-§10 if you change contracts; keep new `LayoutEngine` args
**optional with defaults** (callers use kwargs; `**kwargs` swallows unknowns silently — typos won't error); metres internally, convert at the edges;
respect Windows paths (`os.path.join`), CRLF repo.
- **New indoor room:** `RoomCategory` + `NBC_ROOM_SPECS` (room_specs) → add a `RoomGroup` in `space_program.build_program`
  (kind single/suite/service/servant/core; allowed bands) → `_VASTU_COMPATIBILITY` row (+ `ROOM_IMPORTANCE`, `_STRICT_TABOOS`)
  → door width in `layout_solver.DOOR_WIDTH` / preference in `LayoutEngine._place_doors.circ_pref` → window spec in
  `_place_windows` → label abbreviation in `_common.LABEL_ABBREVIATIONS` → tests. Placement itself is automatic (solver).
- **New site amenity:** category + Vastu row → reservation in `SitePlanner.plan_footprint` if it needs a deep strip →
  `put(...)` call in `place_amenities` (Vastu-ranked) → add to `SITE_CATEGORIES` (layout_engine) and the amenity sets in
  `svg_renderer.py`/`floorplan_editor.py` → `CostEngine.SITE_RATES` → tests.
- **Result tabs (2026-10-12):** `st.tabs(..., key="report_tab", on_change="rerun")` — stateful, only the open tab
  (`tN.open`) is computed. Everything is measured from the plan: `house_report.vastu_ledger` must keep reproducing
  `plan.vastu_score` (same terms as `layout_engine.audit_vastu`: w·(s+50)/60 per placement + door pada, −8 per taboo /
  Brahmasthan violation, −4 N+E<S+W) — change both together. NBC checks use INSIDE sizes (centre-line minus half
  walls, as `_set_clear_areas`): habitable ≥ 9.5 m² (largest) / 7.5 m², width 2.4; kitchen 5.0/1.8; bath+WC 2.8/1.2;
  passage width 1.0. `CostEngine.boq_for_plan(plan, est, quality)` prices `takeoff.material_takeoff` (India only;
  other markets → available False) and shares `est['construction_cost']` over the stages by materials × (1 + STAGE_LABOUR)
  — the total stays CostEngine's (no second cost formula). 'Better Vastu' / 'Best design' = `intent_studio.build_options
  (plan, [], cost_params)` (IntentSolver alternatives with no wishes), scored by `health_card`, cached in
  `st.session_state['report_alternatives']` by id(plan). Modules imported with `importlib.import_module` so edits reload.
- **Site feasibility rule (2026-10-12):** an amenity that costs the house > 4 % (`LOSS`) is 'no' when the house left
  falls below `COMFORT` 0.80 × comfortable footprint (Standard-quality program target × 1.12) or NBC min (a bedroom
  would be dropped); 'tight' (offered with a warning) below `TIGHT` 0.92. Amenities are judged in planner priority
  `AMENITIES` order, each on top of the ones before it that are on. New amenity → add it to `AMENITIES`, `CATEGORY`,
  `LABEL`, `_cfg`, the 'why' dict in `_Checker.judge`, and an `_amenity_toggle` in design_ui. Rule of thumb it yields
  (N, 1:1.33, 2+2 family): car with a 2BHK from ~3.25-3.5 cents, with a 3BHK from ~5 cents; pool from ~5.5-6 cents.
- **New UI input:** add widget in `design_ui.py`, pass through `LayoutEngine(**kwargs)`, store on `GeneratedFloorPlan`, show in SVG title block
  (watch the title-block row count: SVG 6, PDF 6, PNG 5, editor 5), and mirror in Flask `/generate-plan` if the API should support it.
- **New words for the wish box:** add phrases to `wish_parser._P` under the right tag (R: room, A: action,
  D:/S:/P: direction, M: modifier, U: unsupported); multi-word phrases match longest-first, unknown words are
  typo-repaired against the vocabulary (keep real building words in `_STOP` so they are never "corrected").
  Add the sentence to `test_10_everyday_language`.
- **New wish (intent) type:** add it to `INTENT_TYPES`, `normalise_intent`, `describe_intent`, `validate_intent`,
  a penalty + `satisfied` branch in `IntentSolver._intent_eval` (keep unmet ≥ 0.5·W so search is pulled in), a seed
  in `_intent_seeds` if a direct state edit exists, a frame rule in `wish_parser.understand_clause`, a UI action in
  `intent_studio.ACTIONS`, and a test. Rooms named by a wish go in `self.involved` (exempt from disruption,
  anchor increase refunded) — except lock/size (they must stay put) and flip (only the attached room).
  Plain `generate()` must stay byte-identical: intents only act through `_make_solver` / `apply_size_intents`.
- **Cost change:** edit only `CostEngine`. New country/city = add an entry to `MARKETS` (rates, regions, materials ×5 for the
  UI's metric row, site, soft, breakdown summing to 100) — the dropdowns, matrix and test_12 pick it up automatically; add its
  band to test_12. Keep quality keys `QUALITY_KEYS` (UI labels are normalised by `normalize_quality`).
- **Renderer geometry change:** change SVG first; mirror `PPM`/`pad` into `floorplan_editor.py` (`PPM_EDITOR`, `PAD_M`); check the y-axis convention (§10 #7).
- **Detection change:** edit `floor_plan_analyzer.py` (live), not `pipeline.py`/`smart_detection.py`; add tests (none exist).
- **Before finishing a change:** `python -c "import ast,sys;ast.parse(open('<file>').read())"` or run the relevant test file; run the app
  path you touched (`streamlit run model_folder/frontend.py` → Design tab → Generate) because most logic has no automated UI test.
- **Don't:** add a third cost formula; call `subprocess` for new work (import directly); add `isinstance` checks across the reload boundary;
  commit generated artifacts (`static/`, `temp/`, `test_output/`, `.pyc`, `*.pkl`, `*.ply`); touch `GEMINI_MEMORY.md` (kept for the other platform).
