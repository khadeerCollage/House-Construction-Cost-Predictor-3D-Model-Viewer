# AGENTS.md — Project Memory (platform-neutral)

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
floorplan_renderer/           LIVE  plan -> files. SVG(drawsvg) DXF(ezdxf) PDF(reportlab) PNG(matplotlib)
model_folder/
  frontend.py     Streamlit UI (nav radio: Design vs Analyze)                      (1082)
  design_ui.py    `render_design_tab()` — the whole Design tab, direct imports     (675, WIP)
  floorplan_editor.py interactive editor (Fabric.js via streamlit-drawable-canvas) (2346, WIP)
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

### 4a. Generative studio (the active area)
```
design_ui.render_design_tab()              (direct imports; NOT via Flask)
  LandUnitsEngine.solve_parcel(cents, shape, aspect_ratio|frontage_ft) -> plot W,H (m)   [or manual ft/m input]
  FamilyAnalyzer().analyze(FamilyProfile, plot_area_m2) -> recommended_bhk
  importlib.reload(floorplan_generator.layout_engine)   # on EVERY Generate click
  LayoutEngine(plot_w, plot_h, bhk, vastu_enabled, orientation, quality_level, project_name, family_description,
               has_lift, lift_capacity, stair_type, num_floors, building_type, additional_components, **kwargs).generate()
     -> GeneratedFloorPlan
  SVGRenderer / DXFRenderer / PDFRenderer / PNGRenderer .render(plan, path)  -> bytes in st.session_state
  CostEngine.estimate_cost(...), VastuAnalyzer(plan) -> audit tabs; FloorPlanEditor.render_editor_section(plan,...)
```
Flask `/generate-plan` does the same minus the new multi-storey/amenity args (it does **not** pass `stair_type`,
`has_lift`, `num_floors` to the engine, nor `additional_components`). [V]

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
Dimension-label strings are millimetres. Grid snap `GRID_MODULE = 0.15`. Interior = plot inset 0.23 m (external wall).

`layout_engine.py` plain `@dataclass`es (use `field(default_factory=...)` for lists):
| Class | Fields |
|---|---|
| `RoomRect` | name, category(str = `RoomCategory.value`), x, y (bottom-left), w, h, color(hex), area; props cx, cy, right, top |
| `WallSegment` | x1,y1,x2,y2, thickness (0.23 ext / 0.115 int), is_external — **centrelines** |
| `DoorPlacement` | x,y (low-coord corner of opening in generator; renderers treat as hinge), width, wall_side N/S/E/W, swing_direction, room_name, is_main_entrance |
| `WindowPlacement` | x,y (centre), width, height, wall_side, room_name |
| `StaircaseRect` | name, stair_type, x,y,w,h, direction="N", num_steps=15, tread .25, riser .175, has_landing, landing_depth 1.0 |
| `DimLine` | x1,y1,x2,y2, offset, label(mm str), tier (generator emits 1 and 3 only) |
| `GeneratedFloorPlan` | plot_width, plot_height, orientation, rooms, walls, doors, windows, staircases, dimensions, vastu_score, vastu_enabled, total_carpet_area, total_built_up_area, bhk_config, family_description, quality_level, project_name, has_lift, lift_capacity, building_type, num_floors, additional_components |

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
- Amenity/stair/lift **symbols exist only in SVG and the editor**; PDF/PNG draw them as plain room fills; DXF only a text label.
- **Editor**: `FloorPlanEditor` (static methods) — `plan_to_fabric_json(plan, w, h)` builds Fabric.js JSON (v"4.4.0"),
  `sync_canvas_drag_edits` maps drags back to the plan, `render_editor_section(plan, cost_params)` is the UI
  (tabs: Move, Add, Delete, Resize). Constants `PPM_EDITOR=50`, `PAD_M=3.0` must mirror the SVG renderer. Identity is
  smuggled via `strokeDashArray=[tag,0]` (rooms 1000+i, doors 2000+i, windows 3000+i, amenities 4000+i). "Apply Edits"
  deep-copies the plan, rebuilds walls/dims (calls private `LayoutEngine._generate_walls/_generate_dimensions`) and re-renders
  all four formats. No undo, no disk persistence.
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

**CostEngine (`cost_engine.py`) — canonical cost.** `CostEngine.estimate_cost(area_sqm, bedrooms=1, bathrooms=1, kitchens=1,
living_rooms=1, city_tier="Tier-2 (Urban)", quality_level="Standard (Medium)", num_floors=1)` → `total_cost, area_sqft,
cost_per_sqft, cost_breakdown, materials{Cement,Steel,Bricks,Sand,Aggregate:{quantity,unit,rate,cost}}, comparison, description`.
`cost = area_sqft(=m²×10.7639)×floors × rate × city_mult + kitchens×30000×mult + bathrooms×20000×mult`.
Rates ₹/sqft: Basic 1000 · Standard 1550 · Premium 2500. City: Tier-1 1.35 · Tier-2 1.00 · Tier-3 0.80.
**Exact key strings matter:** `"Tier-1 (Metro)" "Tier-2 (Urban)" "Tier-3 (Rural)"`, `"Basic (Low Cost)" "Standard (Medium)"
"Premium (High Quality)"` — anything else silently falls back to Standard / ×1.0 (§10 #4; the Design UI currently hits this on every call).
Material factors/sqft: cement .42 bag @₹410 · steel 2.8 kg @₹68 · bricks 20 @₹8.5 · sand 1.6 cft @₹65 · aggregate 1.35 cft @₹70.
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
- **New room/amenity category:** `RoomCategory` member + `NBC_ROOM_SPECS` entry (room_specs) → `_VASTU_COMPATIBILITY` row if scored
  (missing = 0) → explicit handling in `_place_rooms_vastu`, `_place_doors`, `_place_windows` (no generic placement) → add to `amenity_cats`
  in `generate()` if it should be excluded from carpet area → the 4× duplicated amenity set in `svg_renderer.py`/`floorplan_editor.py`
  → editor `plan_to_fabric_json` branch (+ tag range) → PDF/PNG/DXF if wanted → extend tests.
- **New UI input:** add widget in `design_ui.py`, pass through `LayoutEngine(**kwargs)`, store on `GeneratedFloorPlan`, show in SVG title block
  (watch the title-block row count: SVG 6, PDF 6, PNG 5, editor 5), and mirror in Flask `/generate-plan` if the API should support it.
- **Cost change:** edit only `CostEngine`; if you add a tier/quality, use its exact key strings and fix #4 by mapping UI labels → engine keys in one place.
- **Renderer geometry change:** change SVG first; mirror `PPM`/`pad` into `floorplan_editor.py` (`PPM_EDITOR`, `PAD_M`); check the y-axis convention (§10 #7).
- **Detection change:** edit `floor_plan_analyzer.py` (live), not `pipeline.py`/`smart_detection.py`; add tests (none exist).
- **Before finishing a change:** `python -c "import ast,sys;ast.parse(open('<file>').read())"` or run the relevant test file; run the app
  path you touched (`streamlit run model_folder/frontend.py` → Design tab → Generate) because most logic has no automated UI test.
- **Don't:** add a third cost formula; call `subprocess` for new work (import directly); add `isinstance` checks across the reload boundary;
  commit generated artifacts (`static/`, `temp/`, `test_output/`, `.pyc`, `*.pkl`, `*.ply`); touch `GEMINI_MEMORY.md` (kept for the other platform).
