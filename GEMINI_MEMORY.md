# 🧠 GEMINI MEMORY: HOUSE CONSTRUCTION COST PREDICTOR & 3D MODEL VIEWER

> **Last Deep Audit & Analysis Date:** September 2026  
> **Project Root:** `c:\Users\USER\Documents\vit_project`  
> **Active Target Environment:** Python 3.10 (Windows x64 / Linux Docker)  
> **Corpus Repository:** `khadeerCollage/House-Construction-Cost-Predictor-3D-Model-Viewer`  
> **Purpose:** Permanent comprehensive memory, architecture specification, technical debt ledger, and modernization masterplan for autonomous upgrades.

---

## 📌 TABLE OF CONTENTS
1. [Executive Summary & System Identity](#1-executive-summary--system-identity)
2. [Complete System Architecture & Workflow](#2-complete-system-architecture--workflow)
3. [Deep Component-by-Component Analysis](#3-deep-component-by-component-analysis)
   - [3.1 Computer Vision & Floor Plan Detection Engine (`floorplan_detector/`)](#31-computer-vision--floor-plan-detection-engine)
   - [3.2 Application Layer & Services (`model_folder/`)](#32-application-layer--services)
   - [3.3 Cost Prediction & ML Modeling](#33-cost-prediction--ml-modeling)
   - [3.4 3D Synthesis & WebGL Rendering](#34-3d-synthesis--webgl-rendering)
   - [3.5 Deployment, Containers & Scripts](#35-deployment-containers--scripts)
4. [Identified Technical Debt, Bugs & Bottlenecks](#4-identified-technical-debt-bugs--bottlenecks)
5. [Complete Dependency Matrix & Environment Constraints](#5-complete-dependency-matrix--environment-constraints)
6. [Master Upgrade Roadmap (V2 Next-Gen Architecture)](#6-master-upgrade-roadmap-v2-next-gen-architecture)
7. [Autonomous Execution Guidelines & Protocols for AI](#7-autonomous-execution-guidelines--protocols-for-ai)

---

## 1. EXECUTIVE SUMMARY & SYSTEM IDENTITY

The **House Construction Cost Predictor & 3D Model Viewer** (`vit_project`) is an end-to-end computer vision and architectural engineering software suite.

### Primary Capabilities:
1. **2D Blueprint Ingestion & Preprocessing:** Cleans noisy raster drawings, scanned blueprints, and hand-drawn floor plans via morphological filtering, text inpainting, and CLAHE contrast equalization.
2. **Structural & Spatial Parsing:** Detects structural walls using Probabilistic Hough Transforms, identifies door openings via gap and arc analysis, extracts closed polygon room contours, and semantically classifies room categories (Living Room, Bedroom, Kitchen, Bathroom, Hallway, Storage, etc.).
3. **Zero-Shot & Deep Learning Enrichment:** Incorporates Google OWL-ViT (`google/owlvit-base-patch32`) for zero-shot architectural fixture detection (beds, toilets, sinks, stoves, sofas) and YOLOv8 hooks for symbol identification.
4. **2D-to-3D Extrusion & Mesh Synthesis:** Extrudes 2D coordinate vectors into 3D polygonal wall meshes and floor planes with Open3D, generating vertex-colored 3D models (`.ply` format).
5. **Interactive Three.js WebGL Viewer:** Embeds a client-side Three.js canvas in the Streamlit UI with OrbitControls, rendering base64-encoded 3D geometry without local CORS restrictions.
7. **Generative Architectural Design Studio (Phase 1 — COMPLETED September 2026):**
   - **V-HSP Layout Engine (`floorplan_generator/`):** Vastu-Guided Hierarchical Space Partitioning algorithm converting plot boundaries ($W \times L$) and family intent into NBC 2016-compliant floor plans in $< 10\text{ms}$ on CPU.
   - **Indian Residential Standards Database (`room_specs.py`):** NBC 2016 minimum dimensions, 1BHK–4BHK configurations, standard wall thicknesses (230mm exterior, 115mm interior), and standard doors/windows.
   - **Vastu Purusha Mandala Engine (`vastu_engine.py`):** 3×3 quadrant compass alignment (Master Bed in SW, Kitchen in SE, Pooja in NE, Sanitation in NW/W), taboo detection, and 0-100 scoring.
   - **Family Intent Analyzer (`family_analyzer.py`):** Translates household details (adults, children, elderly, lifestyle rooms) into optimal BHK configurations and reasoning.
   - **CAD & Vector Rendering Pipeline (`floorplan_renderer/`):** 4 export formats:
     - 📐 **AutoCAD DXF (`dxf_renderer.py`):** AIA-standard CAD layers (`A-WALL`, `A-DOOR`, `A-GLAZ`, `A-ANNO-DIMS`), ANSI31 brick hatching, 45° architectural ticks, and ISO 7200 title block.
     - 🎨 **Scalable Vector SVG (`svg_renderer.py`):** Infinite-zoom vector graphics with Blueprint Mode and Classic CAD styling.
     - 📄 **Printable PDF Blueprint (`pdf_renderer.py`):** ISO A3 landscape drawing sheets with title blocks.
     - 🖼️ **High-Res PNG & Wall Mask (`png_renderer.py`):** 300 DPI blueprint image and binary wall mask for direct 3D mesh extrusion.
   - **Full-Stack Integration:** Added `/generate-plan` and `/download-plan` endpoints to Flask (`model_folder/app.py`), and interactive Generative Studio tab in Streamlit (`model_folder/design_ui.py` & `frontend.py`).

---

## 2. COMPLETE SYSTEM ARCHITECTURE & WORKFLOW

```
                                  [ User Input: Floorplan Image (JPG/PNG) ]
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │  Streamlit Frontend UI       │
                                      │  (model_folder/frontend.py)  │
                                      └──────────────┬───────────────┘
                                                     │ HTTP POST
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │  Flask Backend API Server    │
                                      │  (model_folder/app.py:5000)  │
                                      └──────────────┬───────────────┘
                                                     │
                         ┌───────────────────────────┴───────────────────────────┐
                         ▼                                                       ▼
            /detect-walls-rooms                                             /generate-3d
                         │                                                       │
                         ▼                                                       ▼
          ┌─────────────────────────────┐                         ┌─────────────────────────────┐
          │  FloorPlanAnalyzer Engine   │                         │  model_folder/2d-3d.py      │
          │  (floor_plan_analyzer.py)   │                         │  (Subprocess execution)     │
          └──────────────┬──────────────┘                         └──────────────┬──────────────┘
                         │                                                       │
         ┌───────────────┼────────────────┐                                      │
         ▼               ▼                ▼                                      ▼
  [Wall Detection] [Door Detection] [ViT Symbol Detect]                   [Open3D Extrusion]
  - HoughLinesP    - Wall gap search - OWL-ViT zero-shot                  - Wall boxes
  - Merge collin.  - Door arc check  - Sinks, stoves, beds                - Floor planes
         │               │                │                               - Color attribution
         └───────────────┬────────────────┘                                      │
                         ▼                                                       ▼
               [Room Segmentation]                                   [floorplan_3d_model.ply]
               - Connected components                                            │
               - Contour hierarchy                                               ▼
                         │                                        [Three.js OrbitControls WebGL]
                         ▼                                                       │
               [Scale Calibration]                                               ▼
               - PPM calculation                                     [Interactive 3D Preview]
               - Area in m² / sqft
                         │
                         ▼
        ┌────────────────────────────────────────────────────────┐
        │ /estimate-construction-cost (CostEngine)               │
        │ - Base built-up rate (₹1,000 - ₹2,500/sqft)            │
        │ - City multiplier (Tier 1: 1.35x, Tier 2: 1.0x, Tier 3)│
        │ - Wet area surcharges (Kitchens & Bathrooms)           │
        │ - Bill of Quantities (Cement, Steel, Bricks, Sand)     │
        └────────────────────────────────────────────────────────┘
```

---

## 3. DEEP COMPONENT-BY-COMPONENT ANALYSIS

### 3.1 Computer Vision & Floor Plan Detection Engine (`floorplan_detector/`)

| File | Lines | Primary Responsibility | Key Classes / Functions |
|---|---|---|---|
| `floor_plan_analyzer.py` | 1162 | **Main Detection Pipeline**: Combines wall detection, door gap analysis, room contour extraction, and semantic classification. | `FloorPlanAnalyzer`, `WallDoorDetector`, `RoomAnalyzer`, `RoomLabel`, `Wall`, `Door`, `Room` |
| `wall_detection.py` | 455 | Line segment detection, angle categorization (horizontal, vertical, diagonal), and segment merging. | `WallDetector`, `WallSegment` |
| `room_detection.py` | 399 | Contour-based room boundary detection and morphological closure. | `RoomDetector`, `Room`, `RoomType` |
| `smart_detection.py` | 738 | Heuristic spatial parser modeling human spatial intuition (solidity, aspect ratios, corridor vs room). | `SmartFloorPlanDetector`, `SmartWall`, `DetectedSpace`, `SpaceType` |
| `preprocessing.py` | 176 | Image normalization, text annotation removal via MSER + inpainting, CLAHE contrast enhancement. | `FloorPlanPreprocessor` |
| `yolo_room_labeler.py` | 412 | Fixture and furniture symbol detection utilizing YOLOv8. | `YOLORoomLabeler`, `FloorPlanSymbolDetector`, `DetectedObject` |
| `model_3d.py` | 433 | Native Open3D 3D mesh synthesis library (supports PLY, OBJ, GLTF). | `FloorPlan3DGenerator`, `ModelConfig` |
| `pipeline.py` | 331 | Secondary pipeline wrapper bundling preprocessing, walls, rooms, and 3D. | `FloorPlanAnalyzer` (duplicate signature), `AnalysisResult` |
| `__init__.py` | 40 | Package export index. | Exports classes and conditional `HAS_3D` flag. |

#### Detailed Detection Mechanism:
1. **Wall Detection:** Uses `cv2.Canny` (50, 150) followed by Probabilistic Hough Transform (`cv2.HoughLinesP`). Endpoints are flattened (`np.array(line).flatten()`) to prevent NumPy shape mismatches. Collinear lines within 5° and 15px proximity are fused.
2. **Door Gap Detection:** Iterates through wall segments searching for systematic breaks (10px to 60px) in collinear wall chains, verifying door swings or passage openings.
3. **Room Segmentation:** Inverts the wall mask, computes morphological closing, extracts external contours, filters noise regions below 1500 px², and fits minimum bounding boxes.
4. **Room Classification:** Combines:
   - ViT/YOLO detections (e.g., bed detected inside bounding box $\rightarrow$ Bedroom; toilet/bathtub $\rightarrow$ Bathroom; stove $\rightarrow$ Kitchen).
   - Geometric heuristics (aspect ratio $< 0.35 \rightarrow$ Hallway; small square adjacent to bedroom $\rightarrow$ Bathroom; largest central room $\rightarrow$ Living/Dining).

---

### 3.2 Application Layer & Services (`model_folder/`)

| File | Lines | Primary Responsibility |
|---|---|---|
| `app.py` | 737 | **Flask Backend REST API**: Serves endpoints on port 5000 (`/predict`, `/generate-3d`, `/detect-walls-rooms`, `/estimate-construction-cost`, `/update-location-type`, `/download-ply`). |
| `frontend.py` | 1044 | **Streamlit UI Client**: Multi-tab interface, backend health-check & auto-start, scale calibration inputs, room breakdown cards, Three.js 3D viewer, Matplotlib cost charts. |
| `2d-3d.py` | 435 | **3D Conversion CLI**: Called as a subprocess by `app.py` to generate `.ply` models from input images. |
| `cost_engine.py` | 201 | **Civil Cost Engine**: Calculates Indian market construction costs, material breakdowns (cement, steel, bricks, sand, aggregate), and quality tier comparisons. |
| `scale_calibration.py` | 122 | **Scale Calibrator**: Maps image pixels to real-world meters (`PPM`), with fallback room-count heuristic. |
| `vit_detector.py` | 132 | **OWL-ViT Wrapper**: Zero-shot Vision Transformer using `google/owlvit-base-patch32` on CPU/CUDA. |
| `train_cost_model.py` | 114 | **ML Model Trainer**: Generates 10k dataset and trains RandomForestRegressor with R² evaluation. |
| `utils.py` | 309 | **Bridge Module**: Glues `FloorPlanAnalyzer` to Flask with Matplotlib visualization generator. |

---

### 3.3 Cost Prediction & ML Modeling

#### Model 1: Civil Engineering Formula (`CostEngine`)
- **Built-Up Area:** Calculated from calibrated image dimensions:
  $$\text{Area}_{\text{sqft}} = \text{Area}_{\text{m}^2} \times 10.7639 \times \text{Floors}$$
- **Quality Rates per sq.ft:**
  - *Basic (Low Cost):* ₹1,000/sqft
  - *Standard (Medium):* ₹1,550/sqft
  - *Premium (High Quality):* ₹2,500/sqft
- **City Tiers:**
  - *Tier-1 (Metro - Mumbai, Delhi, Bangalore):* $1.35\times$
  - *Tier-2 (Urban - Jaipur, Pune, Lucknow):* $1.00\times$
  - *Tier-3 (Rural / Towns):* $0.80\times$
- **Material Multipliers:**
  - Cement: 0.42 bags/sqft @ ₹410/bag
  - Steel: 2.8 kg/sqft @ ₹68/kg
  - Bricks: 20 pcs/sqft @ ₹8.5/piece
  - Sand: 1.6 CFT/sqft @ ₹65/CFT
  - Aggregate: 1.35 CFT/sqft @ ₹70/CFT

#### Model 2: Machine Learning Random Forest (`house_cost_model_v2.pkl`)
- Trained with `RandomForestRegressor(n_estimators=100, max_depth=12)` on 8 features:
  `["Area_m2", "Rooms", "Bathrooms", "Kitchens", "Living_Rooms", "City_Tier", "Quality_Level", "Num_Floors"]`.
- Output: Total Estimated Cost in INR.
- Accuracy: $R^2 \approx 0.99$, $\text{MAPE} < 3\%$.

---

### 3.4 3D Synthesis & WebGL Rendering

- **Geometry Generation:** `model_folder/2d-3d.py` creates:
  - Extruded wall boxes: 8 vertices per wall segment, 12 triangles, computed vertex normals, painted `#F2EBE0`.
  - Floor plane meshes: Room-bounded horizontal quads colored by room category (Living = `#8DD3C7`, Bed = `#FFFFB3`, Bath = `#BEBADA`, Kitchen = `#FB8072`).
- **Client-Side Rendering:**
  - `model_folder/frontend.py` encodes the generated `.ply` into a Base64 string.
  - Injects Three.js (r128), `OrbitControls.js`, and `PLYLoader.js` inside an `iframe`.
  - Parses binary array buffers in-browser, centers the geometry bounding box, adds directional and ambient lighting, and renders at 60 FPS.

---

### 3.5 Deployment, Containers & Scripts

- `Dockerfile`: Multi-stage Python 3.10 image with system GL libraries (`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`). Installs CPU PyTorch and runs both Flask and Streamlit simultaneously.
- `docker-compose.yml`: Binds ports `8501:8501` (Streamlit) and `5000:5000` (Flask).
- `start.bat`: Automates local Windows setup, virtual environment creation (`.venv310`), CPU torch installation, and process launching.
- `app_streamlit.py`: Standalone single-file Streamlit fallback that operates entirely in memory without requiring Flask.

---

## 4. IDENTIFIED TECHNICAL DEBT, BUGS & BOTTLENECKS

### ⚠️ Critical Performance & Execution Bottlenecks
1. **Redundant Subprocess Execution in 3D Generation:**
   - In `model_folder/app.py`, endpoint `/generate-3d` calls `subprocess.run([sys.executable, '2d-3d.py', ...])`.
   - Spawning a new Python process forces Python to re-import heavy libraries (PyTorch, Transformers, Open3D, OpenCV) on every click, adding 4–8 seconds of unnecessary cold-start overhead.
2. **Double Detection Execution:**
   - Inside `app.py`'s `generate_3d()` endpoint, it runs `2d-3d.py` (which runs `FloorPlanAnalyzer`), and immediately calls `detect_rooms_and_walls()` again on line 152! The entire floor plan is analyzed twice per click.
3. **Synchronous Heavy ViT on CPU:**
   - `VitDetector` loads `google/owlvit-base-patch32` (~600MB weights) and runs inference synchronously in `FloorPlanAnalyzer.analyze()`. On a standard CPU, this adds 3–6 seconds per detection. There is no option to bypass or run asynchronously.

### ⚠️ Architectural Fragmentation & Code Duplication
4. **Duplicate Classes with Conflicting Signatures:**
   - `floorplan_detector/floor_plan_analyzer.py` defines `FloorPlanAnalyzer` (uses `WallDoorDetector` + `RoomAnalyzer`).
   - `floorplan_detector/pipeline.py` ALSO defines `FloorPlanAnalyzer` (uses `WallDetector` + `RoomDetector`).
   - `floorplan_detector/smart_detection.py` defines `SmartFloorPlanDetector`.
   - Three different implementations of wall and room detection coexist, causing maintenance ambiguity.
5. **Divergent Cost Estimation Engines:**
   - `cost_engine.py`: Professional Indian market model in INR (₹) with material BOQ.
   - `app.py` (`/update-location-type`): Hardcoded USD calculation ($1,200/sqm base, $10,000/bedroom).
   - `app_streamlit.py`: Hardcoded USD calculation ($1,500/sqm base).
   - `frontend.py`: Displays INR for 3D results, but has fallback local functions in USD.

### ⚠️ UI & Viewer Limitations
6. **3D Mesh Format is Legacy PLY:**
   - PLY is a vertex-based polygon format without material texture support, PBR shading, or hierarchy.
   - Standard modern 3D web applications use **GLTF / GLB**, which is 60–80% smaller, supports PBR textures (wood floors, brick walls), and loads natively in Three.js with standard Draco compression.
7. **No Interactive 2D-to-3D Room Synchronization:**
   - Clicking a room in the detection table does not highlight or isolate that room in the 3D viewer.

### ⚠️ Deployment & Environment Vulnerabilities
8. **Docker Process Management:**
   - `Dockerfile` runs `CMD python model_folder/app.py & streamlit run ...`. If Flask crashes, Docker does not detect the failure because Streamlit remains active. Needs a supervisor or a unified FastAPI backend serving the frontend static build / Streamlit.
9. **Hardcoded Ports and URLs:**
   - Frontend explicitly makes HTTP calls to `http://127.0.0.1:5000`. If deployed on HuggingFace Spaces or Cloud Run with dynamic ports or subdomains, hardcoded localhost requests fail.

---

## 5. COMPLETE DEPENDENCY MATRIX & ENVIRONMENT CONSTRAINTS

| Category | Package | Minimum Version | Verified Role |
|---|---|---|---|
| **Language** | Python | 3.10.x | **Mandatory**: Open3D wheels and PyTorch CPU are most stable on 3.10. |
| **Vision & Math** | `opencv-python-headless` | $\ge 4.8.0$ | Image processing, Canny, Hough, contouring |
| | `numpy` | $\ge 1.24.0, < 2.0$ | Tensor manipulation and geometry transforms |
| | `scipy` | $\ge 1.10.0$ | Spatial distance computation and KD-trees |
| | `Pillow` | $\ge 10.0.0$ | Image I/O and PIL-to-NumPy conversion |
| | `matplotlib` | $\ge 3.7.0$ | Visualization generation (Agg backend) |
| **Deep Learning** | `torch`, `torchvision` | $\ge 2.0.0$ | Neural network inference engine (CPU/CUDA) |
| | `transformers` | $\ge 4.30.0$ | OWL-ViT zero-shot object detection |
| | `ultralytics` | $\ge 8.0.0$ | YOLOv8 fixture & symbol detection |
| **3D Graphics** | `open3d` | $\ge 0.17.0$ | 3D mesh generation, triangle creation, PLY/GLTF |
| | `plotly` | $\ge 5.0.0$ | Interactive plotting and analytics |
| **Web & API** | `streamlit` | $\ge 1.30.0$ | Primary interactive web dashboard |
| | `flask`, `flask-cors` | $\ge 2.3.0$ | REST backend API server |
| **Data & ML** | `pandas` | $\ge 2.0.0$ | Feature preparation and dataframes |
| | `scikit-learn` | $\ge 1.3.0$ | Random Forest Regressor for house cost |
| | `xgboost` | $\ge 2.0.0$ | Gradient boosting models |
| | `joblib` | $\ge 1.3.0$ | Model persistence serialization (`.pkl`) |

---

## 6. MASTER UPGRADE ROADMAP (V2 NEXT-GEN ARCHITECTURE)

The target upgrade will transform the prototype into a commercial-grade, high-performance architectural AI suite:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      V2 UPGRADE MILESTONE ROADMAP                           │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ Milestone 1       │ Unified Architecture & Engine Consolidation             │
│ (Core Refactor)   │ - Unify FloorPlanAnalyzer into a single clean engine    │
│                   │ - Eliminate redundant subprocess calls (direct import)  │
│                   │ - Remove duplicate image processing passes              │
│                   │ - Establish universal configuration (INR/USD toggle)    │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone 2       │ Next-Gen 3D Synthesis (GLTF/GLB with Textures)          │
│ (3D Revolution)   │ - Replace PLY generation with standard GLTF/GLB exports │
│                   │ - Procedural PBR materials: hardwood floors, plastered  │
│                   │   walls, glazed bathroom tiles                          │
│                   │ - Cutout doors and windows in 3D wall geometry          │
│                   │ - First-person camera walkthrough mode in Three.js      │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone 3       │ Precision AI Detection & CAD Symbol Pipeline            │
│ (AI Enhancements) │ - Toggleable ViT mode (Speed vs Accuracy modes)         │
│                   │ - Fine-tuned architectural fixture detector             │
│                   │ - Scale calibration with visual ruler / auto-door width │
│                   │ - Wall junction alignment (T-junctions & corners snap)  │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone 4       │ Enterprise Cost Estimation & BOQ Export                 │
│ (Cost Engine 2.0) │ - Detailed Bill of Quantities (BOQ) with live market    │
│                   │   material price updates                                │
│                   │ - Exportable Construction Estimate PDF / Excel sheets   │
│                   │ - Multi-currency support (INR ₹, USD $, EUR €)          │
│                   │ - Contractor & Labor schedule timeline estimator        │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Milestone 5       │ Modern High-Performance UI / Full-Stack Integration     │
│ (UI & Cloud)      │ - Single command launch (unified backend/frontend)      │
│                   │ - Responsive modern design with dark/light themes       │
│                   │ - Real-time 3D lighting, shadows, and camera presets    │
│                   │ - Docker container with production gunicorn + supervisor│
└───────────────────┴─────────────────────────────────────────────────────────┘
```

---

## 7. AUTONOMOUS EXECUTION GUIDELINES & PROTOCOLS FOR AI

When working on this codebase in future sessions, follow these operational laws:

1. **Python 3.10 Target Compliance:** Always ensure compatibility with Python 3.10 and avoid syntax/dependencies restricted to newer or older versions.
2. **Preserve Documentation & Comments:** Retain architectural context, docstrings, and comments in files modified.
3. **No Redundant Subprocesses:** Call conversion and detection logic via direct Python module imports rather than `subprocess.run()`.
4. **Single Source of Truth for Cost:** Always rely on `CostEngine` (`model_folder/cost_engine.py`) as the canonical pricing engine; do not introduce conflicting hardcoded cost formulas.
5. **Backward Compatibility:** Maintain API route compatibility (`/detect-walls-rooms`, `/generate-3d`, `/estimate-construction-cost`) so external clients, Docker, and frontend remain operational.
6. **Self-Contained Verification:** Validate every code edit by verifying Python syntax, running import tests, and checking functionality.

---
*End of Gemini Memory Document — Persisted for continuous autonomous upgrades.*
