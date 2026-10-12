"""
Design Your Home UI Component
=============================
Interactive UI for the AI Floor Plan Generator (Phase 1).
Allows civil engineers, architects, and homeowners to design custom,
Vastu-compliant residential floor plans from plot parameters and family intent.
"""

import os
import sys
import time
import base64
import streamlit as st
import numpy as np

# Ensure project root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# A long-running Streamlit keeps old copies of these modules in sys.modules; refresh them
# (dependencies first) whenever this module is (re)loaded so new engine code is picked up.
import importlib as _importlib
for _name in ("floorplan_generator.room_specs", "floorplan_generator.vastu_engine",
              "floorplan_generator.family_analyzer", "floorplan_generator.space_program",
              "floorplan_generator.site_planner", "floorplan_generator.site_feasibility",
              "floorplan_generator.layout_solver", "floorplan_generator.quantities", "floorplan_generator.intent_engine",
              "floorplan_generator.wish_parser",
              "floorplan_generator.layout_engine", "floorplan_generator.vastu_analyzer",
              "floorplan_generator.takeoff", "floorplan_generator.house_report",
              "floorplan_renderer._common", "floorplan_renderer.svg_renderer",
              "floorplan_renderer.dxf_renderer", "floorplan_renderer.pdf_renderer",
              "floorplan_renderer.png_renderer", "model_folder.cost_engine",
              "model_folder.intent_studio", "model_folder.floorplan_editor",
              "model_folder.house_report_ui"):
    if _name in sys.modules:
        _importlib.reload(sys.modules[_name])

from floorplan_generator.family_analyzer import FamilyAnalyzer, FamilyProfile
from floorplan_generator.layout_engine import LayoutEngine, GeneratedFloorPlan
from floorplan_renderer.dxf_renderer import DXFRenderer
from floorplan_renderer.svg_renderer import SVGRenderer
from floorplan_renderer.pdf_renderer import PDFRenderer
from floorplan_renderer.png_renderer import PNGRenderer
from model_folder.cost_engine import CostEngine
from floorplan_generator.vastu_analyzer import VastuAnalyzer
from model_folder.floorplan_editor import FloorPlanEditor
from floorplan_generator.land_units import LandUnitsEngine, LandParcel


_SITE_REQUESTS = [  # (additional_components key, plan room category, label)
    ("car_parking", "PARKING", "🚗 Car parking"), ("bike_parking", "BIKE_PARKING", "🏍️ Two-wheeler parking"),
    ("garden", "GARDEN", "🌿 Garden"), ("swimming_pool", "POOL", "🏊 Swimming pool"),
    ("water_tank", "WATER_TANK", "💧 Water sump"), ("generator_room", "GENERATOR", "⚡ Generator"),
    ("security_cabin", "SECURITY", "🛡️ Security cabin"),
]


def _site_request_status(plan):
    """(not fitted, info) messages for amenities the owner switched on, with the planner's reason."""
    comps = getattr(plan, "additional_components", None) or {}
    placed = {r.category for r in plan.rooms}
    notes = getattr(plan, "design_notes", None) or []
    missing, info = [], []
    for key, cat, label in _SITE_REQUESTS:
        cfg = comps.get(key) or {}
        if not cfg.get("enabled") or cat in placed:
            continue
        word = {"car_parking": "Car parking", "bike_parking": "Two-wheeler", "garden": "Garden",
                "swimming_pool": "Swimming pool", "water_tank": "sump", "generator_room": "enerator",
                "security_cabin": "Security cabin"}[key]
        reason = next((n for n in notes if word in n), "")
        if key == "car_parking" and "stilt" in reason:
            info.append(f"{label}: {reason}")
        else:
            missing.append(f"{label} — {reason or 'no open space was left on this plot.'}")
    return missing, info


def _checker_version() -> float:
    """Part of the cache key, so an edited site_feasibility.py is not answered from the cache."""
    import floorplan_generator.site_feasibility as _sf
    return os.path.getmtime(_sf.__file__)


@st.cache_data(show_spinner=False, max_entries=256)
def _bhk_fit(plot_w, plot_h, facing, family_kw, building_type, floors, vastu, version=0.0):
    """Which BHK sizes fit the bare plot (cached: the form reruns on every click)."""
    from floorplan_generator.site_feasibility import bhk_options
    return bhk_options(plot_w, plot_h, facing, FamilyProfile(**family_kw), building_type, floors, vastu)


@st.cache_data(show_spinner=False, max_entries=256)
def _site_fit(plot_w, plot_h, facing, bhk, family_kw, building_type, floors, quality, vastu,
              has_lift, lift_capacity, stair_code, components, version=0.0):
    """Which site amenities fit with the choices made so far."""
    from floorplan_generator.site_feasibility import check_site
    return check_site(plot_w, plot_h, facing, bhk, FamilyProfile(**family_kw), building_type, floors,
                      quality, vastu, has_lift, lift_capacity, stair_code, components)


def _amenity_toggle(label, key, fit, help=None, show_ok=False) -> bool:
    """A site-amenity switch, locked with the reason when the plot cannot take it."""
    ss = st.session_state
    ss.setdefault(key, False)
    switched_off = fit.status == "no" and bool(ss[key])
    if switched_off:
        ss[key] = False             # it fitted before the owner changed the plot / BHK
    on = st.toggle(label, key=key, disabled=fit.status == "no", help=help)
    if fit.status == "no":
        st.caption(f"🔒 **Won't fit here** — {fit.reason}." + (f"  \n💡 {fit.unlock}" if fit.unlock else "")
                   + ("  \n↩️ Switched off after your last change." if switched_off else ""))
    elif fit.status == "tight":
        st.caption(f"⚠️ {fit.reason[0].upper()}{fit.reason[1:]}.")
    elif on and show_ok:
        st.caption(f"✅ {fit.reason[0].upper()}{fit.reason[1:]}.")
    return on


def _count_input(label, key, lo, top, step, default, unit) -> int:
    """Number of cars / bikes, capped at what fits; a plain note when only the minimum fits."""
    ss = st.session_state
    if top <= lo:
        st.caption(f"Room for {lo} {unit}{'s' if lo > 1 else ''} on this plot.")
        return lo
    ss.setdefault(key, max(lo, min(default, top)))
    if int(ss[key]) > top:
        ss[key] = top
    return int(st.number_input(label, min_value=lo, max_value=top, step=step, key=key,
                               help=f"Up to {top} {unit}s fit on this plot."))


def render_design_tab():
    """Renders the AI Floor Plan Generator tab in Streamlit."""
    st.markdown("""
    ## 🏗️ AI Architect — Generative Floor Plan Studio
    Design **civil-engineer grade**, **Vastu-compliant**, **NBC 2016 standard** floor plans tailored to your land dimensions and family requirements in seconds.
    """)

    # ---------------------------------------------------------
    # Input Form Layout (3 Columns)
    # ---------------------------------------------------------
    col_plot, col_family, col_eng = st.columns([1, 1, 1], gap="medium")

    with col_plot:
        st.markdown("### 📐 1. Land & Plot")
        
        input_mode = st.radio(
            "Input Mode",
            ["🌾 By Land in Cents (Indian Standard)", "📏 By Exact Dimensions (Feet / Meters)"],
            index=0,
            horizontal=False
        )

        if "Cents" in input_mode:
            cents_val = st.number_input(
                "Land Area in Cents",
                min_value=0.8,
                max_value=30.0,
                value=3.0,
                step=0.25,
                help="1 Cent = 435.6 sq ft = 40.47 m² = 48.4 Gaj (Sq Yards)"
            )
            
            shape_val = st.radio(
                "Plot Shape",
                ["Rectangle", "Square"],
                index=0,
                horizontal=True
            )

            if shape_val == "Square":
                parcel = LandUnitsEngine.solve_parcel(cents=cents_val, shape="Square")
                st.caption(f"📐 **Square Constraints:** `{parcel.width_ft:.1f}' × {parcel.length_ft:.1f}'` ({parcel.width_m:.2f}m × {parcel.length_m:.2f}m)")
            else:
                rect_mode = st.selectbox(
                    "Rectangle Proportion",
                    [
                        "Standard Town Plot (1:1.33 — 3:4 ratio)",
                        "Proportional (1:1.50 — 2:3 ratio)",
                        "Golden Vedic Ayatasra (1:1.62)",
                        "Deep Narrow Plot (1:2.00)",
                        "Custom Road Frontage (Feet)"
                    ],
                    index=0
                )
                if "Custom" in rect_mode:
                    frontage_ft = st.number_input(
                        "Road Frontage Width (Feet)",
                        min_value=15.0,
                        max_value=100.0,
                        value=30.0,
                        step=1.0,
                        help="Width of your plot facing the road"
                    )
                    parcel = LandUnitsEngine.solve_parcel(cents=cents_val, shape="Rectangle", frontage_ft=frontage_ft)
                else:
                    ratio_val = 1.33 if "1:1.33" in rect_mode else 1.5 if "1:1.50" in rect_mode else 1.62 if "1:1.62" in rect_mode else 2.0
                    parcel = LandUnitsEngine.solve_parcel(cents=cents_val, shape="Rectangle", aspect_ratio=ratio_val)

                st.caption(f"📐 **Auto-Constrained:** `{parcel.width_ft:.1f}' (W) × {parcel.length_ft:.1f}' (L)` ({parcel.width_m:.2f}m × {parcel.length_m:.2f}m)")

            plot_w = parcel.width_m
            plot_h = parcel.length_m
            plot_area_sqm = parcel.area_sqm
            plot_area_sqft = parcel.area_sqft
            plot_cents = parcel.cents

            # Multi-unit equivalent badge
            st.markdown(f"""
            <div style="background: #1E293B; border-radius: 6px; padding: 8px 12px; margin-top: 6px; margin-bottom: 8px; font-size: 12px; color: #CBD5E1; border-left: 3px solid #0284C7;">
                <b>🌾 {plot_cents:.2f} Cents</b> = <b>{plot_area_sqft:.0f} sq ft</b> ({plot_area_sqm:.1f} m²)<br>
                <span>🏛️ {parcel.area_gaj:.0f} Gaj | {parcel.area_guntha:.2f} Guntha | {parcel.area_ground:.2f} Ground</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            unit_type = st.radio("Dimension Unit", ["Feet", "Meters"], horizontal=True)
            if unit_type == "Feet":
                w_ft = st.number_input("Plot Width / Frontage (Feet)", min_value=15.0, max_value=150.0, value=30.0, step=1.0)
                l_ft = st.number_input("Plot Length / Depth (Feet)", min_value=15.0, max_value=200.0, value=43.5, step=1.0)
                plot_w = round(w_ft * 0.3048, 2)
                plot_h = round(l_ft * 0.3048, 2)
                plot_area_sqft = w_ft * l_ft
                plot_area_sqm = round(plot_area_sqft * 0.092903, 2)
                plot_cents = round(plot_area_sqft / 435.60, 2)
            else:
                plot_w = st.number_input("Plot Width (meters)", min_value=5.0, max_value=50.0, value=12.0, step=0.5)
                plot_h = st.number_input("Plot Length / Depth (meters)", min_value=5.0, max_value=60.0, value=15.0, step=0.5)
                plot_area_sqm = round(plot_w * plot_h, 2)
                plot_area_sqft = round(plot_area_sqm * 10.764, 1)
                plot_cents = round(plot_area_sqft / 435.60, 2)

            st.caption(f"📏 **Plot Area:** {plot_cents:.2f} Cents ({plot_area_sqft:.0f} sq ft | {plot_area_sqm:.1f} m²)")

        orientation = st.selectbox(
            "Compass Facing (Road / Entrance Side)",
            options=["North (N)", "East (E)", "South (S)", "West (W)"],
            index=0,
            help="Which side faces the main access road / North direction"
        )
        orient_code = orientation[0]  # 'N', 'E', 'S', 'W'

    with col_family:
        st.markdown("### 👨‍👩‍👧‍👦 2. Family & Intent")
        adults = st.number_input("Adults (18+)", min_value=1, max_value=12, value=2, step=1)
        children = st.number_input("Children", min_value=0, max_value=8, value=2, step=1)
        elderly = st.number_input("Elderly Parents", min_value=0, max_value=6, value=0, step=1)
        
        st.markdown("**Lifestyle Spaces:**")
        pooja = st.checkbox("🕉️ Pooja / Mandir Room", value=True)
        office = st.checkbox("💻 Study / Home Office", value=False)
        servant = st.checkbox("🧹 Servant Room", value=False)
        store = st.checkbox("📦 Dedicated Store Room", value=False)
        guests = st.checkbox("🛏️ Frequent Guests", value=False)

    family_kw = dict(adults=int(adults), children=int(children), elderly=int(elderly),
                     guests_frequent=bool(guests), needs_home_office=bool(office),
                     needs_pooja_room=bool(pooja), needs_servant_quarter=bool(servant),
                     needs_store_room=bool(store))

    with col_eng:
        st.markdown("### ⚙️ 3. Engineering & Vastu")
        vastu_on = st.toggle("🕉️ Vastu Shastra Compliance", value=True, help="Places rooms according to the ancient 3x3 Vastu Purusha Mandala")

        # BHK choices follow the plot. Building type / floors are chosen further down, so their
        # values for this rerun are read from session state (Streamlit updates it before the rerun).
        _btype_now = st.session_state.get("bldg_type", "Independent House / Villa")
        _floors_now = int(st.session_state.get("num_floors", 1) or 1)
        bhk_fit = _bhk_fit(round(plot_w, 3), round(plot_h, 3), orient_code, family_kw, _btype_now,
                           _floors_now, vastu_on, _checker_version())
        fitting = [b for b, f in bhk_fit.items() if f.status != "no"] or ["1BHK"]
        auto_raw = FamilyAnalyzer().analyze(FamilyProfile(**family_kw), plot_area=plot_area_sqm).recommended_bhk
        auto_bhk = auto_raw if auto_raw in fitting else max([b for b in fitting if b <= auto_raw] or fitting[:1])
        bhk_opts = ["Auto"] + fitting
        if st.session_state.get("bhk_choice") not in bhk_opts:
            st.session_state["bhk_choice"] = "Auto"

        def _bhk_label(opt):
            if opt == "Auto":
                return f"Auto-Detect → {auto_bhk} (Recommended)"
            return f"{opt}  ·  " + ("✅ fits well" if bhk_fit[opt].status == "ok" else "⚠️ small rooms on this plot")

        bhk_choice = st.selectbox("BHK Configuration", options=bhk_opts, key="bhk_choice", format_func=_bhk_label,
                                  help="Only the BHK sizes that fit your plot are offered.")
        no_fit = [b for b, f in bhk_fit.items() if f.status == "no"]
        if no_fit:
            st.caption("🔒 Too big for this plot: " + " · ".join(
                f"{b}" + (f" (needs about {bhk_fit[b].unlock.split('about ')[1].rstrip('.')})"
                          if "about " in bhk_fit[b].unlock else "") for b in no_fit))
        effective_bhk = auto_bhk if bhk_choice == "Auto" else bhk_choice
        # Auto-detect may want more bedrooms than the plot holds — pin the largest that fits
        bhk_override = None if bhk_choice == "Auto" and auto_bhk == auto_raw else effective_bhk

        quality = st.selectbox("Construction Quality", options=["Economy", "Standard", "Premium", "Luxury"], index=1)
        country = st.selectbox(
            "🌍 Country / Market", options=CostEngine.markets(), index=0,
            help="Currency, construction system and 2026 cost benchmarks for the market")
        _regions = CostEngine.regions(country)
        city_tier = st.selectbox(
            "📍 City / Region", options=_regions,
            index=_regions.index(CostEngine.default_region(country)), key=f"cost_region_{country}",
            help="Location cost index applied to labour, materials and site works")
        bldg_type = st.selectbox(
            "🏢 Building Type",
            ["Independent House / Villa", "Apartment / Multi-Storey Building", "Duplex Home"],
            index=0, key="bldg_type",
            help="Select Independent House or Multi-Storey Apartment"
        )

        max_floors = 10 if "Apartment" in bldg_type else 5
        if int(st.session_state.get("num_floors", 1) or 1) > max_floors:
            st.session_state["num_floors"] = max_floors
        st.session_state.setdefault("num_floors", 1)
        floors = st.number_input("Number of Floors", min_value=1, max_value=max_floors, step=1,
                                 key="num_floors")
        has_upper = floors > 1 or "Apartment" in bldg_type or "Duplex" in bldg_type

        # Lift/Elevator toggle — intelligent auto-enable for 4+ floors or Apartments
        is_mandated_lift = (floors >= 4) or ("Apartment" in bldg_type)
        if not has_upper:
            st.toggle("🛗 Lift / Elevator Shaft", value=False, disabled=True, key="lift_locked",
                      help="A single-storey house has no upper level for a lift to serve.")
            st.caption("🔒 A lift needs 2 or more floors.")
            has_lift = False
        elif is_mandated_lift:
            st.info("🛗 **Lift Mandated / Auto-Enabled** (Required per NBC 2016 for 4+ storeys or multi-family apartments).")
            has_lift = st.toggle(
                "🛗 Lift / Elevator Shaft",
                value=True,
                help="IS 14665 compliant lift shaft (min 1.5m × 1.5m). Placed per Vastu in S/SW/W zone."
            )
        else:
            has_lift = st.toggle(
                "🛗 Lift / Elevator Shaft",
                value=False,
                help="IS 14665 compliant lift shaft (min 1.5m × 1.5m). Placed per Vastu in S/SW/W zone."
            )

        if has_lift:
            lift_capacity = st.selectbox(
                "Lift Capacity",
                ["4 Person (630kg) — 1.5m × 1.5m shaft",
                 "6 Person (480kg) — 1.6m × 1.7m shaft",
                 "8 Person (680kg) — 1.8m × 2.0m shaft"],
                index=0
            )
        else:
            lift_capacity = "4 Person"

        # Staircase type
        stair_options = [
            "Dog-Leg (U-Turn) — Space Efficient",
            "Straight Flight — Simple & Direct",
            "L-Shaped (Quarter Turn) — Corner Fit",
            "Open Well (Bifurcated) — Grand Central",
            "Spiral / Circular — Compact Architectural",
            "Crossing / Zigzag — Modern Split-Level",
        ]
        stair_map = {
            "Dog-Leg (U-Turn) — Space Efficient": "dog_leg",
            "Straight Flight — Simple & Direct": "straight",
            "L-Shaped (Quarter Turn) — Corner Fit": "l_shaped",
            "Open Well (Bifurcated) — Grand Central": "open_well",
            "Spiral / Circular — Compact Architectural": "spiral",
            "Crossing / Zigzag — Modern Split-Level": "zigzag",
        }
        if not has_upper:
            st.selectbox("🪜 Staircase Type", ["Outside straight stair to the terrace"], disabled=True,
                         key="stair_locked")
            st.caption("Single storey: a straight stair outside the house leads to the terrace. "
                       "Add a floor to choose an inside staircase.")
            stair_code = "dog_leg"      # unused by the engine for a single storey
        else:
            stair_type = st.selectbox(
                "🪜 Staircase Type",
                stair_options,
                index=0,
                help="Dog-leg fits in compact plots. Open Well needs larger footprint."
            )
            stair_code = stair_map.get(stair_type, "dog_leg")

    # ---------------------------------------------------------
    # What fits on this plot — every site amenity is checked against the plot, BHK and
    # building chosen above (and the amenities before it), using the generator's own site planner
    # ---------------------------------------------------------
    _independent = "Independent" in bldg_type
    ss = st.session_state
    comps_now = {
        'car_parking': {'enabled': bool(ss.get("amen_car")),
                        'slots': int(ss.get("amen_car_slots", 1 if _independent else 3) or 1)},
        'bike_parking': {'enabled': bool(ss.get("amen_bike")),
                         'slots': int(ss.get("amen_bike_slots", 2 if _independent else 6) or 2)},
        'garden': {'enabled': bool(ss.get("amen_garden")), 'pct': int(ss.get("amen_garden_pct", 10) or 10)},
        'swimming_pool': {'enabled': bool(ss.get("amen_pool")), 'size': ss.get("amen_pool_size", "")},
        'water_tank': {'enabled': bool(ss.get("amen_sump"))},
        'generator_room': {'enabled': bool(ss.get("amen_genset"))},
        'security_cabin': {'enabled': bool(ss.get("amen_security"))},
    }
    site_fit = _site_fit(round(plot_w, 3), round(plot_h, 3), orient_code, effective_bhk, family_kw, bldg_type,
                         int(floors), quality, vastu_on, has_lift, lift_capacity, stair_code, comps_now, _checker_version())

    if site_fit.lifestyle_dropped:
        with col_family:
            st.warning("🔒 Not enough room for: **" + ", ".join(site_fit.lifestyle_dropped) +
                       f"** — it will be left out of the {effective_bhk} on this plot.")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
                padding: 18px 24px; border-radius: 10px; border: 1px solid #38BDF8;
                margin-top: 20px; margin-bottom: 12px;">
        <h4 style="color: #38BDF8; margin: 0 0 6px 0;"> Additional Components & Site Amenities</h4>
        <p style="color: #94A3B8; font-size: 13px; margin: 0;">
            Checked against your {plot_cents:.2f}-cent plot and {effective_bhk}: anything that will not fit is
            locked 🔒 with the reason and what would make it possible.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _house = site_fit.house_m2
    _ratio = _house / max(site_fit.comfy_m2, 1e-6)
    _feel = ("comfortable rooms" if _ratio >= 0.92 else
             "rooms a little compact" if _ratio >= 0.80 else "small rooms")
    st.markdown(f"🏠 **Your {effective_bhk} gets about {_house:.0f} m² ({_house * 10.764:.0f} sq ft) per floor** "
                f"— {_feel}.")
    if any(f.status == "no" for f in site_fit.amenities.values()):
        st.info("🔒 " + site_fit.summary())

    ac_col1, ac_col2, ac_col3 = st.columns(3)

    with ac_col1:
        f = site_fit.amenities["car_parking"]
        has_car_parking = _amenity_toggle("🚗 Car Parking Bay", "amen_car", f, show_ok=True)
        car_slots = _count_input("Car Slots", "amen_car_slots", 1, f.max_value or 1, 1,
                                 1 if _independent else 3, "car") if has_car_parking else 0

        f = site_fit.amenities["bike_parking"]
        has_bike_parking = _amenity_toggle("🏍️ Two-Wheeler Parking", "amen_bike", f)
        bike_slots = _count_input("Bike Slots", "amen_bike_slots", 2, f.max_value or 2, 2,
                                  2 if _independent else 6, "two-wheeler") if has_bike_parking else 0

    with ac_col2:
        f = site_fit.amenities["garden"]
        has_garden = _amenity_toggle("🌿 Landscaped Garden / Lawn", "amen_garden", f)
        garden_pct = 0
        if has_garden:
            top = f.max_value or 5
            if top <= 5:
                st.caption(f"Room for a small garden (about {f.amount_m2:.0f} m²).")
                garden_pct = 5
            else:
                if int(ss.get("amen_garden_pct", 10) or 10) > top:
                    ss["amen_garden_pct"] = top
                ss.setdefault("amen_garden_pct", min(10, top))
                garden_pct = st.slider("Garden Size (% of Plot)", min_value=5, max_value=top, step=5,
                                       key="amen_garden_pct", help=f"Up to {top}% fits on this plot.")

        f = site_fit.amenities["swimming_pool"]
        has_pool = _amenity_toggle("🏊 Swimming Pool", "amen_pool", f)
        pool_size = ""
        if has_pool and f.choices:
            if ss.get("amen_pool_size") not in f.choices:
                ss["amen_pool_size"] = f.choices[0]
            pool_size = st.selectbox("Pool Dimension", f.choices, key="amen_pool_size")
            if len(f.choices) < 3:
                st.caption("Larger pools do not fit on this plot.")

    with ac_col3:
        has_sump = _amenity_toggle("💧 Underground Water Sump (NE)", "amen_sump", site_fit.amenities["water_tank"],
                                   help="Underground water reservoir placed in Ishan / NE zone per Vastu")
        has_genset = _amenity_toggle("⚡ Backup Generator / Transformer", "amen_genset",
                                     site_fit.amenities["generator_room"],
                                     help="Acoustic generator room in Agni/Nairutya per engineering codes")
        has_security = _amenity_toggle("🛡️ Security / Watchman Cabin", "amen_security",
                                       site_fit.amenities["security_cabin"],
                                       help="Gatehouse located at the plot entrance boundary")

    additional_components = {
        'car_parking': {'enabled': has_car_parking, 'slots': car_slots if has_car_parking else 0},
        'bike_parking': {'enabled': has_bike_parking, 'slots': bike_slots if has_bike_parking else 0},
        'garden': {'enabled': has_garden, 'pct': garden_pct if has_garden else 0},
        'swimming_pool': {'enabled': has_pool, 'size': pool_size if has_pool else ''},
        'water_tank': {'enabled': has_sump},
        'generator_room': {'enabled': has_genset},
        'security_cabin': {'enabled': has_security},
    }
    # What the engine gets: the same choices plus the checker's hints (e.g. stilt parking)
    from floorplan_generator.site_feasibility import apply_fit
    additional_components = apply_fit(additional_components, site_fit)

    st.write("---")

    # ---------------------------------------------------------
    # Action Button
    # ---------------------------------------------------------
    generate_btn = st.button("🚀 Generate Civil Engineer Blueprint & Cost Estimation", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("🤖 AI Architect is calculating optimal room layout, Vastu alignments, and NBC code standards..."):
            try:
                # Step A: Family Analysis
                profile = FamilyProfile(
                    adults=adults,
                    children=children,
                    elderly=elderly,
                    guests_frequent=guests,
                    needs_home_office=office,
                    needs_pooja_room=pooja,
                    needs_servant_quarter=servant,
                    needs_store_room=store,
                    bhk_override=bhk_override,
                )
                analyzer = FamilyAnalyzer()
                analysis = analyzer.analyze(profile, plot_area=plot_area_sqm)

                # Step B: Core Layout Engine (ensure fresh class reload)
                import importlib
                import inspect
                import floorplan_generator.room_specs
                import floorplan_generator.vastu_engine
                # Reload dependencies first (a long-running Streamlit may hold stale copies),
                # then the modules that import names from them.
                importlib.reload(floorplan_generator.room_specs)
                importlib.reload(floorplan_generator.vastu_engine)
                import floorplan_generator.space_program
                import floorplan_generator.site_planner
                import floorplan_generator.layout_solver
                import floorplan_generator.quantities
                import floorplan_generator.intent_engine
                import floorplan_generator.layout_engine
                for _mod in (floorplan_generator.space_program, floorplan_generator.site_planner,
                             floorplan_generator.layout_solver, floorplan_generator.quantities,
                             floorplan_generator.intent_engine, floorplan_generator.layout_engine):
                    importlib.reload(_mod)
                from floorplan_generator.layout_engine import LayoutEngine, GeneratedFloorPlan

                engine_kwargs = {
                    "plot_width": plot_w,
                    "plot_height": plot_h,
                    "bhk_config": analysis.recommended_bhk,
                    "vastu_enabled": vastu_on,
                    "orientation": orient_code,
                    "quality_level": quality,
                    "project_name": f"{analysis.recommended_bhk} ({plot_cents:.1f} Cents) Blueprint",
                    "family_description": profile.description,
                    "has_lift": has_lift,
                    "lift_capacity": lift_capacity,
                    "stair_type": stair_code,
                    "num_floors": floors,
                    "building_type": bldg_type,
                    "additional_components": additional_components,
                    "family_profile": profile,
                }
                init_sig = inspect.signature(LayoutEngine.__init__)
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values())
                if has_varkw:
                    call_kwargs = engine_kwargs
                else:
                    call_kwargs = {k: v for k, v in engine_kwargs.items() if k in init_sig.parameters}

                engine = LayoutEngine(**call_kwargs)
                if not hasattr(engine, "additional_components") or not engine.additional_components:
                    engine.additional_components = additional_components
                st.session_state['gen_cents'] = plot_cents
                plan = engine.generate()

                # Store lift/stair selections for downstream renderers
                st.session_state['has_lift'] = has_lift
                st.session_state['lift_capacity'] = lift_capacity
                st.session_state['stair_code'] = stair_code
                st.session_state['building_type'] = bldg_type
                st.session_state['additional_components'] = additional_components

                # Step C: Render CAD & Vector outputs in memory/temp
                temp_dir = os.path.join(root_dir, "model_folder", "static", "generated_plans", "session_plan")
                os.makedirs(temp_dir, exist_ok=True)

                dxf_path = os.path.join(temp_dir, "blueprint.dxf")
                svg_path = os.path.join(temp_dir, "blueprint.svg")
                pdf_path = os.path.join(temp_dir, "blueprint.pdf")
                png_path = os.path.join(temp_dir, "blueprint.png")
                mask_path = os.path.join(temp_dir, "wall_mask.png")

                svg_r = SVGRenderer()
                svg_r.render(plan, svg_path, mode='clean')
                svg_clean = svg_r.render_to_string(plan, mode='clean')
                svg_blueprint = svg_r.render_to_string(plan, mode='blueprint')

                png_r = PNGRenderer()
                png_r.render(plan, png_path)
                mask = png_r.render_wall_mask(plan, ppm=50)

                dxf_r = DXFRenderer()
                dxf_r.render(plan, dxf_path)

                pdf_r = PDFRenderer()
                pdf_r.render(plan, pdf_path)

                # Step D: Construction Cost Engine — plinth area × floors, real wet-area
                # counts, lift and site works (UI labels are normalised inside CostEngine)
                cost_data = CostEngine.estimate_for_plan(
                    plan,
                    city_tier=city_tier,
                    quality_level=quality,
                    num_floors=floors,
                    country=country,
                )

                # Read binary file buffers for download buttons
                with open(dxf_path, "rb") as f:
                    dxf_bytes = f.read()
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                with open(png_path, "rb") as f:
                    png_bytes = f.read()
                with open(svg_path, "rb") as f:
                    svg_bytes = f.read()

                # Store in session state
                st.session_state['gen_plan'] = plan
                st.session_state['gen_analysis'] = analysis
                st.session_state['gen_cost'] = cost_data
                st.session_state['gen_svg_clean'] = svg_clean
                st.session_state['gen_svg_blueprint'] = svg_blueprint
                st.session_state['gen_dxf_bytes'] = dxf_bytes
                st.session_state['gen_pdf_bytes'] = pdf_bytes
                st.session_state['gen_png_bytes'] = png_bytes
                st.session_state['gen_svg_bytes'] = svg_bytes
                st.session_state['gen_mask'] = mask

                st.success("✅ Professional floor plan generated successfully!")

            except Exception as e:
                st.error(f"❌ Failed to generate floor plan: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # ---------------------------------------------------------
    # Render Results if Available
    # ---------------------------------------------------------
    if 'gen_plan' in st.session_state:
        plan: GeneratedFloorPlan = st.session_state['gen_plan']
        analysis = st.session_state['gen_analysis']
        # Re-price live so the Country / City / Quality dropdowns update the estimate without regenerating
        _plan_floors = getattr(plan, "num_floors", None) or floors
        cost_data = CostEngine.estimate_for_plan(
            plan, city_tier=city_tier, quality_level=quality, num_floors=_plan_floors, country=country)
        st.session_state['gen_cost'] = cost_data

        st.markdown("---")
        st.markdown(f"### 📋 Architectural Overview: {plan.bhk_config} ({plan.project_name})")

        # Custom CSS to prevent metric value truncation
        st.markdown("""
        <style>
        [data-testid="stMetricValue"] { font-size: 1.6rem !important; white-space: nowrap !important; overflow: visible !important; text-overflow: unset !important; }
        [data-testid="stMetricDelta"] { white-space: nowrap !important; }
        </style>
        """, unsafe_allow_html=True)

        # Top Metric Cards — 4 columns for enough width
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Configuration", plan.bhk_config, f"Vastu: {plan.vastu_score:.0f}/100")
        cents_carpet = (plan.total_carpet_area * 10.764) / 435.6
        cents_plot = (plan.plot_width * plan.plot_height * 10.764) / 435.6
        m2.metric("Carpet Area", f"{plan.total_carpet_area:.1f} m2", f"{cents_carpet:.2f} Cents ({plan.total_carpet_area * 10.764:.0f} sq ft)")
        m3.metric("Built-up Area", f"{plan.total_built_up_area:.1f} m2", f"{cents_plot:.2f} Cents Plot ({plan.total_built_up_area * 10.764:.0f} sq ft)")
        
        m4.metric("Est. Total Cost", cost_data['display']['total_compact'], cost_data['display']['rate'])

        # Amenities the owner asked for but the plot could not take — say so instead of silently dropping them
        _missing, _info = _site_request_status(plan)
        if _missing:
            st.warning("**Requested but not fitted on this plot:**\n\n" + "\n".join(f"- {m}" for m in _missing)
                       + "\n\nTry a bigger plot, fewer rooms, or a second floor.")
        for _m in _info:
            st.info(_m)

        # Style View Selector
        style_col1, style_col2 = st.columns([3, 1])
        with style_col2:
            view_mode = st.radio("Drawing Style", ["Blueprint Mode (Dark)", "Classic CAD (White)"], horizontal=True)

        svg_to_display = st.session_state['gen_svg_blueprint'] if "Blueprint" in view_mode else st.session_state['gen_svg_clean']

        # Display SVG Drawing
        st.markdown(
            f"""
            <div style="background: {'#0A2342' if 'Blueprint' in view_mode else '#FFFFFF'}; 
                        padding: 15px; border-radius: 8px; border: 2px solid #334155; 
                        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); text-align: center; overflow-x: auto;">
                {svg_to_display}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # ---------------------------------------------------------
        # Interactive Editor Section
        # ---------------------------------------------------------
        # keyed so it stays open across reruns (banners above come and go and would otherwise reset it)
        with st.expander("🧠 Smart Re-Plan — change your house by telling us what you want",
                         expanded=False, key="editor_expander"):
            FloorPlanEditor.render_editor_section(
                plan,
                cost_params={"city_tier": city_tier, "quality": quality, "floors": _plan_floors,
                             "country": country}
            )

        st.write("")

        # ---------------------------------------------------------
        # Download Bar
        # ---------------------------------------------------------
        st.markdown("#### 📥 Export to Professional Industry Formats")
        d_cols = st.columns(4)
        
        with d_cols[0]:
            st.download_button(
                label="📐 AutoCAD DXF (.dxf)",
                data=st.session_state['gen_dxf_bytes'],
                file_name=f"{plan.bhk_config}_floor_plan.dxf",
                mime="application/dxf",
                use_container_width=True,
                help="Open directly in AutoCAD, Revit, LibreCAD with full layer controls"
            )
        with d_cols[1]:
            st.download_button(
                label="🎨 Vector SVG (.svg)",
                data=st.session_state['gen_svg_bytes'],
                file_name=f"{plan.bhk_config}_floor_plan.svg",
                mime="image/svg+xml",
                use_container_width=True,
                help="Infinite-zoom scalable vector graphic for web and print"
            )
        with d_cols[2]:
            st.download_button(
                label="📄 Blueprint PDF (.pdf)",
                data=st.session_state['gen_pdf_bytes'],
                file_name=f"{plan.bhk_config}_blueprint_A3.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="ISO Standard A3 Drawing Sheet with title block"
            )
        with d_cols[3]:
            st.download_button(
                label="🖼️ High-Res PNG (.png)",
                data=st.session_state['gen_png_bytes'],
                file_name=f"{plan.bhk_config}_blueprint_300dpi.png",
                mime="image/png",
                use_container_width=True,
                help="300 DPI high-resolution image"
            )

        st.write("---")

        # ---------------------------------------------------------
        # Room Schedule, Vastu Breakdown, Cost BOQ & House Recommendations
        # ---------------------------------------------------------
        # import_module (not `from model_folder import …`) so an edited module is re-imported on rerun
        _report = _importlib.import_module("model_folder.house_report_ui")
        _cost_params = {"city_tier": city_tier, "quality": quality, "floors": _plan_floors, "country": country}
        # stateful tabs: the open tab survives button reruns, and hidden tabs are not computed
        t1, t2, t3, t4 = st.tabs([
            "📊 Room Schedule",
            "🕉️ Vastu Engineering",
            "💰 Cost & Material BOQ",
            "🏡 Best House Recommendation"
        ], key="report_tab", on_change="rerun")
        if t1.open:
            with t1:
                _report.render_room_schedule(plan)
        if t2.open:
            with t2:
                _report.render_vastu(plan, _cost_params)
        if t3.open:
            with t3:
                _report.render_cost(plan, cost_data, _cost_params, quality, _plan_floors)
        if t4.open:
            with t4:
                _report.render_recommendations(plan, cost_data, _cost_params, quality)
