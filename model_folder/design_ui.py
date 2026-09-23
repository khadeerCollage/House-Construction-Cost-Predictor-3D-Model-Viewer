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

    with col_eng:
        st.markdown("### ⚙️ 3. Engineering & Vastu")
        vastu_on = st.toggle("🕉️ Vastu Shastra Compliance", value=True, help="Places rooms according to the ancient 3x3 Vastu Purusha Mandala")
        
        bhk_options = ["Auto-Detect (Recommended)", "1BHK", "2BHK", "3BHK", "4BHK"]
        bhk_choice = st.selectbox("BHK Configuration", options=bhk_options, index=0)
        bhk_override = None if bhk_choice.startswith("Auto") else bhk_choice

        quality = st.selectbox("Construction Quality", options=["Economy", "Standard", "Premium", "Luxury"], index=1)
        city_tier = st.selectbox("Location / City Tier", options=["Tier-1 Metro", "Tier-2 Urban", "Tier-3 Semi-Urban", "Rural"], index=1)
        floors = st.number_input("Number of Floors", min_value=1, max_value=4, value=1, step=1)

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

                # Step B: Core Layout Engine
                engine = LayoutEngine(
                    plot_width=plot_w,
                    plot_height=plot_h,
                    bhk_config=analysis.recommended_bhk,
                    vastu_enabled=vastu_on,
                    orientation=orient_code,
                    quality_level=quality,
                    project_name=f"{analysis.recommended_bhk} ({plot_cents:.1f} Cents) Blueprint",
                    family_description=profile.description,
                )
                st.session_state['gen_cents'] = plot_cents
                plan = engine.generate()

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

                # Step D: Construction Cost Engine
                cost_data = CostEngine.estimate_cost(
                    area_sqm=plan.total_carpet_area,
                    city_tier=city_tier,
                    quality_level=quality,
                    num_floors=floors
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
        cost_data = st.session_state['gen_cost']

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
        
        est_lakhs = cost_data['total_cost'] / 100000.0
        m4.metric("Est. Total Cost", f"Rs {est_lakhs:.2f} Lakhs", f"Rs {cost_data['cost_per_sqft']}/sqft")

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
        with st.expander("✏️ Interactive Floor Plan Studio & Live Customizer — Add Doors, Flat Steps, Adjust Meters & Edit Blueprint", expanded=False):
            FloorPlanEditor.render_editor_section(
                plan,
                cost_params={"city_tier": city_tier, "quality": quality, "floors": floors}
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
        t1, t2, t3, t4 = st.tabs([
            "📊 Room Schedule & Dimensions",
            "🕉️ Ultra-Premium Vastu Audit",
            "💰 Cost & Material BOQ",
            "🏡 Best House Recommendations"
        ])

        with t1:
            st.markdown("##### Room Schedule (NBC 2016 Compliant)")
            room_rows = []
            for r in plan.rooms:
                w_ft = r.w * 3.28084
                h_ft = r.h * 3.28084
                sqft = r.area * 10.764
                room_rows.append({
                    "Room Name": r.name,
                    "Category": r.category,
                    "Width (m)": f"{r.w:.2f} m",
                    "Length (m)": f"{r.h:.2f} m",
                    "Dimensions (ft)": f"{w_ft:.1f}' × {h_ft:.1f}'",
                    "Carpet Area (m²)": f"{r.area:.2f} m²",
                    "Area (sq ft)": f"{sqft:.0f} sq ft",
                })
            st.table(room_rows)

        with t2:
            st.markdown("##### 🕉️ Vastu Shastra Compliance & Planetary Energy Audit")
            analyzer = VastuAnalyzer(plan)
            
            # Overall score & progress bar
            v_score = plan.vastu_score
            rec = analyzer.get_house_recommendation()
            st.progress(v_score / 100.0, text=f"Overall Vastu Harmony Score: {v_score:.0f}% — Grade: {rec['vastu_rating_grade']}")
            
            # Mandala Diagram & 32-Pada Entrance
            v_col1, v_col2 = st.columns([1.1, 1], gap="medium")
            
            with v_col1:
                st.markdown("###### 🌌 Vastu Purusha Mandala (9 Planetary Zones):")
                st.markdown(analyzer.render_mandala_svg(), unsafe_allow_html=True)
                st.caption("Energy vectors: Pranic inflow originates from Ishan (NE) and anchors heavily in Nairutya (SW).")

            with v_col2:
                st.markdown("###### 🚪 32-Pada Main Entrance Gate Audit:")
                entrance = analyzer.audit_entrance()
                
                e_box_color = "#059669" if entrance["rating"] in ("SUPER_AUSPICIOUS", "AUSPICIOUS") else "#D97706" if entrance["rating"] == "NEUTRAL" else "#DC2626"
                st.markdown(f"""
                <div style="background: #0F172A; border-left: 5px solid {e_box_color}; padding: 12px; border-radius: 6px; margin-bottom: 12px;">
                    <div style="font-size: 14px; font-weight: bold; color: #F8FAFC;">Gate {entrance['pada_id']}: {entrance['deity']} ({entrance['direction']} Face)</div>
                    <div style="font-size: 12px; color: #38BDF8;">Auspiciousness: <b>{entrance['rating']}</b> ({entrance['score']:+d}/10)</div>
                    <div style="font-size: 12px; color: #CBD5E1; margin-top: 4px;">{entrance['effect']}</div>
                    {f'<div style="font-size: 11px; color: #F87171; margin-top: 4px;"><b>Remedy:</b> {entrance["remedy"]}</div>' if entrance['remedy'] else ''}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("###### 🌐 Pancha Bhoota (5 Elements) Elemental Balance:")
                balance = analyzer.get_elemental_balance()
                b_cols = st.columns(len(balance))
                for idx, (elem_name, b_info) in enumerate(balance.items()):
                    b_cols[idx].metric(elem_name.split()[0], f"{b_info['percentage']:.0f}%", b_info['health'])

                st.write("")
                # Certificate Download
                cert_text = analyzer.generate_certificate_text()
                st.download_button(
                    label="📜 Download Vastu Compliance Audit Certificate (.txt)",
                    data=cert_text,
                    file_name=f"{plan.bhk_config}_Vastu_Audit_Certificate.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            st.write("---")
            st.markdown("##### 🧭 Zone-by-Zone Elemental & Room Placement Matrix")
            zone_data = analyzer.get_zone_analysis()
            zone_table_rows = []
            for zd in zone_data:
                status_icon = "🟢" if zd['status'] == "OPTIMAL" else "🟡" if zd['status'] == "GOOD" else "🟠" if zd['status'] == "ACCEPTABLE" else "🔴"
                zone_table_rows.append({
                    "Zone": zd["zone_name"],
                    "Element": zd["element"],
                    "Deity": zd["deity"],
                    "Occupying Room(s)": ", ".join(zd["rooms"]),
                    "Score": f"{zd['score']:+d}/10",
                    "Status": f"{status_icon} {zd['status']}",
                    "Observation": zd["observation"]
                })
            st.table(zone_table_rows)

            # Defect & Non-demolition remedies
            defects = analyzer.get_defects_and_remedies()
            if defects:
                st.markdown("##### 🛡️ Detected Energetic Anomalies & Non-Demolition Remedies (Parihara)")
                for d in defects:
                    sev_color = "#DC2626" if d['severity'] == "CRITICAL" else "#EA580C" if d['severity'] == "HIGH" else "#D97706"
                    st.markdown(f"""
                    <div style="background: #1E293B; border-left: 4px solid {sev_color}; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                        <span style="font-weight: bold; color: #F8FAFC;">{d['defect']}</span> 
                        <span style="color: {sev_color}; font-size: 11px; font-weight: bold;">[{d['severity']}]</span><br>
                        <span style="color: #94A3B8; font-size: 12px;"><b>Impact:</b> {d['impact']}</span><br>
                        <span style="color: #38BDF8; font-size: 12px;"><b>Vedic Remedy:</b> {d['remedy']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ **Zero Vastu defects detected!** All primary rooms and circulation paths align flawlessly with canonical guidelines.")

        with t3:
            st.markdown("##### Construction Cost Estimate & Bill of Quantities (BOQ)")
            st.markdown(f"**Total Estimate:** ₹ {cost_data['total_cost']:,.2f} (*{est_lakhs:.2f} Lakhs INR*) at **₹ {cost_data['cost_per_sqft']}/sq ft**")
            
            st.markdown("###### 🧱 Estimated Structural Materials Required:")
            mat_cols = st.columns(5)
            materials = cost_data['materials']
            
            c_info = materials.get('Cement', {})
            mat_cols[0].metric("Cement", f"{c_info.get('quantity', 0)} Bags", f"₹ {c_info.get('cost', 0):,}")
            
            s_info = materials.get('Steel', {})
            mat_cols[1].metric("TMT Steel", f"{s_info.get('quantity', 0):,} Kg", f"₹ {s_info.get('cost', 0):,}")
            
            b_info = materials.get('Bricks', {})
            mat_cols[2].metric("Clay Bricks", f"{b_info.get('quantity', 0):,} Pcs", f"₹ {b_info.get('cost', 0):,}")
            
            sa_info = materials.get('Sand', {})
            mat_cols[3].metric("River Sand", f"{sa_info.get('quantity', 0):,} CFT", f"₹ {sa_info.get('cost', 0):,}")
            
            ag_info = materials.get('Aggregate', {})
            mat_cols[4].metric("Aggregate", f"{ag_info.get('quantity', 0):,} CFT", f"₹ {ag_info.get('cost', 0):,}")

            st.write("")
            st.caption(f"Cost estimates based on {city_tier} market rates and {quality} grade construction materials per Indian CPWD / State PWD schedules.")

        with t4:
            st.markdown("##### 🏡 AI Architect — Best House Archetype & Layout Recommendations")
            rec = analyzer.get_house_recommendation()
            
            st.info(f"🏆 **Recommended House Archetype:** **{rec['house_archetype']}**\n\n{rec['summary']}")
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("###### 📐 Land & Geometry Assessment:")
                st.markdown(f"- **Plot Aspect Ratio:** `{plan.plot_width:.1f}m (W) × {plan.plot_height:.1f}m (L)` (Ratio: {plan.plot_width/max(0.1, plan.plot_height):.2f})")
                st.markdown(f"- **Geometry Classification:** {rec['plot_shape_evaluation']}")
                st.markdown(f"- **Cardinal Alignment:** Road frontage facing **{plan.orientation}**.")
            
            with r_col2:
                st.markdown("###### 🎯 Targeted Optimization Suggestions:")
                for swap in rec['room_swap_recommendations']:
                    st.markdown(f"- {swap}")
