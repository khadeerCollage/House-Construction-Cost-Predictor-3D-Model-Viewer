"""
Standalone Streamlit App for Floor Plan Analysis
Works without Flask backend - suitable for Streamlit Cloud deployment
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import tempfile
import os
import sys
import base64

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import detection modules
from floorplan_detector.floor_plan_analyzer import FloorPlanAnalyzer, RoomLabel, ROOM_NAMES

# ==================== CONFIGURATION ====================

ROOM_CONFIG = {
    "Living/Dining": {"color": "#8DD3C7", "icon": "🛋️", "features": ["Sofa", "TV", "Dining Table"], "description": "Main living space"},
    "Living Room": {"color": "#8DD3C7", "icon": "🛋️", "features": ["Sofa", "TV", "Coffee Table"], "description": "Family room"},
    "Bedroom": {"color": "#FFFFB3", "icon": "🛏️", "features": ["Bed", "Wardrobe"], "description": "Sleeping area"},
    "Bathroom": {"color": "#BEBADA", "icon": "🚿", "features": ["Shower", "Toilet", "Sink"], "description": "Bathroom"},
    "Kitchen": {"color": "#FB8072", "icon": "🍳", "features": ["Stove", "Fridge", "Sink"], "description": "Cooking area"},
    "Hallway": {"color": "#D9D9D9", "icon": "🚶", "features": ["Passage"], "description": "Corridor"},
    "Storage": {"color": "#FDB462", "icon": "📦", "features": ["Shelves"], "description": "Storage space"},
    "Stairs": {"color": "#B3DE69", "icon": "🪜", "features": ["Steps"], "description": "Staircase"},
    "Room": {"color": "#80B1D3", "icon": "🚪", "features": ["Multi-purpose"], "description": "General room"},
}

LABEL_TO_NAME = {
    RoomLabel.LIVING_DINING: "Living/Dining",
    RoomLabel.KITCHEN: "Kitchen",
    RoomLabel.BEDROOM: "Bedroom",
    RoomLabel.BATHROOM: "Bathroom",
    RoomLabel.HALLWAY: "Hallway",
    RoomLabel.STORAGE: "Storage",
    RoomLabel.STAIRS: "Stairs",
    RoomLabel.GARAGE: "Room",
    RoomLabel.OTHER: "Room",
    RoomLabel.BACKGROUND: "Room",
    RoomLabel.OUTDOOR: "Room",
    RoomLabel.WALL: "Room",
}

# ==================== DETECTION FUNCTIONS ====================

@st.cache_resource
def get_analyzer():
    """Cache the analyzer for better performance"""
    return FloorPlanAnalyzer()

def detect_rooms_and_walls(image_array):
    """Detect walls and rooms directly from image array"""
    
    # Get cached analyzer
    analyzer = get_analyzer()
    
    # Resize large images
    max_dimension = 1200
    height, width = image_array.shape[:2]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        image_array = cv2.resize(image_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Run analysis
    result = analyzer.analyze(image_array)
    
    walls = result['walls']
    doors = result['doors']
    rooms = result['rooms']
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    
    # Draw walls in red
    for wall in walls:
        ax.plot([wall.x1, wall.x2], [wall.y1, wall.y2], color='red', linewidth=2, alpha=0.8)
    
    # Draw doors in green
    for door in doors:
        rect = patches.Rectangle(
            (door.x, door.y), door.width, door.height,
            linewidth=2, edgecolor='green', facecolor='green', alpha=0.3
        )
        ax.add_patch(rect)
    
    # Process rooms
    room_data = []
    room_type_counts = {rt: 0 for rt in ROOM_CONFIG.keys()}
    
    for i, room in enumerate(rooms):
        x, y, w, h = room.bounding_box
        area = room.area
        
        room_type = LABEL_TO_NAME.get(room.label, room.name)
        if room_type not in ROOM_CONFIG:
            room_type = "Room"
        
        config = ROOM_CONFIG[room_type]
        color = config["color"]
        icon = config["icon"]
        
        room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
        room_id = room_type_counts[room_type]
        
        # Draw room
        if room.contour is not None and len(room.contour) > 0:
            contour_points = room.contour.reshape(-1, 2)
            polygon = patches.Polygon(contour_points, linewidth=2, edgecolor=color, facecolor=color, alpha=0.4)
            ax.add_patch(polygon)
        else:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor=color, alpha=0.4)
            ax.add_patch(rect)
        
        # Label
        label = f"{icon} {room_type}"
        if room_id > 1:
            label += f" {room_id}"
        
        cx, cy = room.center
        ax.text(cx - 30, cy, label, color='black', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
        room_data.append({
            "id": i + 1,
            "type": room_type,
            "type_id": room_id,
            "icon": icon,
            "area": float(area),
            "bounds": [int(x), int(y), int(w), int(h)],
            "aspect_ratio": float(room.aspect_ratio),
            "features": config["features"],
            "color": color
        })
    
    ax.set_title(f"Floor Plan Analysis\nWalls: {len(walls)} | Doors: {len(doors)} | Rooms: {len(rooms)}", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return {
        'wall_count': len(walls),
        'door_count': len(doors),
        'room_count': len(rooms),
        'visualization': buf,
        'rooms': room_data,
        'room_type_counts': room_type_counts,
    }

def estimate_construction_cost(area, bedrooms, bathrooms, kitchen, living, location_factor=1.0):
    """Estimate construction cost based on room data"""
    base_cost_per_sqm = 1500  # Base cost per square meter
    
    breakdown = {
        'base_area_cost': area * base_cost_per_sqm * location_factor,
        'bedroom_cost': bedrooms * 5000,
        'bathroom_cost': bathrooms * 8000,
        'kitchen_cost': kitchen * 10000,
        'living_cost': living * 3000,
    }
    
    total = sum(breakdown.values())
    
    return {
        'estimated_cost': total,
        'breakdown': breakdown
    }

# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="🏡 House Construction Cost Predictor & 3D Viewer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏡 House Construction Cost Predictor & 3D Model Viewer")

# Sidebar
st.sidebar.markdown("### 🏡 About This Project")
st.sidebar.markdown("""
**AI-powered tool** that helps you:

- ✅ **Detect walls and rooms** from floorplans
- ✅ **Identify room types** automatically  
- ✅ **Estimate construction costs**

Perfect for architects, builders, and homeowners!
""")

st.sidebar.markdown("---")
st.sidebar.info("🚀 **Cloud Version** - No backend server needed!")

# Main content
st.markdown("""
### Upload your floorplan image
- Supported formats: **PNG, JPG, JPEG**
- The AI will detect walls, rooms, and estimate costs
""")

uploaded_file = st.file_uploader(
    "Upload Floorplan Image",
    type=["png", "jpg", "jpeg"],
    key="fileUploader"
)

if not uploaded_file:
    st.markdown("""
    <div style="text-align: center; margin: 30px;">
        <p style="color: #757575; font-style: italic;">
            ✨ Turning floorplans into insights with AI ❤️
        </p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Floor Plan", use_container_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Floor Plan", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Analyzing floor plan..."):
            try:
                # Convert uploaded file to OpenCV format
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is None:
                    st.error("❌ Failed to read image. Please try another file.")
                    st.stop()
                
                # Run detection
                result = detect_rooms_and_walls(img)
                
                # Store in session state
                st.session_state['detection_result'] = result
                st.session_state['image_shape'] = img.shape
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.stop()
    
    # Display results if available
    if 'detection_result' in st.session_state:
        result = st.session_state['detection_result']
        
        st.markdown("---")
        st.markdown("## 🔍 Detection Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧱 Walls Detected", result['wall_count'])
        with col2:
            st.metric("🚪 Doors Detected", result['door_count'])
        with col3:
            st.metric("🏠 Rooms Detected", result['room_count'])
        
        # Visualization
        st.image(result['visualization'], caption="Wall and Room Detection Results", use_container_width=True)
        
        # Room breakdown
        st.markdown("### 🏘️ Room Types Detected")
        
        room_counts = result['room_type_counts']
        detected_rooms = {k: v for k, v in room_counts.items() if v > 0}
        
        if detected_rooms:
            cols = st.columns(len(detected_rooms))
            for i, (room_type, count) in enumerate(detected_rooms.items()):
                icon = ROOM_CONFIG.get(room_type, {}).get('icon', '🏠')
                cols[i].metric(f"{icon} {room_type}", count)
        
        # Room details
        if result['rooms']:
            with st.expander("📋 Detailed Room Information", expanded=False):
                for room in result['rooms']:
                    st.markdown(f"""
                    **{room['icon']} {room['type']} {room['type_id']}**
                    - Area: {room['area']:.0f} px² (~{room['area']/1000:.1f} m²)
                    - Dimensions: {room['bounds'][2]} × {room['bounds'][3]} px
                    - Features: {', '.join(room['features'])}
                    """)
                    st.markdown("---")
        
        # Construction cost estimation
        st.markdown("---")
        st.markdown("## 💰 Construction Cost Estimation")
        
        # Calculate area
        total_area = sum(r['area'] for r in result['rooms']) / 1000  # Convert to approx m²
        bedrooms = room_counts.get('Bedroom', 0)
        bathrooms = room_counts.get('Bathroom', 0)
        kitchens = room_counts.get('Kitchen', 0)
        living = room_counts.get('Living/Dining', 0) + room_counts.get('Living Room', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Estimated Area:** {total_area:.1f} m²")
            st.markdown(f"**Bedrooms:** {bedrooms}")
            st.markdown(f"**Bathrooms:** {bathrooms}")
            st.markdown(f"**Kitchen:** {kitchens}")
        
        with col2:
            location = st.selectbox("📍 Location Type", ["Urban (High Cost)", "Suburban", "Rural (Low Cost)"])
            location_factors = {"Urban (High Cost)": 1.3, "Suburban": 1.0, "Rural (Low Cost)": 0.8}
            location_factor = location_factors[location]
        
        if st.button("💵 Calculate Construction Cost"):
            cost_result = estimate_construction_cost(
                total_area, bedrooms, bathrooms, kitchens, living, location_factor
            )
            
            st.success(f"🏗️ **Estimated Construction Cost: ${cost_result['estimated_cost']:,.2f}**")
            
            with st.expander("📊 Cost Breakdown"):
                breakdown = cost_result['breakdown']
                st.write(f"- Base Area Cost ({total_area:.1f} m²): ${breakdown['base_area_cost']:,.2f}")
                st.write(f"- Bedroom Cost ({bedrooms}): ${breakdown['bedroom_cost']:,.2f}")
                st.write(f"- Bathroom Cost ({bathrooms}): ${breakdown['bathroom_cost']:,.2f}")
                st.write(f"- Kitchen Cost ({kitchens}): ${breakdown['kitchen_cost']:,.2f}")
                st.write(f"- Living Room Cost ({living}): ${breakdown['living_cost']:,.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p style="color: #5A5A5A;">Made with ❤️ for architects and home builders</p>
    <p style="color: #9E9E9E; font-size: 0.8em;">© 2025 House Construction Cost Predictor</p>
</div>
""", unsafe_allow_html=True)
