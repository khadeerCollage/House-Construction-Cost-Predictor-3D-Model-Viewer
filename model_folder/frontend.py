import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import time
import traceback
import subprocess
import platform
import sys
import json
import threading
import requests

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import utilities
try:
    from utils import detect_rooms_and_walls, ROOM_CONFIG
except ImportError:
    try:
        from model_folder.utils import detect_rooms_and_walls, ROOM_CONFIG
    except ImportError:
        ROOM_CONFIG = {}
        detect_rooms_and_walls = None

# Define room colors for frontend display
room_colors = {
    "Living Room": "#8DD3C7",  # Teal
    "Bedroom": "#FFFFB3",      # Light yellow
    "Bathroom": "#BEBADA",     # Lavender
    "Kitchen": "#FB8072",      # Salmon pink
    "Room": "#80B1D3"          # Light blue
}

# ==================== INTEGRATED BACKEND FUNCTIONS ====================

def allowed_file(filename):
    """Check if file is an allowed image type"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def detect_walls_rooms_local(img_path, output_dir=None):
    """Detect walls and rooms locally without Flask API"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    os.makedirs(output_dir, exist_ok=True)
    
    if detect_rooms_and_walls is None:
        raise ImportError("detect_rooms_and_walls function not available")
    
    result = detect_rooms_and_walls(img_path, output_dir=output_dir)
    
    # Get visualization path
    vis_path = result.get('visualization')
    if vis_path:
        result['visualization'] = vis_path
        
        # Create thumbnail
        thumb_path = os.path.join(output_dir, f'thumb_{os.path.basename(vis_path)}')
        try:
            thumb_img = cv2.imread(vis_path)
            thumb_img = cv2.resize(thumb_img, (0, 0), fx=0.5, fy=0.5)
            cv2.imwrite(thumb_path, thumb_img)
            result['thumbnail'] = thumb_path
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
    
    # Add room config details
    result['room_config'] = ROOM_CONFIG
    return result

def generate_3d_local(img_path, output_dir=None):
    """Generate 3D model locally without Flask API"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(output_dir, exist_ok=True)
    
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '2d-3d.py'))
    output_model_path = os.path.join(output_dir, 'floorplan_3d_model.ply')
    output_data_path = os.path.join(output_dir, 'floorplan_data.json')
    
    # Run 2D to 3D conversion script
    result = subprocess.run(
        [sys.executable, script_path, '--input', img_path, '--output', output_model_path],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode != 0:
        return {
            'error': '3D model generation failed.',
            'stderr': result.stderr,
            'stdout': result.stdout
        }
    
    if not os.path.exists(output_model_path):
        return {'error': '3D model file not generated'}
    
    # Copy to static folder
    static_ply_path = os.path.join(static_dir, os.path.basename(output_model_path))
    with open(output_model_path, "rb") as src, open(static_ply_path, "wb") as dst:
        dst.write(src.read())
    
    # Get floorplan data from detection
    floorplan_data = {}
    try:
        detection_result = detect_walls_rooms_local(img_path, output_dir=static_dir)
        
        total_rooms = detection_result.get('total_rooms', 0)
        room_counts = detection_result.get('room_counts', {})
        
        if not room_counts and total_rooms > 0:
            bedrooms = max(1, int(total_rooms * 0.2))
            bathrooms = max(1, int(total_rooms * 0.06))
            kitchen = 1
            living = 1
            other = total_rooms - (bedrooms + bathrooms + kitchen + living)
            
            room_counts = {
                "bedroom": bedrooms,
                "bathroom": bathrooms,
                "kitchen": kitchen,
                "living": living,
                "other": other
            }
        
        if not room_counts or total_rooms <= 0:
            try:
                image = cv2.imread(img_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                potential_rooms = sum(1 for c in contours if cv2.contourArea(c) > 100)
                total_rooms = max(4, potential_rooms)
                
                bedrooms = max(1, int(total_rooms * 0.2))
                bathrooms = max(1, int(total_rooms * 0.06))
                kitchen = 1
                living = 1
                other = total_rooms - (bedrooms + bathrooms + kitchen + living)
                
                room_counts = {
                    "bedroom": bedrooms,
                    "bathroom": bathrooms,
                    "kitchen": kitchen,
                    "living": living,
                    "other": other
                }
            except:
                total_rooms = 4
                room_counts = {"bedroom": 1, "bathroom": 1, "kitchen": 1, "living": 1, "other": 0}
        
        floorplan_data["room_counts"] = room_counts
        floorplan_data["rooms"] = total_rooms
        
        if "estimated_area" in detection_result and detection_result["estimated_area"] > 0:
            floorplan_data["estimated_area"] = detection_result["estimated_area"]
        else:
            floorplan_data["estimated_area"] = max(total_rooms * 6, 30)
        
    except Exception as e:
        floorplan_data = {
            "rooms": 4,
            "room_counts": {"bedroom": 1, "bathroom": 1, "kitchen": 1, "living": 1, "other": 0},
            "estimated_area": 30,
            "error": str(e)
        }
    
    return {
        '3d_model': static_ply_path,
        'floorplan_data': floorplan_data
    }

def estimate_construction_cost_local(area, bedrooms, bathrooms, kitchen, living, location_factor=1.0):
    """Estimate construction cost locally"""
    base_cost_per_sqm = 1200
    
    bedroom_cost = bedrooms * 10000
    bathroom_cost = bathrooms * 15000
    kitchen_cost = kitchen * 20000
    living_cost = living * 8000
    
    total_area_cost = area * base_cost_per_sqm
    total_room_cost = bedroom_cost + bathroom_cost + kitchen_cost + living_cost
    final_cost = (total_area_cost + total_room_cost) * location_factor
    
    return {
        'estimated_cost': round(final_cost, 2),
        'breakdown': {
            'base_area_cost': round(total_area_cost, 2),
            'bedroom_cost': round(bedroom_cost, 2),
            'bathroom_cost': round(bathroom_cost, 2),
            'kitchen_cost': round(kitchen_cost, 2),
            'living_cost': round(living_cost, 2)
        }
    }

# ================================================================

# Function to check if backend is running and reachable
def check_backend_connection(url="http://127.0.0.1:5000/", retry_count=2, timeout=3):
    """Check if backend server is running and reachable"""
    for attempt in range(retry_count):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, "Connected"
            else:
                return False, f"Backend returned status code {response.status_code}"
        except requests.exceptions.ConnectionError:
            if attempt < retry_count - 1:
                time.sleep(1)  # Wait before retry
                continue
            return False, "Connection refused"
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            return False, "Connection timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
    return False, "Could not connect after retries"

# Function to try starting the backend server if not running
def try_start_backend_server():
    """Attempt to start the backend server if it's not running"""
    try:
        # Get the correct path
        model_folder = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(model_folder, "app.py")
        
        if platform.system() == "Windows":
            # Use subprocess.Popen to avoid blocking the Streamlit app
            process = subprocess.Popen(
                ["python", app_path],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Give it a moment to start
            time.sleep(3)
            return True, "Backend server start initiated"
        else:
            # For other platforms like Linux or Mac
            process = subprocess.Popen(
                ["python", app_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)
            return True, "Backend server start initiated"
    except Exception as e:
        return False, f"Failed to start backend server: {str(e)}"

# Function to display project information in the sidebar instead of backend status
def show_sidebar_info():
    """Display project information in the sidebar"""
    st.sidebar.markdown("### 🏡 About This Project")
    
    st.sidebar.markdown("""
    **House Construction Cost Predictor** is an AI-powered tool that helps you:
    
    - ✅ **Detect walls and rooms** from floorplans
    - ✅ **Identify room types** automatically
    - ✅ **Generate 3D models** of your floorplans
    - ✅ **Estimate construction costs** based on area and features
    
    Perfect for architects, builders, and homeowners planning construction projects.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Scale Calibration")
    st.sidebar.markdown("Enter real plot dimensions to calibrate room sizes (optional):")
    st.sidebar.number_input("Plot Width (meters)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="plot_width")
    st.sidebar.number_input("Plot Length (meters)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="plot_length")
    
    # Check backend without showing status - just for functionality
    connected, _ = check_backend_connection()
    if not connected:
        if st.sidebar.button("🔄 Start Backend Server", help="Start the backend server if it's not running"):
            success, msg = try_start_backend_server()
            if success:
                st.sidebar.success("✅ Backend started")
                time.sleep(2)  # Short delay
                st.rerun()  # Updated from experimental_rerun
            else:
                st.sidebar.error(f"Couldn't start backend: {msg}")

st.set_page_config(page_title="🏡 House Construction Cost Predictor & 3D Viewer")
st.title("🏡 House Construction Cost Predictor & 3D Model Viewer")

# Show project info in sidebar instead of backend status
show_sidebar_info()

st.markdown("""
### Upload your floorplan image
- Click the **large + button** or **drag and drop** your file here.
- Supported formats: PNG, JPG, JPEG
""")

uploaded_file = st.file_uploader(
    "Upload Floorplan Image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="fileUploader",
    label_visibility="visible"
)

# Add the small middle note with a heart symbol after the file uploader
if not uploaded_file:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; margin-bottom: 30px;">
            <p style="color: #757575; font-size: 0.9em; font-style: italic;">
                ✨ Turning floorplans into 3D dreams with AI ❤️
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Create columns for the two buttons with better interface flow
    col1, col2 = st.columns(2)
    
    with col1:
        # Add Wall and Room Detection Feature
        if st.button("1️⃣ Detect Walls and Rooms"):
            # Close the columns layout to allow full-width display
            st.write("---")
            
            with st.spinner("Detecting walls and rooms..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_image_path = tmp_file.name
                    
                    # Verify if backend is running with improved error handling and retry
                    connected, status_msg = check_backend_connection(retry_count=3, timeout=5)
                    if not connected:
                        st.error(f"Could not connect to backend: {status_msg}")
                        
                        # Offer to restart the backend
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if st.button("🔄 Start Backend"):
                                success, msg = try_start_backend_server()
                                if success:
                                    st.info(f"{msg}. Please wait a moment and try again.")
                                    time.sleep(3)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with col2:
                            st.code("cd c:\\Users\\USER\\Desktop\\vit_project\\model_folder && python app.py")
                        
                        st.stop()
                    
                    # Send to backend with retry logic
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            with open(temp_image_path, "rb") as f:
                                files = {"file": (os.path.basename(temp_image_path), f, "image/png")}
                                payload = {
                                    "plot_width": st.session_state.get("plot_width", 0.0),
                                    "plot_length": st.session_state.get("plot_length", 0.0)
                                }
                                detection_response = requests.post(
                                    "http://127.0.0.1:5000/detect-walls-rooms", 
                                    files=files, 
                                    data=payload,
                                    timeout=120  # Increased timeout
                                )
                            break  # Success, exit retry loop
                        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                            if attempt < max_retries - 1:
                                st.warning("Connection issue detected. Retrying...")
                                time.sleep(2)
                                continue
                            else:
                                raise  # Re-raise the exception if all retries failed
                    
                    if detection_response.status_code == 200:
                        result = detection_response.json()
                        
                        # Full width header with icon for results
                        st.markdown("""
                        <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:15px;">
                            <h2 style="color:#1E88E5;margin-bottom:0;text-align:center;">🔍 Wall & Room Detection Results</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detection statistics in modern cards
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("🧱 Walls Detected", result['wall_count'])
                        with col_stat2:
                            st.metric("🏠 Rooms Detected", result['room_count'])
                        
                        # Display the visualization from the backend
                        vis_url = f"http://127.0.0.1:5000{result['visualization']}"
                        
                        try:
                            # Fetch and display the image with improved error handling and larger size
                            st.image(vis_url, caption="Wall and Room Detection Results", use_container_width=True, width=800)
                        except Exception as img_error:
                            st.warning(f"Could not load visualization image. Error: {img_error}")
                            
                            # Alternative: Try to download and display locally
                            try:
                                with st.spinner("Downloading visualization image..."):
                                    vis_response = requests.get(vis_url, timeout=30)
                                    if vis_response.status_code == 200:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_tmp:
                                            img_tmp.write(vis_response.content)
                                            img_path = img_tmp.name
                                        
                                        st.image(img_path, caption="Wall and Room Detection Results", use_container_width=True)
                                        try:
                                            os.unlink(img_path)
                                        except:
                                            pass
                                    else:
                                        st.error(f"Failed to download visualization: HTTP {vis_response.status_code}")
                            except Exception as local_err:
                                st.error(f"Could not display visualization: {local_err}")
                        
                        # Store detection results in session state
                        st.session_state.wall_count = result['wall_count']
                        st.session_state.room_count = result['room_count']
                        
                        # Beautiful room type breakdown section
                        st.markdown("""
                        <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;margin-top:20px;margin-bottom:10px;">
                            <h3 style="color:#1E88E5;margin-bottom:0;">🏘️ Room Types</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display room counts by type in a nice grid
                        room_types = ["Living Room", "Bedroom", "Bathroom", "Kitchen", "Room"]
                        room_icons = {"Living Room": "🛋️", "Bedroom": "🛏️", "Bathroom": "🚿", "Kitchen": "🍳", "Room": "🚪"}
                        
                        # Get room counts if available
                        room_type_counts = result.get('room_type_counts', {})
                        
                        # Create columns for room type counters
                        cols = st.columns(len(room_types))
                        for i, room_type in enumerate(room_types):
                            count = room_type_counts.get(room_type, 0)
                            if count > 0:
                                cols[i].metric(f"{room_icons[room_type]} {room_type}s", count)
                        
                        # Beautiful detailed room list with expandable sections
                        if 'rooms' in result and result['rooms']:
                            # Group rooms by type for organized display
                            rooms_by_type = {}
                            for room in result['rooms']:
                                room_type = room['type']
                                if room_type not in rooms_by_type:
                                    rooms_by_type[room_type] = []
                                rooms_by_type[room_type].append(room)
                            
                            # Display rooms organized by type in expandable sections
                            st.markdown("""
                            <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;margin-top:20px;margin-bottom:10px;">
                                <h3 style="color:#1E88E5;margin-bottom:0;">📋 Room Details</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Define room icons consistent with the backend
                            room_icons = {
                                "Living Room": "🛋️", 
                                "Bedroom": "🛏️", 
                                "Bathroom": "🚿", 
                                "Kitchen": "🍳", 
                                "Room": "🚪"
                            }
                            
                            # FIXED: COMPLETELY REMOVED HTML-BASED SUMMARY - Using only Streamlit components
                            if room_type_counts:
                                room_types_with_counts = [rt for rt, count in room_type_counts.items() if count > 0]
                                if len(room_types_with_counts) > 0:
                                    cols = st.columns(len(room_types_with_counts))
                                    for i, room_type in enumerate(room_types_with_counts):
                                        count = room_type_counts[room_type]
                                        icon = room_icons.get(room_type, "🔹")
                                        cols[i].metric(f"{icon} {room_type}s", count)
                            
                            # Create tabbed interface for different room types
                            if rooms_by_type:
                                room_types_with_data = [rt for rt in rooms_by_type if rooms_by_type[rt]]
                                if room_types_with_data:
                                    tabs = st.tabs([f"{room_icons.get(rt, '🔹')} {rt}s ({len(rooms_by_type[rt])})" 
                                                   for rt in room_types_with_data])
                                    
                                    for i, room_type in enumerate(room_types_with_data):
                                        with tabs[i]:
                                            # Get room features and description from room_config
                                            room_config = result.get('room_config', {}).get(room_type, {})
                                            features = room_config.get('features', [])
                                            description = room_config.get('description', "")
                                            
                                            if description:
                                                st.info(f"**{room_type}**: {description}")
                                            
                                            # Use pure Streamlit components for features display
                                            if features:
                                                icon = room_icons.get(room_type, "🔹")
                                                st.write(f"**{icon} Typical features:** {', '.join(features)}")
                                            
                                            # Create a compact grid layout for rooms 
                                            num_rooms = len(rooms_by_type[room_type])
                                            num_cols = min(4, max(2, num_rooms // 3 + 1))
                                            cols = st.columns(num_cols)
                                            
                                            for j, room in enumerate(rooms_by_type[room_type]):
                                                col_idx = j % num_cols
                                                with cols[col_idx]:
                                                    room_id = room.get('type_id', room.get('id', ''))
                                                    area = room['area']
                                                    icon = room.get('icon', room_icons.get(room_type, "🔹"))
                                                    
                                                    # Area is already calibrated square meters
                                                    area_sqm = area
                                                    
                                                    # Get room bounds for visualization
                                                    x, y, width, height = room['bounds']
                                                    
                                                    # Use built-in Streamlit components only
                                                    st.write(f"**{icon} {room_type} {room_id}**")
                                                    
                                                    # Create a more elegant visual representation of the room
                                                    room_color = room_colors.get(room_type, "#80B1D3")
                                                    accent_color = "#1E88E5"  # Blue accent color for all rooms
                                                    room_box_html = f"""
                                                    <div style="
                                                        width: 100%;
                                                        border-radius: 8px;
                                                        margin-bottom: 15px;
                                                        overflow: hidden;
                                                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                                                        border: 1px solid #e0e0e0;
                                                    ">
                                                        <div style="
                                                            padding: 12px;
                                                            display: flex;
                                                            background-color: white;
                                                        ">
                                                            <div style="
                                                                width: 8px;
                                                                background-color: {room_color};
                                                                margin-right: 12px;
                                                                border-radius: 4px;
                                                            "></div>
                                                            <div style="flex-grow: 1;">
                                                                <div style="
                                                                    display: flex;
                                                                    justify-content: space-between;
                                                                    align-items: center;
                                                                    margin-bottom: 8px;
                                                                ">
                                                                    <div style="
                                                                        font-weight: bold;
                                                                        font-size: 1.1em;
                                                                        color: #333;
                                                                    ">{area_sqm:.1f} m²</div>
                                                                    <div style="
                                                                        font-size: 0.9em;
                                                                        color: #666;
                                                                        padding: 2px 8px;
                                                                        background-color: #f5f5f5;
                                                                        border-radius: 12px;
                                                                    ">{width} × {height} px</div>
                                                                </div>
                                                                <div style="
                                                                    display: flex;
                                                                    justify-content: space-between;
                                                                    font-size: 0.85em;
                                                                    color: #666;
                                                                ">
                                                                    <span>Position: X={x}, Y={y}</span>
                                                                    <span>Ratio: {room.get('aspect_ratio', 0):.2f}</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    """
                                                    st.markdown(room_box_html, unsafe_allow_html=True)
                                                    
                                                    # No need for additional captions as all info is in the box
                            
                        # Prompt for next step with an appealing call-to-action
                        st.success("✅ Wall and room detection complete! You can now proceed to generate the 3D model.")
                        
                        # Add a button to return to the original view
                        if st.button("↩️ Return to Selection"):
                            st.rerun()
                        
                    else:
                        try:
                            error_data = detection_response.json()
                            error_message = error_data.get('error', 'Unknown error')
                            st.error(f"Detection failed: {error_message}")
                            if 'traceback' in error_data:
                                with st.expander("Error Details", expanded=False):
                                    st.code(error_data['traceback'], language='python')
                        except:
                            st.error(f"Detection failed with status code {detection_response.status_code}: {detection_response.text}")
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Wall detection request timed out. The server might be busy or the image too large.")
                    st.info("Try using a smaller image or wait a few moments before trying again.")
                    
                    # Offer to restart backend
                    if st.button("🔄 Restart Backend Server"):
                        success, msg = try_start_backend_server()
                        if success:
                            st.info(f"{msg}. Please try again shortly.")
                        else:
                            st.error(msg)
                            
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection to backend was lost. The server may have crashed or the connection was reset.")
                    
                    # Add a convenient button to try restarting the backend
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("🔄 Restart Backend"):
                            success, msg = try_start_backend_server()
                            if success:
                                st.info(f"{msg}. Please try again after a few seconds.")
                                time.sleep(3)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col2:
                        st.code("cd c:\\Users\\USER\\Desktop\\vit_project\\model_folder && python app.py")
                
                except Exception as e:
                    import traceback  # Import here as well for safety
                    st.error(f"❌ Wall detection error: {e}")
                    with st.expander("Error Details", expanded=False):
                        st.code(traceback.format_exc(), language='python')
    
    with col2:
        # 3D Model Conversion button with sequential design
        if st.button("2️⃣ Generate 3D Model"):
            # Close the columns layout to allow full-width display
            st.write("---")
            
            with st.spinner("Generating 3D model, please wait..."):
                try:
                    # First verify if the backend is running with improved error handling
                    connected, status_msg = check_backend_connection(retry_count=3, timeout=5)
                    if not connected:
                        st.error(f"Could not connect to backend: {status_msg}")
                        
                        # Offer to restart the backend
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if st.button("🔄 Start Backend"):
                                success, msg = try_start_backend_server()
                                if success:
                                    st.info(f"{msg}. Please wait a moment and try again.")
                                    time.sleep(3)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with col2:
                            st.code("cd c:\\Users\\USER\\Desktop\\vit_project\\model_folder && python app.py")
                        
                        st.stop()
                    
                    # If we reach here, backend is running, so continue with the request
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_image_path = tmp_file.name

                    with open(temp_image_path, "rb") as f:
                        files = {"file": (os.path.basename(temp_image_path), f, "image/png")}
                        payload = {
                            "plot_width": st.session_state.get("plot_width", 0.0),
                            "plot_length": st.session_state.get("plot_length", 0.0)
                        }
                        response = requests.post("http://127.0.0.1:5000/generate-3d", files=files, data=payload, timeout=180)
                    
                    if response.status_code == 200:
                        data = response.json()
                        ply_path = data.get("3d_model")
                        floorplan_data = data.get("floorplan_data", {})
                        
                        if ply_path:
                            # Full width header for 3D results
                            st.markdown("""
                            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:15px;">
                                <h2 style="color:#1E88E5;margin-bottom:0;text-align:center;">🏗️ 3D Model & Construction Cost Results</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display detected room information in a more prominent layout
                            st.subheader("Detected Floorplan Features")
                            
                            # Use 3 columns for a more balanced layout
                            c1, c2, c3 = st.columns(3)
                            
                            with c1:
                                st.metric("Total Rooms", floorplan_data.get("rooms", "N/A"))
                            with c2:
                                st.metric("Estimated Area (m²)", floorplan_data.get("estimated_area", "N/A"))
                            with c3:
                                room_counts = floorplan_data.get("room_counts", {})
                                bedrooms = room_counts.get("bedroom", 0)
                                bathrooms = room_counts.get("bathroom", 0)
                                st.metric("Bedrooms / Bathrooms", f"{bedrooms} / {bathrooms}")
                            
                            # Calculate construction cost
                            area = floorplan_data.get("estimated_area", 0)
                            
                            if area > 0:
                                st.subheader("Estimated Construction Cost (India)")
                                
                                # Input selectors for cost estimation
                                c_col1, c_col2, c_col3 = st.columns(3)
                                with c_col1:
                                    city_tier = st.selectbox(
                                        "City Type", 
                                        options=["Tier-2 (Urban)", "Tier-1 (Metro)", "Tier-3 (Rural)"],
                                        help="Tier-1: Metro cities like Mumbai, Delhi, Bangalore. Tier-2: Cities like Jaipur, Pune. Tier-3: Towns/Rural areas."
                                    )
                                with c_col2:
                                    quality_level = st.selectbox(
                                        "Material Quality",
                                        options=["Standard (Medium)", "Basic (Low Cost)", "Premium (High Quality)"],
                                        help="Basic: Local materials. Standard: Branded tiles/fixtures. Premium: Italian marble, wooden flooring, modular features."
                                    )
                                with c_col3:
                                    num_floors = st.number_input(
                                        "Number of Floors",
                                        min_value=1, max_value=5, value=1, step=1
                                    )
                                
                                try:
                                    cost_response = requests.post(
                                        "http://127.0.0.1:5000/estimate-construction-cost",
                                        json={
                                            "area": area,
                                            "bedrooms": bedrooms,
                                            "bathrooms": bathrooms,
                                            "kitchen": floorplan_data.get("room_counts", {}).get("kitchen", 1),
                                            "living": floorplan_data.get("room_counts", {}).get("living", 1),
                                            "city_tier": city_tier,
                                            "quality_level": quality_level,
                                            "num_floors": num_floors
                                        },
                                        timeout=10
                                    )
                                    
                                    if cost_response.status_code == 200:
                                        cost_data = cost_response.json()
                                        estimated_cost = cost_data.get("estimated_cost", 0)
                                        breakdown = cost_data.get("breakdown", {})
                                        materials = cost_data.get("materials", {})
                                        comparison = cost_data.get("comparison", {})
                                        description = cost_data.get("description", "")
                                        cost_per_sqft = cost_data.get("cost_per_sqft", 0)
                                        area_sqft = cost_data.get("area_sqft", 0)
                                        
                                        # Display Currency in lakhs/crores
                                        def format_indian_currency(num):
                                            if num >= 10000000:
                                                return f"₹ {num / 10000000:.2f} Crores"
                                            elif num >= 100000:
                                                return f"₹ {num / 100000:.2f} Lakhs"
                                            else:
                                                return f"₹ {num:,.2f}"
                                                
                                        st.success(f"🏗️ Total Estimated Construction Cost: **{format_indian_currency(estimated_cost)}**")
                                        st.info(f"**Specifications**: {description} (Estimated Area: {area:.1f} m² / {area_sqft:.1f} sqft at ~₹{cost_per_sqft:.0f}/sqft)")
                                        
                                        # Render Pie Chart of cost breakdown
                                        try:
                                            import matplotlib
                                            matplotlib.use('Agg')
                                            import matplotlib.pyplot as plt
                                            
                                            fig, ax = plt.subplots(figsize=(6, 4))
                                            labels = list(breakdown.keys())
                                            sizes = list(breakdown.values())
                                            
                                            # Filter out zero costs
                                            non_zero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
                                            if non_zero:
                                                labels, sizes = zip(*non_zero)
                                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                                                       colors=['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69'])
                                                ax.axis('equal')
                                                plt.title("Cost Breakdown")
                                                st.pyplot(fig)
                                            plt.close(fig)
                                        except Exception as chart_err:
                                            st.warning(f"Could not render breakdown chart: {chart_err}")
                                            
                                        # Display Materials and quantities
                                        with st.expander("🛠️ Estimated Raw Material Requirements"):
                                            st.write("These are approximate material quantities required for construction:")
                                            for mat_name, mat_info in materials.items():
                                                st.markdown(f"- **{mat_name}**: {mat_info['quantity']:,} {mat_info['unit']} (Est. Cost: {format_indian_currency(mat_info['cost'])})")
                                        
                                        # Display quality tier comparison
                                        with st.expander("📊 Cost Comparison across Quality Tiers"):
                                            st.write("Estimated costs for other material quality levels:")
                                            for qual_name, qual_cost in comparison.items():
                                                selected_marker = "👈 (Selected)" if qual_name == quality_level else ""
                                                st.markdown(f"- **{qual_name}**: {format_indian_currency(qual_cost)} {selected_marker}")
                                                
                                    else:
                                        st.error(f"Failed to estimate cost: {cost_response.text}")
                                        
                                except Exception as e:
                                    st.error(f"Error estimating construction cost: {e}")
                                    
                                # Ensure ply_url is correct and does not duplicate /static/
                                if ply_path.startswith("/static/"):
                                    ply_url = f"http://127.0.0.1:5000{ply_path}"
                                else:
                                    ply_url = f"http://127.0.0.1:5000/static/{os.path.basename(ply_path)}"
                                
                                # Download button
                                ply_content = requests.get(ply_url).content
                                try:
                                    st.download_button(
                                        label="Download 3D Model (.ply)",
                                        data=ply_content,
                                        file_name="floorplan_3d_model.ply",
                                        mime="application/octet-stream"
                                    )
                                except Exception as e:
                                    st.error(f"Error creating download button: {e}")
                                
                                # Create base64 encoded version of the PLY content to avoid CORS issues
                                ply_base64 = base64.b64encode(ply_content).decode('utf-8')
                                
                                st.markdown("### 3D Model Viewer")
                                
                                # Create a simpler 3D viewer that doesn't rely on external files
                                viewer_html = f"""
                                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
                                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>
                                
                                <div id="viewer" style="width:100%;height:500px;background:#222;position:relative;"></div>
                                <div id="status" style="position:absolute;bottom:10px;left:10px;color:white;background:rgba(0,0,0,0.7);padding:10px;border-radius:5px;z-index:1000;">Loading model...</div>
                                
                                <script>
                                    // Initialize Three.js scene
                                    const container = document.getElementById('viewer');
                                    const status = document.getElementById('status');
                                    
                                    const scene = new THREE.Scene();
                                    scene.background = new THREE.Color(0x222222);
                                    
                                    const camera = new THREE.PerspectiveCamera(75, container.clientWidth/container.clientHeight, 0.1, 1000);
                                    camera.position.set(0, 5, 10);
                                    
                                    const renderer = new THREE.WebGLRenderer({{antialias: true}});
                                    renderer.setSize(container.clientWidth, container.clientHeight);
                                    container.appendChild(renderer.domElement);
                                    
                                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                                    scene.add(ambientLight);
                                    
                                    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                                    directionalLight.position.set(1, 1, 1);
                                    scene.add(directionalLight);
                                    
                                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                                    controls.enableDamping = true;
                                    controls.dampingFactor = 0.25;
                                    
                                    // Function to decode base64 to array buffer
                                    function base64ToArrayBuffer(base64) {{
                                        const binary_string = window.atob(base64);
                                        const len = binary_string.length;
                                        const bytes = new Uint8Array(len);
                                        for (let i = 0; i < len; i++) {{
                                            bytes[i] = binary_string.charCodeAt(i);
                                        }}
                                        return bytes.buffer;
                                    }}
                                    
                                    // Load model from embedded base64 data to avoid CORS issues
                                    const loader = new THREE.PLYLoader();
                                    function loadPLY() {{
                                        try {{
                                            status.textContent = 'Preparing model data...';
                                            
                                            // Use the base64 encoded PLY data directly
                                            const plyData = base64ToArrayBuffer("{ply_base64}");
                                            
                                            // Parse PLY data directly from array buffer
                                            const geometry = loader.parse(plyData);
                                            geometry.computeVertexNormals();
                                            
                                            const material = new THREE.MeshStandardMaterial({{
                                                color: 0xffffff,
                                                flatShading: true,
                                                side: THREE.DoubleSide
                                            }});
                                            
                                            const mesh = new THREE.Mesh(geometry, material);
                                            
                                            // Center the model
                                            geometry.computeBoundingBox();
                                            const center = geometry.boundingBox.getCenter(new THREE.Vector3());
                                            mesh.position.x = -center.x;
                                            mesh.position.y = -center.y;
                                            mesh.position.z = -center.z;
                                            
                                            scene.add(mesh);
                                            
                                            // Adjust camera to fit model
                                            const box = new THREE.Box3().setFromObject(mesh);
                                            const size = box.getSize(new THREE.Vector3());
                                            const maxDim = Math.max(size.x, size.y, size.z);
                                            camera.position.z = maxDim * 2;
                                            
                                            status.textContent = 'Model loaded successfully. Use mouse to rotate, scroll to zoom.';
                                            setTimeout(() => {{ status.style.opacity = '0'; }}, 3000);
                                        }} catch(error) {{
                                            console.error('Error loading PLY:', error);
                                            status.textContent = 'Error: ' + error.message;
                                        }}
                                    }}
                                    
                                    // Handle window resize
                                    window.addEventListener('resize', () => {{
                                        camera.aspect = container.clientWidth / container.clientHeight;
                                        camera.updateProjectionMatrix();
                                        renderer.setSize(container.clientWidth, container.clientHeight);
                                    }});
                                    
                                    // Animation loop
                                    function animate() {{
                                        requestAnimationFrame(animate);
                                        controls.update();
                                        renderer.render(scene, camera);
                                    }}
                                    
                                    // Start everything
                                    loadPLY();
                                    animate();
                                </script>
                                """
                                
                                try:
                                    st.components.v1.html(viewer_html, height=520)
                                    st.info("You can also download and view the .ply file in MeshLab, Blender, or other tools.")
                                except Exception as viewer_error:
                                    st.error(f"Error in 3D viewer: {viewer_error}")
                                    st.info("Please download the PLY file using the button above and view it in an external 3D viewer.")
                                
                            # Add a button to return to the original view
                            if st.button("↩️ Return to Selection"):
                                st.rerun()
                        
                        else:
                            st.error("3D model file not found or not generated.")
                    else:
                        try:
                            err = response.json()
                            st.error(f"3D model generation failed: {err.get('error', response.text)}")
                            if 'stderr' in err:
                                st.error(f"STDERR: {err['stderr']}")
                            if 'stdout' in err:
                                st.error(f"STDOUT: {err['stdout']}")
                            if 'traceback' in err:
                                st.error(f"Traceback: {err['traceback']}")
                        except Exception:
                            st.error(f"3D model generation failed: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("3D model generation timed out. Try a smaller image.")
                except requests.exceptions.ConnectionError:
                    st.error("Connection to backend was lost. Make sure the Flask server is still running.")
                    st.info("To start the Flask server, open a terminal and run: `cd c:\\Users\\USER\\Desktop\\vit_project\\model_folder && python app.py`")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    
    # Clean up temp image file
    try:
        os.remove(temp_image_path)
    except Exception:
        pass

def display_footer():
    """Display a beautiful footer with a made-with-love message"""
    # First add a divider
    st.markdown("---")
    
    # Create footer with columns for better layout
    footer_cols = st.columns([1, 3, 1])
    
    # Middle column with the main message
    with footer_cols[1]:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px;">
                <p style="color: #5A5A5A; font-size: 0.9em; margin-bottom: 5px;">
                    Made with ❤️ for architects and home builders
                </p>
                <p style="color: #757575; font-size: 0.8em; font-style: italic;">
                    Turn your floorplans into intelligent 3D models with AI
                </p>
                <p style="color: #9E9E9E; font-size: 0.7em; margin-top: 15px;">
                    © 2026 House Construction Cost Predictor | All Rights Reserved
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
# Call the footer function at the end of the file
display_footer()
