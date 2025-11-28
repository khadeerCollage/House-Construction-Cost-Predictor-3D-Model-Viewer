import streamlit as st
import requests
import tempfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import time
import traceback  # Add this import to fix traceback error
import subprocess
import platform
import socket
import threading
import sys
import json
import pandas as pd

# Flask imports
from flask import Flask, request as flask_request, jsonify, send_from_directory

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

# ==================== INTEGRATED FLASK SERVER ====================
# Create Flask app - disable default static handling so we can use our custom route
model_folder_path = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.join(model_folder_path, 'static')
os.makedirs(static_folder_path, exist_ok=True)

flask_app = Flask(__name__, static_folder=static_folder_path, static_url_path='/static')

# CORS for local development
try:
    from flask_cors import CORS
    CORS(flask_app)
except ImportError:
    pass

# Load models at startup
try:
    model_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_folder, "house_cost_model.pkl")
    import joblib
    cost_model = joblib.load(model_path) if os.path.exists(model_path) else None
except Exception as e:
    cost_model = None
    print(f"Error loading house_cost_model.pkl: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@flask_app.route('/')
def home():
    return "Welcome to House Cost Prediction API! Use POST /predict to get predictions."

@flask_app.route('/predict', methods=['POST'])
def predict():
    if cost_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    data = flask_request.get_json()
    if not data:
        return jsonify({'error': 'No input data'}), 400
    try:
        features = pd.DataFrame([data])
        prediction = cost_model.predict(features)
        estimated_cost = float(prediction[0])
        return jsonify({'estimated_cost': round(estimated_cost, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@flask_app.route('/generate-3d', methods=['POST'])
def generate_3d():
    if 'file' not in flask_request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = flask_request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        model_folder = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(model_folder, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        img_path = os.path.join(temp_dir, file.filename)
        file.save(img_path)
        script_path = os.path.join(model_folder, '2d-3d.py')
        output_model_path = os.path.join(temp_dir, 'floorplan_3d_model.ply')
        output_data_path = os.path.join(temp_dir, 'floorplan_data.json')
        
        try:
            result = subprocess.run(
                [sys.executable, script_path, '--input', img_path, '--output', output_model_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
            if result.returncode != 0:
                return jsonify({
                    'error': '3D model generation failed.',
                    'stderr': result.stderr,
                    'stdout': result.stdout
                }), 500
                
        except subprocess.TimeoutExpired:
            return jsonify({'error': '3D model generation timed out. Try a smaller image.'}), 500
        except Exception as e:
            return jsonify({'error': f'Unexpected error: {str(e)}', 'traceback': traceback.format_exc()}), 500
            
        if not os.path.exists(output_model_path):
            return jsonify({'error': '3D model file not generated'}), 500
            
        # Move to static for frontend viewing
        static_dir = os.path.join(model_folder, 'static')
        os.makedirs(static_dir, exist_ok=True)
        static_ply_path = os.path.join(static_dir, os.path.basename(output_model_path))
        
        # Extract detected rooms and metrics from output
        floorplan_data = {}
        
        try:
            if detect_rooms_and_walls:
                detection_result = detect_rooms_and_walls(img_path, output_dir=static_dir, save_visualization=False)
                total_rooms = detection_result.get('total_rooms', 0)
                room_counts = detection_result.get('room_counts', {})
                
                if not room_counts and total_rooms > 0:
                    bedrooms = max(1, int(total_rooms * 0.2))
                    bathrooms = max(1, int(total_rooms * 0.06))
                    kitchen = 1
                    living = 1
                    other = total_rooms - (bedrooms + bathrooms + kitchen + living)
                    room_counts = {"bedroom": bedrooms, "bathroom": bathrooms, "kitchen": kitchen, "living": living, "other": other}
                
                floorplan_data["room_counts"] = room_counts
                floorplan_data["rooms"] = total_rooms
                
                if "estimated_area" in detection_result and detection_result["estimated_area"] > 0:
                    floorplan_data["estimated_area"] = detection_result["estimated_area"]
                else:
                    floorplan_data["estimated_area"] = max(total_rooms * 6, 30)
            else:
                floorplan_data = {"rooms": 4, "room_counts": {"bedroom": 1, "bathroom": 1, "kitchen": 1, "living": 1}, "estimated_area": 30}
                
            with open(output_data_path, 'w') as f:
                json.dump(floorplan_data, f)
                
        except Exception as e:
            print(f"Error estimating room metrics: {e}")
            floorplan_data = {"rooms": 4, "room_counts": {"bedroom": 1, "bathroom": 1, "kitchen": 1, "living": 1}, "estimated_area": 30, "error": str(e)}
        
        # Copy PLY file to static dir
        with open(output_model_path, "rb") as src, open(static_ply_path, "wb") as dst:
            dst.write(src.read())
            
        return jsonify({
            '3d_model': f"/static/{os.path.basename(output_model_path)}",
            'floorplan_data': floorplan_data
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@flask_app.route('/detect-walls-rooms', methods=['POST'])
def detect_walls_rooms():
    if 'file' not in flask_request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = flask_request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        model_folder = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(model_folder, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = int(time.time())
        unique_filename = f'{timestamp}_{file.filename}'
        img_path = os.path.join(temp_dir, unique_filename)
        file.save(img_path)
        
        try:
            static_dir = os.path.join(model_folder, 'static')
            os.makedirs(static_dir, exist_ok=True)
            
            if detect_rooms_and_walls:
                result = detect_rooms_and_walls(img_path, output_dir=static_dir)
                
                vis_path = result.get('visualization')
                if vis_path:
                    # Use /files/ route which we handle explicitly
                    filename = os.path.basename(vis_path)
                    rel_path = f'/files/{filename}'
                    result['visualization'] = rel_path
                    print(f"Visualization saved to: {vis_path}")
                    print(f"Visualization URL path: {rel_path}")
                    
                result['room_config'] = ROOM_CONFIG
            else:
                result = {'error': 'Detection function not available', 'wall_count': 0, 'room_count': 0}
            
            try:
                os.remove(img_path)
            except:
                pass
            
            return jsonify(result)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Wall detection error: {str(e)}\n{error_trace}")
            return jsonify({'error': str(e), 'traceback': error_trace}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@flask_app.route('/files/<path:filename>')
def serve_files(filename):
    """Serve files from static folder using /files/ route to avoid conflicts"""
    file_path = os.path.join(static_folder_path, filename)
    
    print(f"Serving file: {filename}")
    print(f"Full path: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {filename}'}), 404
    
    return send_from_directory(static_folder_path, filename)

@flask_app.route('/estimate-construction-cost', methods=['POST'])
def estimate_construction_cost():
    data = flask_request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
        
    try:
        area = data.get('area', 0)
        bedrooms = data.get('bedrooms', 0)
        bathrooms = data.get('bathrooms', 0)
        kitchen = data.get('kitchen', 0) 
        living = data.get('living', 0)
        location_factor = data.get('location_factor', 1.0)
        
        base_cost_per_sqm = 1200
        bedroom_cost = bedrooms * 10000
        bathroom_cost = bathrooms * 15000
        kitchen_cost = kitchen * 20000
        living_cost = living * 8000
        
        total_area_cost = area * base_cost_per_sqm
        total_room_cost = bedroom_cost + bathroom_cost + kitchen_cost + living_cost
        final_cost = (total_area_cost + total_room_cost) * location_factor
        
        return jsonify({
            'estimated_cost': round(final_cost, 2),
            'breakdown': {
                'base_area_cost': round(total_area_cost, 2),
                'bedroom_cost': round(bedroom_cost, 2),
                'bathroom_cost': round(bathroom_cost, 2),
                'kitchen_cost': round(kitchen_cost, 2),
                'living_cost': round(living_cost, 2),
                'location_factor': location_factor
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error calculating cost: {str(e)}', 'traceback': traceback.format_exc()}), 500

@flask_app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

def run_flask_server():
    """Run Flask server in a thread"""
    model_folder = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(model_folder, 'static')
    os.makedirs(static_dir, exist_ok=True)
    flask_app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

def start_flask_in_background():
    """Start Flask server in a background thread"""
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    print("Flask server started in background thread on http://127.0.0.1:5000/")

def ensure_backend_running():
    """Check if backend is running, start it if not"""
    try:
        response = requests.get("http://127.0.0.1:5000/", timeout=2)
        if response.status_code == 200:
            return True  # Already running
    except:
        pass
    
    # Not running, start it in background thread
    start_flask_in_background()
    
    # Wait for it to start (max 10 seconds)
    for i in range(10):
        time.sleep(0.5)
        try:
            response = requests.get("http://127.0.0.1:5000/", timeout=2)
            if response.status_code == 200:
                return True
        except:
            continue
    
    return False

# Auto-start Flask when frontend loads (only once per session)
if 'flask_started' not in st.session_state:
    st.session_state.flask_started = ensure_backend_running()

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
        # Start Flask in background thread (integrated)
        start_flask_in_background()
        time.sleep(2)
        return True, "Backend server started in background"
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
                        if st.button("🔄 Start Backend", key="start_backend_1"):
                            success, msg = try_start_backend_server()
                            if success:
                                st.info(f"{msg}. Please wait a moment and try again.")
                                time.sleep(3)
                                st.rerun()
                            else:
                                st.error(msg)
                        
                        st.stop()
                    
                    # Send to backend with retry logic
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            with open(temp_image_path, "rb") as f:
                                files = {"file": (os.path.basename(temp_image_path), f, "image/png")}
                                detection_response = requests.post(
                                    "http://127.0.0.1:5000/detect-walls-rooms", 
                                    files=files, 
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
                        vis_path = result.get('visualization')
                        if vis_path:
                            vis_url = f"http://127.0.0.1:5000{vis_path}"
                            print(f"Visualization URL: {vis_url}")
                            
                            # Always download the image first since st.image() doesn't work well with localhost URLs
                            try:
                                vis_response = requests.get(vis_url, timeout=30)
                                if vis_response.status_code == 200:
                                    # Display image directly from bytes
                                    st.image(vis_response.content, caption="Wall and Room Detection Results", use_container_width=True)
                                else:
                                    st.error(f"Failed to load visualization image: HTTP {vis_response.status_code}")
                                    # Show more details
                                    try:
                                        error_detail = vis_response.json()
                                        st.error(f"Error details: {error_detail}")
                                    except:
                                        st.error(f"Response: {vis_response.text[:500]}")
                            except Exception as img_error:
                                st.warning(f"Could not load visualization image. Error: {img_error}")
                        else:
                            st.warning("No visualization path returned from detection")
                        
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
                                                    
                                                    # Calculate area in square meters (roughly)
                                                    area_sqm = area / 1000  # Approximate conversion
                                                    
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
                    if st.button("🔄 Restart Backend", key="restart_backend_1"):
                        success, msg = try_start_backend_server()
                        if success:
                            st.info(f"{msg}. Please try again after a few seconds.")
                            time.sleep(3)
                            st.rerun()
                        else:
                            st.error(msg)
                
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
                        if st.button("🔄 Start Backend", key="start_backend_2"):
                            success, msg = try_start_backend_server()
                            if success:
                                st.info(f"{msg}. Please wait a moment and try again.")
                                time.sleep(3)
                                st.rerun()
                            else:
                                st.error(msg)
                        
                        st.stop()
                    
                    # If we reach here, backend is running, so continue with the request
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_image_path = tmp_file.name

                    with open(temp_image_path, "rb") as f:
                        files = {"file": (os.path.basename(temp_image_path), f, "image/png")}
                        response = requests.post("http://127.0.0.1:5000/generate-3d", files=files, timeout=180)
                    
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
                                st.subheader("Estimated Construction Cost")
                                location_options = {
                                    "Urban (High Cost)": 1.3,
                                    "Suburban": 1.0,
                                    "Rural (Low Cost)": 0.8
                                }
                                location = st.selectbox("Location Type", options=list(location_options.keys()))
                                location_factor = location_options[location]
                                
                                try:
                                    cost_response = requests.post(
                                        "http://127.0.0.1:5000/estimate-construction-cost",
                                        json={
                                            "area": area,
                                            "bedrooms": bedrooms,
                                            "bathrooms": bathrooms,
                                            "kitchen": floorplan_data.get("room_counts", {}).get("kitchen", 0),
                                            "living": floorplan_data.get("room_counts", {}).get("living", 0),
                                            "location_factor": location_factor
                                        },
                                        timeout=10
                                    )
                                    
                                    if cost_response.status_code == 200:
                                        cost_data = cost_response.json()
                                        estimated_cost = cost_data.get("estimated_cost", 0)
                                        breakdown = cost_data.get("breakdown", {})
                                        
                                        st.success(f"🏗️ Total Estimated Construction Cost: ${estimated_cost:,.2f}")
                                        
                                        with st.expander("Cost Breakdown"):
                                            st.write(f"Base Area Cost (${area} m²): ${breakdown.get('base_area_cost', 0):,.2f}")
                                            st.write(f"Bedroom Cost ({bedrooms}): ${breakdown.get('bedroom_cost', 0):,.2f}")
                                            st.write(f"Bathroom Cost ({bathrooms}): ${breakdown.get('bathroom_cost', 0):,.2f}")
                                            st.write(f"Kitchen Cost ({floorplan_data.get('room_counts', {}).get('kitchen', 0)}): ${breakdown.get('kitchen_cost', 0):,.2f}")
                                            st.write(f"Living Room Cost ({floorplan_data.get('room_counts', {}).get('living', 0)}): ${breakdown.get('living_cost', 0):,.2f}")
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
                    st.error("Connection to backend was lost. Trying to restart...")
                    success, msg = try_start_backend_server()
                    if success:
                        st.info(f"{msg}. Please try again.")
                    else:
                        st.error(msg)
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
                    © 2025 House Construction Cost Predictor | All Rights Reserved
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Call the footer function at the end of the file
display_footer()
