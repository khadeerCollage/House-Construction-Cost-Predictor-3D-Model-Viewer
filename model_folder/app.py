from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os
import subprocess
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches

try:
    from utils import detect_rooms_and_walls, ROOM_CONFIG
except ImportError:
    from model_folder.utils import detect_rooms_and_walls, ROOM_CONFIG

app = Flask(__name__, static_folder='static')

# CORS for local development (so Streamlit can access)
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass

# Load models at startup
try:
    model = joblib.load("house_cost_model.pkl")
except Exception as e:
    model = None
    print(f"Error loading house_cost_model.pkl: {e}")

# No YOLO detection
yolo_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return "Welcome to House Cost Prediction API! Use POST /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data'}), 400
    try:
        features = pd.DataFrame([data])
        prediction = model.predict(features)
        estimated_cost = float(prediction[0])
        return jsonify({'estimated_cost': round(estimated_cost, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-3d', methods=['POST'])
def generate_3d():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        os.makedirs('temp', exist_ok=True)
        img_path = os.path.join('temp', file.filename)
        file.save(img_path)
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '2d-3d.py'))
        output_model_path = os.path.join('temp', 'floorplan_3d_model.ply')
        output_data_path = os.path.join('temp', 'floorplan_data.json')
        
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
            import traceback
            return jsonify({'error': f'Unexpected error: {str(e)}', 'traceback': traceback.format_exc()}), 500
            
        if not os.path.exists(output_model_path):
            return jsonify({'error': '3D model file not generated'}), 500
            
        # Move to static for frontend viewing
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        static_ply_path = os.path.join(static_dir, os.path.basename(output_model_path))
        
        # Extract detected rooms and metrics from output
        floorplan_data = {}
        
        # Use our dedicated room detection function for consistency
        try:
            # Use the same detection function as the detect-walls-rooms endpoint
            try:
                from utils import detect_rooms_and_walls, ROOM_CONFIG
            except ImportError:
                from model_folder.utils import detect_rooms_and_walls, ROOM_CONFIG
                
            # Get room counts directly from detection - don't use hardcoded values
            detection_result = detect_rooms_and_walls(img_path, output_dir=static_dir, save_visualization=False)
            print("Detection result for 3D model:", detection_result)
            
            # Get room counts from detection
            total_rooms = detection_result.get('total_rooms', 0)
            room_counts = detection_result.get('room_counts', {})
            
            # Calculate distribution only if no specific room counts detected
            if not room_counts and total_rooms > 0:
                # Calculate a sensible distribution based on actual detection
                bedrooms = max(1, int(total_rooms * 0.2))  # ~20% bedrooms
                bathrooms = max(1, int(total_rooms * 0.06))  # ~6% bathrooms
                kitchen = 1  # Always at least one kitchen
                living = 1  # Always at least one living room
                other = total_rooms - (bedrooms + bathrooms + kitchen + living)
                
                room_counts = {
                    "bedroom": bedrooms,
                    "bathroom": bathrooms,
                    "kitchen": kitchen,
                    "living": living,
                    "other": other
                }
            
            # If room counts is still empty or no total rooms detected
            if not room_counts or total_rooms <= 0:
                # Try to calculate from contour analysis without hardcoded values
                try:
                    image = cv2.imread(img_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Count significant contours as potential rooms
                    potential_rooms = 0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 100:  # Minimum area to be considered a room
                            potential_rooms += 1
                    
                    total_rooms = max(4, potential_rooms)  # At least 4 rooms (minimum functional home)
                    
                    # Create a sensible distribution
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
                except Exception as e:
                    print(f"Error in contour analysis: {e}")
                    # If everything failed, use minimal functional values
                    total_rooms = 4
                    room_counts = {
                        "bedroom": 1,
                        "bathroom": 1,
                        "kitchen": 1,
                        "living": 1,
                        "other": 0
                    }
                    
            floorplan_data["room_counts"] = room_counts
            floorplan_data["rooms"] = total_rooms
            
            # Set area based on detection or calculate from room count
            if "estimated_area" in detection_result and detection_result["estimated_area"] > 0:
                floorplan_data["estimated_area"] = detection_result["estimated_area"]
            else:
                # Calculate based on room count - use a smaller average size to avoid inflation
                avg_room_size = 6  # Minimum average room size in m²
                calculated_area = total_rooms * avg_room_size
                
                # Ensure a minimum reasonable dwelling size
                floorplan_data["estimated_area"] = max(calculated_area, 30)
            
            # Save the floorplan data to json
            with open(output_data_path, 'w') as f:
                json.dump(floorplan_data, f)
                
            # Copy data file to static dir
            static_data_path = os.path.join(static_dir, os.path.basename(output_data_path))
            with open(output_data_path, "r") as src, open(static_data_path, "w") as dst:
                dst.write(src.read())
                
        except Exception as e:
            import traceback
            print(f"Error estimating room metrics: {e}")
            print(traceback.format_exc())
            # Try to use contour-based estimation as a last resort
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Count significant contours
                    potential_rooms = 0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 100:
                            potential_rooms += 1
                    
                    # Create minimal data
                    floorplan_data = {
                        "rooms": max(4, potential_rooms),
                        "room_counts": {
                            "bedroom": 1,
                            "bathroom": 1,
                            "kitchen": 1,
                            "living": 1,
                            "other": max(0, potential_rooms - 4)
                        },
                        "estimated_area": max(potential_rooms * 6, 30),
                        "error": str(e)
                    }
                else:
                    raise ValueError("Failed to load image")
            except Exception as e2:
                print(f"Error in contour analysis fallback: {e2}")
                # If all else fails, use absolute minimum values
                floorplan_data = {
                    "rooms": 4,
                    "room_counts": {
                        "bedroom": 1,
                        "bathroom": 1,
                        "kitchen": 1,
                        "living": 1,
                        "other": 0
                    },
                    "estimated_area": 30,
                    "error": str(e)
                }
        
        # Copy PLY file to static dir
        with open(output_model_path, "rb") as src, open(static_ply_path, "wb") as dst:
            dst.write(src.read())
            
        return jsonify({
            '3d_model': f"/static/{os.path.basename(output_model_path)}",
            'floorplan_data': floorplan_data
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detect-walls-rooms', methods=['POST'])
def detect_walls_rooms():
    """Detect walls and rooms in the uploaded floorplan"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        os.makedirs('temp', exist_ok=True)
        timestamp = int(time.time())
        unique_filename = f'{timestamp}_{file.filename}'
        img_path = os.path.join('temp', unique_filename)
        file.save(img_path)
        
        try:
            # Create static dir for output visualization
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(static_dir, exist_ok=True)
            
            # Use our utility function to detect rooms and walls
            # Import locally to avoid circular imports
            try:
                from utils import detect_rooms_and_walls, ROOM_CONFIG
            except ImportError:
                from model_folder.utils import detect_rooms_and_walls, ROOM_CONFIG
                
            result = detect_rooms_and_walls(img_path, output_dir=static_dir)
            
            # Get path relative to static folder
            vis_path = result.get('visualization')
            if vis_path:
                rel_path = '/' + os.path.join('static', os.path.basename(vis_path))
                result['visualization'] = rel_path
                
                # Create a thumbnail for quicker loading
                thumb_path = os.path.join(static_dir, f'thumb_{os.path.basename(vis_path)}')
                try:
                    thumb_img = cv2.imread(vis_path)
                    thumb_img = cv2.resize(thumb_img, (0, 0), fx=0.5, fy=0.5)
                    cv2.imwrite(thumb_path, thumb_img)
                    result['thumbnail'] = f'/static/thumb_{os.path.basename(vis_path)}'
                except Exception as e:
                    print(f"Error creating thumbnail: {e}")
            
            # Add room config details
            result['room_config'] = ROOM_CONFIG
            
            # Clean up temp files
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")
            
            return jsonify(result)
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Wall detection error: {str(e)}\n{error_trace}")
            return jsonify({'error': str(e), 'traceback': error_trace}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/estimate-cost', methods=['POST'])
def estimate_cost():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data'}), 400
    try:
        features = pd.DataFrame([data])
        prediction = model.predict(features)
        estimated_cost = float(prediction[0])
        return jsonify({'estimated_cost': round(estimated_cost, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update-location-type', methods=['POST'])
def update_location_type():
    """Update location type for cost estimation without regenerating 3D model"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
        
    try:
        print("Received location update request:", data)
        location_type = data.get('location_type', 'standard')
        floorplan_data = data.get('floorplan_data', {})
        
        # Update the location type in floorplan data
        floorplan_data['location_type'] = location_type
        
        # If we have a stored floorplan data file, update it
        output_data_path = None
        if 'filename' in data:
            filename = data['filename']
            if filename:
                static_dir = os.path.join(os.path.dirname(__file__), 'static')
                output_data_path = os.path.join(static_dir, f"{os.path.splitext(filename)[0]}_data.json")
                
                if os.path.exists(output_data_path):
                    try:
                        with open(output_data_path, 'r') as f:
                            stored_data = json.load(f)
                        
                        stored_data['location_type'] = location_type
                        
                        with open(output_data_path, 'w') as f:
                            json.dump(stored_data, f)
                        print(f"Updated location type in {output_data_path}")
                        
                        # Use the stored data as it's more reliable
                        floorplan_data = stored_data
                    except Exception as e:
                        print(f"Error updating location in file: {e}")
        
        # Extract room data from floorplan_data
        area = floorplan_data.get('estimated_area', 0)
        room_counts = floorplan_data.get('room_counts', {})
        bedrooms = room_counts.get('bedroom', 0)
        bathrooms = room_counts.get('bathroom', 0)
        kitchen = room_counts.get('kitchen', 0)
        living = room_counts.get('living', 0)
        total_rooms = floorplan_data.get('rooms', 0)
        
        # If floorplan data is empty or invalid, try to get it from detection
        if area == 0 or total_rooms == 0 or bedrooms == 0:
            try:
                # Try to use detection data if available
                print("Trying to use detection data for room values")
                # Get uploaded image path or use stored image path
                img_path = data.get('img_path')
                temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
                # Look for any image files in temp directory if path not provided
                if not img_path or not os.path.exists(img_path):
                    # Try to find an image in temp directory
                    if os.path.exists(temp_dir):
                        for file in os.listdir(temp_dir):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(temp_dir, file)
                                print(f"Using image from temp directory: {img_path}")
                                break
                
                if img_path and os.path.exists(img_path):
                    # Import room detection utilities
                    try:
                        from utils import detect_rooms_and_walls
                    except ImportError:
                        from model_folder.utils import detect_rooms_and_walls
                    
                    # Detect rooms from image
                    detection_result = detect_rooms_and_walls(img_path, save_visualization=False)
                    print("Detection result:", detection_result)
                    
                    # Use detected values, don't fall back to defaults
                    if 'total_rooms' in detection_result:
                        total_rooms = detection_result.get('total_rooms')
                    
                    if 'estimated_area' in detection_result:
                        area = detection_result.get('estimated_area')
                        
                    # Get room counts from detection
                    detected_rooms = detection_result.get('room_counts', {})
                    if detected_rooms:
                        bedrooms = detected_rooms.get('bedroom', 0)
                        bathrooms = detected_rooms.get('bathroom', 0)
                        kitchen = detected_rooms.get('kitchen', 0)
                        living = detected_rooms.get('living', 0)
                        
                    print(f"Updated from detection: rooms={total_rooms}, area={area}, bedrooms={bedrooms}")
            except Exception as e:
                print(f"Error using detection for room values: {e}")
        
        # We still need to handle the case when detection truly failed
        # Use minimum values that make sense rather than hardcoded defaults
        # These values are only used when absolutely nothing else is available
        if area <= 0:
            # Get from adjacent room calculation if possible
            if total_rooms > 0:
                area = total_rooms * 6  # Minimum 6m² per room
            else:
                area = 30  # Absolute minimum size for any dwelling
            print(f"Using calculated minimum area: {area}")
            
        if total_rooms <= 0:
            # Calculate from specific rooms if possible
            specific_rooms = sum(v for k, v in room_counts.items() if v > 0)
            if specific_rooms > 0:
                total_rooms = specific_rooms
            else:
                # Set a reasonable minimum
                total_rooms = 3
            print(f"Using calculated total rooms: {total_rooms}")
        
        # Calculate other rooms - any room that isn't specifically categorized
        specific_rooms = sum([bedrooms, bathrooms, kitchen, living])
        other_rooms = max(0, total_rooms - specific_rooms)
        
        # Ensure we have at least one bedroom, bathroom, kitchen, living room
        # only if we have at least 4 rooms total
        if total_rooms >= 4:
            if bedrooms <= 0:
                if other_rooms > 0:
                    bedrooms = 1
                    other_rooms -= 1
                elif total_rooms > 3:
                    bedrooms = 1
                    total_rooms += 1
                    
            if bathrooms <= 0:
                if other_rooms > 0:
                    bathrooms = 1
                    other_rooms -= 1
                elif total_rooms > 3:
                    bathrooms = 1
                    total_rooms += 1
                    
            if kitchen <= 0:
                if other_rooms > 0:
                    kitchen = 1
                    other_rooms -= 1
                elif total_rooms > 3:
                    kitchen = 1
                    total_rooms += 1
                    
            if living <= 0:
                if other_rooms > 0:
                    living = 1
                    other_rooms -= 1
                elif total_rooms > 3:
                    living = 1
                    total_rooms += 1
        
        # Location cost factors
        location_factors = {
            'rural': 0.8,
            'suburban': 1.0,
            'urban': 1.2,
            'urban_high': 1.3,
            'premium': 1.5
        }
        
        # Map UI-friendly location terms to internal factors
        location_mapping = {
            'Rural (Low Cost)': 'rural',
            'Suburban (Standard)': 'suburban',
            'Urban': 'urban',
            'Urban (High Cost)': 'urban_high',
            'Premium Location': 'premium'
        }
        
        # Get location factor from mapping or use direct value
        location_key = location_mapping.get(location_type, 'suburban')
        location_factor = location_factors.get(location_key, 1.0)
        
        # Base construction cost
        base_cost_per_sqm = 1200
        
        # Additional costs based on room types
        bedroom_cost = bedrooms * 10000
        bathroom_cost = bathrooms * 15000
        kitchen_cost = kitchen * 20000
        living_cost = living * 8000
        other_cost = other_rooms * 5000
        
        # Calculate total cost
        total_area_cost = area * base_cost_per_sqm
        total_room_cost = bedroom_cost + bathroom_cost + kitchen_cost + living_cost + other_cost
        final_cost = (total_area_cost + total_room_cost) * location_factor
        
        print(f"Final room counts: bedrooms={bedrooms}, bathrooms={bathrooms}, kitchen={kitchen}, living={living}, other={other_rooms}")
        print(f"Calculated cost for location {location_type}: {final_cost}")
        
        # Return the updated cost estimation directly with no-cache headers
        response = jsonify({
            'estimated_cost': round(final_cost, 2),
            'breakdown': {
                'base_area_cost': round(total_area_cost, 2),
                'bedroom_cost': round(bedroom_cost, 2),
                'bathroom_cost': round(bathroom_cost, 2),
                'kitchen_cost': round(kitchen_cost, 2),
                'living_cost': round(living_cost, 2),
                'other_rooms_cost': round(other_cost, 2),
                'location_type': location_type,
                'location_factor': location_factor,
                'total_rooms': {
                    'detected': total_rooms,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'kitchen': kitchen,
                    'living': living,
                    'other': other_rooms
                }
            },
            'reload_required': False,
            'success': True,
            'message': 'Location updated successfully'
        })
        
        # Add no-cache headers to prevent browser caching
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in location type update: {e}\n{error_trace}")
        return jsonify({
            'error': f'Error updating location type: {str(e)}',
            'traceback': error_trace
        }), 500

@app.route('/download-ply/<filename>', methods=['GET'])
def download_ply(filename):
    """Download PLY file without triggering model regeneration"""
    try:
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        filepath = os.path.join(static_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        directory = os.path.dirname(filepath)
        return send_from_directory(
            directory, 
            os.path.basename(filepath),
            as_attachment=True,
            attachment_filename=filename
        )
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error downloading file: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/estimate-construction-cost', methods=['POST'])
def estimate_construction_cost():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
        
    try:
        # Extract features from request
        area = data.get('area', 0)
        bedrooms = data.get('bedrooms', 0)
        bathrooms = data.get('bathrooms', 0)
        kitchen = data.get('kitchen', 0) 
        living = data.get('living', 0)
        total_rooms = data.get('total_rooms', 0)
        
        # Validate that data matches detected rooms
        if total_rooms > 0 and total_rooms != (bedrooms + bathrooms + kitchen + living):
            # Adjust room distribution if total doesn't match
            remaining_rooms = total_rooms - (bedrooms + bathrooms + kitchen + living)
            if remaining_rooms > 0:
                # Assign remaining rooms as "other" spaces
                other_rooms = remaining_rooms
            else:
                # If overallocated, don't make adjustments
                other_rooms = 0
        else:
            other_rooms = 0
            
        # Base cost per square meter - adjusted for different quality levels
        location_type = data.get('location_type', 'standard')
        
        # Location cost factors
        location_factors = {
            'rural': 0.8,
            'suburban': 1.0,
            'urban': 1.2,
            'urban_high': 1.3,  # Urban high cost areas
            'premium': 1.5      # Premium locations
        }
        
        # Map UI-friendly location terms to internal factors
        location_mapping = {
            'Rural (Low Cost)': 'rural',
            'Suburban (Standard)': 'suburban',
            'Urban': 'urban',
            'Urban (High Cost)': 'urban_high',
            'Premium Location': 'premium'
        }
        
        # Store original location type for returning to client
        original_location_type = location_type
        
        # Get location factor from mapping or use direct value
        location_key = location_mapping.get(location_type, 'suburban')
        location_factor = location_factors.get(location_key, 1.0)
        
        # Base construction cost
        base_cost_per_sqm = 1200  # $1200 per square meter
        
        # Additional costs based on room types
        bedroom_cost = bedrooms * 10000  # $10,000 per bedroom
        bathroom_cost = bathrooms * 15000  # $15,000 per bathroom
        kitchen_cost = kitchen * 20000  # $20,000 per kitchen
        living_cost = living * 8000  # $8,000 per living room
        other_cost = other_rooms * 5000  # $5,000 per other room
        
        # Calculate total cost
        total_area_cost = area * base_cost_per_sqm
        total_room_cost = bedroom_cost + bathroom_cost + kitchen_cost + living_cost + other_cost
        
        # Calculate final cost with location factor
        final_cost = (total_area_cost + total_room_cost) * location_factor
        
        # Create detailed response
        return jsonify({
            'estimated_cost': round(final_cost, 2),
            'breakdown': {
                'base_area_cost': round(total_area_cost, 2),
                'bedroom_cost': round(bedroom_cost, 2),
                'bathroom_cost': round(bathroom_cost, 2),
                'kitchen_cost': round(kitchen_cost, 2),
                'living_cost': round(living_cost, 2),
                'other_rooms_cost': round(other_cost, 2),
                'location_type': original_location_type,
                'location_factor': location_factor,
                'total_rooms': {
                    'detected': total_rooms,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'kitchen': kitchen,
                    'living': living,
                    'other': other_rooms
                }
            },
            'reload_required': False  # Flag to indicate no reload needed
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error calculating cost: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    # Ensure static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Make sure we have the 2d-3d.py script or copy it if needed
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '2d-3d.py'))
    if not os.path.exists(script_path):
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2d-3d.py'))
        if os.path.exists(alt_path):
            import shutil
            shutil.copy2(alt_path, script_path)
            print(f"Copied 2d-3d.py from {alt_path} to {script_path}")
        else:
            print(f"WARNING: 2d-3d.py not found at {script_path} or {alt_path}")
    
    print("Starting Flask server on http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True)
