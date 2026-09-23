from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os
import subprocess
import sys
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    model_path = os.path.join(os.path.dirname(__file__), "house_cost_model_v2.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "house_cost_model.pkl")
    model = joblib.load(model_path)
    print(f"Successfully loaded model from: {model_path}")
except Exception as e:
    model = None
    print(f"Error loading house_cost_model: {e}")

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
        # Standardize features for model
        features_dict = {
            "Area_m2": data.get("Area_m2", data.get("area", 100)),
            "Rooms": data.get("Rooms", data.get("bedrooms", 3)),
            "Bathrooms": data.get("Bathrooms", data.get("bathrooms", 2)),
            "Kitchens": data.get("Kitchens", data.get("kitchen", 1)),
            "Living_Rooms": data.get("Living_Rooms", data.get("living", 1)),
            "City_Tier": data.get("City_Tier", 1), # Default standard (Tier-2)
            "Quality_Level": data.get("Quality_Level", 1), # Default standard (Medium)
            "Num_Floors": data.get("Num_Floors", 1)
        }
        features = pd.DataFrame([features_dict])
        
        # Check model requirements
        if hasattr(model, 'n_features_in_'):
            if model.n_features_in_ == 3:
                features = features[["Area_m2", "Rooms", "Bathrooms"]]
                features.columns = ["Area_m2", "Rooms", "Bathrooms"]
            else:
                expected_cols = ["Area_m2", "Rooms", "Bathrooms", "Kitchens", "Living_Rooms", "City_Tier", "Quality_Level", "Num_Floors"]
                features = features[expected_cols]
                
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
        
        try:
            # Extract plot dimensions if provided
            plot_width = request.form.get('plot_width', type=float)
            plot_length = request.form.get('plot_length', type=float)

            # Use our dedicated room detection function for consistency
            try:
                # Use the same detection function as the detect-walls-rooms endpoint
                try:
                    from utils import detect_rooms_and_walls, ROOM_CONFIG
                except ImportError:
                    from model_folder.utils import detect_rooms_and_walls, ROOM_CONFIG
                    
                # Get room counts directly from detection - don't use hardcoded values
                detection_result = detect_rooms_and_walls(
                    img_path, output_dir=static_dir, 
                    plot_width=plot_width, plot_length=plot_length, 
                    save_visualization=False
                )
            except Exception as e_det:
                print(f"Error calling detect_rooms_and_walls inside generate_3d: {e_det}")
                detection_result = {}
                
            print("Detection result for 3D model:", detection_result)
            
            # Get room counts from detection
            total_rooms = detection_result.get('room_count', detection_result.get('total_rooms', 0))
            room_counts = detection_result.get('room_type_counts', detection_result.get('room_counts', {}))
            
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
            
            # Extract plot dimensions if provided
            plot_width = request.form.get('plot_width', type=float)
            plot_length = request.form.get('plot_length', type=float)
            
            # Use our utility function to detect rooms and walls
            # Import locally to avoid circular imports
            try:
                from utils import detect_rooms_and_walls, ROOM_CONFIG
            except ImportError:
                from model_folder.utils import detect_rooms_and_walls, ROOM_CONFIG
                
            result = detect_rooms_and_walls(img_path, output_dir=static_dir, plot_width=plot_width, plot_length=plot_length)
            
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
        
        city_tier = data.get('city_tier', 'Tier-2 (Urban)')
        quality_level = data.get('quality_level', 'Standard (Medium)')
        num_floors = data.get('num_floors', 1)
        
        try:
            from cost_engine import CostEngine
        except ImportError:
            from model_folder.cost_engine import CostEngine
            
        estimate = CostEngine.estimate_cost(
            area_sqm=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            kitchens=kitchen,
            living_rooms=living,
            city_tier=city_tier,
            quality_level=quality_level,
            num_floors=num_floors
        )
        
        return jsonify({
            'estimated_cost': estimate['total_cost'],
            'breakdown': estimate['cost_breakdown'],
            'materials': estimate['materials'],
            'comparison': estimate['comparison'],
            'description': estimate['description'],
            'area_sqft': estimate['area_sqft'],
            'cost_per_sqft': estimate['cost_per_sqft'],
            'reload_required': False
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error calculating cost: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

# =============================================================================
# GENERATIVE AI FLOOR PLAN DESIGN ENDPOINTS (PHASE 1)
# =============================================================================

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from floorplan_generator.family_analyzer import FamilyAnalyzer, FamilyProfile
    from floorplan_generator.layout_engine import LayoutEngine
    from floorplan_renderer.dxf_renderer import DXFRenderer
    from floorplan_renderer.svg_renderer import SVGRenderer
    from floorplan_renderer.pdf_renderer import PDFRenderer
    from floorplan_renderer.png_renderer import PNGRenderer
    from model_folder.cost_engine import CostEngine
    GENERATOR_AVAILABLE = True
except Exception as e:
    print(f"Warning: floorplan_generator or renderer import failed in app.py: {e}")
    GENERATOR_AVAILABLE = False


@app.route('/generate-plan', methods=['POST'])
def generate_floor_plan_endpoint():
    """
    Generate a professional residential floor plan from land dimensions,
    family intent, and Vastu preferences.
    """
    if not GENERATOR_AVAILABLE:
        return jsonify({'error': 'Generative floor plan engine is not available on this server.'}), 503

    try:
        data = request.get_json() or {}

        # 1. Parse plot dimensions
        plot_width = float(data.get('plot_width', 12.0))
        plot_height = float(data.get('plot_height', 15.0))
        orientation = str(data.get('orientation', 'N')).upper()

        # 2. Parse family intent
        adults = int(data.get('adults', 2))
        children = int(data.get('children', 0))
        elderly = int(data.get('elderly', 0))
        guests = bool(data.get('guests_frequent', False))
        needs_office = bool(data.get('needs_home_office', False))
        needs_pooja = bool(data.get('needs_pooja_room', True))
        needs_servant = bool(data.get('needs_servant_quarter', False))
        needs_store = bool(data.get('needs_store_room', False))
        bhk_override = data.get('bhk_override') or None

        # 3. Engineering options
        vastu_enabled = bool(data.get('vastu_enabled', True))
        quality_level = str(data.get('quality_level', 'Standard'))
        city_tier = str(data.get('city_tier', 'Tier-2 Urban'))
        num_floors = int(data.get('num_floors', 1))
        project_name = str(data.get('project_name', 'Custom Villa Blueprint'))

        # Step A: Family intent analysis
        profile = FamilyProfile(
            adults=adults,
            children=children,
            elderly=elderly,
            guests_frequent=guests,
            needs_home_office=needs_office,
            needs_pooja_room=needs_pooja,
            needs_servant_quarter=needs_servant,
            needs_store_room=needs_store,
            bhk_override=bhk_override,
        )
        analyzer = FamilyAnalyzer()
        analysis = analyzer.analyze(profile, plot_area=plot_width * plot_height)

        # Step B: Core layout generation (V-HSP)
        engine = LayoutEngine(
            plot_width=plot_width,
            plot_height=plot_height,
            bhk_config=analysis.recommended_bhk,
            vastu_enabled=vastu_enabled,
            orientation=orientation,
            quality_level=quality_level,
            project_name=project_name,
            family_description=profile.description,
        )
        plan = engine.generate()

        # Step C: Render to all 4 professional CAD formats
        plan_id = f"plan_{int(time.time())}_{np.random.randint(1000, 9999)}"
        plan_dir = os.path.join(os.path.dirname(__file__), 'static', 'generated_plans', plan_id)
        os.makedirs(plan_dir, exist_ok=True)

        dxf_path = os.path.join(plan_dir, 'plan.dxf')
        svg_path = os.path.join(plan_dir, 'plan.svg')
        pdf_path = os.path.join(plan_dir, 'plan.pdf')
        png_path = os.path.join(plan_dir, 'plan.png')
        mask_path = os.path.join(plan_dir, 'wall_mask.png')

        svg_r = SVGRenderer()
        svg_r.render(plan, svg_path, mode='clean')
        svg_content = svg_r.render_to_string(plan, mode='clean')

        png_r = PNGRenderer()
        png_r.render(plan, png_path)
        mask = png_r.render_wall_mask(plan, ppm=50)
        cv2.imwrite(mask_path, mask) if cv2 is not None else None

        dxf_r = DXFRenderer()
        dxf_r.render(plan, dxf_path)

        pdf_r = PDFRenderer()
        pdf_r.render(plan, pdf_path)

        # Step D: Cost Estimation with Indian Market BOQ
        cost_data = CostEngine.estimate_cost(
            area_sqm=plan.total_carpet_area,
            city_tier=city_tier,
            quality_level=quality_level,
            num_floors=num_floors
        )

        # Step E: Construct comprehensive response
        rooms_data = [
            {
                'name': r.name,
                'category': r.category,
                'x': r.x,
                'y': r.y,
                'width': r.w,
                'height': r.h,
                'area': r.area,
                'color': r.color
            }
            for r in plan.rooms
        ]

        return jsonify({
            'success': True,
            'plan_id': plan_id,
            'project_name': project_name,
            'bhk_config': plan.bhk_config,
            'family_description': plan.family_description,
            'reasoning': analysis.reasoning,
            'plot_width': plan.plot_width,
            'plot_height': plan.plot_height,
            'orientation': plan.orientation,
            'vastu_enabled': plan.vastu_enabled,
            'vastu_score': plan.vastu_score,
            'total_carpet_area': plan.total_carpet_area,
            'total_carpet_area_sqft': round(plan.total_carpet_area * 10.764, 1),
            'total_built_up_area': plan.total_built_up_area,
            'rooms': rooms_data,
            'room_count': len(plan.rooms),
            'door_count': len(plan.doors),
            'window_count': len(plan.windows),
            'cost_estimation': cost_data,
            'svg_content': svg_content,
            'download_urls': {
                'dxf': f'/download-plan/dxf/{plan_id}',
                'svg': f'/download-plan/svg/{plan_id}',
                'pdf': f'/download-plan/pdf/{plan_id}',
                'png': f'/download-plan/png/{plan_id}',
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error generating floor plan: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/download-plan/<fmt>/<plan_id>', methods=['GET'])
def download_generated_plan(fmt, plan_id):
    """Serve generated CAD and blueprint files for download."""
    try:
        fmt = fmt.lower()
        if fmt not in {'dxf', 'svg', 'pdf', 'png'}:
            return jsonify({'error': 'Invalid format requested'}), 400

        filename = f"plan.{fmt}"
        plan_dir = os.path.join(os.path.dirname(__file__), 'static', 'generated_plans', plan_id)
        filepath = os.path.join(plan_dir, filename)

        if not os.path.exists(filepath):
            return jsonify({'error': f'Generated plan file not found for plan_id {plan_id}'}), 404

        # Modern Flask supports download_name, fallback to attachment_filename for older Flask
        download_filename = f"{plan_id}_{filename}"
        try:
            return send_from_directory(
                plan_dir,
                filename,
                as_attachment=True,
                download_name=download_filename
            )
        except TypeError:
            return send_from_directory(
                plan_dir,
                filename,
                as_attachment=True,
                attachment_filename=download_filename
            )
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


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
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
