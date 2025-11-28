# 🏠 Floor Plan to 3D Model Converter & Cost Estimator

Transform 2D floor plan images into interactive 3D models with automatic room detection and construction cost estimation.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-orange)

---

## 🚀 Quick Start

### Step 1: Clone & Setup
```bash
git clone <repository-url>
cd vit_project
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

**Terminal 1 - Start Backend:**
```bash
cd model_folder
python app.py
```

**Terminal 2 - Start Frontend:**
```bash
cd model_folder
streamlit run frontend.py
```

### Step 5: Open in Browser
- 🌐 **Frontend**: http://localhost:8501
- 🔧 **Backend API**: http://127.0.0.1:5000

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📐 **Room Detection** | Automatically detects rooms (Living Room, Kitchen, Bedroom, Bathroom) |
| 🧱 **Wall Detection** | Identifies walls and structural elements |
| 🏗️ **3D Model Generation** | Converts 2D floor plans to 3D PLY models |
| 💰 **Cost Estimation** | Calculates construction costs based on area and location |
| 📥 **Download 3D Model** | Export PLY files for use in Blender, MeshLab, etc. |

---

## 📖 How to Use

### 1. Upload Floor Plan
- Click "Upload Floor Plan" button
- Select a clear floor plan image (JPG, PNG)
- Wait for automatic analysis

### 2. View Results
- See detected rooms with colored overlays
- Review wall detection visualization
- Check room breakdown (types and areas)

### 3. Generate 3D Model
- Click "Generate 3D Model"
- View interactive 3D preview
- Download PLY file for external viewers

### 4. Get Cost Estimate
- Select your location type (Urban/Suburban/Rural)
- View detailed cost breakdown by room
- Adjust parameters as needed

---

## 📁 Project Structure

```
vit_project/
├── model_folder/
│   ├── app.py              # Flask backend API
│   ├── frontend.py         # Streamlit UI
│   ├── 2d-3d.py            # 3D model generator
│   ├── utils.py            # Detection utilities
│   └── temp/               # Generated files
├── floorplan_detector/
│   ├── floor_plan_analyzer.py   # Main analyzer
│   ├── wall_detection.py        # Wall detection
│   ├── room_detection.py        # Room segmentation
│   └── smart_detection.py       # Advanced detection
├── .venv/                  # Virtual environment
└── README.md               # This file
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect-walls-rooms` | POST | Detect rooms and walls from image |
| `/generate-3d` | POST | Generate 3D model from floor plan |
| `/estimate-construction-cost` | POST | Calculate construction cost |
| `/download-ply/<filename>` | GET | Download generated 3D model |

### Example API Usage

```python
import requests

# Upload floor plan for detection
with open('floorplan.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/detect-walls-rooms',
        files={'file': f}
    )
    result = response.json()
    print(f"Detected {result['room_count']} rooms")
```

---

## 🛠️ Requirements

### Core Dependencies
```
flask>=2.0.0
streamlit>=1.20.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
open3d>=0.15.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

### Install All
```bash
pip install flask streamlit opencv-python numpy matplotlib open3d pandas scikit-learn
```

---

## ❓ Troubleshooting

### Port Already in Use
```bash
# Find and kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Then restart the backend
python app.py
```

### Module Not Found
```bash
# Ensure virtual environment is activated
.\.venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### 3D Model Not Loading
- Ensure Open3D is installed: `pip install open3d`
- Check that the floor plan image has clear walls
- Try a different floor plan image

### Poor Room Detection
- Use high-contrast floor plan images
- Ensure walls are clearly visible
- Avoid floor plans with furniture/clutter

---

## 📊 Supported Room Types

| Room Type | Color in Visualization |
|-----------|----------------------|
| Living/Dining | 🟢 Light Green |
| Bedroom | 🔵 Light Blue |
| Kitchen | 🟠 Orange |
| Bathroom | 🔷 Cyan |
| Hallway | 🟣 Light Purple |
| Storage | 🟤 Tan |

---

## 🖥️ System Requirements

- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.10+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for dependencies

---

## 📝 Tips for Best Results

1. **Use Clear Images**: High resolution (1000x1000+) works best
2. **Black & White Plans**: Clean architectural drawings give best results
3. **Visible Walls**: Ensure all walls are clearly drawn
4. **No Furniture**: Remove furniture from floor plans if possible
5. **Proper Scale**: Images should show the complete floor plan

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

---

## 📜 License

© 2024 VIT Project. All Rights Reserved.

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Open an issue on GitHub
3. Contact the development team

---

Made with ❤️ for architectural visualization
