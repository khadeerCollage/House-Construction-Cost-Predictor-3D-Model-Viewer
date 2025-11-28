# 🏠 Floor Plan to 3D Model Converter & Cost Estimator

Transform 2D floor plan images into interactive 3D models with automatic room detection and construction cost estimation.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-orange)

---

## 🐳 Quick Start with Docker (Recommended)

**The easiest way to run this application - just one command!**

### Prerequisites
- [Docker](https://docker.com/get-started) installed on your system

### Run the App
```bash
docker run -d -p 8501:8501 -p 5000:5000 khadee/house-3d-cost-design-and-detection
```

### Open in Browser
- 🌐 **App URL**: http://localhost:8501

That's it! The app will automatically download and start. 🎉

### Stop the App
```bash
docker stop $(docker ps -q --filter ancestor=khadee/house-3d-cost-design-and-detection)
```

---

## 📦 Docker Hub

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-khadee%2Fhouse--3d--cost--design--and--detection-blue?logo=docker)](https://hub.docker.com/r/khadee/house-3d-cost-design-and-detection)

**Pull the image:**
```bash
docker pull khadee/house-3d-cost-design-and-detection:latest
```

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

### 2. Detect Walls & Rooms
- Click "1️⃣ Detect Walls and Rooms"
- See detected rooms with colored overlays
- Review room breakdown (types and areas)

### 3. Generate 3D Model
- Click "2️⃣ Generate 3D Model"
- View interactive 3D preview
- Download PLY file for external viewers

### 4. Get Cost Estimate
- Select your location type (Urban/Suburban/Rural)
- View detailed cost breakdown by room

---

## 🛠️ Alternative: Run from Source

<details>
<summary>Click to expand manual setup instructions</summary>

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
```bash
cd model_folder
streamlit run frontend.py
```

### Step 5: Open in Browser
- 🌐 **Frontend**: http://localhost:8501

</details>

---

## 📁 Project Structure

```
vit_project/
├── model_folder/
│   ├── frontend.py         # Streamlit UI + Flask backend
│   ├── 2d-3d.py            # 3D model generator
│   ├── utils.py            # Detection utilities
│   └── temp/               # Generated files
├── floorplan_detector/
│   ├── floor_plan_analyzer.py   # Main analyzer
│   ├── wall_detection.py        # Wall detection
│   ├── room_detection.py        # Room segmentation
│   └── model_3d.py              # 3D generation
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose config
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect-walls-rooms` | POST | Detect rooms and walls from image |
| `/generate-3d` | POST | Generate 3D model from floor plan |
| `/estimate-construction-cost` | POST | Calculate construction cost |

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

## 📝 Tips for Best Results

1. **Use Clear Images**: High resolution (1000x1000+) works best
2. **Black & White Plans**: Clean architectural drawings give best results
3. **Visible Walls**: Ensure all walls are clearly drawn
4. **No Furniture**: Remove furniture from floor plans if possible

---

## ❓ Troubleshooting

### Docker Issues
```bash
# Check if container is running
docker ps

# View logs
docker logs $(docker ps -q --filter ancestor=khadee/house-3d-cost-design-and-detection)

# Restart container
docker restart $(docker ps -q --filter ancestor=khadee/house-3d-cost-design-and-detection)
```

### Port Already in Use
```bash
# Use different ports
docker run -d -p 8502:8501 -p 5001:5000 khadee/house-3d-cost-design-and-detection
# Then access at http://localhost:8502
```

---

## 🖥️ System Requirements

- **Docker**: Latest version recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for Docker image

---

## 📜 License

© 2025 House Construction Cost Predictor. All Rights Reserved.

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Open an issue on GitHub

---

Made with ❤️ for architects and home builders


