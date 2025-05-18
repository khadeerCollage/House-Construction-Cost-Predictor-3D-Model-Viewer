# Floorplan Analysis and Cost Estimation Tool

This project provides an automated tool for analyzing floorplan images, detecting rooms and walls, generating 3D models, and estimating construction costs.

## Features

- Floorplan image analysis with room and wall detection
- 3D model generation from 2D floorplans
- Construction cost estimation based on detected rooms and area
- Interactive web interface for uploading and analyzing floorplans
- Support for different location types affecting construction costs
- Downloadable 3D models in PLY format
- Real-time cost updates based on location changes without page reloading
- Intelligent room detection with automatic room type classification
- Dynamic area calculation based on detected room sizes
- Cost breakdown by room type and area
- Customizable construction cost factors for different regions
- Visualization of detected walls and room boundaries
- Thumbnail generation for quick previews
- Error handling with detailed feedback

## Technology Stack

### Backend

- **Python 3.10**: Core programming language
- **Flask**: Web server framework for API endpoints
- **OpenCV**: Computer vision library for image processing and room detection
- **NumPy/Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualization of detected rooms and walls
- **Scikit-learn**: Machine learning for cost prediction model
- **Joblib**: Model serialization and loading

### Frontend

- **Streamlit**: Main frontend framework providing the user interface
- **HTML/CSS/JavaScript**: Additional frontend enhancements and interactivity
- **Plotly**: Interactive data visualization

## Project Structure

### Backend Components

- `model_folder/app.py`: Main Flask application providing API endpoints
- `model_folder/utils.py`: Utility functions for wall and room detection
- `model_folder/2d-3d.py`: Script for converting 2D floorplans to 3D models
- `model_folder/house_cost_model.pkl`: Trained model for cost prediction

### Frontend Components

- `frontend.py`: Streamlit application for user interface
- `static/`: Directory containing generated visualizations and 3D models

## Setup and Installation

### Prerequisites

- Python 3.10
- Virtual Environment
- Git (for cloning the repository)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vit_project
   ```

2. Activate the virtual environment:
   
   **For Windows:**
   ```bash
   .\venv310\Scripts\activate
   ```
   
   **For macOS/Linux:**
   ```bash
   source venv310/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Large File Downloads

Some large files needed for this project exceed GitHub's file size limit (25MB) and must be downloaded separately:

1. **SAM Vision Transformer Model**:
   - File: `sam_vit_h_4b839.pth`
   - Download from: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b839.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b839.pth)
   - Place in: `c:\Users\USER\Desktop\vit_project\model_folder\`

2. **House Cost Prediction Model**:
   - File: `house_cost_model.pkl`
   - Download from: [https://huggingface.co/datasets/vit-project/house-cost-model/resolve/main/house_cost_model.pkl](https://huggingface.co/datasets/vit-project/house-cost-model/resolve/main/house_cost_model.pkl)
   - Place in: `c:\Users\USER\Desktop\vit_project\model_folder\`

3. **Cubic5k Dataset** (optional - for training only):
   - Download from: [https://huggingface.co/datasets/vit-project/cubic5k/resolve/main/cubic5k.zip](https://huggingface.co/datasets/vit-project/cubic5k/resolve/main/cubic5k.zip)
   - Extract to: `c:\Users\USER\Desktop\vit_project\datasets\cubic5k\`

After downloading these files, ensure they are placed in the correct directories before running the application.

### Running the Application

1. Start the Flask backend:
   ```bash
   cd model_folder
   python app.py
   ```

2. In a new terminal, start the Streamlit frontend:
   ```bash
   # Activate the virtual environment again if needed
   cd <project-root>
   streamlit run frontend.py
   ```

3. Open your browser and navigate to:
   - Backend API: http://127.0.0.1:5000/
   - Frontend UI: http://localhost:8501/

## API Endpoints

- `POST /detect-walls-rooms`: Upload a floorplan image to detect rooms and walls
- `POST /generate-3d`: Generate a 3D model from a floorplan image
- `POST /estimate-construction-cost`: Calculate construction cost based on room data
- `POST /update-location-type`: Update location type for cost estimation
- `GET /download-ply/<filename>`: Download a generated 3D model

## Usage

1. Upload a floorplan image through the Streamlit interface
2. View detected rooms and walls with visualization
3. Generate a 3D model of the floorplan
4. Modify parameters like location type to see different cost estimates
5. Download the 3D model for use in other applications

## Common Issues and Troubleshooting

- If the Flask server fails to start, ensure port 5000 is not in use by another application
- If room detection gives unexpected results, try using a clearer floorplan image
- For 3D model generation issues, ensure the floorplan has clear walls and room boundaries
- If you get errors about missing model files, check that you've downloaded the large files mentioned in the "Large File Downloads" section

## License

All rights reserved. This project and its contents are proprietary and confidential.

© 2023 Visual Intelligence Technologies Project. Unauthorized copying or distribution of this project or any of its contents is strictly prohibited.

## Acknowledgments

This project utilizes several open-source technologies:
- Python, Flask, Streamlit
- OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn
- Additional libraries and tools mentioned in the Technology Stack section

---

*Note: This README.md file is part of the VIT Project and is subject to the same proprietary restrictions as the rest of the codebase.*
