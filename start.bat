@echo off
echo ========================================================
echo Starting House Construction Cost Predictor 3D
echo ========================================================

echo.
echo [1] Setting up Python 3.10 Virtual Environment...
if not exist ".venv310" (
    py -3.10 -m venv .venv310
)

echo [2] Activating Environment...
call .venv310\Scripts\activate.bat

echo [3] Installing Requirements (this will take a few minutes the first time)...
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo.
echo [4] Starting Backend (Flask) on Port 5000...
start cmd /k "call .venv310\Scripts\activate.bat && python model_folder\app.py"

echo [5] Starting Frontend (Streamlit) on Port 8501...
streamlit run model_folder\frontend.py
