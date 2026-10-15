@echo off
rem Voices for the 3D guided walk (Telugu / Hindi / Indian English), run once.
rem A small separate environment OUTSIDE the project (%USERPROFILE%\.vit_voice\venv, about 190 MB),
rem so the app's own environment and the project folder stay untouched.
echo ========================================================
echo Voices for the guided walk (Piper: Telugu, Hindi, Indian English)
echo ========================================================
set VHOME=%USERPROFILE%\.vit_voice
if not exist "%VHOME%\venv\Scripts\python.exe" (
    echo [1] Creating the voice environment in %VHOME% ...
    py -3.12 -m venv "%VHOME%\venv" || py -3.11 -m venv "%VHOME%\venv" || python -m venv "%VHOME%\venv"
)
set VPY=%VHOME%\venv\Scripts\python.exe
echo [2] Installing Piper (about 190 MB, the first time only)...
"%VPY%" -m pip install --no-cache-dir -r requirements-voice.txt
echo [3] Downloading the voices: Telugu Venkatesh, Hindi Rohan, English Ryan (about 250 MB, the first time only)...
"%VPY%" model_folder\voice_worker.py download
"%VPY%" model_folder\voice_worker.py check
echo.
echo Done. Restart the app; open "The guide's voice" above the 3D house.
