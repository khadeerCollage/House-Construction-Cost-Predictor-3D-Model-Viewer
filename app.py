# Entry point for Hugging Face Spaces
# This file allows deployment to HF Spaces with Streamlit SDK

import sys
import os

# Add model_folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_folder'))

# Import and run the main app
exec(open('model_folder/frontend.py', encoding='utf-8').read())
