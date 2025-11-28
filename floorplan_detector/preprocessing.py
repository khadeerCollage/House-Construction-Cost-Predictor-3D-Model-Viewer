"""
Floor Plan Preprocessing Module
================================
This module handles image preprocessing specifically designed for hand-drawn floor plans.
It addresses the common issues with floor plan images:
1. Removes noise and text annotations
2. Enhances wall lines
3. Normalizes image quality
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloorPlanPreprocessor:
    """
    Preprocessor specifically designed for floor plan images.
    Handles hand-drawn, scanned, and digital floor plans.
    """
    
    def __init__(self, target_size: int = 800):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target size for the longest dimension
        """
        self.target_size = target_size
        
    def load_and_resize(self, image_path: str) -> Tuple[np.ndarray, float]:
        """
        Load and resize image while maintaining aspect ratio.
        
        Returns:
            Tuple of (resized image, scale factor)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            
        logger.info(f"Loaded image: {image_path}, Original: {w}x{h}, Resized: {img.shape[1]}x{img.shape[0]}")
        return img, scale
    
    def remove_text_regions(self, img: np.ndarray) -> np.ndarray:
        """
        Remove text annotations from floor plan.
        Uses morphological operations to identify and remove text.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect text-like regions (small, dense areas)
        # Use MSER for text detection
        mser = cv2.MSER_create()
        mser.setMinArea(10)
        mser.setMaxArea(500)
        
        regions, _ = mser.detectRegions(gray)
        
        # Create mask for text regions
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], -1, 255, -1)
        
        # Dilate mask to cover entire text areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Inpaint text regions
        result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        
        logger.info(f"Removed {len(regions)} text-like regions")
        return result
    
    def enhance_walls(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance wall lines in the floor plan.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def extract_wall_binary(self, gray: np.ndarray) -> np.ndarray:
        """
        Extract binary image with walls clearly marked.
        Uses adaptive thresholding optimized for floor plans.
        """
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=5
        )
        
        # Remove small noise
        kernel_small = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Connect nearby wall segments
        kernel_connect = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_connect)
        
        return binary
    
    def preprocess(self, image_path: str, remove_text: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path: Path to floor plan image
            remove_text: Whether to attempt text removal
            
        Returns:
            Tuple of (original resized, processed color, binary walls, scale factor)
        """
        # Load and resize
        img, scale = self.load_and_resize(image_path)
        original = img.copy()
        
        # Optionally remove text
        if remove_text:
            try:
                img = self.remove_text_regions(img)
            except Exception as e:
                logger.warning(f"Text removal failed: {e}")
        
        # Enhance walls
        gray = self.enhance_walls(img)
        
        # Extract binary walls
        binary = self.extract_wall_binary(gray)
        
        return original, img, binary, scale


def preprocess_floorplan(image_path: str, target_size: int = 800, remove_text: bool = True):
    """
    Convenience function for preprocessing a floor plan.
    
    Returns:
        dict with 'original', 'processed', 'binary', 'scale'
    """
    preprocessor = FloorPlanPreprocessor(target_size=target_size)
    original, processed, binary, scale = preprocessor.preprocess(image_path, remove_text)
    
    return {
        'original': original,
        'processed': processed,
        'binary': binary,
        'scale': scale
    }
