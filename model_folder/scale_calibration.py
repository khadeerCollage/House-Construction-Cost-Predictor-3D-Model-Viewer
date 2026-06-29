"""
Scale Calibration Module
========================
Calculates the pixels-to-meters ratio and converts pixel measurements (lengths, areas) 
into real-world metric units (meters, square meters) for floor plan analysis.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScaleCalibrator:
    """
    Handles calibration calculations to map pixel measurements to real-world meters.
    """
    
    def __init__(
        self,
        image_shape = None,  # (height, width)
        plot_width_m = None,
        plot_length_m = None,
        pixels_per_meter = None
    ):
        """
        Initialize the calibrator.
        
        Args:
            image_shape: Tuple of (height, width) of the floorplan image
            plot_width_m: User-defined plot width in meters
            plot_length_m: User-defined plot length/height in meters
            pixels_per_meter: Manually set scale factor (if known)
        """
        self.image_shape = image_shape
        self.plot_width_m = plot_width_m
        self.plot_length_m = plot_length_m
        self.ppm = pixels_per_meter
        
        if self.ppm is None:
            self.ppm = self._calculate_ppm()
            
        logger.info(f"Initialized ScaleCalibrator with ppm = {self.ppm:.2f} pixels/meter")
        
    def _calculate_ppm(self) -> float:
        """
        Calculate pixels per meter scale factor.
        """
        # If manual ppm is not set and plot size is given:
        if self.image_shape is not None and len(self.image_shape) >= 2:
            img_h, img_w = self.image_shape[:2]
            
            # Case 1: Both width and length are provided
            if self.plot_width_m and self.plot_length_m:
                ppm_w = img_w / self.plot_width_m
                ppm_h = img_h / self.plot_length_m
                # Average to get uniform scale
                return (ppm_w + ppm_h) / 2.0
                
            # Case 2: Only width is provided
            elif self.plot_width_m:
                return img_w / self.plot_width_m
                
            # Case 3: Only length is provided
            elif self.plot_length_m:
                return img_h / self.plot_length_m
        
        # Fallback default: assume 40 pixels per meter (approx. 1 pixel = 2.5 cm, common for blueprints)
        # Or estimate based on average size
        return 40.0

    def pixel_length_to_meters(self, pixels: float) -> float:
        """Convert pixel distance to meters."""
        if self.ppm <= 0:
            return 0.0
        return pixels / self.ppm

    def pixel_area_to_sqm(self, pixel_area: float) -> float:
        """Convert pixel area to square meters."""
        if self.ppm <= 0:
            return 0.0
        # Area scale is ppm squared
        return pixel_area / (self.ppm ** 2)

    def sqm_to_pixel_area(self, sqm: float) -> float:
        """Convert square meters back to pixel area."""
        return sqm * (self.ppm ** 2)

    @classmethod
    def get_fallback_ppm_for_rooms(cls, room_count: int, image_shape) -> float:
        """
        Estimate a reasonable scale if the user did not calibrate,
        assuming an average Indian home size (approx. 100 sqm for a standard 3BHK).
        """
        if image_shape is None or len(image_shape) < 2:
            return 40.0
            
        img_h, img_w = image_shape[:2]
        total_pixel_area = img_h * img_w
        
        # Guess total sqm based on room count:
        # 1 room (studio/other) ~ 30 sqm
        # 2 rooms ~ 50 sqm
        # 3-4 rooms ~ 80 sqm
        # 5-6 rooms ~ 120 sqm
        # 7+ rooms ~ 180 sqm
        if room_count <= 1:
            guessed_sqm = 30.0
        elif room_count == 2:
            guessed_sqm = 50.0
        elif room_count <= 4:
            guessed_sqm = 85.0
        elif room_count <= 6:
            guessed_sqm = 130.0
        else:
            guessed_sqm = 200.0
            
        # We assume the floor plan occupes about 65% of the total canvas area
        plan_pixel_area = total_pixel_area * 0.65
        ppm = np.sqrt(plan_pixel_area / guessed_sqm)
        return float(ppm)
