"""
Indian Land Units & Shape Constraint Engine
============================================
Converts and auto-constrains land parcels based on traditional and official
Indian land measurement systems:
  - Cents (Ubiquitous in South/Central India: TN, Kerala, AP, Telangana, Karnataka)
  - Guntha (Maharashtra, Karnataka, Telangana: 1 Guntha = 2.5 Cents = 1,089 sq ft)
  - Ground (Tamil Nadu: 1 Ground = 5.51 Cents = 2,400 sq ft)
  - Gaj / Square Yards (North India: 1 Gaj = 9 sq ft = 0.02066 Cents)
  - Ankanam (Andhra Pradesh: 1 Ankanam = 72 sq ft = 0.165 Cents)
  - Square Feet & Square Meters (Universal Indian Municipal Standards)

Geometry Solvers:
  - Square (Chaturasra) — Width = Length = sqrt(Area)
  - Rectangle (Ayatasra) — Proportional aspect ratio or fixed frontage width
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import math


# =============================================================================
# Indian Land Unit Constants
# =============================================================================
SQFT_PER_CENT = 435.60
SQM_PER_CENT = 40.4685642
GAJ_PER_CENT = 48.40              # 1 Gaj = 1 Sq Yard = 9 sq ft
GUNTHA_PER_CENT = 0.40            # 1 Guntha = 2.5 Cents = 1089 sq ft
GROUND_PER_CENT = 0.1815          # 1 Ground = 2400 sq ft = 5.5096 Cents
ANKANAM_PER_CENT = 6.05           # 1 Ankanam = 72 sq ft = 0.165289 Cents
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 0.3048


@dataclass
class LandParcel:
    """Represents a fully constrained land parcel with multi-unit equivalents."""
    cents: float
    shape: str               # 'Square' or 'Rectangle'
    aspect_ratio: float      # Length / Width
    width_m: float
    length_m: float
    width_ft: float
    length_ft: float
    area_sqm: float
    area_sqft: float
    area_gaj: float          # Sq Yards
    area_guntha: float
    area_ground: float
    area_ankanam: float
    recommended_bhk: str
    vastu_geometry_grade: str
    description: str


class LandUnitsEngine:
    """Mathematical solver and converter for Indian land parcels."""

    @staticmethod
    def cents_to_sqm(cents: float) -> float:
        """Converts Cents to square meters."""
        return cents * SQM_PER_CENT

    @staticmethod
    def cents_to_sqft(cents: float) -> float:
        """Converts Cents to square feet."""
        return cents * SQFT_PER_CENT

    @staticmethod
    def sqm_to_cents(sqm: float) -> float:
        """Converts square meters to Cents."""
        return sqm / SQM_PER_CENT

    @staticmethod
    def sqft_to_cents(sqft: float) -> float:
        """Converts square feet to Cents."""
        return sqft / SQFT_PER_CENT

    @staticmethod
    def recommend_bhk_for_cents(cents: float) -> str:
        """
        Recommends the ideal BHK configuration based on land Cents.
        Guidelines:
          < 1.8 Cents (< 780 sq ft)   -> 1BHK
          1.8 - 3.2 Cents (780 - 1400 sq ft) -> 2BHK
          3.3 - 5.2 Cents (1400 - 2265 sq ft) -> 3BHK
          > 5.2 Cents (> 2265 sq ft)  -> 4BHK
        """
        if cents < 1.8:
            return "1BHK"
        elif cents < 3.2:
            return "2BHK"
        elif cents < 5.2:
            return "3BHK"
        else:
            return "4BHK"

    @classmethod
    def solve_parcel(
        cls,
        cents: float,
        shape: str = "Rectangle",
        aspect_ratio: float = 1.33,
        frontage_ft: Optional[float] = None
    ) -> LandParcel:
        """
        Constrains a plot geometry based on Cents and desired shape.

        Args:
            cents: Total plot area in Cents (e.g. 2.5, 3.0, 5.0).
            shape: 'Square' or 'Rectangle'.
            aspect_ratio: Length/Width ratio (ignored if shape is Square or frontage_ft is given).
            frontage_ft: Optional road frontage width in feet. If supplied, depth is auto-calculated.

        Returns:
            LandParcel dataclass with synchronized dimensions and multi-unit conversions.
        """
        cents = max(0.5, float(cents))
        total_sqft = cents * SQFT_PER_CENT
        total_sqm = cents * SQM_PER_CENT

        if shape.lower().startswith("square"):
            # Square: Width = Length = sqrt(Area)
            side_m = math.sqrt(total_sqm)
            width_m = round(side_m, 2)
            length_m = round(side_m, 2)
            width_ft = round(width_m * METERS_TO_FEET, 1)
            length_ft = round(length_m * METERS_TO_FEET, 1)
            actual_ratio = 1.0
            vastu_grade = "Chaturasra (Square) — Supreme Vedic Auspiciousness"
            desc = f"Square plot of {cents:.2f} Cents ({width_ft:.1f}' x {length_ft:.1f}'). Flawless 1:1 symmetry."

        else:
            # Rectangle
            if frontage_ft and frontage_ft > 5.0:
                # Custom frontage in feet
                width_ft = float(frontage_ft)
                length_ft = total_sqft / width_ft
                width_m = round(width_ft * FEET_TO_METERS, 2)
                length_m = round(length_ft * FEET_TO_METERS, 2)
                actual_ratio = round(length_ft / width_ft, 2)
            else:
                # Ratio-based: Length = ratio * Width
                # Area = Width * Length = Width * (ratio * Width) = ratio * Width^2
                # Width = sqrt(Area / ratio)
                ratio = max(0.5, min(4.0, float(aspect_ratio)))
                w_m = math.sqrt(total_sqm / ratio)
                l_m = w_m * ratio
                width_m = round(w_m, 2)
                length_m = round(l_m, 2)
                width_ft = round(width_m * METERS_TO_FEET, 1)
                length_ft = round(length_m * METERS_TO_FEET, 1)
                actual_ratio = round(ratio, 2)

            if 1.1 <= actual_ratio <= 1.65:
                vastu_grade = "Ayatasra (Harmonic Rectangle) — Highly Auspicious Vedic Proportion"
            elif actual_ratio < 1.1:
                vastu_grade = "Near-Square Chaturasra — Auspicious"
            else:
                vastu_grade = "Dirgha (Deep/Narrow) — Requires internal partitioning for optimal ventilation"

            desc = f"Rectangular plot of {cents:.2f} Cents ({width_ft:.1f}' x {length_ft:.1f}') with aspect ratio 1:{actual_ratio:.2f}."

        rec_bhk = cls.recommend_bhk_for_cents(cents)

        return LandParcel(
            cents=round(cents, 2),
            shape="Square" if shape.lower().startswith("square") else "Rectangle",
            aspect_ratio=actual_ratio,
            width_m=width_m,
            length_m=length_m,
            width_ft=width_ft,
            length_ft=length_ft,
            area_sqm=round(total_sqm, 2),
            area_sqft=round(total_sqft, 1),
            area_gaj=round(total_sqft / 9.0, 1),
            area_guntha=round(cents / 2.50, 2),
            area_ground=round(cents / 5.5096, 2),
            area_ankanam=round(total_sqft / 72.0, 1),
            recommended_bhk=rec_bhk,
            vastu_geometry_grade=vastu_grade,
            description=desc
        )
