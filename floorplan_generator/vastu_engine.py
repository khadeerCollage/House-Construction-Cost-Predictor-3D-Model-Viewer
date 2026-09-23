"""
Vastu Shastra Compliance Engine
================================
Implements the traditional Indian architectural science of Vastu Shastra for
optimal compass-based room placement in residential floor plans.

Based on the Vastu Purusha Mandala (a 3×3 grid of 9 zones/padas) aligned
with cardinal compass directions and the five elements (Pancha Bhoota).

Reference: IS 15500 (Indian Standard Guide for Vastu Shastra)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class VastuZone(str, Enum):
    """The 9 zones of the Vastu Purusha Mandala."""
    NE = "NE"        # Ishan (Water/Divine) — Most sacred
    E = "E"          # Indra (Sun/Light)
    SE = "SE"        # Agni (Fire)
    S = "S"          # Yama (Dharma/Rest)
    SW = "SW"        # Niruthi (Earth) — Heaviest
    W = "W"          # Varuna (Water)
    NW = "NW"        # Vayu (Air)
    N = "N"          # Kubera (Wealth)
    CENTER = "CENTER" # Brahmasthan (Ether/Space)


@dataclass
class VastuZoneBounds:
    """Physical boundaries of a Vastu zone in the plot."""
    zone: VastuZone
    x: float      # left edge in meters
    y: float      # bottom edge in meters
    w: float      # width in meters
    h: float      # height in meters


# =============================================================================
# Vastu Compatibility Matrix
# =============================================================================
# Scores: +10 = Optimal, +5 = Acceptable, 0 = Neutral, -20 = Undesirable, -50 = Strict Taboo

_VASTU_COMPATIBILITY: Dict[str, Dict[str, int]] = {
    # Room Category → Zone → Score
    "BEDROOM": {
        "SW": 10,   # Master bedroom — earth, stability, head of family
        "S": 8,     # Good for bedrooms
        "W": 7,     # Good for children's bedroom
        "NW": 5,    # Acceptable for guest bedroom
        "N": 0,     # Neutral
        "E": 0,     # Neutral
        "SE": -20,  # Fire zone — not ideal for rest
        "NE": -20,  # Sacred zone — avoid sleeping
        "CENTER": -50,  # Brahmasthan — strict taboo
    },
    "LIVING": {
        "N": 10,    # Kubera — wealth, prosperity, main gathering
        "NE": 8,    # Divine corner — auspicious for living
        "E": 8,     # Morning sun — excellent for living
        "CENTER": 5, # Brahmasthan — can be open living hall
        "NW": 0,
        "W": 0,
        "S": -5,
        "SW": -10,
        "SE": -5,
    },
    "DINING": {
        "E": 10,    # Morning sun while eating
        "W": 8,     # Acceptable
        "N": 5,     # Good
        "CENTER": 5,
        "S": 0,
        "NE": 0,
        "NW": 0,
        "SE": -5,
        "SW": -10,
    },
    "KITCHEN": {
        "SE": 10,   # Agni (Fire) corner — OPTIMAL for kitchen
        "NW": 5,    # Vayu — second best (wind fans the flames)
        "E": 5,     # Cook facing east — good
        "S": 0,
        "W": -5,
        "N": -10,
        "CENTER": -50,  # Strict taboo
        "NE": -50,      # Sacred — strict taboo for kitchen
        "SW": -20,      # Earth — not for fire
    },
    "BATHROOM": {
        "NW": 10,   # Vayu (Air) — optimal for waste/elimination
        "W": 8,     # Water element
        "N": 0,
        "S": 0,
        "E": -5,
        "CENTER": -50,  # Strict taboo
        "NE": -50,      # Most sacred — NEVER toilet here
        "SE": -10,
        "SW": -20,      # Earth zone — avoid waste
    },
    "POOJA": {
        "NE": 10,   # Ishan — OPTIMAL, most sacred corner
        "E": 8,     # Indra — deities face west
        "N": 5,     # Kubera — acceptable
        "CENTER": 0,
        "NW": -10,
        "W": -10,
        "S": -20,
        "SW": -20,
        "SE": -50,  # Fire zone — taboo for prayer
    },
    "STUDY": {
        "E": 10,    # Indra — morning sun, knowledge
        "NE": 8,    # Divine — concentration
        "N": 8,     # Kubera — wealth through knowledge
        "W": 5,
        "NW": 0,
        "S": 0,
        "CENTER": 0,
        "SE": -5,
        "SW": -10,
    },
    "CORRIDOR": {
        "CENTER": 10,  # Brahmasthan — ideal for open passage
        "N": 5,
        "E": 5,
        "S": 5,
        "W": 5,
        "NE": 5,
        "NW": 5,
        "SE": 5,
        "SW": 5,
    },
    "UTILITY": {
        "NW": 10,   # Vayu — utilities, storage
        "W": 5,
        "S": 5,
        "SE": 0,
        "N": 0,
        "E": -5,
        "CENTER": -20,
        "NE": -20,
        "SW": 0,
    },
    "SERVANT": {
        "NW": 10,   # Traditional placement
        "SE": 5,
        "W": 5,
        "S": 0,
        "N": -5,
        "E": -5,
        "CENTER": -20,
        "NE": -50,  # Sacred — taboo
        "SW": -20,  # Master's corner
    },
    "FOYER": {
        "N": 10,    # Best entrance direction
        "E": 10,    # Second best entrance
        "NE": 8,    # Auspicious
        "NW": 5,    # Acceptable
        "SE": 0,
        "W": 0,
        "S": -5,
        "CENTER": 0,
        "SW": -50,  # Niruthi — NEVER main entrance here
    },
    "BALCONY": {
        "N": 10,
        "E": 10,
        "NE": 10,
        "NW": 5,
        "SE": 5,
        "W": 0,
        "S": -5,
        "SW": -10,
        "CENTER": -50,
    },
    "STORE": {
        "NW": 10,
        "SW": 5,
        "S": 5,
        "W": 5,
        "SE": 0,
        "N": 0,
        "E": -5,
        "NE": -10,
        "CENTER": -20,
    },
}

# Strict taboo pairs — these room-zone combos are architectural sins
_STRICT_TABOOS = [
    ("BATHROOM", "NE"),   # Toilet in sacred corner
    ("KITCHEN", "NE"),    # Kitchen fire in divine space
    ("KITCHEN", "CENTER"),# Kitchen in Brahmasthan
    ("BATHROOM", "CENTER"),# Toilet in Brahmasthan
    ("BEDROOM", "CENTER"),# Sleeping in Brahmasthan
    ("FOYER", "SW"),      # Main entrance in Niruthi
    ("SERVANT", "NE"),    # Servant quarter in sacred corner
]


class VastuEngine:
    """
    Vastu Shastra compliance engine for room placement optimization.

    The engine computes a 3×3 Vastu Purusha Mandala grid over the plot
    and scores room placements based on traditional compass-direction rules.
    """

    def __init__(self, plot_width: float, plot_height: float,
                 north_direction: str = "N"):
        """
        Initialize the Vastu engine.

        Args:
            plot_width: Plot width in meters (along X axis).
            plot_height: Plot height in meters (along Y axis).
            north_direction: Which edge of the plot faces North.
                           'N' = top edge is North (default, standard).
                           'E' = right edge is North (plot rotated 90° CCW).
                           'S' = bottom edge is North (plot rotated 180°).
                           'W' = left edge is North (plot rotated 90° CW).
        """
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.north_direction = north_direction.upper()
        self._zone_bounds = self._compute_zone_grid()

    def _compute_zone_grid(self) -> Dict[VastuZone, VastuZoneBounds]:
        """
        Compute the 3×3 Vastu Mandala grid boundaries.

        The grid divides the plot into 9 equal zones. The mapping of
        compass directions to physical (x, y) positions depends on the
        north_direction setting.

        Returns:
            Dictionary mapping each VastuZone to its physical bounds.
        """
        w3 = self.plot_width / 3.0
        h3 = self.plot_height / 3.0

        # Physical grid positions (column, row) where (0,0) is bottom-left
        # Each cell is (col * w3, row * h3, w3, h3)
        physical_grid = {}
        for col in range(3):
            for row in range(3):
                physical_grid[(col, row)] = (col * w3, row * h3, w3, h3)

        # Map compass zones to (col, row) based on north direction
        # Standard (N = top): North is row=2, South is row=0, East is col=2, West is col=0
        if self.north_direction == "N":
            zone_map = {
                VastuZone.SW: (0, 0), VastuZone.S:  (1, 0), VastuZone.SE: (2, 0),
                VastuZone.W:  (0, 1), VastuZone.CENTER: (1, 1), VastuZone.E:  (2, 1),
                VastuZone.NW: (0, 2), VastuZone.N:  (1, 2), VastuZone.NE: (2, 2),
            }
        elif self.north_direction == "E":
            # North is to the right (col=2)
            zone_map = {
                VastuZone.SE: (0, 0), VastuZone.E:  (1, 0), VastuZone.NE: (2, 0),
                VastuZone.S:  (0, 1), VastuZone.CENTER: (1, 1), VastuZone.N:  (2, 1),
                VastuZone.SW: (0, 2), VastuZone.W:  (1, 2), VastuZone.NW: (2, 2),
            }
        elif self.north_direction == "S":
            # North is at bottom (row=0)
            zone_map = {
                VastuZone.NE: (0, 0), VastuZone.N:  (1, 0), VastuZone.NW: (2, 0),
                VastuZone.E:  (0, 1), VastuZone.CENTER: (1, 1), VastuZone.W:  (2, 1),
                VastuZone.SE: (0, 2), VastuZone.S:  (1, 2), VastuZone.SW: (2, 2),
            }
        elif self.north_direction == "W":
            # North is to the left (col=0)
            zone_map = {
                VastuZone.NW: (0, 0), VastuZone.W:  (1, 0), VastuZone.SW: (2, 0),
                VastuZone.N:  (0, 1), VastuZone.CENTER: (1, 1), VastuZone.S:  (2, 1),
                VastuZone.NE: (0, 2), VastuZone.E:  (1, 2), VastuZone.SE: (2, 2),
            }
        else:
            raise ValueError(f"Invalid north_direction: '{self.north_direction}'. Use 'N', 'E', 'S', or 'W'.")

        bounds = {}
        for zone, (col, row) in zone_map.items():
            x, y, w, h = physical_grid[(col, row)]
            bounds[zone] = VastuZoneBounds(zone=zone, x=x, y=y, w=w, h=h)

        return bounds

    def get_zone_bounds(self, zone: VastuZone) -> VastuZoneBounds:
        """Get the physical bounds of a Vastu zone."""
        return self._zone_bounds[zone]

    def get_all_zones(self) -> Dict[VastuZone, VastuZoneBounds]:
        """Get all zone boundaries."""
        return dict(self._zone_bounds)

    def get_zone_for_position(self, x: float, y: float) -> VastuZone:
        """
        Determine which Vastu zone a point falls into.

        Args:
            x: X coordinate in meters.
            y: Y coordinate in meters.

        Returns:
            The VastuZone containing the point.
        """
        for zone, bounds in self._zone_bounds.items():
            if (bounds.x <= x < bounds.x + bounds.w and
                    bounds.y <= y < bounds.y + bounds.h):
                return zone
        # Edge case: point on right/top boundary
        return VastuZone.CENTER

    def get_room_score(self, room_category: str, zone: str) -> int:
        """
        Get the Vastu compatibility score for placing a room category in a zone.

        Args:
            room_category: Room category string (e.g., 'BEDROOM', 'KITCHEN').
            zone: Zone string (e.g., 'SW', 'NE', 'CENTER').

        Returns:
            Compatibility score (-50 to +10).
        """
        category = room_category.upper()
        zone_key = zone.upper()

        if category in _VASTU_COMPATIBILITY:
            return _VASTU_COMPATIBILITY[category].get(zone_key, 0)
        return 0  # Unknown category = neutral

    def is_taboo(self, room_category: str, zone: str) -> bool:
        """Check if placing a room category in a zone is a strict taboo."""
        return (room_category.upper(), zone.upper()) in _STRICT_TABOOS

    def get_optimal_zone(self, room_category: str) -> VastuZone:
        """
        Get the optimal Vastu zone for a room category.

        Args:
            room_category: Room category string.

        Returns:
            The VastuZone with the highest compatibility score.
        """
        category = room_category.upper()
        if category not in _VASTU_COMPATIBILITY:
            return VastuZone.CENTER

        scores = _VASTU_COMPATIBILITY[category]
        best_zone = max(scores, key=scores.get)
        return VastuZone(best_zone)

    def get_acceptable_zones(self, room_category: str,
                              min_score: int = 0) -> List[VastuZone]:
        """
        Get all acceptable zones for a room category (score >= min_score).

        Args:
            room_category: Room category string.
            min_score: Minimum acceptable score (default 0 = neutral or better).

        Returns:
            List of VastuZones sorted by score (best first).
        """
        category = room_category.upper()
        if category not in _VASTU_COMPATIBILITY:
            return list(VastuZone)

        scores = _VASTU_COMPATIBILITY[category]
        acceptable = [
            (VastuZone(zone), score)
            for zone, score in scores.items()
            if score >= min_score
        ]
        acceptable.sort(key=lambda x: x[1], reverse=True)
        return [zone for zone, _ in acceptable]

    def compute_plan_score(self, room_placements: List[Tuple[str, str]]) -> float:
        """
        Compute the overall Vastu compliance score for a complete floor plan.

        Args:
            room_placements: List of (room_category, zone) tuples.

        Returns:
            Normalized score from 0 to 100.
        """
        if not room_placements:
            return 100.0

        total_score = 0
        max_possible = 0
        taboo_count = 0

        for category, zone in room_placements:
            score = self.get_room_score(category, zone)
            total_score += score
            max_possible += 10  # Maximum possible score per room

            if self.is_taboo(category, zone):
                taboo_count += 1

        if max_possible == 0:
            return 100.0

        # Normalize: shift from [-50..+10] range to [0..100]
        # With taboo penalty
        raw_ratio = (total_score + max_possible * 5) / (max_possible * 1.5)
        normalized = max(0.0, min(100.0, raw_ratio * 100))

        # Heavy penalty for each taboo violation
        normalized -= taboo_count * 15

        return max(0.0, min(100.0, normalized))

    def get_placement_report(self, room_placements: List[Tuple[str, str, str]]) -> Dict:
        """
        Generate a detailed Vastu compliance report.

        Args:
            room_placements: List of (room_name, room_category, zone) tuples.

        Returns:
            Dictionary with score, details, warnings, and recommendations.
        """
        details = []
        warnings = []
        taboos = []

        for name, category, zone in room_placements:
            score = self.get_room_score(category, zone)
            optimal = self.get_optimal_zone(category)

            detail = {
                "room": name,
                "category": category,
                "zone": zone,
                "score": score,
                "optimal_zone": optimal.value,
                "status": "optimal" if score >= 8 else
                          "good" if score >= 5 else
                          "acceptable" if score >= 0 else
                          "undesirable" if score > -50 else "TABOO"
            }
            details.append(detail)

            if self.is_taboo(category, zone):
                taboos.append(
                    f"⛔ TABOO: {name} ({category}) in {zone} zone — "
                    f"move to {optimal.value}"
                )
            elif score < 0:
                warnings.append(
                    f"⚠️ {name} ({category}) in {zone} zone (score: {score}) — "
                    f"better in {optimal.value}"
                )

        category_zone_pairs = [(cat, zone) for _, cat, zone in room_placements]
        overall_score = self.compute_plan_score(category_zone_pairs)

        return {
            "overall_score": round(overall_score, 1),
            "rating": (
                "Excellent" if overall_score >= 85 else
                "Good" if overall_score >= 70 else
                "Acceptable" if overall_score >= 50 else
                "Poor" if overall_score >= 30 else
                "Non-Compliant"
            ),
            "details": details,
            "warnings": warnings,
            "taboos": taboos,
            "total_rooms": len(room_placements),
            "taboo_count": len(taboos),
        }
