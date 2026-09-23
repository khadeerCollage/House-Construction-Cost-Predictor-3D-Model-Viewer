"""
Family Intent Analyzer
=======================
Converts household composition and lifestyle preferences into an optimal
BHK configuration and room schedule.

This module bridges the gap between human-readable family descriptions
("2 adults, 2 children, need home office") and the technical room
specifications consumed by the layout engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from floorplan_generator.room_specs import (
    BHKConfig, RoomTemplate, RoomCategory,
    get_bhk_config, recommend_bhk_for_plot, get_all_configs,
    NBC_ROOM_SPECS,
    DOOR_BEDROOM, DOOR_BATHROOM, DOOR_INTERNAL,
    WINDOW_BEDROOM, WINDOW_BATHROOM,
)


@dataclass
class FamilyProfile:
    """
    Household composition and preferences.

    This is the primary user input — everything needed to determine
    the optimal floor plan configuration.
    """
    adults: int = 2               # Number of adults (18+)
    children: int = 0             # Number of children (< 18)
    elderly: int = 0              # Number of elderly members (65+)
    guests_frequent: bool = False # Frequent overnight guests?

    # Lifestyle preferences
    needs_home_office: bool = False   # Dedicated study/office room
    needs_pooja_room: bool = True     # Prayer room (default for Indian homes)
    needs_servant_quarter: bool = False  # Servant room with bath
    needs_store_room: bool = False    # Dedicated storage room

    # Override
    bhk_override: Optional[str] = None  # Force a specific BHK config

    @property
    def total_members(self) -> int:
        """Total family members."""
        return self.adults + self.children + self.elderly

    @property
    def description(self) -> str:
        """Human-readable family description."""
        parts = []
        if self.adults:
            parts.append(f"{self.adults} Adult{'s' if self.adults > 1 else ''}")
        if self.children:
            parts.append(f"{self.children} Child{'ren' if self.children > 1 else ''}")
        if self.elderly:
            parts.append(f"{self.elderly} Elder{'s' if self.elderly > 1 else ''}")
        if self.guests_frequent:
            parts.append("Frequent Guests")
        return ", ".join(parts) if parts else "Single Person"


@dataclass
class AnalysisResult:
    """
    Result of family analysis — the recommended configuration.
    """
    recommended_bhk: str              # '1BHK', '2BHK', etc.
    config: BHKConfig                 # Full BHK configuration
    family_description: str           # Human-readable description
    reasoning: List[str]              # Why this config was chosen
    bedrooms_needed: int              # Calculated bedroom count
    bathrooms_needed: int             # Calculated bathroom count
    additional_rooms: List[str]       # Extra rooms added (study, pooja, etc.)
    min_plot_area: float              # Minimum plot area needed (m²)
    recommended_plot_area: float      # Recommended plot area (m²)


class FamilyAnalyzer:
    """
    Analyzes family composition and recommends the optimal BHK configuration.

    Usage:
        analyzer = FamilyAnalyzer()
        profile = FamilyProfile(adults=2, children=2, needs_pooja_room=True)
        result = analyzer.analyze(profile)
        print(f"Recommended: {result.recommended_bhk}")
        print(f"Reasoning: {result.reasoning}")
    """

    def analyze(self, profile: FamilyProfile,
                plot_area: Optional[float] = None) -> AnalysisResult:
        """
        Analyze a family profile and recommend a BHK configuration.

        Args:
            profile: FamilyProfile with household details.
            plot_area: Optional plot area in m² (used to constrain recommendation).

        Returns:
            AnalysisResult with the recommended configuration and reasoning.
        """
        reasoning = []
        additional_rooms = []

        # Step 1: Calculate bedroom needs
        bedrooms = self._calculate_bedrooms(profile, reasoning)

        # Step 2: Calculate bathroom needs
        bathrooms = self._calculate_bathrooms(profile, bedrooms, reasoning)

        # Step 3: Determine additional rooms
        if profile.needs_home_office:
            additional_rooms.append("Study / Home Office")
            reasoning.append("Home office added for remote work needs")

        if profile.needs_pooja_room and bedrooms >= 2:
            additional_rooms.append("Pooja Room")
            reasoning.append("Pooja room added (Indian residential tradition)")

        if profile.needs_servant_quarter and bedrooms >= 3:
            additional_rooms.append("Servant Quarter")
            reasoning.append("Servant quarter added for household help")

        if profile.needs_store_room:
            additional_rooms.append("Store Room")
            reasoning.append("Dedicated storage room added")

        # Step 4: Map to BHK configuration
        if profile.bhk_override:
            bhk = profile.bhk_override.upper().replace(" ", "")
            reasoning.insert(0, f"Using user-specified configuration: {bhk}")
        else:
            bhk = self._determine_bhk(bedrooms, additional_rooms, reasoning)

        # Step 5: Constrain by plot area if provided
        if plot_area:
            max_bhk = recommend_bhk_for_plot(plot_area)
            bhk_order = ["1BHK", "2BHK", "3BHK", "4BHK"]
            if bhk_order.index(bhk) > bhk_order.index(max_bhk):
                reasoning.append(
                    f"Downgraded from {bhk} to {max_bhk} due to plot area "
                    f"constraint ({plot_area:.0f} m²)"
                )
                bhk = max_bhk

        # Step 6: Get the full configuration
        config = get_bhk_config(bhk)

        return AnalysisResult(
            recommended_bhk=bhk,
            config=config,
            family_description=profile.description,
            reasoning=reasoning,
            bedrooms_needed=bedrooms,
            bathrooms_needed=bathrooms,
            additional_rooms=additional_rooms,
            min_plot_area=config.min_plot_area,
            recommended_plot_area=config.recommended_plot_area,
        )

    def _calculate_bedrooms(self, profile: FamilyProfile,
                            reasoning: List[str]) -> int:
        """
        Calculate the number of bedrooms needed.

        Indian residential norms:
        - Couple: 1 master bedroom
        - Children: share until age 10, then separate (1 room per 2 children)
        - Elderly: 1 separate bedroom (ground floor preferred)
        - Frequent guests: 1 additional guest room
        """
        bedrooms = 0

        # Adults need at least 1 master bedroom
        if profile.adults >= 1:
            bedrooms += 1
            reasoning.append("1 Master Bedroom for adult couple/single")

        # Additional adult couples (joint family)
        extra_adult_pairs = max(0, (profile.adults - 2) // 2)
        if extra_adult_pairs > 0:
            bedrooms += extra_adult_pairs
            reasoning.append(f"{extra_adult_pairs} additional bedroom(s) for extra adults")

        # Children's bedrooms
        if profile.children >= 1:
            child_rooms = max(1, (profile.children + 1) // 2)  # 1 room per 2 children
            bedrooms += child_rooms
            reasoning.append(f"{child_rooms} bedroom(s) for {profile.children} children")

        # Elderly bedroom
        if profile.elderly >= 1:
            bedrooms += 1
            reasoning.append("1 bedroom for elderly (ground floor, attached bath)")

        # Guest bedroom
        if profile.guests_frequent:
            bedrooms += 1
            reasoning.append("1 guest bedroom for frequent overnight visitors")

        # Cap at 4 for practical Indian residential
        bedrooms = min(bedrooms, 4)

        return bedrooms

    def _calculate_bathrooms(self, profile: FamilyProfile,
                             bedrooms: int,
                             reasoning: List[str]) -> int:
        """
        Calculate bathroom count.

        Indian residential norms:
        - Master bedroom: attached bathroom (always)
        - Elderly bedroom: attached bathroom (accessibility)
        - Common bathroom: 1 per 2-3 remaining bedrooms
        - 4BHK+: each bedroom gets attached bath
        """
        bathrooms = 1  # Master bath (always)
        reasoning.append("1 attached bathroom for master bedroom")

        if profile.elderly >= 1:
            bathrooms += 1
            reasoning.append("1 attached bathroom for elderly accessibility")

        if bedrooms >= 3:
            # Common bathroom + additional attached
            bathrooms += 1
            reasoning.append("1 common bathroom for family/guests")

        if bedrooms >= 4:
            bathrooms += 1
            reasoning.append("1 additional bathroom for 4-bedroom layout")

        return bathrooms

    def _determine_bhk(self, bedrooms: int,
                       additional_rooms: List[str],
                       reasoning: List[str]) -> str:
        """Map bedroom count to BHK configuration."""
        if bedrooms <= 1:
            bhk = "1BHK"
            reasoning.append(f"=> Recommended: 1BHK (1 bedroom needed)")
        elif bedrooms == 2:
            bhk = "2BHK"
            reasoning.append(f"=> Recommended: 2BHK (2 bedrooms needed)")
        elif bedrooms == 3:
            bhk = "3BHK"
            reasoning.append(f"=> Recommended: 3BHK (3 bedrooms needed)")
        else:
            bhk = "4BHK"
            reasoning.append(f"=> Recommended: 4BHK ({bedrooms} bedrooms needed)")

        # If many additional rooms are needed, consider upgrading
        if len(additional_rooms) >= 2 and bhk in ("1BHK", "2BHK"):
            old_bhk = bhk
            bhk_order = ["1BHK", "2BHK", "3BHK", "4BHK"]
            idx = bhk_order.index(bhk)
            if idx < 3:
                bhk = bhk_order[idx + 1]
                reasoning.append(
                    f"Upgraded {old_bhk} => {bhk} to accommodate "
                    f"{len(additional_rooms)} additional rooms"
                )

        return bhk

    def get_plot_recommendation(self, bhk: str) -> Dict[str, float]:
        """
        Get recommended plot dimensions for a BHK configuration.

        Returns:
            Dictionary with 'min_area', 'recommended_area',
            'min_width', 'min_height', 'recommended_width', 'recommended_height'.
        """
        config = get_bhk_config(bhk)

        # Standard Indian plot proportions (roughly 2:3 or 3:4)
        rec_area = config.recommended_plot_area
        ratio = 0.75  # width/height ratio (3:4)

        rec_height = math.sqrt(rec_area / ratio)
        rec_width = rec_area / rec_height

        min_area = config.min_plot_area
        min_height = math.sqrt(min_area / ratio)
        min_width = min_area / min_height

        return {
            "min_area": round(min_area, 1),
            "recommended_area": round(rec_area, 1),
            "min_width": round(min_width, 1),
            "min_height": round(min_height, 1),
            "recommended_width": round(rec_width, 1),
            "recommended_height": round(rec_height, 1),
        }


# Need math for plot recommendation
import math
