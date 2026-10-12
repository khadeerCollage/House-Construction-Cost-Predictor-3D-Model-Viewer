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

    # ── Role-specific bedroom rows (used when the occupant role is known) ──
    "MASTER_BEDROOM": {
        "SW": 10, "S": 7, "W": 6, "NW": 0, "N": -5, "E": -5,
        "SE": -20, "NE": -30, "CENTER": -50,
    },
    "CHILD_BEDROOM": {
        "W": 10, "NW": 8, "S": 7, "E": 6, "N": 6, "SW": 5,
        "SE": -10, "NE": -20, "CENTER": -50,
    },
    "GUEST_BEDROOM": {
        "NW": 10, "W": 8, "N": 5, "E": 4, "S": 4, "SW": 2,
        "SE": -10, "NE": -20, "CENTER": -50,
    },
    "ELDER_BEDROOM": {
        "SW": 9, "S": 8, "W": 8, "NW": 5, "E": 3, "N": 3,
        "SE": -15, "NE": -20, "CENTER": -50,
    },

    # ── Vertical circulation ────────────────────────────────────────────
    "STAIRS": {
        # Heavy, clockwise-ascending stairs belong to S / W / SW (Yama, Varuna, Niruthi)
        "S": 10, "W": 10, "SW": 8, "SE": 5, "NW": 5,
        "N": -10, "E": -10, "NE": -50, "CENTER": -50,
    },
    "LIFT": {
        "S": 8, "W": 8, "SW": 6, "NW": 6, "SE": 4, "N": 0, "E": 0,
        "NE": -30, "CENTER": -50,
    },

    # ── Site amenities (scored against the PLOT mandala) ────────────────
    "PARKING": {
        "NW": 10, "SE": 8, "N": 5, "E": 5, "W": 3, "S": 3,
        "SW": -20, "NE": -20, "CENTER": -50,
    },
    "BIKE_PARKING": {
        "NW": 10, "SE": 8, "N": 5, "E": 5, "W": 3, "S": 3,
        "SW": -20, "NE": -20, "CENTER": -50,
    },
    "GARDEN": {
        "NE": 10, "N": 10, "E": 10, "NW": 5, "CENTER": 5, "SE": 3, "W": 0,
        "S": -5, "SW": -10,
    },
    "POOL": {
        "NE": 10, "N": 9, "E": 8, "NW": 0, "W": -10,
        "SE": -30, "S": -40, "SW": -50, "CENTER": -50,
    },
    "WATER_TANK": {
        # Underground sump = water body below ground → Ishan (NE) only
        "NE": 10, "N": 8, "E": 8, "NW": 0, "W": -10,
        "SE": -40, "S": -40, "SW": -50, "CENTER": -50,
    },
    "GENERATOR": {
        "SE": 10, "NW": 6, "S": 5, "E": 0, "W": 0, "SW": -10,
        "N": -20, "NE": -50, "CENTER": -50,
    },
    "SECURITY": {
        "NW": 10, "SE": 8, "N": 5, "E": 5, "W": 3, "S": 3,
        "NE": -20, "SW": -20, "CENTER": -50,
    },
}

# Strict taboo pairs — these room-zone combos are architectural sins
_STRICT_TABOOS = [
    ("BATHROOM", "NE"),   # Toilet in sacred corner
    ("KITCHEN", "NE"),    # Kitchen fire in divine space
    ("KITCHEN", "CENTER"),# Kitchen in Brahmasthan
    ("BATHROOM", "CENTER"),# Toilet in Brahmasthan
    ("BEDROOM", "CENTER"),# Sleeping in Brahmasthan
    ("MASTER_BEDROOM", "CENTER"),
    ("CHILD_BEDROOM", "CENTER"),
    ("GUEST_BEDROOM", "CENTER"),
    ("ELDER_BEDROOM", "CENTER"),
    ("FOYER", "SW"),      # Main entrance in Niruthi
    ("SERVANT", "NE"),    # Servant quarter in sacred corner
    ("STAIRS", "NE"),     # Heavy stair mass blocks the Ishan inflow
    ("STAIRS", "CENTER"), # Stairs pierce the Brahmasthan
    ("LIFT", "CENTER"),
    ("POOL", "SW"),       # Water body in Niruthi
    ("WATER_TANK", "SW"), # Underground sump in Niruthi
    ("WATER_TANK", "SE"), # Water in Agni — fire/water clash
    ("GENERATOR", "NE"),  # Fire equipment in Ishan
]

# Relative importance of each room type in the overall Vastu score.
# Classical texts weight the master suite, kitchen, pooja and main entrance most.
ROOM_IMPORTANCE: Dict[str, float] = {
    "MASTER_BEDROOM": 3.0, "KITCHEN": 3.0, "POOJA": 2.5, "ENTRANCE": 3.0,
    "LIVING": 2.0, "WATER_TANK": 2.0, "ELDER_BEDROOM": 1.75,
    "BEDROOM": 1.5, "CHILD_BEDROOM": 1.5, "GUEST_BEDROOM": 1.25,
    "BATHROOM": 1.5, "STAIRS": 1.5, "POOL": 1.5,
    "DINING": 1.0, "STUDY": 1.0, "LIFT": 1.0, "FOYER": 1.0, "GENERATOR": 1.0,
    "UTILITY": 0.75, "STORE": 0.75, "SERVANT": 0.75, "PARKING": 0.75, "GARDEN": 0.75,
    "BIKE_PARKING": 0.5, "SECURITY": 0.5, "CORRIDOR": 0.5, "BALCONY": 0.5,
}


# =============================================================================
# Compass ↔ plan-frame geometry helpers
# =============================================================================
# Plan frame convention used across the generator: +x = right, +y = up, and the
# road / main-entrance side ("facing") is the TOP edge (y = plot_height).

COMPASS_ORDER = ["N", "E", "S", "W"]
_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _rot(direction: str, quarter_turns_cw: int) -> str:
    idx = COMPASS_ORDER.index(direction.upper())
    return COMPASS_ORDER[(idx + quarter_turns_cw) % 4]


def plan_sides_for_facing(facing: str) -> Dict[str, str]:
    """
    Compass direction of each plan edge when the road/entrance side is `facing`
    and that side is drawn at the top of the plan.

    >>> plan_sides_for_facing("E")
    {'top': 'E', 'bottom': 'W', 'left': 'N', 'right': 'S'}
    """
    f = (facing or "N").upper()[0]
    if f not in COMPASS_ORDER:
        f = "N"
    return {"top": f, "bottom": _OPPOSITE[f], "left": _rot(f, -1), "right": _rot(f, 1)}


def north_edge_for_facing(facing: str) -> str:
    """
    Convert a road-facing direction to the VastuEngine `north_direction` code
    (which plan edge points North: 'N' top, 'E' right, 'S' bottom, 'W' left).
    """
    sides = plan_sides_for_facing(facing)
    edge_code = {"top": "N", "right": "E", "bottom": "S", "left": "W"}
    for edge, compass in sides.items():
        if compass == "N":
            return edge_code[edge]
    return "N"


def compass_vector(direction: str, facing: str) -> Tuple[float, float]:
    """Unit plan-frame vector pointing towards a compass direction."""
    vec = {"top": (0.0, 1.0), "bottom": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
    for edge, compass in plan_sides_for_facing(facing).items():
        if compass == direction.upper():
            return vec[edge]
    return (0.0, 1.0)


def compose_zone(*directions: Optional[str]) -> str:
    """Combine a N/S and an E/W component into a zone code ('N'+'E' → 'NE')."""
    dirs = {d.upper() for d in directions if d}
    ns = "N" if "N" in dirs else "S" if "S" in dirs else ""
    ew = "E" if "E" in dirs else "W" if "W" in dirs else ""
    return (ns + ew) or "CENTER"


# =============================================================================
# 32-Pada entrance system (8 padas per side, clockwise from the corner)
# =============================================================================

@dataclass
class EntranceGateInfo:
    pada_id: str
    deity: str
    direction: str
    angle_start: float   # legacy fields (kept for backward compatibility)
    angle_end: float
    rating: str          # 'SUPER_AUSPICIOUS', 'AUSPICIOUS', 'NEUTRAL', 'NEGATIVE', 'TABOO'
    score: int           # -50 to +10
    effect: str          # Traditional effect description
    remedy: str = ""     # Remedy if negative


_PADA_DATABASE: Dict[str, EntranceGateInfo] = {
    # North side, counted from the NW corner towards NE
    "N1": EntranceGateInfo("N1", "Roga", "North", 348.75, 0.0, "NEGATIVE", -20, "Chronic illness, fear of opposition", "Install Brass pyramid on door frame"),
    "N2": EntranceGateInfo("N2", "Naga", "North", 0.0, 11.25, "NEGATIVE", -15, "Jealousy and unnecessary friction", "Hang Copper Swastik on lintel"),
    "N3": EntranceGateInfo("N3", "Mukhya", "North", 11.25, 22.5, "SUPER_AUSPICIOUS", 10, "Supreme wealth, continuous cashflow, commercial success"),
    "N4": EntranceGateInfo("N4", "Bhallata", "North", 22.5, 33.75, "SUPER_AUSPICIOUS", 10, "Abundant inheritance, property multiplication, prosperity"),
    "N5": EntranceGateInfo("N5", "Soma (Kubera)", "North", 33.75, 45.0, "AUSPICIOUS", 9, "Spiritual clarity, financial stability, peace of mind"),
    "N6": EntranceGateInfo("N6", "Bhujanga", "North", 45.0, 56.25, "NEGATIVE", -10, "Discontentment among youth, deceitful guests", "Place yellow mustard seeds at threshold"),
    "N7": EntranceGateInfo("N7", "Aditi", "North", 56.25, 67.5, "NEUTRAL", 0, "High household expenses, restlessness", "Add brass plate under threshold"),
    "N8": EntranceGateInfo("N8", "Diti", "North", 67.5, 78.75, "NEGATIVE", -10, "Petty misunderstandings, divided attention", "Install Zinc wire below entry sill"),

    # East side, from NE towards SE
    "E1": EntranceGateInfo("E1", "Shikhi", "East", 78.75, 90.0, "NEGATIVE", -30, "Mental agitation, fire hazard risk", "Install Green fluorite pyramid over entrance"),
    "E2": EntranceGateInfo("E2", "Parjanya", "East", 90.0, 101.25, "NEUTRAL", 0, "Excess luxury expenditures, casual approach", "Balance with indoor Tulsi plant nearby"),
    "E3": EntranceGateInfo("E3", "Jayanta", "East", 101.25, 112.5, "SUPER_AUSPICIOUS", 10, "Victory, social influence, rapid career advancement"),
    "E4": EntranceGateInfo("E4", "Indra", "East", 112.5, 123.75, "SUPER_AUSPICIOUS", 10, "Executive power, authoritative stature, government honors"),
    "E5": EntranceGateInfo("E5", "Surya", "East", 123.75, 135.0, "NEGATIVE", -10, "Excessive pride, frequent conflicts with authority", "Keep copper bowl of water with white flowers"),
    "E6": EntranceGateInfo("E6", "Satya", "East", 135.0, 146.25, "NEGATIVE", -15, "Broken commitments, disputes in partnerships", "Place Brass Vishnu Yantra"),
    "E7": EntranceGateInfo("E7", "Bhrisha", "East", 146.25, 157.5, "NEGATIVE", -20, "Hasty impulsive decisions, burnout", "Add 3 brass pyramids above door"),
    "E8": EntranceGateInfo("E8", "Akash", "East", 157.5, 168.75, "TABOO", -30, "Financial drainage, vulnerability to deceit", "Embed copper strip in threshold floor"),

    # South side, from SE towards SW
    "S1": EntranceGateInfo("S1", "Anila (Vayu)", "South", 168.75, 180.0, "NEGATIVE", -20, "Fiery temper, stress for children", "Hang red jasper crystal on right door jamb"),
    "S2": EntranceGateInfo("S2", "Pusha", "South", 180.0, 191.25, "NEGATIVE", -15, "Subjugation, feeling of servitude", "Add brass bell on door"),
    "S3": EntranceGateInfo("S3", "Vitatha", "South", 191.25, 202.5, "AUSPICIOUS", 8, "Clever strategies, lucrative trade, wealth retention"),
    "S4": EntranceGateInfo("S4", "Brihatkshata", "South", 202.5, 213.75, "SUPER_AUSPICIOUS", 10, "Lineage glory, robust male lineage health, abundant wealth"),
    "S5": EntranceGateInfo("S5", "Yama", "South", 213.75, 225.0, "TABOO", -40, "Heavy debt liabilities, chronic health obstacles", "Install Lead metal strip under threshold with Rahu Yantra"),
    "S6": EntranceGateInfo("S6", "Gandharva", "South", 225.0, 236.25, "NEGATIVE", -20, "Loss of public reputation, hollow pride", "Place Rock salt lamp near entrance"),
    "S7": EntranceGateInfo("S7", "Bhrigaraja", "South", 236.25, 247.5, "NEGATIVE", -25, "Wasted efforts, sudden fund shortages", "Embed copper wire boundary"),
    "S8": EntranceGateInfo("S8", "Mriga", "South", 247.5, 258.75, "TABOO", -35, "Weakness in family foundation, vitality loss", "Install Panchadhatu strip with lead plate"),

    # West side, from SW towards NW
    "W1": EntranceGateInfo("W1", "Pitri", "West", 258.75, 270.0, "TABOO", -50, "Strict taboo: obstacles to progeny, severe instability", "Embed Lead strip, place yellow stone threshold, Rahu helix"),
    "W2": EntranceGateInfo("W2", "Dauvarika", "West", 270.0, 281.25, "NEGATIVE", -15, "Insecure income, career volatility", "Install 9 Brass pyramids on frame"),
    "W3": EntranceGateInfo("W3", "Sugriva", "West", 281.25, 292.5, "SUPER_AUSPICIOUS", 10, "Steady business expansion, scholastic excellence, gold accumulation"),
    "W4": EntranceGateInfo("W4", "Pushpadanta", "West", 292.5, 303.75, "SUPER_AUSPICIOUS", 10, "Blessing of children, abundance, fame and public goodwill"),
    "W5": EntranceGateInfo("W5", "Varuna", "West", 303.75, 315.0, "NEUTRAL", -5, "Over-ambition leading to fluctuating savings", "Hang wind chime with 5 metal hollow rods"),
    "W6": EntranceGateInfo("W6", "Asura", "West", 315.0, 326.25, "NEGATIVE", -20, "Chronic exhaustion, depressive moods, tax troubles", "Add Lead helix on West wall"),
    "W7": EntranceGateInfo("W7", "Sosha", "West", 326.25, 337.5, "NEGATIVE", -25, "Bodily weakness, respiratory complaints", "Place sea salt bowl renewed weekly"),
    "W8": EntranceGateInfo("W8", "Papayakshma", "West", 337.5, 348.75, "TABOO", -40, "Financial leaks, addiction tendencies", "Embed thick Brass strip and install Shani Yantra"),
}


def pada_for_point(px: float, py: float,
                   rect: Tuple[float, float, float, float],
                   facing: str) -> EntranceGateInfo:
    """
    Identify the 32-pada gate for a point on (or near) the boundary of `rect`.

    Each side is divided into 8 equal padas counted clockwise from its corner:
    North NW→NE, East NE→SE, South SE→SW, West SW→NW (Manasara / Mayamata).

    Args:
        px, py: Point in plan coordinates (e.g. the centre of the main door).
        rect: (x, y, w, h) of the building (or plot) the gate belongs to.
        facing: Road-facing direction ('N'/'E'/'S'/'W') — defines plan↔compass.
    """
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    nvx, nvy = compass_vector("N", facing)
    evx, evy = compass_vector("E", facing)
    dx, dy = px - cx, py - cy
    n = dx * nvx + dy * nvy          # northing relative to centre
    e = dx * evx + dy * evy          # easting relative to centre
    hn = abs(w * nvx) / 2.0 + abs(h * nvy) / 2.0   # half extent along N
    he = abs(w * evx) / 2.0 + abs(h * evy) / 2.0   # half extent along E
    hn = max(hn, 1e-6)
    he = max(he, 1e-6)

    # Which side is the point closest to (normalised distance to each edge)?
    d_side = {"N": 1 - n / hn, "S": 1 + n / hn, "E": 1 - e / he, "W": 1 + e / he}
    side = min(d_side, key=d_side.get)

    if side == "N":
        t = (e + he) / (2 * he)
    elif side == "E":
        t = (hn - n) / (2 * hn)
    elif side == "S":
        t = (he - e) / (2 * he)
    else:
        t = (n + hn) / (2 * hn)
    idx = min(8, max(1, int(t * 8) + 1))
    return _PADA_DATABASE[f"{side}{idx}"]


class VastuEngine:
    """
    Vastu Shastra compliance engine for room placement optimization.

    The engine computes a 3×3 Vastu Purusha Mandala grid over the plot
    and scores room placements based on traditional compass-direction rules.
    """

    def __init__(self, plot_width: float, plot_height: float,
                 north_direction: str = "N",
                 origin_x: float = 0.0, origin_y: float = 0.0):
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
            origin_x, origin_y: Bottom-left corner of the mandala in plan
                           coordinates. Use the building footprint origin to
                           lay the Vastu Purusha Mandala over the house itself.
        """
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.north_direction = north_direction.upper()
        self.origin_x = origin_x
        self.origin_y = origin_y
        self._zone_bounds = self._compute_zone_grid()

    @classmethod
    def for_facing(cls, width: float, height: float, facing: str,
                   origin_x: float = 0.0, origin_y: float = 0.0) -> "VastuEngine":
        """Build an engine from the road-facing direction (road side = plan top)."""
        return cls(width, height, north_edge_for_facing(facing), origin_x, origin_y)

    @classmethod
    def for_plan(cls, plan, scope: str = "building") -> "VastuEngine":
        """
        Build the correct mandala for a GeneratedFloorPlan.

        scope='building' lays the mandala over the house footprint (rooms);
        scope='plot' lays it over the whole site (gates, amenities).
        Plans from older engines without footprint/north_edge fall back to the plot.
        """
        north = getattr(plan, "north_edge", "") or north_edge_for_facing(getattr(plan, "orientation", "N"))
        fp = getattr(plan, "building_footprint", None)
        if scope == "building" and fp and fp[2] > 0 and fp[3] > 0:
            return cls(fp[2], fp[3], north, fp[0], fp[1])
        return cls(plan.plot_width, plan.plot_height, north)

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
            bounds[zone] = VastuZoneBounds(zone=zone, x=x + self.origin_x, y=y + self.origin_y, w=w, h=h)

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
        # Points on the far boundary or outside the mandala: snap to the nearest cell
        w3 = self.plot_width / 3.0
        h3 = self.plot_height / 3.0
        col = min(2, max(0, int((x - self.origin_x) // w3))) if w3 > 0 else 1
        row = min(2, max(0, int((y - self.origin_y) // h3))) if h3 > 0 else 1
        for zone, bounds in self._zone_bounds.items():
            if (abs(bounds.x - (self.origin_x + col * w3)) < 1e-6 and
                    abs(bounds.y - (self.origin_y + row * h3)) < 1e-6):
                return zone
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

    def compute_plan_score(self, room_placements: List[Tuple[str, str]],
                           weights: Optional[List[float]] = None,
                           extra_scores: Optional[List[Tuple[float, float]]] = None) -> float:
        """
        Compute the overall Vastu compliance score for a complete floor plan.

        Each placement's raw score (−50…+10) is mapped linearly to 0…1 and
        averaged with importance weights (master bedroom, kitchen, pooja and
        entrance weigh most). Every strict taboo then costs 8 points. Unlike a
        plain average this does not saturate: a −20 placement visibly lowers
        the score.

        Args:
            room_placements: List of (room_category, zone) tuples.
            weights: Optional per-placement weights (defaults to ROOM_IMPORTANCE).
            extra_scores: Optional (raw_score, weight) pairs for non-room items,
                          e.g. the main-entrance pada score.

        Returns:
            Normalized score from 0 to 100.
        """
        if not room_placements and not extra_scores:
            return 100.0

        acc = 0.0
        total_w = 0.0
        taboo_count = 0

        for i, (category, zone) in enumerate(room_placements):
            w = weights[i] if weights and i < len(weights) else ROOM_IMPORTANCE.get(category.upper(), 1.0)
            score = self.get_room_score(category, zone)
            acc += w * (score + 50) / 60.0
            total_w += w
            if self.is_taboo(category, zone):
                taboo_count += 1

        for raw, w in (extra_scores or []):
            acc += w * (max(-50.0, min(10.0, raw)) + 50) / 60.0
            total_w += w

        if total_w <= 0:
            return 100.0

        normalized = 100.0 * acc / total_w - 8.0 * taboo_count
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
