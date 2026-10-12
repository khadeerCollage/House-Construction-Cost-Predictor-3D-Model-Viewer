"""
Construction Cost Estimation Engine (multi-market)
==================================================
Estimates the total project cost of a house (excluding land) the way quantity
surveyors price early-stage designs worldwide:

    effective rate = base rate (quality grade, national reference city)
                     × location index (city/region — cf. RSMeans City Cost Index,
                       CPWD cost index, BCIS location factors)
                     × size adjustment (small homes cost more per unit area)
                     × storey factor (upper floors share the foundation and roof)
    construction   = built-up area × effective rate
                     + wet-area fit-out (kitchens, bathrooms)
                     + site works & amenities + lift
    project total  = construction + professional fees + approvals / permits
                     + contingency + taxes not already inside market rates

Rates are 2026 planning benchmarks compiled from published contractor / QS guides
(India: Ambuja, JK Cement, InfraLens city tables; US: NAHB cost survey shares and
RSMeans-style location factors; UK: self-build / BCIS-style ranges; UAE: Knight
Frank / Engel & Völkers villa rates; Canada: Altus-based guides; Australia:
detached-house QS ranges). They are estimates, not contractor quotations.

Backwards compatible: `estimate_cost(area_sqm, ..., city_tier, quality_level)` still
works with the old Indian tier labels; `country` / `region` select other markets.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

SQFT_PER_M2 = 10.7639

QUALITY_KEYS = ["Basic (Low Cost)", "Standard (Medium)",
                "Premium (High Quality)", "Luxury (Ultra Premium)"]
QUALITY_SHORT = {"Basic (Low Cost)": "Economy", "Standard (Medium)": "Standard",
                 "Premium (High Quality)": "Premium", "Luxury (Ultra Premium)": "Luxury"}

# ---------------------------------------------------------------------------
# Market database
#   rates      : hard cost per market unit (sqft or m²) in the reference city,
#                incl. contractor overhead & profit, excl. land / fees / permits
#   regions    : location index (1.00 = reference) or {"m": index, overrides...}
#   materials  : (key, label, unit, qty per market unit, unit price, basis)
#                basis 'floor' = per built-up area, 'footprint' = per ground floor
#   site       : site-works rates in local currency (reference city)
#   soft       : % of construction cost; tax applies to construction + fees
# ---------------------------------------------------------------------------
MARKETS: Dict[str, Dict[str, Any]] = {
    "India": {
        "currency": "INR", "symbol": "₹", "unit": "sqft", "number_style": "indian",
        "system": "RCC frame with brick / block masonry",
        "rates": {"Basic (Low Cost)": 1500, "Standard (Medium)": 2000,
                  "Premium (High Quality)": 2800, "Luxury (Ultra Premium)": 4200},
        "reference_area_sqft": 1200,
        "wet": {"kitchen": 30000, "bath": 20000},
        "soft": {"fees": 4.0, "approvals": 2.5, "contingency": 7.0, "tax": 0.0},
        "tax_label": "GST",
        "tax_note": "Contractor package rates are quoted inclusive of GST on the works contract.",
        "default_region": "Other Tier-2 City (average)",
        "regions": {
            "Mumbai / MMR (Metro)": 1.55,
            "Delhi NCR (Metro)": 1.30,
            "Pune (Metro)": 1.18,
            "Bengaluru (Metro)": 1.18,
            "Chennai (Metro)": 1.10,
            "Hyderabad (Metro)": 1.08,
            "Kolkata (Metro)": 1.05,
            "Ahmedabad (Metro)": 1.02,
            "Other Tier-1 Metro": 1.25,
            "Kochi / Ernakulam (Kerala)": 1.25,
            "Thiruvananthapuram (Kerala)": 1.18,
            "Kozhikode / Thrissur (Kerala)": 1.15,
            "Goa": 1.22,
            "Hill Station (Shimla / Ooty / Munnar)": 1.20,
            "Chandigarh / Mohali": 1.10,
            "Guwahati / North-East": 1.08,
            "Jaipur": 1.03,
            "Mysuru / Mangaluru": 1.02,
            "Lucknow": 1.02,
            "Coimbatore / Madurai": 1.00,
            "Surat / Vadodara": 1.00,
            "Visakhapatnam / Vijayawada": 0.98,
            "Indore / Bhopal": 0.97,
            "Nagpur": 0.97,
            "Bhubaneswar": 0.95,
            "Patna / Ranchi": 0.93,
            "Other Tier-2 City (average)": 1.00,
            "Tier-3 Town / Semi-Urban": 0.88,
            "Rural / Village": 0.80,
        },
        "materials": [
            ("Cement", "Cement (OPC 53, 50 kg)", "Bags", 0.42, 390, "floor"),
            ("Steel", "TMT Steel (Fe 500D)", "Kg", 3.6, 72, "floor"),
            ("Bricks", "Clay Bricks", "Pcs", 18, 9.0, "floor"),
            ("Sand", "M-Sand / River Sand", "CFT", 1.8, 70, "floor"),
            ("Aggregate", "Aggregate (20 mm)", "CFT", 1.35, 60, "floor"),
        ],
        "site": {
            "PARKING": 1200, "BIKE_PARKING": 1000, "GARDEN": 1500, "POOL": 38000,
            "WATER_TANK": 14000, "GENERATOR": 18000, "SECURITY": 19000,
            "EXT_STAIR": 120000, "COMPOUND_WALL": 3800,
            "LIFT_BASE": 950000, "LIFT_PER_STOP": 120000,
        },
        "breakdown": None,   # quality-specific, see INDIA_BREAKDOWN
        "sources": "2026 city rate tables (Ambuja, JK Cement, InfraLens), CPWD plinth-area method",
    },
    "United States": {
        "currency": "USD", "symbol": "$", "unit": "sqft", "number_style": "western",
        "system": "Wood (stick) frame on concrete slab / crawlspace",
        "rates": {"Basic (Low Cost)": 150, "Standard (Medium)": 215,
                  "Premium (High Quality)": 330, "Luxury (Ultra Premium)": 520},
        "reference_area_sqft": 2200,
        "wet": {"kitchen": 6000, "bath": 4000},
        "soft": {"fees": 7.0, "approvals": 4.5, "contingency": 10.0, "tax": 0.0},
        "tax_label": "Sales tax",
        "tax_note": "State sales tax on materials is already inside builder rates.",
        "default_region": "National Average",
        "regions": {
            "New York City (NY)": 1.38,
            "San Francisco Bay Area (CA)": 1.35,
            "Honolulu (HI)": 1.30,
            "Boston (MA)": 1.22,
            "Los Angeles (CA)": 1.20,
            "Chicago (IL)": 1.16,
            "San Diego (CA)": 1.14,
            "Seattle (WA)": 1.14,
            "Philadelphia (PA)": 1.14,
            "Minneapolis (MN)": 1.08,
            "Portland (OR)": 1.06,
            "Washington DC Metro": 1.04,
            "Las Vegas (NV)": 1.02,
            "Denver (CO)": 1.00,
            "Miami (FL)": 0.98,
            "Phoenix (AZ)": 0.92,
            "Austin (TX)": 0.92,
            "Orlando / Tampa (FL)": 0.90,
            "Atlanta (GA)": 0.90,
            "Nashville (TN)": 0.90,
            "Dallas-Fort Worth (TX)": 0.88,
            "Houston (TX)": 0.88,
            "Charlotte / Raleigh (NC)": 0.87,
            "National Average": 1.00,
            "Rural / Small Town": 0.82,
        },
        "materials": [
            ("Concrete", "Ready-mix Concrete", "yd³", 0.0176, 175, "footprint"),
            ("Lumber", "Framing Lumber", "board-ft", 6.5, 0.65, "floor"),
            ("Sheathing", "OSB / Plywood Sheathing", "Sheets", 0.10, 18, "floor"),
            ("Drywall", "Drywall (½in 4×8)", "Sheets", 0.12, 16, "floor"),
            ("Roofing", "Architectural Shingles", "Squares", 0.0115, 130, "footprint"),
        ],
        "site": {
            "PARKING": 140, "BIKE_PARKING": 110, "GARDEN": 90, "POOL": 2000,
            "WATER_TANK": 900, "GENERATOR": 650, "SECURITY": 2500,
            "EXT_STAIR": 9000, "COMPOUND_WALL": 300,
            "LIFT_BASE": 45000, "LIFT_PER_STOP": 6000,
        },
        "breakdown": {
            "Site Work": 6, "Foundation": 10, "Framing": 17,
            "Exterior Finishes (roof, siding, windows)": 14,
            "Major Systems (plumbing, electrical, HVAC)": 19,
            "Interior Finishes": 25,
            "Final Steps (driveway, landscaping, clean-up)": 6,
            "Builder Overheads & Other": 3,
        },
        "sources": "NAHB Cost of Constructing a Home shares, RSMeans-style city cost indices, 2026 builder guides",
    },
    "United Kingdom": {
        "currency": "GBP", "symbol": "£", "unit": "m²", "number_style": "western",
        "system": "Masonry cavity wall, concrete foundations, timber roof",
        "rates": {"Basic (Low Cost)": 1700, "Standard (Medium)": 2200,
                  "Premium (High Quality)": 2900, "Luxury (Ultra Premium)": 3800},
        "reference_area_sqft": 1400,
        "wet": {"kitchen": 4000, "bath": 2500},
        "soft": {"fees": 10.0, "approvals": 3.5, "contingency": 10.0, "tax": 0.0},
        "tax_label": "VAT",
        "tax_note": "New-build dwellings are zero-rated for VAT (self-builders reclaim VAT on materials).",
        "default_region": "Midlands (UK average)",
        "regions": {
            "Greater London": 1.30,
            "South East England": 1.15,
            "East of England": 1.08,
            "South West England": 1.04,
            "Scotland - Edinburgh / Aberdeen": 1.00,
            "Midlands (UK average)": 1.00,
            "North West England": 0.94,
            "Yorkshire & the Humber": 0.93,
            "Wales": 0.93,
            "Scotland - other": 0.93,
            "North East England": 0.90,
            "Northern Ireland": 0.86,
        },
        "materials": [
            ("Concrete", "Concrete (foundations & slab)", "m³", 0.28, 150, "footprint"),
            ("Bricks", "Facing Bricks", "Pcs", 66, 0.85, "floor"),
            ("Blocks", "Concrete / Aircrete Blocks", "Pcs", 20, 2.40, "floor"),
            ("Timber", "Structural Timber (C24)", "m³", 0.05, 600, "floor"),
            ("Plasterboard", "Plasterboard", "m²", 3.0, 4.5, "floor"),
        ],
        "site": {
            "PARKING": 110, "BIKE_PARKING": 90, "GARDEN": 70, "POOL": 2500,
            "WATER_TANK": 800, "GENERATOR": 450, "SECURITY": 2200,
            "EXT_STAIR": 8000, "COMPOUND_WALL": 320,
            "LIFT_BASE": 35000, "LIFT_PER_STOP": 5000,
        },
        "breakdown": {
            "Substructure (foundations, ground floor)": 11,
            "Superstructure (walls, upper floors, roof)": 30,
            "Windows & External Doors": 9,
            "Internal Finishes": 15,
            "Fittings (kitchen, bathrooms, joinery)": 10,
            "Services (M&E, heating)": 15,
            "Preliminaries & External Works": 10,
        },
        "sources": "2026 UK self-build / new-build cost guides, BCIS-style regional factors",
    },
    "United Arab Emirates": {
        "currency": "AED", "symbol": "AED", "unit": "sqft", "number_style": "western",
        "system": "RCC frame with hollow concrete block walls",
        "rates": {"Basic (Low Cost)": 330, "Standard (Medium)": 450,
                  "Premium (High Quality)": 720, "Luxury (Ultra Premium)": 1100},
        "reference_area_sqft": 4000,
        "wet": {"kitchen": 20000, "bath": 12000},
        "soft": {"fees": 7.0, "approvals": 3.0, "contingency": 10.0, "tax": 5.0},
        "tax_label": "VAT",
        "tax_note": "5% VAT applies to construction and consultancy services.",
        "default_region": "Dubai",
        "regions": {
            "Abu Dhabi": 1.04,
            "Dubai": 1.00,
            "Al Ain": 0.92,
            "Sharjah": 0.88,
            "Ras Al Khaimah": 0.86,
            "Ajman": 0.84,
            "Fujairah / Umm Al Quwain": 0.83,
        },
        "materials": [
            ("Cement", "Cement (50 kg)", "Bags", 0.45, 16, "floor"),
            ("Steel", "Rebar Steel", "Kg", 4.0, 2.7, "floor"),
            ("Blocks", "Hollow Concrete Blocks (8in)", "Pcs", 2.9, 3.2, "floor"),
            ("Sand", "Washed Sand", "m³", 0.051, 70, "floor"),
            ("Aggregate", "Aggregate (20 mm)", "m³", 0.038, 65, "floor"),
        ],
        "site": {
            "PARKING": 150, "BIKE_PARKING": 130, "GARDEN": 220, "POOL": 6000,
            "WATER_TANK": 1800, "GENERATOR": 1500, "SECURITY": 3500,
            "EXT_STAIR": 30000, "COMPOUND_WALL": 900,
            "LIFT_BASE": 120000, "LIFT_PER_STOP": 15000,
        },
        "breakdown": {
            "RCC Structure (foundation, frame, slabs)": 32,
            "Blockwork & Plaster": 12,
            "Finishes (flooring, paint, ceilings)": 18,
            "MEP & HVAC": 22,
            "Doors, Windows & Joinery": 8,
            "Waterproofing & Insulation": 4,
            "Preliminaries": 4,
        },
        "sources": "2026 villa construction rates (Knight Frank / Engel & Völkers, Dubai contractor guides)",
    },
    "Canada": {
        "currency": "CAD", "symbol": "C$", "unit": "sqft", "number_style": "western",
        "system": "Wood frame on full concrete basement",
        "rates": {"Basic (Low Cost)": 230, "Standard (Medium)": 320,
                  "Premium (High Quality)": 470, "Luxury (Ultra Premium)": 700},
        "reference_area_sqft": 2000,
        "wet": {"kitchen": 7000, "bath": 5000},
        "soft": {"fees": 6.0, "approvals": 6.0, "contingency": 10.0, "tax": 5.0},
        "tax_label": "GST/HST",
        "tax_note": "GST/HST on contracted work before any new-housing rebate; provincial PST on materials is inside rates.",
        "default_region": "National Average",
        "regions": {
            "Vancouver (BC)": {"m": 1.25, "approvals": 8.0, "tax": 5.0},
            "Toronto / GTA (ON)": {"m": 1.22, "approvals": 9.0, "tax": 13.0},
            "Victoria (BC)": {"m": 1.15, "tax": 5.0},
            "Ottawa (ON)": {"m": 1.05, "tax": 13.0},
            "Montreal (QC)": {"m": 0.95, "tax": 14.975},
            "Calgary (AB)": {"m": 0.95, "tax": 5.0},
            "Halifax (NS)": {"m": 0.95, "tax": 14.0},
            "Edmonton (AB)": {"m": 0.92, "tax": 5.0},
            "Winnipeg (MB)": {"m": 0.90, "tax": 5.0},
            "Quebec City (QC)": {"m": 0.88, "tax": 14.975},
            "National Average": 1.00,
            "Rural / Small Town": 0.85,
        },
        "materials": [
            ("Concrete", "Ready-mix Concrete (basement)", "m³", 0.028, 260, "footprint"),
            ("Lumber", "Framing Lumber", "board-ft", 6.5, 0.85, "floor"),
            ("Sheathing", "OSB / Plywood Sheathing", "Sheets", 0.10, 24, "floor"),
            ("Drywall", "Drywall (½in 4×8)", "Sheets", 0.12, 19, "floor"),
            ("Roofing", "Asphalt Shingles", "Squares", 0.0115, 160, "footprint"),
        ],
        "site": {
            "PARKING": 160, "BIKE_PARKING": 120, "GARDEN": 110, "POOL": 2600,
            "WATER_TANK": 1100, "GENERATOR": 800, "SECURITY": 3000,
            "EXT_STAIR": 12000, "COMPOUND_WALL": 380,
            "LIFT_BASE": 60000, "LIFT_PER_STOP": 8000,
        },
        "breakdown": {
            "Site Work": 6, "Foundation & Basement": 13, "Framing": 16,
            "Exterior Finishes (roof, siding, windows)": 14,
            "Major Systems (plumbing, electrical, HVAC)": 19,
            "Interior Finishes": 24,
            "Final Steps (driveway, landscaping, clean-up)": 5,
            "Builder Overheads & Other": 3,
        },
        "sources": "2026 Canadian custom-home guides (Altus-based cost ranges)",
    },
    "Australia": {
        "currency": "AUD", "symbol": "A$", "unit": "m²", "number_style": "western",
        "system": "Brick veneer on timber frame, concrete slab",
        "rates": {"Basic (Low Cost)": 2100, "Standard (Medium)": 2800,
                  "Premium (High Quality)": 4000, "Luxury (Ultra Premium)": 5800},
        "reference_area_sqft": 2475,
        "wet": {"kitchen": 8000, "bath": 5000},
        "soft": {"fees": 6.0, "approvals": 3.0, "contingency": 10.0, "tax": 0.0},
        "tax_label": "GST",
        "tax_note": "Builder quotes include 10% GST.",
        "default_region": "National Average",
        "regions": {
            "Darwin (NT)": 1.18,
            "Sydney (NSW)": 1.12,
            "Canberra (ACT)": 1.06,
            "Hobart (TAS)": 1.04,
            "Melbourne (VIC)": 1.03,
            "Gold Coast (QLD)": 1.02,
            "Brisbane (QLD)": 1.00,
            "Perth (WA)": 0.98,
            "Adelaide (SA)": 0.95,
            "National Average": 1.00,
            "Regional / Country": 0.93,
        },
        "materials": [
            ("Concrete", "Concrete (slab & footings)", "m³", 0.18, 310, "footprint"),
            ("Bricks", "Face Bricks (veneer)", "Pcs", 55, 1.30, "floor"),
            ("Timber", "Frame Timber (MGP10)", "m³", 0.06, 1100, "floor"),
            ("Plasterboard", "Plasterboard", "m²", 3.0, 9, "floor"),
            ("Roofing", "Roof Sheeting (Colorbond)", "m²", 1.15, 38, "footprint"),
        ],
        "site": {
            "PARKING": 130, "BIKE_PARKING": 100, "GARDEN": 120, "POOL": 2800,
            "WATER_TANK": 800, "GENERATOR": 700, "SECURITY": 3000,
            "EXT_STAIR": 12000, "COMPOUND_WALL": 350,
            "LIFT_BASE": 70000, "LIFT_PER_STOP": 9000,
        },
        "breakdown": {
            "Site Preparation & Slab": 12, "Frame & Roof": 18,
            "External Walls (brick veneer)": 12, "Windows & Doors": 7,
            "Services (plumbing, electrical, HVAC)": 16,
            "Internal Linings & Finishes": 20, "Kitchen & Wet Areas": 10,
            "Preliminaries & External Works": 5,
        },
        "sources": "2026 Australian detached-house QS ranges (capital-city indices)",
    },
}

INDIA_DESCRIPTIONS = {
    "Basic (Low Cost)": "Simple RCC/brick structure, standard flooring, basic paint, local plumbing & electric fixtures.",
    "Standard (Medium)": "RCC structure, vitrified tile flooring, premium emulsion paint, branded fittings (Jaquar/Anchor class).",
    "Premium (High Quality)": "Heavy structure, Italian marble/wooden flooring, luxury finishes, modular kitchen & premium branded fittings.",
    "Luxury (Ultra Premium)": "Architect-designed structure, imported stone & engineered wood, home automation, VRF air-conditioning, designer sanitaryware and façade glazing.",
}
GENERIC_DESCRIPTIONS = {
    "Basic (Low Cost)": "Builder-grade spec: standard materials, laminate/vinyl floors, stock cabinetry, basic fixtures.",
    "Standard (Medium)": "Mid-range spec: quality windows, engineered wood/tile floors, branded fixtures, mid-range kitchen.",
    "Premium (High Quality)": "High spec: architect-designed, hardwood/stone floors, custom joinery, premium kitchen & baths, smart-home pre-wiring.",
    "Luxury (Ultra Premium)": "Bespoke spec: complex structure, imported stone, high-end appliances, full home automation, designer fit-out.",
}
INDIA_BREAKDOWN = {
    "Basic (Low Cost)": {"Cement & Steel (Structure)": 35, "Bricks & Sand (Masonry)": 20, "Flooring & Tiles": 10,
                         "Plumbing & Sanitation": 8, "Electrical Fittings": 7, "Painting & Finishes": 8,
                         "Labor & Supervision": 12},
    "Standard (Medium)": {"Cement & Steel (Structure)": 33, "Bricks & Sand (Masonry)": 18, "Flooring & Tiles": 12,
                          "Plumbing & Sanitation": 10, "Electrical Fittings": 9, "Painting & Finishes": 10,
                          "Labor & Supervision": 8},
    "Premium (High Quality)": {"Cement & Steel (Structure)": 30, "Bricks & Sand (Masonry)": 15, "Flooring & Tiles": 15,
                               "Plumbing & Sanitation": 12, "Electrical Fittings": 11, "Painting & Finishes": 11,
                               "Labor & Supervision": 6},
    "Luxury (Ultra Premium)": {"Cement & Steel (Structure)": 26, "Bricks & Sand (Masonry)": 12, "Flooring & Tiles": 17,
                               "Plumbing & Sanitation": 12, "Electrical Fittings": 13, "Painting & Finishes": 14,
                               "Labor & Supervision": 6},
}

# Old Indian tier labels -> India regions
LEGACY_TIER_REGION = {
    "Tier-1 (Metro)": "Other Tier-1 Metro",
    "Tier-2 (Urban)": "Other Tier-2 City (average)",
    "Tier-3 (Semi-Urban)": "Tier-3 Town / Semi-Urban",
    "Tier-3 (Rural)": "Rural / Village",
}

COUNTRY_ALIASES = {
    "india": "India", "in": "India", "bharat": "India",
    "us": "United States", "usa": "United States", "united states": "United States", "america": "United States",
    "uk": "United Kingdom", "gb": "United Kingdom", "united kingdom": "United Kingdom", "england": "United Kingdom",
    "uae": "United Arab Emirates", "united arab emirates": "United Arab Emirates", "dubai": "United Arab Emirates",
    "canada": "Canada", "ca": "Canada",
    "australia": "Australia", "au": "Australia",
}


def _indian_group(n: float) -> str:
    """12345678 -> '1,23,45,678'."""
    s = f"{abs(int(round(n)))}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return ("-" if n < 0 else "") + s


class CostEngine:
    """Multi-market construction cost engine (India default, INR)."""

    MARKETS = MARKETS
    QUALITY_KEYS = QUALITY_KEYS

    # Back-compat views of the Indian market
    CITY_MULTIPLIERS = {k: MARKETS["India"]["regions"][v] for k, v in LEGACY_TIER_REGION.items()}
    QUALITY_RATES = {q: {"rate_per_sqft": MARKETS["India"]["rates"][q],
                         "description": INDIA_DESCRIPTIONS[q],
                         "breakdown_pct": INDIA_BREAKDOWN[q]} for q in QUALITY_KEYS}
    SITE_RATES = MARKETS["India"]["site"]
    MATERIAL_PRICES = {m[0]: m[4] for m in MARKETS["India"]["materials"]}

    # ------------------------------------------------------------------ lookups
    @classmethod
    def markets(cls) -> List[str]:
        return list(MARKETS.keys())

    @classmethod
    def regions(cls, country: str) -> List[str]:
        return list(MARKETS[cls.normalize_country(country)]["regions"].keys())

    @classmethod
    def default_region(cls, country: str) -> str:
        return MARKETS[cls.normalize_country(country)]["default_region"]

    @classmethod
    def normalize_country(cls, country: Optional[str]) -> str:
        if country in MARKETS:
            return country
        c = (country or "").strip().lower()
        if c in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[c]
        for name in MARKETS:
            if c and name.lower().startswith(c):
                return name
        return "India"

    @classmethod
    def normalize_city_tier(cls, city_tier: str) -> str:
        """Map old Indian UI labels ('Tier-1 Metro', 'Rural', …) onto the legacy tier keys."""
        if city_tier in cls.CITY_MULTIPLIERS:
            return city_tier
        t = (city_tier or "").lower()
        if "1" in t or "metro" in t:
            return "Tier-1 (Metro)"
        if "semi" in t:
            return "Tier-3 (Semi-Urban)"
        if "rural" in t or "village" in t:
            return "Tier-3 (Rural)"
        if "3" in t:
            return "Tier-3 (Semi-Urban)"
        return "Tier-2 (Urban)"

    @classmethod
    def normalize_quality(cls, quality_level: str) -> str:
        """Map UI labels ('Economy', 'Luxury', …) onto the engine's keys."""
        if quality_level in QUALITY_KEYS:
            return quality_level
        q = (quality_level or "").lower()
        if "lux" in q or "ultra" in q:
            return "Luxury (Ultra Premium)"
        if "prem" in q or "high" in q:
            return "Premium (High Quality)"
        if "econ" in q or "basic" in q or "low" in q:
            return "Basic (Low Cost)"
        return "Standard (Medium)"

    @classmethod
    def resolve_location(cls, country: Optional[str] = None, region: Optional[str] = None,
                         city_tier: Optional[str] = None) -> Tuple[str, str]:
        """(country, region) from any mix of new and legacy inputs."""
        if not country and region:
            for name, m in MARKETS.items():
                if region in m["regions"]:
                    country = name
                    break
        country = cls.normalize_country(country)
        regs = MARKETS[country]["regions"]
        if region in regs:
            return country, region
        if city_tier in regs:
            return country, city_tier
        if country == "India" and city_tier:
            return country, LEGACY_TIER_REGION[cls.normalize_city_tier(city_tier)]
        return country, MARKETS[country]["default_region"]

    @staticmethod
    def _region_params(market: Dict[str, Any], region: str) -> Tuple[float, Dict[str, float]]:
        v = market["regions"][region]
        if isinstance(v, dict):
            return float(v.get("m", 1.0)), {k: float(x) for k, x in v.items() if k != "m"}
        return float(v), {}

    # --------------------------------------------------------------- formatting
    @classmethod
    def format_money(cls, amount: float, country: str = "India", compact: bool = False) -> str:
        m = MARKETS[cls.normalize_country(country)]
        sym = m["symbol"]
        if m["number_style"] == "indian":
            if compact and abs(amount) >= 1e7:
                return f"{sym} {amount / 1e7:.2f} Crore"
            if compact and abs(amount) >= 1e5:
                return f"{sym} {amount / 1e5:.2f} Lakhs"
            return f"{sym} {_indian_group(amount)}"
        if compact and abs(amount) >= 1e6:
            return f"{sym} {amount / 1e6:.2f} M"
        return f"{sym} {amount:,.0f}"

    # -------------------------------------------------------------- core math
    @staticmethod
    def size_factor(area_sqft: float, reference_sqft: float) -> float:
        """Economy of scale: fixed costs spread over less area make small homes dearer per unit."""
        if area_sqft <= 0:
            return 1.0
        return max(0.90, min(1.15, (reference_sqft / area_sqft) ** 0.10))

    @staticmethod
    def storey_factor(floors: int) -> float:
        """Upper floors reuse the foundation and roof (≈7% cheaper); G+2+ need heavier columns."""
        n = max(1, int(floors))
        return (1 + 0.93 * (n - 1)) / n * (1 + 0.02 * max(0, n - 2))

    @classmethod
    def _totals(cls, m: Dict[str, Any], mult: float, over: Dict[str, float], quality: str,
                area_sqft: float, floors: int, kitchens: int, bathrooms: int,
                extras_raw: Dict[str, float]) -> Dict[str, float]:
        unit_area = area_sqft if m["unit"] == "sqft" else area_sqft / SQFT_PER_M2
        base_rate = m["rates"][quality]
        size = cls.size_factor(area_sqft, m["reference_area_sqft"])
        storey = cls.storey_factor(floors)
        eff_rate = base_rate * mult * size * storey
        building = unit_area * eff_rate
        q_factor = base_rate / m["rates"]["Standard (Medium)"]
        wet = (kitchens * m["wet"]["kitchen"] + bathrooms * m["wet"]["bath"]) * q_factor * mult
        lift = extras_raw.get("Lift Installation", 0.0) * mult
        site = sum(v for k, v in extras_raw.items() if k != "Lift Installation") * mult
        construction = building + wet + site + lift
        soft = dict(m["soft"])
        soft.update({k: v for k, v in over.items() if k in soft})
        fees = construction * soft["fees"] / 100
        approvals = construction * soft["approvals"] / 100
        contingency = construction * soft["contingency"] / 100
        tax = (construction + fees) * soft["tax"] / 100
        total = construction + fees + approvals + contingency + tax
        return {"unit_area": unit_area, "base_rate": base_rate, "size": size, "storey": storey,
                "eff_rate": eff_rate, "building": building, "wet": wet, "site": site, "lift": lift,
                "construction": construction, "fees": fees, "approvals": approvals,
                "contingency": contingency, "tax": tax, "total": total, "soft": soft}

    @classmethod
    def estimate_cost(
        cls,
        area_sqm: float,
        bedrooms: int = 1,
        bathrooms: int = 1,
        kitchens: int = 1,
        living_rooms: int = 1,
        city_tier: str = "Tier-2 (Urban)",
        quality_level: str = "Standard (Medium)",
        num_floors: int = 1,
        country: Optional[str] = None,
        region: Optional[str] = None,
        _extras_raw: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate total project cost (excl. land) and a material BOQ.

        Args:
            area_sqm: built-up (plinth) area of ONE floor in m².
            num_floors: floors with that area (total area = area × floors).
            country / region: market and city (see `markets()` / `regions()`);
                when omitted, the legacy Indian `city_tier` label is used.
        """
        country, region = cls.resolve_location(country, region, city_tier)
        quality = cls.normalize_quality(quality_level)
        m = MARKETS[country]
        mult, over = cls._region_params(m, region)
        floors = max(1, int(num_floors or 1))
        extras_raw = dict(_extras_raw or {})
        area_sqft = max(0.0, float(area_sqm or 0)) * SQFT_PER_M2 * floors
        kitchens, bathrooms = max(0, int(kitchens)), max(0, int(bathrooms))

        t = cls._totals(m, mult, over, quality, area_sqft, floors, kitchens, bathrooms, extras_raw)
        unit = m["unit"]
        fmt = lambda v, c=False: cls.format_money(v, country, compact=c)  # noqa: E731

        # Building-cost split by trade, then site / lift / soft-cost lines
        shares = m["breakdown"] or INDIA_BREAKDOWN[quality]
        bldg = t["building"] + t["wet"]
        breakdown = {k: round(bldg * pct / 100.0, 2) for k, pct in shares.items()}
        if t["site"]:
            breakdown["Site Works & Amenities"] = round(t["site"], 2)
        if t["lift"]:
            breakdown["Lift & Elevator"] = round(t["lift"], 2)
        breakdown["Professional Fees (design & engineering)"] = round(t["fees"], 2)
        breakdown["Approvals, Permits & Utility Connections"] = round(t["approvals"], 2)
        breakdown["Contingency"] = round(t["contingency"], 2)
        if t["tax"]:
            breakdown[f"Taxes ({m['tax_label']})"] = round(t["tax"], 2)

        # Material BOQ (prices move about half as much as the overall location index)
        mat_index = 1 + 0.5 * (mult - 1)
        footprint_units = t["unit_area"] / floors
        materials = {}
        for key, label, mu, qty_per, price, basis in m["materials"]:
            qty = qty_per * (t["unit_area"] if basis == "floor" else footprint_units)
            qty = round(qty, 1) if qty < 100 else round(qty)
            rate = round(price * mat_index, 2)
            materials[key] = {"label": label, "quantity": qty, "unit": mu, "rate": rate,
                              "cost": round(qty * rate, 2)}

        comparison = {}
        for q in QUALITY_KEYS:
            comparison[q] = round(cls._totals(m, mult, over, q, area_sqft, floors, kitchens,
                                              bathrooms, extras_raw)["total"], 2)

        soft = t["soft"]
        steps = [
            {"step": "Base rate", "basis": f"{QUALITY_SHORT[quality]} spec, {m['system']}",
             "value": f"{fmt(t['base_rate'])}/{unit}"},
            {"step": "× Location index", "basis": region, "value": f"× {mult:.2f}"},
            {"step": "× Size adjustment", "basis": f"{area_sqft:,.0f} sqft vs {m['reference_area_sqft']:,} sqft typical",
             "value": f"× {t['size']:.3f}"},
            {"step": "× Storey factor", "basis": f"{floors} floor(s)", "value": f"× {t['storey']:.3f}"},
            {"step": "= Effective rate", "basis": "", "value": f"{fmt(t['eff_rate'])}/{unit}"},
            {"step": "Building cost", "basis": f"{t['unit_area']:,.0f} {unit} × effective rate",
             "value": fmt(t["building"])},
            {"step": "＋ Wet-area fit-out", "basis": f"{kitchens} kitchen(s), {bathrooms} bath(s)",
             "value": fmt(t["wet"])},
        ]
        if t["site"]:
            steps.append({"step": "＋ Site works & amenities", "basis": "parking, landscape, utilities, compound wall",
                          "value": fmt(t["site"])})
        if t["lift"]:
            steps.append({"step": "＋ Lift", "basis": "supply, install, commissioning", "value": fmt(t["lift"])})
        steps += [
            {"step": "= Construction cost", "basis": "hard cost", "value": fmt(t["construction"])},
            {"step": "＋ Professional fees", "basis": f"{soft['fees']:g}% (architect, structural, MEP)",
             "value": fmt(t["fees"])},
            {"step": "＋ Approvals & connections", "basis": f"{soft['approvals']:g}% (permits, utility hook-ups)",
             "value": fmt(t["approvals"])},
            {"step": "＋ Contingency", "basis": f"{soft['contingency']:g}%", "value": fmt(t["contingency"])},
        ]
        if t["tax"]:
            steps.append({"step": f"＋ {m['tax_label']}", "basis": f"{soft['tax']:g}% on construction + fees",
                          "value": fmt(t["tax"])})
        steps.append({"step": "= Total project cost", "basis": "excl. land", "value": fmt(t["total"])})

        cost_per_unit = t["total"] / t["unit_area"] if t["unit_area"] else 0.0
        notes = [
            f"{country} · {region}: location index ×{mult:.2f} on labour, materials and site works.",
            f"Construction system assumed: {m['system']}.",
            m["tax_note"],
            "Excludes land, loose furniture, landscaping beyond the listed site works, and financing.",
            f"Rate sources: {m['sources']}. Planning estimate — confirm with 2-3 itemised local quotes.",
        ]
        return {
            "total_cost": round(t["total"], 2),
            "construction_cost": round(t["construction"], 2),
            "building_cost": round(bldg, 2),
            "area_sqft": round(area_sqft, 2),
            "area_unit": unit,
            "area_in_unit": round(t["unit_area"], 2),
            "cost_per_sqft": round(t["total"] / area_sqft, 2) if area_sqft else 0,
            "cost_per_unit": round(cost_per_unit, 2),
            "base_rate": t["base_rate"],
            "effective_rate": round(t["eff_rate"], 2),
            "region_multiplier": mult,
            "factors": {"location": mult, "size": round(t["size"], 4), "storey": round(t["storey"], 4)},
            "soft_costs": {"fees": round(t["fees"], 2), "approvals": round(t["approvals"], 2),
                           "contingency": round(t["contingency"], 2), "tax": round(t["tax"], 2),
                           "pct": soft},
            "cost_breakdown": breakdown,
            "materials": materials,
            "comparison": comparison,
            "calculation": steps,
            "notes": notes,
            "description": (INDIA_DESCRIPTIONS if country == "India" else GENERIC_DESCRIPTIONS)[quality],
            "quality_level": quality,
            "city_tier": region,
            "country": country,
            "region": region,
            "currency": m["currency"],
            "symbol": m["symbol"],
            "display": {
                "total": fmt(t["total"]),
                "total_compact": fmt(t["total"], True),
                "rate": f"{fmt(cost_per_unit)}/{unit}",
                "construction": fmt(t["construction"]),
            },
        }

    # ----------------------------------------------------------- plan support
    @classmethod
    def _plan_inputs(cls, plan, country: str, floors: int):
        """(area_sqm, kitchens, baths, bedrooms, extras_raw) for a GeneratedFloorPlan."""
        R = MARKETS[country]["site"]
        rooms = list(getattr(plan, "rooms", []))
        kitchens = max(1, sum(1 for r in rooms if r.category == "KITCHEN"))
        baths = max(1, sum(1 for r in rooms if r.category == "BATHROOM"))
        bedrooms = sum(1 for r in rooms if r.category == "BEDROOM")
        area = getattr(plan, "total_built_up_area", 0) or getattr(plan, "total_carpet_area", 0)
        extras: Dict[str, float] = {}
        for r in rooms:
            a = r.w * r.h
            if r.category in ("PARKING", "BIKE_PARKING", "GARDEN", "POOL", "SECURITY"):
                extras[r.name] = R[r.category] * a
            elif r.category == "WATER_TANK":
                extras[r.name] = R["WATER_TANK"] * a * 1.5
            elif r.category == "GENERATOR":
                mt = re.search(r"([\d.]+)\s*kVA", r.name)
                extras[r.name] = R["GENERATOR"] * (float(mt.group(1)) if mt else 10.0)
            elif "External" in r.name and r.category == "STAIRS":
                extras[r.name] = R["EXT_STAIR"]
        fp = getattr(plan, "building_footprint", (0, 0, 0, 0))
        if fp and fp[2] > 0 and (fp[2] < plan.plot_width - 0.5 or fp[3] < plan.plot_height - 0.5):
            extras["Compound Wall"] = R["COMPOUND_WALL"] * 2 * (plan.plot_width + plan.plot_height)
        if getattr(plan, "has_lift", False) and any(r.category == "LIFT" for r in rooms):
            extras["Lift Installation"] = R["LIFT_BASE"] + R["LIFT_PER_STOP"] * max(0, floors - 2)
        return area, kitchens, baths, bedrooms, extras

    # Arrangement-dependent work items, indicative 2026 India rates (₹, Standard quality,
    # Tier-2 average). Other markets scale by their Standard rate per m² relative to India's.
    LAYOUT_RATES_INDIA = {
        "partition_m": ("Internal partition walls", "m", 3900.0),   # 115 mm brick, plaster both faces, 3.0 m high
        "doors": ("Internal doors", "nos", 14000.0),                 # frame, flush shutter, hardware, paint
        "glazing_m2": ("Windows & ventilators", "m²", 6500.0),       # UPVC sliding with MS grille
        "plumbing_m": ("Plumbing & drainage runs", "m", 1100.0),     # CPVC supply + PVC drain, fittings, chasing
    }
    LAYOUT_LINE = "Layout changes vs original design"

    @classmethod
    def layout_delta(cls, quantities: Dict[str, float], reference: Dict[str, float],
                     country: Optional[str] = None, quality_level: str = "Standard (Medium)") -> Dict[str, Any]:
        """
        Cost of the arrangement-dependent work that differs from `reference` (before the
        location index and soft costs, which the estimate applies). Items carry quantity
        deltas so the UI can say *why* an edited design costs more or less.
        """
        country = cls.normalize_country(country)
        m = MARKETS[country]
        india = MARKETS["India"]
        std = "Standard (Medium)"
        per_m2 = lambda mk: mk["rates"][std] * (SQFT_PER_M2 if mk["unit"] == "sqft" else 1.0)
        scale = per_m2(m) / per_m2(india)
        qf = m["rates"][cls.normalize_quality(quality_level)] / m["rates"][std]
        items, total = [], 0.0
        for key, (label, unit, rate) in cls.LAYOUT_RATES_INDIA.items():
            dq = float(quantities.get(key, 0.0)) - float(reference.get(key, 0.0))
            if abs(dq) < 0.05:
                continue
            r = rate * scale * qf
            items.append({"key": key, "label": label, "unit": unit, "delta_qty": round(dq, 2),
                          "rate": round(r, 2), "amount": round(dq * r, 2)})
            total += dq * r
        return {"items": items, "total": round(total, 2), "country": country}

    @classmethod
    def estimate_for_plan(cls, plan, city_tier: str = "Tier-2 (Urban)",
                          quality_level: str = "Standard (Medium)",
                          num_floors: int = None, country: Optional[str] = None,
                          region: Optional[str] = None) -> Dict[str, Any]:
        """
        Estimate a GeneratedFloorPlan: built-up (plinth) area × floors, the plan's
        real kitchen/bath counts, plus lift and site works (parking, garden, pool,
        sump, genset, security cabin, external stair, compound wall) and soft costs.
        """
        country, region = cls.resolve_location(country, region, city_tier)
        floors = max(1, int(num_floors or getattr(plan, "num_floors", 1) or 1))
        area, kitchens, baths, bedrooms, extras_raw = cls._plan_inputs(plan, country, floors)
        # An owner-edited design: price the partitions / doors / glazing / plumbing that differ
        # from the design it started from (unedited plans have no difference → no line)
        layout = None
        ref = (getattr(plan, "layout_state", None) or {}).get("ref_quantities")
        if ref:
            from floorplan_generator.quantities import layout_quantities
            layout = cls.layout_delta(layout_quantities(plan), ref, country, quality_level)
            if abs(layout["total"]) >= 1.0:
                extras_raw[cls.LAYOUT_LINE] = layout["total"] * floors
        est = cls.estimate_cost(area_sqm=area, bedrooms=bedrooms, bathrooms=baths, kitchens=kitchens,
                                living_rooms=1, quality_level=quality_level, num_floors=floors,
                                country=country, region=region, _extras_raw=extras_raw)
        mult = est["region_multiplier"]
        est["extras"] = {k: round(v * mult, 2) for k, v in extras_raw.items()}
        if layout is not None:
            for it in layout["items"]:
                it["amount"] = round(it["amount"] * mult * floors, 2)
            layout["total"] = round(layout["total"] * mult * floors, 2)
            est["layout_delta"] = layout
        return est

    # ---------------------------------------------------------------- drawing BOQ
    # Supply rates (material + fixtures, no labour), India 2026, Tier-2 average, by quality grade
    FINISH_RATES_INDIA = {
        #                      Economy  Standard  Premium  Luxury
        "floor_tile_m2":      (450,     750,      1600,    3500),
        "wet_tile_m2":        (400,     600,      1100,    2500),
        "waterproofing_m2":   (120,     180,      260,     380),
        "main_door":          (18000,   35000,    70000,   150000),
        "room_door":          (7000,    11000,    20000,   40000),
        "toilet_door":        (5000,    7500,     12000,   25000),
        "window_m2":          (2800,    4500,     7500,    14000),
        "paint_in_m2":        (45,      70,       110,     180),
        "paint_out_m2":       (40,      60,       95,      150),
        "sanitary_set":       (15000,   28000,    60000,   150000),
        "sink_set":           (4000,    7000,     15000,   35000),
        "electrical_point":   (650,     900,      1400,    2600),
        "pipe_m":             (250,     350,      500,     800),
        "excavation_m3":      (350,     350,      350,     350),
        "filling_m3":         (600,     600,      600,     600),
    }
    BOQ_ITEM_RATE = {   # take-off item → supply rate key
        "Excavation for footings": "excavation_m3", "Plinth filling (earth / murram)": "filling_m3",
        "Vitrified floor tiles": "floor_tile_m2", "Anti-skid tiles (toilets, utility)": "wet_tile_m2",
        "Wall dado tiles": "wet_tile_m2", "Toilet waterproofing": "waterproofing_m2",
        "Terrace waterproofing": "waterproofing_m2", "Main door (frame + shutter)": "main_door",
        "Room doors": "room_door", "Toilet doors (WPC / FRP)": "toilet_door", "Windows & ventilators": "window_m2",
        "Inside: putty, primer, 2 coats emulsion": "paint_in_m2",
        "Outside: primer + 2 coats weatherproof paint": "paint_out_m2",
        "Toilet sanitary sets": "sanitary_set", "Kitchen sink sets": "sink_set",
        "Electrical points (conduit + wiring)": "electrical_point", "Water supply & drainage pipes": "pipe_m",
    }
    # Labour / material ratio by stage (CPWD analysis of rates, typical Indian site practice)
    STAGE_LABOUR = {"foundation": 0.70, "plinth": 0.60, "frame": 0.60, "masonry": 0.80, "rough_in": 0.80,
                    "plaster": 2.00, "flooring": 0.60, "openings": 0.15, "finishing": 0.80}
    STAGE_WEEKS = {"foundation": 0.12, "plinth": 0.05, "frame": 0.22, "masonry": 0.14, "rough_in": 0.05,
                   "plaster": 0.12, "flooring": 0.10, "openings": 0.05, "finishing": 0.15}

    @classmethod
    def boq_for_plan(cls, plan, est: Dict[str, Any], quality_level: str = "Standard (Medium)",
                     design_floors: Optional[int] = None) -> Dict[str, Any]:
        """
        Bill of quantities measured from the drawing (floorplan_generator.takeoff), priced at
        local material rates, and split into construction stages. The stage amounts are the
        project's own construction cost (`est`, from estimate_for_plan) shared out in proportion
        to each stage's materials + its typical labour — one cost formula, not two.
        """
        country = est.get("country", "India")
        if country != "India":
            return {"available": False,
                    "reason": "Measured take-off covers India's RCC-frame + brick construction; "
                              f"{country} uses {MARKETS[country]['system'].lower()}, priced by area here."}
        from floorplan_generator.takeoff import material_takeoff, STAGES
        per_floor_sqft = (getattr(plan, "total_built_up_area", 0) or 0) * SQFT_PER_M2
        floors = max(1, int(round(est.get("area_sqft", 0) / per_floor_sqft))) if per_floor_sqft else \
            max(1, int(getattr(plan, "num_floors", 1) or 1))          # the floors this estimate priced
        to = material_takeoff(plan, floors=floors, design_floors=design_floors)
        m = MARKETS["India"]
        mult = est.get("region_multiplier", 1.0)
        mat_index = 1 + 0.5 * (mult - 1)
        quality = cls.normalize_quality(quality_level)
        qi = QUALITY_KEYS.index(quality)
        base = {k: price * mat_index for k, _, _, _, price, _ in m["materials"]}
        sand_cft, agg_cft = base["Sand"], base["Aggregate"]

        def basic_cost(line):
            return (line["cement_bags"] * base["Cement"] + line["steel_kg"] * base["Steel"]
                    + line["bricks"] * base["Bricks"] + line["sand_m3"] * 35.3147 * sand_cft
                    + line["aggregate_m3"] * 35.3147 * agg_cft)

        lines = []
        for l in to["lines"]:
            key = cls.BOQ_ITEM_RATE.get(l["item"])
            rate = cls.FINISH_RATES_INDIA[key][qi] * mat_index if key else 0.0
            amount = basic_cost(l) + l["qty"] * rate
            lines.append(dict(l, rate=round(rate, 2), amount=round(amount, 2)))

        site_lines = [l for l in lines if l["group"] == "Site works"]
        bld_lines = [l for l in lines if l["group"] != "Site works"]
        breakdown = est.get("cost_breakdown", {})
        site_cost = breakdown.get("Site Works & Amenities", 0.0)
        lift_cost = breakdown.get("Lift & Elevator", 0.0)
        building = max(0.0, est.get("construction_cost", 0.0) - site_cost - lift_cost)
        materials_total = sum(l["amount"] for l in bld_lines)

        # stage weights = materials × (1 + labour); scaled so the stages add up to the building cost
        raw = {k: 0.0 for k, _, _ in STAGES}
        mats = {k: {"cement_bags": 0.0, "steel_kg": 0.0, "bricks": 0.0, "materials": 0.0} for k, _, _ in STAGES}
        for l in bld_lines:
            raw[l["stage"]] += l["amount"] * (1 + cls.STAGE_LABOUR[l["stage"]])
            for k in ("cement_bags", "steel_kg", "bricks"):
                mats[l["stage"]][k] += l[k]
            mats[l["stage"]]["materials"] += l["amount"]
        scale = building / max(1.0, sum(raw.values()))
        sqft = to["basis"]["built_up_sqft"]
        weeks_total = round(16 + 0.011 * sqft + 6 * (floors - 1))
        stages, cum, week = [], 0.0, 0.0
        for key, label, what in STAGES:
            amt = raw[key] * scale
            cum += amt
            w = max(1.0, cls.STAGE_WEEKS[key] * weeks_total)
            stages.append({"key": key, "label": label, "what": what, "amount": round(amt, 2),
                           "materials": round(mats[key]["materials"], 2),
                           "pct": round(100 * amt / max(1.0, building + site_cost + lift_cost), 1),
                           "cumulative": round(cum, 2), "start_week": round(week), "weeks": round(w),
                           "cement_bags": round(mats[key]["cement_bags"]), "steel_kg": round(mats[key]["steel_kg"]),
                           "bricks": round(mats[key]["bricks"])})
            week += w
        if site_cost or lift_cost:
            cum += site_cost + lift_cost
            stages.append({"key": "site", "label": "Site works & amenities" + (" + lift" if lift_cost else ""),
                           "what": "Compound wall, gate, parking, garden, sump …" + (", lift" if lift_cost else ""),
                           "amount": round(site_cost + lift_cost, 2),
                           "materials": round(sum(l["amount"] for l in site_lines), 2),
                           "pct": round(100 * (site_cost + lift_cost) / max(1.0, building + site_cost + lift_cost), 1),
                           "cumulative": round(cum, 2), "start_week": round(week - stages[-1]["weeks"]),
                           "weeks": stages[-1]["weeks"],
                           "cement_bags": round(sum(l["cement_bags"] for l in site_lines)), "steel_kg": 0,
                           "bricks": round(sum(l["bricks"] for l in site_lines))})

        t = to["totals"]
        thumb = {k: v.get("quantity", 0) for k, v in (est.get("materials") or {}).items()}
        key_materials = [
            ("Cement", "Cement (OPC 53, 50 kg)", t["cement_bags"], "bags", base["Cement"]),
            ("Steel", "TMT steel (Fe 500D)", t["steel_kg"], "kg", base["Steel"]),
            ("Bricks", "Clay bricks", t["bricks"], "nos", base["Bricks"]),
            ("Sand", "M-sand / river sand", t["sand_cft"], "cft", sand_cft),
            ("Aggregate", "Aggregate (20/40 mm)", t["aggregate_cft"], "cft", agg_cft),
        ]
        return {
            "available": True,
            "lines": lines,
            "materials": [{"key": k, "label": lab, "qty": round(q), "unit": u, "rate": round(r, 2),
                           "amount": round(q * r, 2), "per_sqft": round(q / max(1, sqft), 2),
                           "thumb_qty": thumb.get(k)} for k, lab, q, u, r in key_materials],
            "finishes": {"floor_tiles_m2": t["floor_tiles_m2"], "paint_litres": t["paint_litres"],
                         "putty_kg": t["putty_kg"], "concrete_m3": t["concrete_m3"],
                         "brickwork_m3": t["brickwork_m3"]},
            "groups": {g: round(sum(l["amount"] for l in bld_lines if l["group"] == g), 2)
                       for g in dict.fromkeys(l["group"] for l in bld_lines)},
            "stages": stages,
            "materials_total": round(materials_total, 2),
            "labour_total": round(max(0.0, building - materials_total), 2),
            "building_cost": round(building, 2),
            "site_cost": round(site_cost + lift_cost, 2),
            "weeks": int(round(week)),
            "basis": to["basis"],
            "country": country,
        }

    @classmethod
    def region_matrix(cls, area_sqm: float, country: str, num_floors: int = 1,
                      kitchens: int = 1, bathrooms: int = 1,
                      extras_raw: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Total project cost for every region of a market × every quality grade."""
        country = cls.normalize_country(country)
        m = MARKETS[country]
        floors = max(1, int(num_floors or 1))
        area_sqft = max(0.0, float(area_sqm or 0)) * SQFT_PER_M2 * floors
        rows = []
        for reg in m["regions"]:
            mult, over = cls._region_params(m, reg)
            row: Dict[str, Any] = {"region": reg, "index": mult}
            for q in QUALITY_KEYS:
                row[QUALITY_SHORT[q]] = cls._totals(m, mult, over, q, area_sqft, floors, kitchens,
                                                    bathrooms, dict(extras_raw or {}))["total"]
            rows.append(row)
        return sorted(rows, key=lambda r: -r["Standard"])

    @classmethod
    def region_matrix_for_plan(cls, plan, country: str, num_floors: int = None) -> List[Dict[str, Any]]:
        country = cls.normalize_country(country)
        floors = max(1, int(num_floors or getattr(plan, "num_floors", 1) or 1))
        area, kitchens, baths, _, extras_raw = cls._plan_inputs(plan, country, floors)
        return cls.region_matrix(area, country, floors, kitchens, baths, extras_raw)
