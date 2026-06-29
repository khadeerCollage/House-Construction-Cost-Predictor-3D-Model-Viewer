"""
Indian Construction Cost Estimation Engine
===========================================
Calculates realistic construction costs for residential buildings in India
based on build area (sqm/sqft), city tiers, material quality, and number of floors.
Also calculates estimated material quantities (cement, steel, bricks, sand, aggregate) and costs.
"""

from typing import Dict, Any

class CostEngine:
    """
    Cost estimation engine with realistic Indian construction market rates (INR - ₹).
    """
    
    # City Multipliers based on labor and material cost differences:
    # Tier-1: Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata
    # Tier-2: Jaipur, Lucknow, Nagpur, Indore, Patna, etc.
    # Tier-3: Small towns, villages
    CITY_MULTIPLIERS = {
        "Tier-1 (Metro)": 1.35,
        "Tier-2 (Urban)": 1.00,
        "Tier-3 (Rural)": 0.80
    }
    
    # Base Rates per Square Foot for different quality levels:
    QUALITY_RATES = {
        "Basic (Low Cost)": {
            "rate_per_sqft": 1000,
            "description": "Simple brick structure, standard flooring, basic paint, local plumbing & electric fixtures.",
            "breakdown_pct": {
                "Cement & Steel (Structure)": 35,
                "Bricks & Sand (Masonry)": 20,
                "Flooring & Tiles": 10,
                "Plumbing & Sanitation": 8,
                "Electrical Fittings": 7,
                "Painting & Finishes": 8,
                "Labor & Supervision": 12
            }
        },
        "Standard (Medium)": {
            "rate_per_sqft": 1550,
            "description": "RCC structure, vitrified tile flooring, premium paint (Apex/Emulsion), brand fittings (Jaguaar/Anchor).",
            "breakdown_pct": {
                "Cement & Steel (Structure)": 33,
                "Bricks & Sand (Masonry)": 18,
                "Flooring & Tiles": 12,
                "Plumbing & Sanitation": 10,
                "Electrical Fittings": 9,
                "Painting & Finishes": 10,
                "Labor & Supervision": 8
            }
        },
        "Premium (High Quality)": {
            "rate_per_sqft": 2500,
            "description": "Heavy structure, Italian marble/wooden flooring, luxury finishes, automated/modular kitchen & branded premium fittings.",
            "breakdown_pct": {
                "Cement & Steel (Structure)": 30,
                "Bricks & Sand (Masonry)": 15,
                "Flooring & Tiles": 15,
                "Plumbing & Sanitation": 12,
                "Electrical Fittings": 11,
                "Painting & Finishes": 11,
                "Labor & Supervision": 6
            }
        }
    }
    
    # Material Constants per Square Foot (Average requirements in India)
    # E.g. Cement: 0.4 bags/sqft, Steel: 2.5 kg/sqft, Bricks: 22 bricks/sqft
    MATERIAL_FACTORS = {
        "cement_bags_per_sqft": 0.42,
        "steel_kg_per_sqft": 2.8,
        "bricks_count_per_sqft": 20,
        "sand_cft_per_sqft": 1.6,
        "aggregate_cft_per_sqft": 1.35
    }
    
    # Material Unit Prices in INR
    MATERIAL_PRICES = {
        "cement_per_bag": 410,       # Avg rate for UltraTech/Ambuja
        "steel_per_kg": 68,          # Avg rate for Tata/Jindal
        "brick_per_piece": 8.5,      # Avg clay brick rate
        "sand_per_cft": 65,          # Sand/m-sand rate
        "aggregate_per_cft": 70      # Crusher metal rate
    }

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
        num_floors: int = 1
    ) -> Dict[str, Any]:
        """
        Estimate realistic Indian construction cost and material breakdown.
        
        Args:
            area_sqm: Calibrated building area in square meters.
            bedrooms: Number of bedrooms.
            bathrooms: Number of bathrooms.
            kitchens: Number of kitchens.
            living_rooms: Number of living/dining spaces.
            city_tier: Metro, Urban or Rural tier multiplier.
            quality_level: Basic, Standard or Premium pricing tier.
            num_floors: Number of floors/levels.
            
        Returns:
            Dict containing detailed cost and material breakdowns.
        """
        # Convert area from sqm to sqft
        area_sqft = area_sqm * 10.7639
        # Multiply area by floor count for total built-up area
        total_built_up_sqft = area_sqft * num_floors
        
        # Get base rate and city multiplier
        rate_info = cls.QUALITY_RATES.get(quality_level, cls.QUALITY_RATES["Standard (Medium)"])
        base_rate = rate_info["rate_per_sqft"]
        city_mult = cls.CITY_MULTIPLIERS.get(city_tier, 1.0)
        
        # Total base structure cost
        base_cost = total_built_up_sqft * base_rate * city_mult
        
        # Add room complexity surcharge (bathrooms and kitchens are more expensive due to plumbing/fittings)
        # Wet areas have extra costs (₹25,000 for standard kitchen, ₹15,000 for standard bathroom)
        kitchen_premium = kitchens * 30000 * city_mult
        bathroom_premium = bathrooms * 20000 * city_mult
        room_surcharge = kitchen_premium + bathroom_premium
        
        total_estimated_cost = base_cost + room_surcharge
        
        # Calculate component breakdown (INR)
        breakdown_pct = rate_info["breakdown_pct"]
        cost_breakdown = {}
        for category, pct in breakdown_pct.items():
            cost_breakdown[category] = round((total_estimated_cost * pct) / 100.0, 2)
            
        # Estimate material quantities and their costs
        cement_bags = round(total_built_up_sqft * cls.MATERIAL_FACTORS["cement_bags_per_sqft"])
        steel_kg = round(total_built_up_sqft * cls.MATERIAL_FACTORS["steel_kg_per_sqft"])
        bricks = round(total_built_up_sqft * cls.MATERIAL_FACTORS["bricks_count_per_sqft"])
        sand_cft = round(total_built_up_sqft * cls.MATERIAL_FACTORS["sand_cft_per_sqft"])
        aggregate_cft = round(total_built_up_sqft * cls.MATERIAL_FACTORS["aggregate_cft_per_sqft"])
        
        material_breakdown = {
            "Cement": {
                "quantity": cement_bags,
                "unit": "Bags",
                "rate": cls.MATERIAL_PRICES["cement_per_bag"],
                "cost": round(cement_bags * cls.MATERIAL_PRICES["cement_per_bag"], 2)
            },
            "Steel": {
                "quantity": steel_kg,
                "unit": "Kg",
                "rate": cls.MATERIAL_PRICES["steel_per_kg"],
                "cost": round(steel_kg * cls.MATERIAL_PRICES["steel_per_kg"], 2)
            },
            "Bricks": {
                "quantity": bricks,
                "unit": "Pcs",
                "rate": cls.MATERIAL_PRICES["brick_per_piece"],
                "cost": round(bricks * cls.MATERIAL_PRICES["brick_per_piece"], 2)
            },
            "Sand": {
                "quantity": sand_cft,
                "unit": "CFT",
                "rate": cls.MATERIAL_PRICES["sand_per_cft"],
                "cost": round(sand_cft * cls.MATERIAL_PRICES["sand_per_cft"], 2)
            },
            "Aggregate": {
                "quantity": aggregate_cft,
                "unit": "CFT",
                "rate": cls.MATERIAL_PRICES["aggregate_per_cft"],
                "cost": round(aggregate_cft * cls.MATERIAL_PRICES["aggregate_per_cft"], 2)
            }
        }
        
        # Calculate comparison for UI (side-by-side)
        comparison = {}
        for qual_name, q_info in cls.QUALITY_RATES.items():
            q_base_cost = total_built_up_sqft * q_info["rate_per_sqft"] * city_mult
            # Wet areas premiums adjust slightly by quality
            qual_mult = q_info["rate_per_sqft"] / cls.QUALITY_RATES["Standard (Medium)"]["rate_per_sqft"]
            q_surcharge = (kitchen_premium + bathroom_premium) * qual_mult
            comparison[qual_name] = round(q_base_cost + q_surcharge, 2)
            
        return {
            "total_cost": round(total_estimated_cost, 2),
            "area_sqft": round(total_built_up_sqft, 2),
            "cost_per_sqft": round(total_estimated_cost / total_built_up_sqft, 2) if total_built_up_sqft > 0 else 0,
            "cost_breakdown": cost_breakdown,
            "materials": material_breakdown,
            "comparison": comparison,
            "description": rate_info["description"]
        }
