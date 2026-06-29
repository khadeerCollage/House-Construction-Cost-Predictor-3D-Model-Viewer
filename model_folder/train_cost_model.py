"""
ML Model Training Script
========================
Generates a realistic dataset based on Indian construction cost rules,
trains a Random Forest Regressor, and saves it as a joblib model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import joblib
import os

def generate_enhanced_data_and_train():
    print("Generating enhanced training dataset...")
    # Load original dataset as base to preserve same data distribution style
    base_file = os.path.join(os.path.dirname(__file__), "house_cost_prediction_dataset_10000.csv")
    
    if os.path.exists(base_file):
        df_base = pd.read_csv(base_file)
        n_samples = len(df_base)
        areas = df_base['Area_m2'].values
        bedrooms = df_base['Rooms'].values
        bathrooms = df_base['Bathrooms'].values
    else:
        # Generate synthetically if base file not found
        n_samples = 10000
        np.random.seed(42)
        areas = np.random.randint(40, 400, size=n_samples)
        bedrooms = np.random.randint(1, 6, size=n_samples)
        bathrooms = np.random.randint(1, 4, size=n_samples)

    # Generate additional realistic features for Indian market
    # 0: Tier-3 (Rural), 1: Tier-2 (Urban), 2: Tier-1 (Metro)
    city_tiers = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.50, 0.25])
    # 0: Basic, 1: Standard, 2: Premium
    quality_levels = np.random.choice([0, 1, 2], size=n_samples, p=[0.30, 0.55, 0.15])
    num_floors = np.random.choice([1, 2, 3], size=n_samples, p=[0.70, 0.25, 0.05])
    kitchens = np.ones(n_samples, dtype=int)
    # Extra kitchen for larger multi-floor homes
    kitchens[(areas > 180) & (num_floors > 1)] = 2
    
    living_rooms = np.ones(n_samples, dtype=int)
    living_rooms[areas > 150] = 2

    # Map multipliers
    city_multipliers = np.array([0.8, 1.0, 1.35])
    quality_rates = np.array([1000, 1550, 2500]) # per sqft

    # Calculate target (Cost in INR) using the logic from CostEngine:
    # 1. Convert sqm to sqft
    areas_sqft = areas * 10.7639
    total_built_up_sqft = areas_sqft * num_floors
    
    # 2. Base cost calculation
    base_cost = total_built_up_sqft * quality_rates[quality_levels] * city_multipliers[city_tiers]
    
    # 3. Add wet area surcharges
    room_surcharges = (kitchens * 30000 + bathrooms * 20000) * city_multipliers[city_tiers]
    
    # Total Cost with some normal noise (+-5%) to make it look realistic for ML training
    noise = np.random.normal(1.0, 0.03, size=n_samples)
    target_costs = (base_cost + room_surcharges) * noise

    # Construct DataFrame
    df = pd.DataFrame({
        "Area_m2": areas,
        "Rooms": bedrooms,
        "Bathrooms": bathrooms,
        "Kitchens": kitchens,
        "Living_Rooms": living_rooms,
        "City_Tier": city_tiers,
        "Quality_Level": quality_levels,
        "Num_Floors": num_floors,
        "Estimated_Cost_INR": target_costs
    })

    print(f"Data shape: {df.shape}")
    
    # Save the new dataset
    dataset_out = os.path.join(os.path.dirname(__file__), "house_cost_prediction_dataset_enhanced.csv")
    df.to_csv(dataset_out, index=False)
    print(f"Enhanced dataset saved to {dataset_out}")

    # Prepare features and target
    X = df.drop(columns=["Estimated_Cost_INR"])
    y = df["Estimated_Cost_INR"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    print(f"Model Training Completed.")
    print(f"R² Score: {r2:.4f} (Accuracy)")
    print(f"Mean Absolute Percentage Error: {mape*100:.2f}%")

    # Save model
    model_out = os.path.join(os.path.dirname(__file__), "house_cost_model_v2.pkl")
    joblib.dump(model, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    generate_enhanced_data_and_train()
