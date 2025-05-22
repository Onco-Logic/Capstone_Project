import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Load and Clean Data
df = pd.read_csv("Data/survival_data.csv")

# Standardize column names
df.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)

# Define Features and Target Variables
X = df[['Disease', 'Gender', 'Age_Group']]  # Categorical variables
y_columns = ['Survival_5Y', 'Survival_10Y', 'Survival_15Y']  # Separate targets

# Encode Categorical Variables using One-Hot Encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Save Encoder and Scaler for Future Predictions
os.makedirs("models", exist_ok=True)
joblib.dump(encoder, "models/survival_encoder.pkl")
joblib.dump(scaler, "models/survival_scaler.pkl")
# os.makedirs("models", exist_ok=True)
# joblib.dump(encoder, "models/models/survival_encoder.pkl")
# joblib.dump(scaler, "models/models/survival_scaler.pkl")

# Define Base Models
base_models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Linear Regression": LinearRegression(),
    "Support Vector Machine": SVR(),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    "CatBoost": cb.CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, verbose=0)
}

# Stacking Ensemble (Combining Models)
stacking_regressor = StackingRegressor(
    estimators=[("Random Forest", base_models["Random Forest"]),
                ("XGBoost", base_models["XGBoost"]),
                ("LightGBM", base_models["LightGBM"])],
    final_estimator=LinearRegression()
)

# Add Stacking Model to the list
base_models["Stacking Ensemble"] = stacking_regressor

# Hyperparameter Optimization (GridSearchCV for RandomForest)
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring="r2", n_jobs=-1)

# Train and Save Models
for survival_target in y_columns:
    print(f"\n🔹 Training models for {survival_target}...")

    y = df[survival_target]  # Select only one survival column at a time

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    os.makedirs("models/data", exist_ok=True)  # Ensure the directory exists
    # os.makedirs("models/models/data", exist_ok=True)  # Ensure the directory exists

    # Save Test Data
    pd.DataFrame(X_test).to_csv("models/data/X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"models/data/y_test_{survival_target}.csv", index=False)
    # pd.DataFrame(X_test).to_csv("models/models/data/X_test.csv", index=False)
    # pd.DataFrame(y_test).to_csv(f"models/models/data/y_test_{survival_target}.csv", index=False)

    for name, model in base_models.items():
        print(f"   Training {name} for {survival_target}...")

        # If Random Forest, apply GridSearchCV for tuning
        if name == "Random Forest":
            model = rf_grid.fit(X_train, y_train).best_estimator_

        model.fit(X_train, y_train)

        # Save Model
        model_filename = f"models/{name.lower().replace(' ', '_')}_{survival_target}.pkl"
        # model_filename = f"models/models/{name.lower().replace(' ', '_')}_{survival_target}.pkl"
        joblib.dump(model, model_filename)

print("\n✅ All survival models trained and saved successfully!")
