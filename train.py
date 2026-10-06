import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Generate Synthetic Exercise & Calorie Dataset
np.random.seed(42)
n_samples = 2000

gender = np.random.choice([0, 1], size=n_samples) # 0: Female, 1: Male
age = np.random.randint(18, 65, size=n_samples)
height = np.random.normal(170, 10, size=n_samples)
weight = np.random.normal(70, 15, size=n_samples)
duration = np.random.randint(5, 60, size=n_samples)
heart_rate = np.random.randint(80, 170, size=n_samples)
body_temp = np.random.normal(38.5, 0.8, size=n_samples)

# Formula to generate realistic burned calories output
calories = (
    (age * -0.2) +
    (weight * 0.5) +
    (duration * 4.2) +
    (heart_rate * 1.5) +
    (gender * 15.0) +
    np.random.normal(0, 5, size=n_samples)
)

df = pd.DataFrame({
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'Duration': duration,
    'Heart_Rate': heart_rate,
    'Body_Temp': body_temp,
    'Calories': calories
})

X = df.drop(columns=['Calories'])
y = df['Calories']

# 2. Train/Test Split & Feature Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Hybrid Ensemble Model (XGBoost + Random Forest Regressor)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

hybrid_model = VotingRegressor(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model)
])

# Fit Hybrid Model
hybrid_model.fit(X_train_scaled, y_train)

# Evaluate Model Performance
predictions = hybrid_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("--- Model Training Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} kcal")
print(f"R2 Score: {r2:.4f}")

# 4. Save Model Artifacts
joblib.dump(hybrid_model, 'model_artifacts/calorie_model.pkl')
joblib.dump(scaler, 'model_artifacts/scaler.pkl')
print("\nModel and Scaler successfully saved in 'model_artifacts/' directory.")
