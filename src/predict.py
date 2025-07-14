import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model_perc = joblib.load("models/lpg_percentage_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Function to predict LPG percentage and status
def predict_lpg(tare_weight, lpg_weight, gross_weight, temperature, humidity, air_pressure):
    # Prepare input data
    input_data = np.array([[tare_weight, lpg_weight, gross_weight, temperature, humidity, air_pressure]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict LPG Percentage
    lpg_percentage = model_perc.predict(input_scaled)[0]

    # Determine status
    if temperature == 29:  
        status = "Safe"
    elif lpg_percentage > 60:
        status = "High"
    elif lpg_percentage > 30:
        status = "Medium"
    else:
        status = "Low"

    return round(lpg_percentage, 2), status

# Example test
if __name__ == "__main__":
    tare_weight = 15.0
    lpg_weight = 5.0
    gross_weight = 20.0
    temperature = 25.0
    humidity = 50.0
    air_pressure = 1000.0

    percentage, status = predict_lpg(tare_weight, lpg_weight, gross_weight, temperature, humidity, air_pressure)
    print(f"Predicted LPG Percentage: {percentage}%")
    print(f"Status: {status}")
