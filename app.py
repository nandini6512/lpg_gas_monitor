from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
model_percentage = joblib.load("models/lpg_percentage_model.pkl")
model_status = joblib.load("models/lpg_status_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        tare_weight = float(request.form['tare_weight'])
        lpg_weight = float(request.form['lpg_weight'])
        gross_weight = float(request.form['gross_weight'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        air_pressure = float(request.form['air_pressure'])

        # Prepare input for models
        input_data = np.array([[tare_weight, lpg_weight, gross_weight, temperature, humidity, air_pressure]])

        # Predict LPG Percentage
        lpg_percentage = model_percentage.predict(input_data)[0]

        # Predict Status
        status_code = int(model_status.predict(input_data)[0])  # Ensure it's an integer

        # Convert numeric status to readable text
        status_labels = {0: "Safe", 1: "Low", 2: "Critical"}
        status_str = status_labels.get(status_code, "Unknown")

        return render_template('result.html', percentage=round(lpg_percentage, 2), status=status_str.lower())

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
