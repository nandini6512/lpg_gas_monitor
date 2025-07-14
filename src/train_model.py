import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset with correct encoding
file_path = "dataset/lpg_india_dataset_250.csv"
df = pd.read_csv(file_path, encoding='utf-8')  # Change encoding if needed

# Print column names to check formatting issues
print("Original column names:", df.columns.tolist())

# Rename columns to remove any encoding issues or spaces
df.columns = df.columns.str.replace(r"[^\x00-\x7F]+", "", regex=True)  # Remove special characters
df.columns = df.columns.str.strip()  # Remove extra spaces

# Print column names after cleaning
print("Updated column names:", df.columns.tolist())

# Define features (X) and target variables (y)
X = df[['Tare Weight (kg)', 'LPG Weight (kg)', 'Gross Weight (kg)', 'Temperature (C)', 'Humidity (%)', 'Air Pressure (hPa)']]

# Target 1: LPG Percentage Prediction
y_percentage = (df['LPG Weight (kg)'] / df['Gross Weight (kg)']) * 100  # Compute LPG % dynamically

# Target 2: Status Prediction (Convert categories into numbers)
label_encoder = LabelEncoder()
df['Status'] = label_encoder.fit_transform(df['Status'])  # Convert 'Safe', 'Low', 'Critical' to numbers
y_status = df['Status']

# Split data into training and testing sets
X_train, X_test, y_percentage_train, y_percentage_test = train_test_split(X, y_percentage, test_size=0.2, random_state=42)
X_train, X_test, y_status_train, y_status_test = train_test_split(X, y_status, test_size=0.2, random_state=42)

# Train LPG Percentage Model (Regression)
model_percentage = LinearRegression()
model_percentage.fit(X_train, y_percentage_train)

# Train Status Model (Classification)
model_status = RandomForestClassifier(n_estimators=100, random_state=42)
model_status.fit(X_train, y_status_train)

# Save models
joblib.dump(model_percentage, "models/lpg_percentage_model.pkl")
joblib.dump(model_status, "models/lpg_status_model.pkl")
joblib.dump(label_encoder, "models/status_label_encoder.pkl")  # Save label encoder for later decoding

print("✅ Models trained and saved successfully!")
