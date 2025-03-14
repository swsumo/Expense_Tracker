import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/transactions_data_extended.csv", parse_dates=["date"], dayfirst=True)
df = df.sort_values(by="date")

# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

# Drop rows with invalid dates
df = df.dropna(subset=['date'])

# Preprocessing
df["amount"] = df["amount"].astype(float)
scaler = MinMaxScaler()
df["scaled_amount"] = scaler.fit_transform(df[["amount"]])

# Feature Engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Prepare time-series data
def create_sequences(data, seq_length=10):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i: i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 10
data = df["scaled_amount"].values
X, y = create_sequences(data, seq_length)

# Split data properly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Flatten input for RandomForest
train_samples, test_samples = X_train.shape[0], X_test.shape[0]
X_train = X_train.reshape(train_samples, -1)
X_test = X_test.reshape(test_samples, -1)

# Train RandomForest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "financial_forecast_rf_model.pkl")
joblib.dump(scaler, "amount_scaler.pkl")

# Evaluate model
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# Predict future spending
future_sequence = X[-1].reshape(1, -1)
future_predictions = []

for _ in range(15):  # Predict next 15 time periods
    next_pred = model.predict(future_sequence)[0]
    future_predictions.append(next_pred)
    future_sequence = np.append(future_sequence[:, 1:], [[next_pred]], axis=1)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Plot Results
plt.plot(actual_values, label="Actual")
plt.plot(test_predictions, label="Predicted")
plt.legend()
plt.show()
