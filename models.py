import joblib

# Load the files
data1 = joblib.load("expense_tracker_model.pkl")
data2 = joblib.load("amount_scaler.pkl")  # MinMaxScaler
data3 = joblib.load("budget_alert_model.pkl")  # Tuple: (model, encoder, category_limits)
data4 = joblib.load("financial_forecast_rf_model.pkl")  # RandomForestRegressor
data5 = joblib.load("fraud_detection_model.pkl")  # IsolationForest

# Extract models and other components
model1 = data1["model"]
scaler = data2  # MinMaxScaler
model3, encoder3, category_limits3 = data3  # Unpacking tuple
model4 = data4
model5 = data5

# Print formatted output
print("\n===== Expense Tracker Model Parameters =====")
print(model1.get_params())

print("\n===== MinMax Scaler Parameters =====")
print(scaler.get_params())

print("\n===== Budget Alert Model Parameters =====")
print("Model:", model3.__class__.__name__)  # Print model type
print("Parameters:", model3.get_params())

print("\n===== Budget Alert Additional Components =====")
print("Encoder:", encoder3.__class__.__name__)  # Print encoder type
print("Category Limits:", category_limits3)

print("\n===== Financial Forecast Model Parameters =====")
print("Model:", model4.__class__.__name__)  # Print model type
print("Parameters:", model4.get_params())

print("\n===== Fraud Detection Model Parameters =====")
print("Model:", model5.__class__.__name__)  # Print model type
print("Parameters:", model5.get_params())
