import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle


df = pd.read_csv("data/transactions.csv")

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%d-%m-%Y %H:%M:%S", dayfirst=True)


df['amount'] = pd.to_numeric(df['amount'], errors='coerce')  
df = df.dropna(subset=['amount'])

features = df[['amount']].copy() 
model = IsolationForest(contamination=0.02, random_state=42)
df['anomaly_score'] = model.fit_predict(features)


with open("fraud_detection_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as 'fraud_detection_model.pkl'.")

