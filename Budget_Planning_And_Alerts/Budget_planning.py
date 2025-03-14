import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/transactions.csv")  

# User-defined budgets
categories = df["category"].unique()
budgets = {}
for category in categories:
    budget = float(input(f"Set budget for {category}: "))
    budgets[category] = budget

# Calculate total spending per category
df["cumulative_spending"] = df.groupby("category")["amount"].cumsum()
df["budget_limit"] = df["category"].map(budgets)


# 0 = Normal, 1 = Warning (near budget), 2 = Over budget
def alert_level(row):
    if row["cumulative_spending"] >= row["budget_limit"]:
        return 2  
    elif row["cumulative_spending"] >= 0.8 * row["budget_limit"]:
        return 1 
    return 0 

df["alert"] = df.apply(alert_level, axis=1)

# Encode categorical features
le = LabelEncoder()
df["category"] = le.fit_transform(df["category"])

# Train ML Model
X = df[["category", "amount", "cumulative_spending", "budget_limit"]]
y = df["alert"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("budget_alert_model.pkl", "wb") as f:
    pickle.dump((model, le, budgets), f)

print("Budget alert model trained and saved successfully!")
