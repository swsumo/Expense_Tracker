import os
import datetime
import calendar
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

EXPENSE_FILE = "data/cleaned_transactions.csv"
CATEGORY_MAPPING_FILE = os.path.join(os.path.dirname(__file__), "category_mapping.json")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "expense_model.pkl")

class Expense:
    def __init__(self, name, amount, category, date=None):
        self.name = name
        self.amount = float(amount)
        self.category = category
        self.date = date if date else datetime.datetime.now().strftime("%Y-%m-%d")
    
    def to_dict(self):
        return {
            "name": self.name,
            "amount": self.amount,
            "category": self.category,
            "date": self.date
        }

class ExpensePredictor:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.trained = False
    
    def prepare_data(self, df):
        # Convert date to features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_month'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Extract text features from name
        df['name_length'] = df['name'].apply(len)
        
        # Encode categorical features
        if self.encoder is None:
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            category_encoded = self.encoder.fit_transform(df[['category']])
        else:
            category_encoded = self.encoder.transform(df[['category']])
        
        category_cols = [f'category_{i}' for i in range(category_encoded.shape[1])]
        category_df = pd.DataFrame(category_encoded, columns=category_cols)
        
        # Create feature matrix
        features_df = pd.concat([
            df[['day_of_month', 'day_of_week', 'month', 'name_length']].reset_index(drop=True),
            category_df.reset_index(drop=True)
        ], axis=1)
        
        return features_df, df['amount']
    
    def train(self, expenses, force_retrain=False):
        if not expenses or len(expenses) < 10:
            print("Not enough data to train the model (minimum 10 expenses required).")
            return False
        
        if self.trained and not force_retrain:
            print("Model already trained. Use force_retrain=True to retrain.")
            return True
        
        # Convert expenses to DataFrame
        df = pd.DataFrame([exp.to_dict() for exp in expenses])
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"Model trained successfully!")
            print(f"Mean Absolute Error: ${mae:.2f}")
            print(f"Root Mean Squared Error: ${rmse:.2f}")
            
            self.trained = True
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_expense(self, category, name, date=None):
        if not self.trained or self.model is None:
            print("Model not trained yet.")
            return None
        
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create a dataframe with one row
        df = pd.DataFrame([{
            'category': category,
            'name': name,
            'date': date
        }])
        
        # Prepare features
        X, _ = self.prepare_data(df)
        
        # Make prediction
        predicted_amount = self.model.predict(X)[0]
        
        return max(0.01, predicted_amount) 
    
    def save_model(self):
        if not self.trained or self.model is None:
            print("No trained model to save.")
            return False
        
        try:
            import pickle
            model_data = {
                'model': self.model,
                'encoder': self.encoder
            }
            
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(model_data, f)
            
            print("Model saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        if not os.path.exists(MODEL_FILE):
            print("No saved model found.")
            return False
        
        try:
            import pickle
            with open(MODEL_FILE, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.encoder = model_data['encoder']
            self.trained = True
            
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class ExpenseManager:
    def __init__(self):
        self.expenses = []
        self.category_mapping = {}
        self.predictor = ExpensePredictor()
        self.load_data()
        self.load_category_mapping()
        self.predictor.load_model()
    
    def load_data(self):
        if os.path.exists(EXPENSE_FILE):
            try:
                df = pd.read_csv(EXPENSE_FILE)
                
                df.columns = [col.lower() for col in df.columns]
                column_mapping = {
                    'name': 'name',
                    'amount': 'amount',
                    'category': 'category',
                    'date': 'date'
                }

                for expected_col, actual_col in column_mapping.items():
                    if actual_col not in df.columns:
                        for col in df.columns:
                            if col.lower() == actual_col.lower():
                                df = df.rename(columns={col: actual_col})
                                break
                
                for _, row in df.iterrows():
                    self.expenses.append(Expense(
                        row['name'],
                        row['amount'],
                        row['category'],
                        row['date']
                    ))
                print(f"Loaded {len(self.expenses)} expenses from file.")
            except Exception as e:
                print(f"Error loading expense data: {e}")
                self.save_data()
        else:
            self.save_data()
    
    def load_category_mapping(self):
        if os.path.exists(CATEGORY_MAPPING_FILE):
            try:
                with open(CATEGORY_MAPPING_FILE, 'r') as f:
                    self.category_mapping = json.load(f)
                print("Loaded category mappings.")
            except Exception as e:
                print(f"Error loading category mappings: {e}")
                self.initialize_category_mapping()
        else:
            self.initialize_category_mapping()
    
    def initialize_category_mapping(self):
        sample_categories = [
            "food", "transportation", "education", "utilities", "entertainment",
            "groceries", "fitness", "health", "housing", "shopping", "fuel",
            "dining", "travel", "personal care", "gifts", "donations", "sports",
            "electronics", "grooming", "income"
        ]
        
        self.category_mapping = {cat: cat for cat in sample_categories}
        self.save_category_mapping()
    
    def save_data(self):
        data = []
        for expense in self.expenses:
            data.append(expense.to_dict())
        
        df = pd.DataFrame(data)
        
        dir_path = os.path.dirname(EXPENSE_FILE)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save to CSV
        if not df.empty:
            df.to_csv(EXPENSE_FILE, index=False, columns=["name", "amount", "category", "date"], 
                      header=["Name", "Amount", "Category", "Date"])
        else:
            pd.DataFrame(columns=["Name", "Amount", "Category", "Date"]).to_csv(EXPENSE_FILE, index=False)
        
        print("Expense data saved.")
    
    def save_category_mapping(self):
        with open(CATEGORY_MAPPING_FILE, 'w') as f:
            json.dump(self.category_mapping, f, indent=4)
        print("Category mappings saved.")
    
    def add_expense(self, name, amount, category, date=None):
        normalized_category = self.normalize_category(category)
        
        expense = Expense(name, amount, normalized_category, date)
        self.expenses.append(expense)
        self.save_data()
        print(f"Added expense: {name} - ${amount} ({normalized_category})")
        
        # Train or update model if we have enough data
        if len(self.expenses) > 10 and len(self.expenses) % 10 == 0:
            self.predictor.train(self.expenses)
            self.predictor.save_model()
        
        return expense
    
    def normalize_category(self, category):
        category = category.lower().strip()

        for key, value in self.category_mapping.items():
            if category == key or category == value:
                return value

        self.category_mapping[category] = category
        self.save_category_mapping()
        return category
    
    def get_expenses_by_time_period(self, period_type, period_value=None):
        if not self.expenses:
            return [], 0
        
        df = pd.DataFrame([exp.to_dict() for exp in self.expenses])
        df['date'] = pd.to_datetime(df['date'])
        
        # Add time period columns
        df['week'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        filtered_df = df.copy()
        
        # Filter by the specified time period
        if period_type == 'weekly' and period_value:
            year, week = period_value
            filtered_df = df[(df['year'] == year) & (df['week'] == week)]
        elif period_type == 'monthly' and period_value:
            year, month = period_value
            filtered_df = df[(df['year'] == year) & (df['month'] == month)]
        elif period_type == 'yearly' and period_value:
            # period_value should be year
            filtered_df = df[df['year'] == period_value]
        
        # Convert filtered dataframe back to Expense objects
        filtered_expenses = []
        for _, row in filtered_df.iterrows():
            filtered_expenses.append(Expense(
                row['name'],
                row['amount'],
                row['category'],
                row['date'].strftime('%Y-%m-%d')
            ))
        
        total = filtered_df['amount'].sum() if not filtered_df.empty else 0
        
        return filtered_expenses, total
    
    def get_expense_summary(self):
        if not self.expenses:
            return "No expenses recorded yet."
        
        df = pd.DataFrame([exp.to_dict() for exp in self.expenses])
        df['date'] = pd.to_datetime(df['date'])
        
        # Summary statistics
        total_spent = df['amount'].sum()
        category_totals = df.groupby('category')['amount'].sum().to_dict()
        
        # Time-based summaries
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month
        
        # Current month expenses
        month_expenses = df[(df['date'].dt.year == current_year) & 
                           (df['date'].dt.month == current_month)]['amount'].sum()
        
        # Current year expenses
        year_expenses = df[df['date'].dt.year == current_year]['amount'].sum()
        
        # ML model performance metrics
        model_status = "Trained" if self.predictor.trained else "Not trained"
        
        summary = {
            'total_spent': total_spent,
            'category_totals': category_totals,
            'month_expenses': month_expenses,
            'year_expenses': year_expenses,
            'model_status': model_status
        }
        
        return summary
    
    def visualize_expenses(self, chart_type='pie', period=None, category=None):
        if not self.expenses:
            print("No expenses to visualize.")
            return
        
        df = pd.DataFrame([exp.to_dict() for exp in self.expenses])
        df['date'] = pd.to_datetime(df['date'])
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'pie':
            # Category breakdown
            category_totals = df.groupby('category')['amount'].sum()
            plt.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%')
            plt.axis('equal')
            plt.title('Expense Distribution by Category')
            
        elif chart_type == 'bar':
            category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
            category_totals.plot(kind='bar')
            plt.title('Expense by Category')
            plt.ylabel('Amount')
            plt.xticks(rotation=45)
            
        elif chart_type == 'line':
            # Time series of spending
            if period == 'monthly':
                df['period'] = df['date'].dt.strftime('%Y-%m')
                period_label = 'Month'
            elif period == 'weekly':
                df['period'] = df['date'].dt.strftime('%Y-W%U')
                period_label = 'Week'
            else:  # daily
                df['period'] = df['date'].dt.strftime('%Y-%m-%d')
                period_label = 'Day'
            
            time_series = df.groupby('period')['amount'].sum()
            time_series.plot(kind='line', marker='o')
            plt.title(f'Expenses Over Time ({period_label})')
            plt.ylabel('Amount')
            plt.xticks(rotation=45)
        
        elif chart_type == 'prediction':
            if not self.predictor.trained:
                print("ML model not trained yet. Add more expenses first.")
                plt.close()
                return
                
            # Get top 5 categories
            top_categories = df.groupby('category')['amount'].sum().nlargest(5).index.tolist()
            
            # Create date range for next 30 days
            today = datetime.datetime.now()
            dates = [(today + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30)]
            
            # Create a sample expense name
            sample_expense = "Predicted expense"
            
            # Make predictions for each category
            predictions = {}
            for category in top_categories:
                predictions[category] = [
                    self.predictor.predict_expense(category, sample_expense, date) 
                    for date in dates
                ]
            
            # Plot predictions
            plt.figure(figsize=(12, 8))
            for category, amounts in predictions.items():
                plt.plot(range(len(dates)), amounts, marker='o', label=category)
            
            plt.title('30-Day Expense Predictions by Category')
            plt.xlabel('Days from today')
            plt.ylabel('Predicted Amount ($)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def predict_expense_amount(self, category, name, date=None):
        if not self.predictor.trained:
            print("ML model not trained yet. Add more expenses first.")
            return None
        
        predicted_amount = self.predictor.predict_expense(category, name, date)
        return predicted_amount
    
    def save_model(self):
        if not self.trained or self.model is None:
            print("No trained model to save.")
            return False
        
        try:
            import pickle
            model_data = {
                'model': self.model,
                'encoder': self.encoder
            }
            
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(model_data, f)
            
            print("Model saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

def main():
    manager = ExpenseManager()
    
    while True:
        print("\n===== Expense Tracker with ML =====")
        print("1. Add Expense")
        print("4. View Weekly Expenses")
        print("5. View Monthly Expenses")
        print("6. View Yearly Expenses")
        print("7. View All Expenses")
        print("8. View Expense Summary")
        print("9. Visualize Expenses")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == "1":
            name = input("Enter expense name: ")
            category = input("Enter category: ")
            
            # Try to predict the amount
            if manager.predictor.trained:
                predicted = manager.predict_expense_amount(category, name)
                if predicted is not None:
                    print(f"Predicted amount based on your history: ${predicted:.2f}")
                    use_prediction = input("Use this prediction? (y/n): ").lower()
                    if use_prediction == 'y':
                        amount = predicted
                    else:
                        amount = input("Enter amount: ")
            else:
                amount = input("Enter amount: ")
    
            custom_date = input("Is this for a past expense? (y/n): ").lower()
    
            try:
                if custom_date == 'y':
                    date_input = input("Enter date (YYYY-MM-DD): ")
                    date = datetime.datetime.strptime(date_input, "%Y-%m-%d").strftime("%Y-%m-%d")
                else:
                    date = None 
                
                if isinstance(amount, str):
                    amount = float(amount)
                    
                manager.add_expense(name, amount, category, date)
            except ValueError as e:
                 print(f"Error: {e}")
        
        elif choice == "4":  # Weekly
            year = int(input("Enter year (YYYY): "))
            week = int(input("Enter week number (1-53): "))
            
            expenses, total = manager.get_expenses_by_time_period('weekly', (year, week))
            
            print(f"\n--- Week {week}, {year} Expenses ---")
            for exp in expenses:
                print(f"{exp.date}: {exp.name} - ${exp.amount:.2f} ({exp.category})")
            
            print(f"\nTotal for Week {week}: ${total:.2f}")
        
        elif choice == "5":  # Monthly
            year = int(input("Enter year (YYYY): "))
            month = int(input("Enter month (1-12): "))
            
            expenses, total = manager.get_expenses_by_time_period('monthly', (year, month))
            
            month_name = calendar.month_name[month]
            print(f"\n--- {month_name} {year} Expenses ---")
            for exp in expenses:
                print(f"{exp.date}: {exp.name} - ${exp.amount:.2f} ({exp.category})")
            
            print(f"\nTotal for {month_name}: ${total:.2f}")
        
        elif choice == "6":  # Yearly
            year = int(input("Enter year (YYYY): "))
            
            expenses, total = manager.get_expenses_by_time_period('yearly', year)
            
            print(f"\n--- {year} Expenses ---")
            for exp in expenses:
                print(f"{exp.date}: {exp.name} - ${exp.amount:.2f} ({exp.category})")
            
            print(f"\nTotal for {year}: ${total:.2f}")
        
        elif choice == "7":  # All Expenses
            print("\n--- All Expenses ---")
            for exp in manager.expenses:
                print(f"{exp.date}: {exp.name} - ${exp.amount:.2f} ({exp.category})")
            
            total = sum(exp.amount for exp in manager.expenses)
            print(f"\nTotal expenses: ${total:.2f}")
        
        elif choice == "8":  # Expense Summary
            summary = manager.get_expense_summary()
            
            if isinstance(summary, str):
                print(summary)
            else:
                print("\n--- Expense Summary ---")
                print(f"Total spent: ${summary['total_spent']:.2f}")
                print(f"Current month expenses: ${summary['month_expenses']:.2f}")
                print(f"Current year expenses: ${summary['year_expenses']:.2f}")
                print(f"ML model status: {summary['model_status']}")
                
                print("\nCategory breakdown:")
                for category, amount in sorted(summary['category_totals'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {category}: ${amount:.2f}")
        
        elif choice == "9":  # Visualize
            print("Chart types:")
            print("1. Pie chart (category breakdown)")
            print("2. Bar chart (category comparison)")
            print("3. Line chart (spending over time)")
            print("4. ML Predictions (next 30 days)")
            
            viz_choice = input("Select chart type (1-4): ")
            
            if viz_choice == "1":
                manager.visualize_expenses('pie')
            elif viz_choice == "2":
                manager.visualize_expenses('bar')
            elif viz_choice == "3":
                period = input("Select time period (daily/weekly/monthly): ").lower()
                manager.visualize_expenses('line', period)
            elif viz_choice == "4":
                manager.visualize_expenses('prediction')
            else:
                print("Invalid visualization choice.")
        
        elif choice == "0":
            print("Exiting Expense Tracker. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()