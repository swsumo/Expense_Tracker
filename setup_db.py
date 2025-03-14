import sqlite3

# Connect to SQLite database (or create if it doesn't exist)
conn = sqlite3.connect("finance.db")
cursor = conn.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    amount REAL,
    category TEXT,
    date TEXT,
    time TEXT
);
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS budgets (
    user_id INTEGER,
    category TEXT,
    budget_amount REAL,
    PRIMARY KEY (user_id, category)
);
""")

# sample data
cursor.executemany("""
INSERT OR IGNORE INTO budgets (user_id, category, budget_amount)
VALUES (?, ?, ?);
""", [
    (101, "food", 500),
    (101, "transportation", 300),
    (101, "entertainment", 200)
])

conn.commit()
conn.close()
print("Database setup complete!")
