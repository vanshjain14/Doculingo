import sqlite3

DB_PATH = "feedback.db"  

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedback ORDER BY timestamp DESC")
    rows = cursor.fetchall()

for row in rows:
    print(f"ID: {row[0]}, Question: {row[1]}, Answer: {row[2]}, Rating: {row[3]}, Timestamp: {row[4]}")
