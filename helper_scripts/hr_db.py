import pandas as pd, sqlite3, os

DB_PATH = os.path.abspath("hr_dataset.db")
CSV_PATH = os.path.abspath("data/employee_attrition.csv")
TABLE = "employees"

def create_hr_db():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE, conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Created {DB_PATH} with table '{TABLE}' ({len(df)} rows).")

if __name__ == "__main__":
    create_hr_db()
