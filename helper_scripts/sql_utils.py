import sqlite3, pandas as pd, os

DB_PATH = os.path.abspath("hr_dataset.db")

def run_query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()

sample_queries = {
    "Attrition by Department": """
        SELECT Department, COUNT(*) as total, 
               SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) as left_count
        FROM employees
        GROUP BY Department
        ORDER BY left_count DESC;
    """,
    "Attrition by Age Bucket": """
        SELECT 
          CASE 
            WHEN Age < 30 THEN '<30'
            WHEN Age BETWEEN 30 AND 40 THEN '30-40'
            WHEN Age BETWEEN 41 AND 50 THEN '41-50'
            ELSE '50+' END as AgeGroup,
          COUNT(*) as total,
          SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) as left_count
        FROM employees
        GROUP BY AgeGroup;
    """
}
