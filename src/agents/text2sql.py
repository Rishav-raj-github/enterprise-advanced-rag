import sqlite3
import pandas as pd
from src.config import get_llm, SQL_DB_PATH

class Text2SQLAgent:
    """
    Structured DB Agent that converts natural language queries to valid SQLite SQL,
    executes them, and returns structured data results.
    """
    def __init__(self, db_path: str = SQL_DB_PATH):
        self.db_path = db_path
        self.llm = get_llm()

    def get_schema_info(self) -> str:
        """
        Connects to the SQLite instance and extracts full schema definitions
        to ground the SQL generator LLM.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fetch table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]
            
            schema_details = []
            for table in tables:
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
                create_sql = cursor.fetchone()[0]
                
                # Fetch 2 sample rows to showcase data representation
                cursor.execute(f"SELECT * FROM {table} LIMIT 2;")
                sample_rows = cursor.fetchall()
                
                schema_details.append(f"Table: {table}\nDDL: {create_sql}\nSample Rows: {sample_rows}\n")
                
            conn.close()
            return "\n".join(schema_details)
        except Exception as e:
            return f"Error retrieving schema metadata: {e}"

    def generate_sql(self, query: str) -> str:
        """
        Leverages the LLM to translate natural language into a functional SQLite query.
        """
        schemas = self.get_schema_info()
        
        prompt = f"""You are an elite SQL database administrator and systems engineer.
Your task is to write a single, syntactically correct SQLite query to answer the user's natural language request.
Use the database schemas provided below.

SQLite Database Schemas:
{schemas}

Instructions:
1. Write ONLY a valid SELECT SQL query. Do not include any explanations, formatting ticks like ```sql, or surrounding paragraphs.
2. Rely only on tables and fields present in the schema.
3. If aggregate columns are queried, use meaningful column aliases (e.g. SELECT SUM(total_amount) AS total_revenue).

User Query: "{query}"

SQLite SQL Query:"""
        try:
            sql = self.llm.generate_content(prompt).strip()
            
            # Clean Markdown code fence block wrapping if present
            sql = re.sub(r"```(sql)?", "", sql).strip()
            # Clean leading/trailing spaces or ending semicolons if double printed
            sql = sql.split(";")[0].strip() + ";"
            return sql
        except Exception as e:
            print(f"[Text2SQL Error] Generation failed: {e}")
            return "SELECT * FROM sqlite_master;"

    def execute_query(self, sql: str) -> dict:
        """
        Runs the generated query and outputs results packaged as records.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            # Use pandas for fast structured record ingestion
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            records = df.to_dict(orient="records")
            markdown_table = df.to_markdown(index=False) if not df.empty else "No results found."
            
            return {
                "success": True,
                "sql": sql,
                "records": records,
                "markdown": markdown_table,
                "columns": list(df.columns)
            }
        except Exception as e:
            return {
                "success": False,
                "sql": sql,
                "error": str(e),
                "markdown": f"Execution failed: {e}"
            }

    def process(self, query: str) -> dict:
        """
        Executes complete flow: Generates SQL, runs query, and formats.
        """
        sql = self.generate_sql(query)
        result = self.execute_query(sql)
        return result
import re # Explicit import for internal re usage inside class
