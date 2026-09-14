# database.py
# This file handles all interactions with the SQLite database.
# SQLite is a simple file-based database — no server setup needed!
# The database is saved as a file called "resume_history.db" in the project folder.

import sqlite3      # Built-in Python library for SQLite databases
import pandas as pd # Used to read database results as a nice table


# Name of our database file
DB_FILE = "resume_history.db"


def get_connection():
    """
    Creates and returns a connection to the SQLite database.
    If the database file doesn't exist, SQLite automatically creates it.
    
    Returns:
        A sqlite3.Connection object
    """
    connection = sqlite3.connect(DB_FILE)
    return connection


def create_table():
    """
    Creates the 'analysis_history' table if it doesn't already exist.
    This function should be called once when the app starts.
    
    Table columns:
    - id: Auto-incremented unique ID for each record
    - resume_filename: Name of the uploaded resume file
    - final_score: The overall match score (0-100)
    - skill_match_percentage: Skill-based match percentage
    - tfidf_similarity: The TF-IDF cosine similarity score (0-1)
    - matching_skills: Comma-separated list of matching skills
    - missing_skills: Comma-separated list of missing skills
    - analyzed_at: Date and time when the analysis was done
    """
    connection = get_connection()
    cursor = connection.cursor()  # A cursor lets us execute SQL commands
    
    # CREATE TABLE IF NOT EXISTS → only creates if it doesn't exist (safe to call multiple times)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_filename      TEXT,
            final_score          REAL,
            skill_match_percentage REAL,
            tfidf_similarity     REAL,
            matching_skills      TEXT,
            missing_skills       TEXT,
            analyzed_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    connection.commit()   # Save the changes
    connection.close()    # Always close the connection when done


def save_analysis(resume_filename, final_score, skill_match_percentage,
                  tfidf_similarity, matching_skills, missing_skills):
    """
    Saves one analysis result to the database.
    
    Parameters:
        resume_filename: Name of the resume file (string)
        final_score: The combined final score (float)
        skill_match_percentage: Skill match % (float)
        tfidf_similarity: TF-IDF similarity score (float)
        matching_skills: Set of matching skills — we convert to comma-separated string
        missing_skills: Set of missing skills — we convert to comma-separated string
    """
    connection = get_connection()
    cursor = connection.cursor()
    
    # Convert skill sets to comma-separated strings for storage
    # Example: {"python", "sql"} → "python, sql"
    matching_str = ", ".join(sorted(matching_skills)) if matching_skills else "None"
    missing_str = ", ".join(sorted(missing_skills)) if missing_skills else "None"
    
    # INSERT command adds a new row to the table
    # The ? placeholders are filled by the tuple of values (prevents SQL injection)
    cursor.execute("""
        INSERT INTO analysis_history 
        (resume_filename, final_score, skill_match_percentage, tfidf_similarity,
         matching_skills, missing_skills)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (resume_filename, final_score, skill_match_percentage,
          tfidf_similarity, matching_str, missing_str))
    
    connection.commit()
    connection.close()


def get_all_history():
    """
    Retrieves all past analysis records from the database.
    
    Returns:
        A pandas DataFrame with all records (easy to display in Streamlit)
        Returns an empty DataFrame if no records exist
    """
    connection = get_connection()
    
    try:
        # pd.read_sql_query runs a SQL query and returns a pandas DataFrame
        # This is a simple SELECT * to get all rows
        df = pd.read_sql_query(
            "SELECT * FROM analysis_history ORDER BY analyzed_at DESC",
            connection
        )
    except Exception as e:
        print(f"Error reading history: {e}")
        df = pd.DataFrame()  # Return empty table on error
    finally:
        connection.close()
    
    return df


def delete_all_history():
    """
    Deletes all records from the analysis_history table.
    Useful for clearing old data during testing.
    """
    connection = get_connection()
    cursor = connection.cursor()
    
    cursor.execute("DELETE FROM analysis_history")
    
    connection.commit()
    connection.close()


def get_total_analyses():
    """
    Returns the total count of analyses stored in the database.
    
    Returns:
        An integer count
    """
    connection = get_connection()
    cursor = connection.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM analysis_history")
    count = cursor.fetchone()[0]  # fetchone() gets the first (and only) result row
    
    connection.close()
    return count
