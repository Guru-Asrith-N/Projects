"""
database.py — All database-related code lives here.

Why SQLite?
-----------
SQLite stores the entire database in a single file (tasks.db).
It requires NO server process, NO credentials, and NO installation.
This makes it ideal for learning and small projects.

Limitation: SQLite is NOT suitable for production apps with many
concurrent writers, as it locks the entire file on writes.

Why a separate database.py?
-----------------------------
Separation of concerns: app.py handles HTTP logic,
database.py handles data storage logic.
This keeps each file focused and easy to understand.
"""

import sqlite3
import os

# The database file will be created in the same folder as database.py
# You can override this path for testing (see test_api.py)
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "tasks.db")


def get_connection(db_path=None):
    """
    Opens and returns a connection to the SQLite database.

    sqlite3.Row makes rows behave like Python dictionaries,
    so we can access columns by name: row["title"] instead of row[1].

    We pass db_path so tests can point to a different (temporary) database.
    """
    path = db_path or DATABASE_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def init_db(db_path=None):
    """
    Creates the 'tasks' table if it does not already exist.
    Called once when the Flask app starts.

    Column notes:
        id          — SQLite auto-increments INTEGER PRIMARY KEY
        title       — must have a value (NOT NULL)
        description — optional, may be empty/NULL
        status      — either 'pending' or 'completed'; default 'pending'
        created_at  — SQLite fills this in automatically via CURRENT_TIMESTAMP
    """
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                description TEXT,
                status      TEXT    NOT NULL DEFAULT 'pending',
                created_at  TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP)
            )
        """)
        conn.commit()
    finally:
        conn.close()   # always close, even if an error occurs


# ---------------------------------------------------------------------------
# CRUD helpers — each function opens its own connection and closes it.
# ---------------------------------------------------------------------------

def get_all_tasks(db_path=None):
    """Return a list of all tasks, newest first."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC"
        )
        # sqlite3.Row objects → plain Python dicts so Flask can serialize them
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_task_by_id(task_id, db_path=None):
    """
    Return a single task dict, or None if not found.

    IMPORTANT — Parameterized query:
        We write  WHERE id = ?  and pass (task_id,) separately.
        SQLite substitutes the value SAFELY, preventing SQL injection.

        NEVER write:  f"WHERE id = {task_id}"   ← dangerous!
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,)          # always a tuple, even for one value
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_task(title, description, db_path=None):
    """
    Insert a new task and return the auto-generated id.

    We let the database set 'status' and 'created_at' via column defaults.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO tasks (title, description) VALUES (?, ?)",
            (title, description)
        )
        conn.commit()
        return cursor.lastrowid   # the id SQLite assigned to the new row
    finally:
        conn.close()


def update_task(task_id, title, description, status, db_path=None):
    """
    Update an existing task's fields.
    Returns True if a row was actually changed, False if id not found.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            UPDATE tasks
               SET title       = ?,
                   description = ?,
                   status      = ?
             WHERE id = ?
            """,
            (title, description, status, task_id)
        )
        conn.commit()
        # rowcount == 0 means no row matched the WHERE clause
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_task(task_id, db_path=None):
    """
    Delete a task by id.
    Returns True if deleted, False if the id did not exist.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM tasks WHERE id = ?",
            (task_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
