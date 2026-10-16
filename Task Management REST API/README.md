# Task Manager REST API

A beginner-friendly REST API for managing tasks, built with Python 3, Flask, and SQLite.
This project is intentionally simple — it is designed to be understood, run, and defended in a software engineering interview.

---

## Features

- Create, read, update, and delete tasks (full CRUD)
- JSON request and response format
- Input validation with clear error messages
- Parameterized SQL queries (SQL injection prevention)
- Automatic database creation on first run
- Isolated unit tests with a temporary database

---

## Technology Stack

| Layer       | Technology              | Why?                                        |
|-------------|-------------------------|---------------------------------------------|
| Language    | Python 3                | Readable, widely used, beginner-friendly    |
| Web framework | Flask               | Lightweight, minimal boilerplate            |
| Database    | SQLite                  | Zero-config, file-based, built into Python  |
| DB driver   | Built-in `sqlite3`      | No extra installation needed                |
| Testing     | Built-in `unittest`     | No extra packages required                  |

---

## Folder Structure

```
Task Management REST API/
├── task_manager_api/
│   ├── app.py          ← Flask routes and HTTP logic
│   ├── database.py     ← All SQLite database functions
│   ├── requirements.txt
│   └── tests/
│       └── test_api.py ← All unit/integration tests
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip` (included with Python)

### Steps

```bash
# 1. Clone the repository (or navigate to the project folder)
cd "Task Management REST API/task_manager_api"

# 2. Create a virtual environment (keeps your system Python clean)
python -m venv venv

# 3. Activate the virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
# Make sure you are in task_manager_api/ with venv active
python app.py
```

You should see:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

The database file `tasks.db` is created automatically on the first run.
The server restarts automatically when you save a file (debug mode).

---

## API Endpoints

### GET /
Returns a message confirming the API is running.

**Example response:**
```json
{"message": "Task Manager API is running!"}
```

---

### GET /tasks
Returns a list of all tasks (empty array if none exist).

**Example response:**
```json
[
  {
    "id": 1,
    "title": "Buy groceries",
    "description": "Milk and eggs",
    "status": "pending",
    "created_at": "2024-01-15 10:30:00"
  }
]
```

---

### GET /tasks/\<id\>
Returns a single task by ID.

**Success (200):**
```json
{"id": 1, "title": "Buy groceries", "description": "Milk", "status": "pending", "created_at": "..."}
```

**Not found (404):**
```json
{"error": "Task not found"}
```

---

### POST /tasks
Creates a new task.

**Request body:**
```json
{
  "title": "Buy groceries",
  "description": "Milk and eggs"
}
```

- `title` is **required** and must not be empty.
- `description` is **optional**.
- `status` defaults to `pending`.

**Success (201):**
```json
{"id": 1, "title": "Buy groceries", "description": "Milk and eggs", "status": "pending", "created_at": "..."}
```

**Error (400):**
```json
{"error": "Title is required and cannot be empty"}
```

---

### PUT /tasks/\<id\>
Updates an existing task. You may send any combination of fields.

**Request body:**
```json
{
  "title": "Buy groceries and bread",
  "status": "completed"
}
```

- Valid values for `status`: `pending`, `completed`

**Success (200):** Returns the updated task object.

**Error (400):**
```json
{"error": "Invalid status 'done'. Must be one of: ['completed', 'pending']"}
```

---

### DELETE /tasks/\<id\>
Deletes a task permanently.

**Success (200):**
```json
{"message": "Task 1 deleted successfully"}
```

**Not found (404):**
```json
{"error": "Task not found"}
```

---

## HTTP Status Code Summary

| Code | Meaning              | When used                          |
|------|----------------------|------------------------------------|
| 200  | OK                   | Successful GET, PUT, DELETE        |
| 201  | Created              | Successful POST (new task created) |
| 400  | Bad Request          | Invalid input or malformed JSON    |
| 404  | Not Found            | Task ID does not exist             |
| 405  | Method Not Allowed   | Wrong HTTP method for a route      |
| 500  | Internal Server Error| Unexpected server-side failure     |

---

## Running Tests

```bash
# From inside task_manager_api/
python -m unittest discover tests -v
```

Tests use a temporary database file — your real `tasks.db` is never affected.

Expected output:

```
test_01_health_check ... ok
test_02_get_tasks_empty ... ok
test_03_create_task_valid ... ok
test_04_create_task_empty_title ... ok
test_04b_create_task_missing_title ... ok
test_05_get_existing_task ... ok
test_06_get_nonexistent_task ... ok
test_07_update_task ... ok
test_08_update_invalid_status ... ok
test_09_delete_task ... ok
test_10_deleted_task_returns_404 ... ok
test_11_malformed_json ... ok
----------------------------------------------------------------------
Ran 12 tests in X.XXXs
OK
```

---

## Manual Testing with curl

```bash
# Health check
curl http://127.0.0.1:5000/

# Get all tasks
curl http://127.0.0.1:5000/tasks

# Create a task
curl -X POST http://127.0.0.1:5000/tasks \
     -H "Content-Type: application/json" \
     -d '{"title": "Learn Flask", "description": "Build a REST API"}'

# Get one task
curl http://127.0.0.1:5000/tasks/1

# Update a task
curl -X PUT http://127.0.0.1:5000/tasks/1 \
     -H "Content-Type: application/json" \
     -d '{"status": "completed"}'

# Delete a task
curl -X DELETE http://127.0.0.1:5000/tasks/1
```

On Windows PowerShell, replace single quotes with double quotes and escape inner quotes:

```powershell
curl -X POST http://127.0.0.1:5000/tasks `
     -H "Content-Type: application/json" `
     -d '{"title": "Learn Flask"}'
```

Or use a GUI tool like **Postman** or **Insomnia** (free to download).

---

## Git Setup

```bash
# From the project root (Task Management REST API/)
git init
git add .
git commit -m "Initial commit: Task Manager REST API"

# Push to GitHub (create repo first at github.com)
git remote add origin https://github.com/YOUR_USERNAME/task-manager-api.git
git branch -M main
git push -u origin main
```

---

## Known Limitations

1. **No authentication** — anyone with the URL can read, create, or delete tasks.
2. **SQLite concurrency** — SQLite locks the file on writes; not suitable for multiple simultaneous users.
3. **No pagination** — GET /tasks returns all rows, which could be slow with thousands of tasks.
4. **In-memory only** — the database file lives on the server; no backup or replication.
5. **No HTTPS** — the dev server uses plain HTTP; production would need TLS.

---

## Possible Future Improvements

1. Add user accounts and JWT authentication.
2. Replace SQLite with PostgreSQL for production concurrency.
3. Add pagination (e.g., `GET /tasks?page=1&limit=20`).
4. Add task due dates and priority levels.
5. Deploy to a cloud platform (Render, Railway, Fly.io — all have free tiers).
