"""
app.py — The Flask application entry point.

Flask is a micro web framework.  Think of it as a post office:
  - Clients send letters (HTTP requests) to specific addresses (URL routes).
  - Flask reads the address, finds the matching route handler, runs it,
    and sends back a reply (HTTP response).

Running this file starts a local web server on http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify
import database

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Initialize the database (create table if needed) as soon as the app loads.
# Subsequent starts are safe because we use CREATE TABLE IF NOT EXISTS.
database.init_db()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

VALID_STATUSES = {"pending", "completed"}


def task_not_found():
    """Return a consistent 404 JSON response."""
    return jsonify({"error": "Task not found"}), 404


def bad_request(message):
    """Return a consistent 400 JSON response with a caller-supplied message."""
    return jsonify({"error": message}), 400


# ---------------------------------------------------------------------------
# Routes / Endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health_check():
    """
    Endpoint: GET /
    Purpose : Confirm the API is running.

    How it works:
        Flask calls this function when a GET request arrives at "/".
        jsonify() converts a Python dict to a JSON HTTP response.
        Flask defaults to HTTP 200 OK when no status code is supplied.

    Why 200? — The request succeeded; the server is alive and responding.
    """
    return jsonify({"message": "Task Manager API is running!"}), 200


@app.route("/tasks", methods=["GET"])
def get_tasks():
    """
    Endpoint: GET /tasks
    Purpose : Return ALL tasks in the database.

    How it works:
        1. Calls database.get_all_tasks() which runs SELECT * FROM tasks.
        2. The list of dicts is serialized to a JSON array by jsonify().
        3. An empty database returns [] — still a 200, not a 404.

    Why 200? — The request was valid; returning an empty list is fine.
    """
    tasks = database.get_all_tasks()
    return jsonify(tasks), 200


@app.route("/tasks/<int:task_id>", methods=["GET"])
def get_task(task_id):
    """
    Endpoint: GET /tasks/<id>
    Purpose : Return one task by its integer ID.

    How it works:
        Flask extracts <int:task_id> from the URL and passes it as a parameter.
        database.get_task_by_id() returns None if no row matches.
        We return 404 if None.

    Why 404? — The resource (task) does not exist at the requested URL.
    """
    task = database.get_task_by_id(task_id)
    if task is None:
        return task_not_found()
    return jsonify(task), 200


@app.route("/tasks", methods=["POST"])
def create_task():
    """
    Endpoint: POST /tasks
    Purpose : Create a new task.

    Expected JSON body:
        {
            "title":       "Buy groceries",   ← required, non-empty
            "description": "Milk and eggs"    ← optional
        }

    How it works:
        1. request.get_json() parses the body. silent=True returns None
           instead of raising an exception when the body is not valid JSON.
        2. We validate the title (required, non-empty).
        3. database.create_task() inserts the row and returns its new id.
        4. We fetch and return the full new task so the client sees the
           server-generated id and created_at timestamp.

    Why 201? — A new resource was created. 200 means "OK, nothing new."
    """
    data = request.get_json(silent=True)

    # Guard: no body or invalid JSON
    if not data:
        return bad_request("Request body must be valid JSON")

    # Guard: title is required and must not be blank
    title = data.get("title", "").strip()
    if not title:
        return bad_request("Title is required and cannot be empty")

    description = data.get("description", "")

    new_id = database.create_task(title, description)
    new_task = database.get_task_by_id(new_id)
    return jsonify(new_task), 201


@app.route("/tasks/<int:task_id>", methods=["PUT"])
def update_task(task_id):
    """
    Endpoint: PUT /tasks/<id>
    Purpose : Replace all editable fields of an existing task.

    Expected JSON body (all fields required for a full update):
        {
            "title":       "Buy groceries",
            "description": "Also bread",
            "status":      "completed"
        }

    How it works:
        1. Confirm the task exists first (return 404 if not).
        2. Parse and validate the JSON body.
        3. Validate that status is one of the allowed values.
        4. database.update_task() runs the SQL UPDATE.
        5. Return the updated task.

    Why 200? — The resource was updated successfully; returning the new state.
    """
    # First check the task exists
    existing = database.get_task_by_id(task_id)
    if existing is None:
        return task_not_found()

    data = request.get_json(silent=True)
    if not data:
        return bad_request("Request body must be valid JSON")

    # Use existing values as defaults so partial updates work too
    title = data.get("title", existing["title"])
    if isinstance(title, str):
        title = title.strip()
    if not title:
        return bad_request("Title is required and cannot be empty")

    description = data.get("description", existing["description"])
    status      = data.get("status",      existing["status"])

    if status not in VALID_STATUSES:
        return bad_request(
            f"Invalid status '{status}'. Must be one of: {sorted(VALID_STATUSES)}"
        )

    database.update_task(task_id, title, description, status)
    updated_task = database.get_task_by_id(task_id)
    return jsonify(updated_task), 200


@app.route("/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    """
    Endpoint: DELETE /tasks/<id>
    Purpose : Permanently remove a task.

    How it works:
        1. database.delete_task() runs DELETE FROM tasks WHERE id = ?
        2. Returns False when no row was deleted (task did not exist).
        3. Returns a confirmation message on success.

    Why 200 (not 204)? — We return a JSON body, so the client gets
    confirmation text. HTTP 204 means "no body", which is also valid.
    """
    deleted = database.delete_task(task_id)
    if not deleted:
        return task_not_found()
    return jsonify({"message": f"Task {task_id} deleted successfully"}), 200


# ---------------------------------------------------------------------------
# Error handlers — called automatically by Flask
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found_handler(e):
    """Handles requests to URLs that Flask has no route for."""
    return jsonify({"error": "The requested URL was not found on this server"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handles wrong HTTP method on a valid URL (e.g. DELETE /tasks)."""
    return jsonify({"error": "Method not allowed on this endpoint"}), 405


@app.errorhandler(500)
def internal_error(e):
    """Catch-all for unexpected server-side errors. Never expose internals."""
    return jsonify({"error": "An unexpected server error occurred"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # debug=True: Flask auto-reloads when you save a file and shows tracebacks
    # in the browser.  NEVER use debug=True in production.
    app.run(debug=True)
