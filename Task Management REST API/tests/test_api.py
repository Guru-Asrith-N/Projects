"""
tests/test_api.py — Unit tests for the Task Manager API.

Strategy
--------
- We use a TEMPORARY database file for every test run so tests
  never corrupt the real tasks.db.
- We patch database.DATABASE_PATH before importing app so every
  database call inside app.py uses the temp file.
- Flask's built-in test client lets us send fake HTTP requests
  without starting a real server.

Run all tests with:
    python -m pytest tests/ -v          (if pytest is installed)
    python -m unittest discover tests   (built-in, no extra package needed)
"""

import os
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Point the database module at a temporary file BEFORE importing the app.
# This is the standard pattern for isolating tests from production data.
# ---------------------------------------------------------------------------
import database

# Create a temp file once for the whole test session
_TMP_FD, _TMP_PATH = tempfile.mkstemp(suffix=".db")
os.close(_TMP_FD)          # We only need the path, not the open file descriptor
database.DATABASE_PATH = _TMP_PATH   # Monkey-patch before app is imported

import app as flask_app     # noqa: E402  (import after patching is intentional)


class TestTaskManagerAPI(unittest.TestCase):
    """
    Each test method must start with 'test_' for unittest to find it.
    setUp / tearDown run before / after EACH individual test.
    """

    def setUp(self):
        """
        Prepare a clean state before every test:
          1. Initialise (or re-initialise) the temp database.
          2. Create a Flask test client — simulates HTTP without a real server.
          3. Enable TESTING mode so Flask gives us better error messages.
        """
        database.init_db(_TMP_PATH)
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def tearDown(self):
        """
        Wipe all rows after each test so tests are independent.
        We drop and recreate the table rather than DELETE so the
        auto-increment counter resets too — easier to reason about IDs.
        """
        import sqlite3
        conn = sqlite3.connect(_TMP_PATH)
        conn.execute("DROP TABLE IF EXISTS tasks")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _create_task(self, title="Test task", description="A description"):
        """Convenience method: POST a task and return the response."""
        return self.client.post(
            "/tasks",
            json={"title": title, "description": description},
        )

    # ------------------------------------------------------------------
    # 1. Health check
    # ------------------------------------------------------------------

    def test_01_health_check(self):
        """GET / should return 200 and a running message."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("message", data)
        self.assertIn("running", data["message"].lower())

    # ------------------------------------------------------------------
    # 2. Empty task list
    # ------------------------------------------------------------------

    def test_02_get_tasks_empty(self):
        """GET /tasks on a fresh database should return 200 and empty list."""
        response = self.client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    # ------------------------------------------------------------------
    # 3. Create a valid task
    # ------------------------------------------------------------------

    def test_03_create_task_valid(self):
        """POST /tasks with valid data should return 201 and the new task."""
        response = self._create_task(title="Buy groceries", description="Milk and eggs")
        self.assertEqual(response.status_code, 201)
        data = response.get_json()

        # Check all expected fields are present
        self.assertIn("id", data)
        self.assertIn("title", data)
        self.assertIn("description", data)
        self.assertIn("status", data)
        self.assertIn("created_at", data)

        # Check field values
        self.assertEqual(data["title"], "Buy groceries")
        self.assertEqual(data["description"], "Milk and eggs")
        self.assertEqual(data["status"], "pending")   # default status

    # ------------------------------------------------------------------
    # 4. Reject empty title
    # ------------------------------------------------------------------

    def test_04_create_task_empty_title(self):
        """POST /tasks with an empty title should return 400 Bad Request."""
        response = self.client.post("/tasks", json={"title": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_04b_create_task_missing_title(self):
        """POST /tasks with no title key should also return 400."""
        response = self.client.post("/tasks", json={"description": "no title"})
        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------------
    # 5. Retrieve an existing task
    # ------------------------------------------------------------------

    def test_05_get_existing_task(self):
        """GET /tasks/<id> for an existing task should return 200 and task data."""
        create_resp = self._create_task(title="Walk the dog")
        task_id = create_resp.get_json()["id"]

        get_resp = self.client.get(f"/tasks/{task_id}")
        self.assertEqual(get_resp.status_code, 200)
        data = get_resp.get_json()
        self.assertEqual(data["id"], task_id)
        self.assertEqual(data["title"], "Walk the dog")

    # ------------------------------------------------------------------
    # 6. 404 for nonexistent task
    # ------------------------------------------------------------------

    def test_06_get_nonexistent_task(self):
        """GET /tasks/9999 when no such task exists should return 404."""
        response = self.client.get("/tasks/9999")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("error", data)

    # ------------------------------------------------------------------
    # 7. Update a task
    # ------------------------------------------------------------------

    def test_07_update_task(self):
        """PUT /tasks/<id> with valid data should return 200 and updated task."""
        task_id = self._create_task(title="Read book").get_json()["id"]

        response = self.client.put(
            f"/tasks/{task_id}",
            json={"title": "Read two books", "status": "completed"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["title"], "Read two books")
        self.assertEqual(data["status"], "completed")

    # ------------------------------------------------------------------
    # 8. Reject invalid status
    # ------------------------------------------------------------------

    def test_08_update_invalid_status(self):
        """PUT /tasks/<id> with an invalid status should return 400."""
        task_id = self._create_task().get_json()["id"]
        response = self.client.put(
            f"/tasks/{task_id}",
            json={"status": "in-progress"},   # not a valid status
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    # ------------------------------------------------------------------
    # 9. Delete a task
    # ------------------------------------------------------------------

    def test_09_delete_task(self):
        """DELETE /tasks/<id> should return 200 and a confirmation message."""
        task_id = self._create_task(title="Clean room").get_json()["id"]

        response = self.client.delete(f"/tasks/{task_id}")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("message", data)

    # ------------------------------------------------------------------
    # 10. Deleted task cannot be retrieved
    # ------------------------------------------------------------------

    def test_10_deleted_task_returns_404(self):
        """After deletion, GET /tasks/<id> should return 404."""
        task_id = self._create_task(title="Temp task").get_json()["id"]
        self.client.delete(f"/tasks/{task_id}")  # delete it

        response = self.client.get(f"/tasks/{task_id}")
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # Bonus: malformed JSON body
    # ------------------------------------------------------------------

    def test_11_malformed_json(self):
        """POST /tasks with a non-JSON body should return 400."""
        response = self.client.post(
            "/tasks",
            data="this is not json",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
