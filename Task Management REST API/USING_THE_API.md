# How to Use the Task Manager API

This file explains every way you can interact with your API.
You do not need to use all of them — pick the one that feels most comfortable.

---

## What You Are Actually Doing

Your API is a running program that listens for requests on your computer.
Think of it like a waiter at a restaurant:

- **You** are the customer (the client)
- **The API** is the waiter (the server)
- **The database** is the kitchen

You place an order (send a request) → the waiter takes it to the kitchen → the kitchen prepares it → the waiter brings back your food (the response).

---

## The Base URL

Every request starts with:
```
http://127.0.0.1:5000
```
- `127.0.0.1` means "this computer" (also called `localhost`)
- `5000` is the port number (like a door number on the computer)
- This only works while `python app.py` is running

---

## Available Actions (Endpoints)

| What you want to do | Method | URL |
|---|---|---|
| Check if API is running | GET | `http://127.0.0.1:5000/` |
| See all tasks | GET | `http://127.0.0.1:5000/tasks` |
| See one task | GET | `http://127.0.0.1:5000/tasks/1` |
| Create a new task | POST | `http://127.0.0.1:5000/tasks` |
| Update a task | PUT | `http://127.0.0.1:5000/tasks/1` |
| Delete a task | DELETE | `http://127.0.0.1:5000/tasks/1` |

> Replace `/1` with the actual ID of your task.

---

---

# METHOD 1 — Web Browser (simplest, but limited)

The browser can only do GET requests (reading data).
You cannot create, update, or delete tasks from the browser alone.

### What you can do in the browser

**Check if the API is alive:**
Open your browser and go to:
```
http://127.0.0.1:5000
```
Expected response:
```json
{"message": "Task Manager API is running!"}
```

---

**See all tasks:**
```
http://127.0.0.1:5000/tasks
```
Expected response (empty at first):
```json
[]
```
After creating some tasks:
```json
[
  {
    "id": 1,
    "title": "Study for interview",
    "description": "Review REST API concepts",
    "status": "pending",
    "created_at": "2024-01-15 10:30:00"
  }
]
```

---

**See one specific task:**
```
http://127.0.0.1:5000/tasks/1
```
Replace `1` with the ID of the task you want to view.

Expected response:
```json
{
  "id": 1,
  "title": "Study for interview",
  "description": "Review REST API concepts",
  "status": "pending",
  "created_at": "2024-01-15 10:30:00"
}
```

If the task does not exist:
```json
{"error": "Task not found"}
```

---

**Limitation:** For creating, updating, and deleting tasks, use Method 2 or 3 below.

---

---

# METHOD 2 — Postman (recommended, beginner-friendly)

Postman is a free desktop app that lets you send any type of HTTP request
with a visual interface. No code required.

---

## Installing Postman

### Step 1 — Download Postman
Go to:
```
https://www.postman.com/downloads/
```
Click **"Download the App"** — choose the Windows version.

### Step 2 — Install it
Run the downloaded `.exe` file and follow the installer.
Postman will open automatically when done.

### Step 3 — Create a free account (optional)
Postman will ask you to sign in or create an account.
You can click **"Skip and go to the app"** at the bottom — no account required.

---

## Using Postman

### The Postman interface has these parts:
- **Method dropdown** (top left) — choose GET, POST, PUT, DELETE
- **URL bar** — where you type the address
- **Body tab** — where you type the JSON data you want to send
- **Send button** — sends the request
- **Response area** (bottom) — shows what the API sent back

---

### Action 1 — Check if the API is running

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://127.0.0.1:5000/` |
| Body | (none) |

Click **Send**.

Expected response (bottom panel):
```json
{"message": "Task Manager API is running!"}
```
Status shown: `200 OK`

---

### Action 2 — See all tasks

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://127.0.0.1:5000/tasks` |
| Body | (none) |

Click **Send**.

---

### Action 3 — Create a new task

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://127.0.0.1:5000/tasks` |

**Body setup:**
1. Click the **Body** tab (below the URL bar)
2. Select **raw**
3. Change the dropdown on the right from `Text` to `JSON`
4. Type this in the text area:

```json
{
    "title": "Study for interview",
    "description": "Review REST API concepts"
}
```

Click **Send**.

Expected response:
```json
{
    "id": 1,
    "title": "Study for interview",
    "description": "Review REST API concepts",
    "status": "pending",
    "created_at": "2024-01-15 10:30:00"
}
```
Status shown: `201 Created`

> 📌 Note the `"id"` value — you will need it for the next actions.

---

### Action 4 — View one specific task

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://127.0.0.1:5000/tasks/1` |
| Body | (none) |

Replace `1` with your task's ID. Click **Send**.

---

### Action 5 — Update a task

| Field | Value |
|---|---|
| Method | `PUT` |
| URL | `http://127.0.0.1:5000/tasks/1` |

**Body (raw → JSON):**
```json
{
    "status": "completed"
}
```

> You only need to include the fields you want to change.
> Title and description will stay the same automatically.

You can also update the title:
```json
{
    "title": "Study harder for interview",
    "status": "completed"
}
```

Click **Send**. Expected response: the updated task with status `200 OK`.

---

### Action 6 — Delete a task

| Field | Value |
|---|---|
| Method | `DELETE` |
| URL | `http://127.0.0.1:5000/tasks/1` |
| Body | (none needed) |

Click **Send**.

Expected response:
```json
{"message": "Task 1 deleted successfully"}
```
Status shown: `200 OK`

---

### What the status codes mean in Postman

Postman shows the status code in the top-right of the response panel.

| Status | Color in Postman | Meaning |
|---|---|---|
| `200 OK` | Green | Request succeeded |
| `201 Created` | Green | New task was created |
| `400 Bad Request` | Orange/Red | You sent invalid data |
| `404 Not Found` | Orange/Red | That task ID does not exist |

---

---

# METHOD 3 — PowerShell (built into Windows, no install needed)

Use this if you want to test from the terminal without installing anything.
Make sure the server (`python app.py`) is running in a separate terminal first.

---

### Action 1 — Check if the API is running

```powershell
Invoke-RestMethod http://127.0.0.1:5000/
```

---

### Action 2 — See all tasks

```powershell
Invoke-RestMethod http://127.0.0.1:5000/tasks
```

---

### Action 3 — Create a new task

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:5000/tasks `
  -ContentType "application/json" `
  -Body '{"title": "Study for interview", "description": "Review REST API"}'
```

> Note the backtick `` ` `` at the end of each line — this continues the command on the next line.

---

### Action 4 — View one specific task

```powershell
Invoke-RestMethod http://127.0.0.1:5000/tasks/1
```

---

### Action 5 — Update a task

```powershell
Invoke-RestMethod -Method PUT http://127.0.0.1:5000/tasks/1 `
  -ContentType "application/json" `
  -Body '{"status": "completed"}'
```

---

### Action 6 — Delete a task

```powershell
Invoke-RestMethod -Method DELETE http://127.0.0.1:5000/tasks/1
```

---

---

# METHOD 4 — curl (command line tool)

curl is a command-line tool that sends HTTP requests.
It may already be installed on your machine.

**Check if curl is installed:**
```powershell
curl --version
```
If you see a version number, it is installed. If not, download it from:
```
https://curl.se/windows/
```

---

### Action 1 — Check if the API is running

```bash
curl http://127.0.0.1:5000/
```

---

### Action 2 — See all tasks

```bash
curl http://127.0.0.1:5000/tasks
```

---

### Action 3 — Create a new task

```bash
curl -X POST http://127.0.0.1:5000/tasks ^
     -H "Content-Type: application/json" ^
     -d "{\"title\": \"Study for interview\", \"description\": \"Review REST API\"}"
```

> On Windows Command Prompt, use `^` to continue on the next line and `\"` for quotes inside the JSON.

---

### Action 4 — View one specific task

```bash
curl http://127.0.0.1:5000/tasks/1
```

---

### Action 5 — Update a task

```bash
curl -X PUT http://127.0.0.1:5000/tasks/1 ^
     -H "Content-Type: application/json" ^
     -d "{\"status\": \"completed\"}"
```

---

### Action 6 — Delete a task

```bash
curl -X DELETE http://127.0.0.1:5000/tasks/1
```

---

---

# Validation Rules — What the API Accepts and Rejects

### Creating a task (POST /tasks)

| Field | Required? | Rules |
|---|---|---|
| `title` | ✅ Yes | Must not be empty or just spaces |
| `description` | ❌ No | Any text, or leave it out entirely |
| `status` | ❌ No | Cannot set on creation — always starts as `pending` |

**Rejected examples:**
```json
{}
```
```json
{"title": ""}
```
```json
{"title": "   "}
```
All return: `{"error": "Title is required and cannot be empty"}` with status `400`

---

### Updating a task (PUT /tasks/id)

| Field | Required? | Rules |
|---|---|---|
| `title` | ❌ No | If included, must not be empty |
| `description` | ❌ No | Any text |
| `status` | ❌ No | Must be exactly `"pending"` or `"completed"` |

**Rejected example:**
```json
{"status": "in-progress"}
```
Returns: `{"error": "Invalid status 'in-progress'. Must be one of: ['completed', 'pending']"}` with status `400`

---

---

# Full Workflow — Try This in Order

This is the complete lifecycle of one task.

```
Step 1:  Start the server         →  python app.py
Step 2:  Check it is alive        →  GET  /
Step 3:  See empty task list      →  GET  /tasks          (returns [])
Step 4:  Create a task            →  POST /tasks          (returns id: 1)
Step 5:  See the task             →  GET  /tasks/1
Step 6:  See all tasks            →  GET  /tasks          (now shows 1 task)
Step 7:  Mark it completed        →  PUT  /tasks/1        body: {"status": "completed"}
Step 8:  Confirm it updated       →  GET  /tasks/1        (status is now "completed")
Step 9:  Delete the task          →  DELETE /tasks/1
Step 10: Confirm it is gone       →  GET  /tasks/1        (returns 404)
```

If all 10 steps work without errors, your API is fully functional.

---

---

# Troubleshooting Common Errors

### "Connection refused" or "Unable to connect"
The server is not running.
Fix: Open a terminal and run `python app.py`

---

### `400 Bad Request` when creating a task
Most common causes:
1. You forgot to set the body type to **JSON** in Postman
2. The title is missing or empty
3. There is a typo in your JSON (e.g. missing comma or quote)

Fix: Check your JSON carefully. Every key and value must be in double quotes.

---

### `404 Not Found` when getting a task
The task ID you used does not exist.
Fix: Run `GET /tasks` first to see which IDs actually exist.

---

### Response shows HTML instead of JSON
You are accessing a URL that Flask does not recognise.
Fix: Double-check the URL — make sure it matches exactly (e.g. `/tasks` not `/task`).

---

# Summary — Which Method Should You Use?

| Method | Best for | Requires install? |
|---|---|---|
| Browser | Quick read-only checks | No |
| Postman | All actions, visual interface | Yes (free) |
| PowerShell | Quick tests from terminal | No (built into Windows) |
| curl | Command line, scripting | Maybe (check with `curl --version`) |

**Recommendation for beginners:** Use **Postman** for exploring the API and **PowerShell** for quick checks.
