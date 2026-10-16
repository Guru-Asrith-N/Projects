# How to Start, Run, and Stop the Task Manager API

Follow these steps in order every time you want to work on the project.

---

## BEFORE YOU BEGIN — Do this only ONCE (first time setup)

These steps create the virtual environment and install Flask.
You only need to do this once. Skip to **Step 1** on future runs.

### 1. Open the project folder in VS Code
- Open VS Code
- Go to **File → Open Folder**
- Select this folder:
  ```
  c:\Projects\SDE\Task Management REST API
  ```

### 2. Open the terminal inside VS Code
- Press `` Ctrl+` `` (the backtick key, top-left of keyboard)
- A terminal panel opens at the bottom

### 3. Navigate into the app folder
```powershell
cd task_manager_api
```

### 4. Create a virtual environment
```powershell
python -m venv venv
```
Wait for it to finish. A new `venv` folder will appear in `task_manager_api`.

### 5. Allow scripts to run (Windows only — do this once)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Type `Y` and press Enter when asked to confirm.

### 6. Install Flask
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Wait for the installation to finish. You should see `Successfully installed flask-...`

---

## EVERY TIME — Starting the app

### Step 1 — Open the project in VS Code
- Open VS Code
- Go to **File → Open Folder**
- Select:
  ```
  c:\Projects\SDE\Task Management REST API
  ```

### Step 2 — Open the terminal
Press `` Ctrl+` `` to open the terminal at the bottom of VS Code.

### Step 3 — Navigate into the app folder
```powershell
cd task_manager_api
```
> If the terminal already shows you are inside `task_manager_api`, skip this step.

### Step 4 — Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
✅ **You will know it worked** when you see `(venv)` at the start of the line, like this:
```
(venv) PS c:\Projects\SDE\Task Management REST API\task_manager_api>
```
> ⚠️ You must do this every time you open a new terminal. Flask will not be found without it.

### Step 5 — Start the server
```powershell
python app.py
```
✅ **You will know it worked** when you see this output:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 6 — Open the API in your browser
Open your browser and go to:
```
http://127.0.0.1:5000
```
You should see:
```json
{"message": "Task Manager API is running!"}
```

🎉 **The app is now running.**

---

## WHILE THE APP IS RUNNING

- The terminal will stay busy — that is normal. The server is listening for requests.
- Every time you **save a change** to `app.py` or `database.py`, Flask will **automatically restart**. You do not need to stop and re-run it.
- To test the API, open a **second terminal** by clicking the `+` button in the terminal panel.

### Opening a second terminal for testing
1. Click the **`+` icon** in the terminal panel (top-right of the terminal area)
2. In the new terminal, navigate and activate the venv:
   ```powershell
   cd task_manager_api
   .\venv\Scripts\Activate.ps1
   ```
3. Now you can run tests or send requests from here while the server keeps running.

---

## RUNNING THE TESTS

Open a second terminal (see above), then run:
```powershell
python -m unittest discover tests -v
```
✅ **You will know all tests passed** when you see:
```
Ran 12 tests in X.XXXs
OK
```
> Tests use a separate temporary database. Your real data is never affected.

---

## STOPPING THE APP

### To stop the Flask server
- Click inside the terminal where `python app.py` is running
- Press `Ctrl+C`

You will see:
```
 * Detected change in 'app.py', reloading       ← (if you saved a file)
 * Restarting with stat
```
or the server will simply stop and return you to the prompt.

✅ **You will know it stopped** when you see the `(venv)` prompt again:
```
(venv) PS c:\Projects\SDE\Task Management REST API\task_manager_api>
```

---

## QUICK REFERENCE (cheat sheet)

| What you want to do | Command |
|---|---|
| Navigate into app folder | `cd task_manager_api` |
| Activate virtual environment | `.\venv\Scripts\Activate.ps1` |
| Start the server | `python app.py` |
| Stop the server | `Ctrl+C` |
| Run all tests | `python -m unittest discover tests -v` |
| Open a new terminal in VS Code | Click `+` in the terminal panel |

---

## TROUBLESHOOTING

### "ModuleNotFoundError: No module named flask"
You forgot to activate the virtual environment.
Run this first:
```powershell
.\venv\Scripts\Activate.ps1
```
Then try `python app.py` again.

### "execution of scripts is disabled on this system"
Run this once in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate again.

### "Address already in use" or port 5000 conflict
The server is already running somewhere. Either:
- Find the other terminal and press `Ctrl+C` there, or
- Restart VS Code

### The browser shows "This site can't be reached"
The server is not running. Go back to the terminal and run `python app.py`.
