# 📄 AI Resume Analyzer

A beginner-friendly Python project that analyzes how well a resume matches a job description using skill extraction, TF-IDF cosine similarity, and a transparent scoring formula.

---

## 🚀 Features

- Upload PDF or DOCX resumes
- Paste any job description
- Extract and compare skills
- Calculate a clear match score
- View strengths and improvement suggestions
- Store all analyses in a local SQLite database
- View analysis history with a trend chart

---

## 🗂️ Project Structure

```
Resume_Analyzer/
│
├── app.py              ← Main Streamlit web app (run this!)
├── analyzer.py         ← Core analysis logic (skills, TF-IDF, scoring)
├── skill_extractor.py  ← File reading (PDF/DOCX) and skill detection
├── skills.py           ← Predefined list of skills to detect
├── database.py         ← SQLite database operations
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🛠️ Installation & Setup (Windows)

### Step 1 — Make sure Python is installed
Open Command Prompt and type:
```
python --version
```
You should see something like `Python 3.10.x`. If not, download Python from https://python.org

### Step 2 — Navigate to the project folder
```
cd c:\Projects\SDE\Resume_Analyzer
```

### Step 3 — (Optional but recommended) Create a virtual environment
A virtual environment keeps your project's packages isolated from the rest of your system.
```
python -m venv venv
venv\Scripts\activate
```
You'll see `(venv)` appear in your terminal — that means it's active.

### Step 4 — Install required packages
```
pip install -r requirements.txt
```
This installs all the libraries the project needs.

### Step 5 — Run the app
```
streamlit run app.py
```
A browser window will open automatically at `http://localhost:8501`

---

## 📖 How to Use

1. **Home Page** — Read about the project and how scoring works
2. **Analyze Resume** — Upload your resume (PDF or DOCX) and paste a job description, then click "Analyze"
3. **Analysis History** — View all your past analyses stored in the database

---

## 📦 Dependencies Explained

| Package | What It Does |
|---|---|
| `streamlit` | Builds the interactive web UI |
| `pdfplumber` | Extracts text from PDF files |
| `python-docx` | Extracts text from DOCX (Word) files |
| `scikit-learn` | Provides TF-IDF and cosine similarity |
| `pandas` | Displays data as tables, reads SQLite results |
| `sqlite3` | Built into Python — stores analysis history |

---

## 🏆 Scoring Formula

```
Final Score = (Skill Match % × 0.60) + (TF-IDF Similarity × 100 × 0.40)
```

- **60%** from skill match (did the resume have the required skills?)
- **40%** from TF-IDF similarity (how similar is the overall text?)

---

## 👤 Author

Final Year Computer Science Student Project
