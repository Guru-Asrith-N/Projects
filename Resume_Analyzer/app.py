# app.py
# This is the MAIN file of the project.
# Run this file with: streamlit run app.py
#
# It builds the web interface using Streamlit and connects all other modules:
#   - skill_extractor.py  → extracts text from files and detects skills
#   - analyzer.py         → runs the matching and scoring logic
#   - database.py         → saves and retrieves analysis history

import streamlit as st
import pandas as pd
from skill_extractor import extract_text_from_pdf, extract_text_from_docx
from analyzer import analyze_resume
from database import create_table, save_analysis, get_all_history, delete_all_history, get_total_analyses

# ─────────────────────────────────────────────
# PAGE CONFIGURATION (must be first Streamlit command)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Makes the app look more professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Sidebar: dark navy gradient, white text ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        background-color: transparent !important;
    }

    /* ── Cards: dark slate background, bright white text ── */
    .result-card {
        background: #1e2a3a !important;
        border-radius: 12px;
        padding: 22px;
        margin: 10px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        border-left: 4px solid #4fa3e0;
        color: #f0f4f8 !important;
    }
    .result-card h3 {
        color: #7dd3fc !important;
        font-size: 1.15rem;
        margin-bottom: 10px;
    }
    .result-card p, .result-card li {
        color: #e2e8f0 !important;
        line-height: 1.7;
    }
    .result-card strong {
        color: #93c5fd !important;
    }
    .result-card ol li, .result-card ul li {
        color: #e2e8f0 !important;
    }
    .result-card h4 {
        color: #7dd3fc !important;
    }

    /* ── Info box (formula box): dark teal, white text ── */
    .info-box {
        background: #0d3349 !important;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #38bdf8 !important;
        margin: 10px 0;
        color: #e0f2fe !important;
    }
    .info-box * {
        color: #e0f2fe !important;
        background-color: transparent !important;
    }
    .info-box strong {
        color: #7dd3fc !important;
    }
    .info-box code {
        color: #a5f3fc !important;
        font-size: 1rem;
        font-weight: 600;
    }
    .info-box ul li {
        color: #bae6fd !important;
    }

    /* ── Score badge ── */
    .score-badge {
        font-size: 3rem;
        font-weight: bold;
        color: #38bdf8 !important;
        text-align: center;
    }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #7dd3fc !important;
        border-bottom: 2px solid #4fa3e0;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }

    /* ── Skill pills: matching (green tones) ── */
    .skill-pill {
        display: inline-block;
        background: #14532d !important;
        color: #86efac !important;
        border-radius: 20px;
        padding: 4px 13px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #4ade80;
    }

    /* ── Skill pills: missing (red tones) ── */
    .skill-pill-missing {
        display: inline-block;
        background: #4c0519 !important;
        color: #fca5a5 !important;
        border-radius: 20px;
        padding: 4px 13px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #f87171;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Initialize database on startup
# ─────────────────────────────────────────────
create_table()  # Creates the SQLite table if it doesn't exist yet

# ─────────────────────────────────────────────
# SIDEBAR — Navigation
# ─────────────────────────────────────────────
st.sidebar.markdown("## 📄 AI Resume Analyzer")
st.sidebar.markdown("---")

# Navigation options
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🔍 Analyze Resume", "📊 Analysis History"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"📁 **Total Analyses:** {get_total_analyses()}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Made with ❤️ using Python & Streamlit**")


# ═══════════════════════════════════════════════
# PAGE 1: HOME PAGE
# ═══════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📄 AI Resume Analyzer")
    st.markdown("### *Match your resume to any job description instantly*")
    st.markdown("---")

    # Introduction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="result-card">
        <h3>👋 Welcome!</h3>
        <p>
        The <strong>AI Resume Analyzer</strong> helps you understand how well your resume 
        matches a specific job description. It uses simple, transparent techniques:
        </p>
        <ul>
            <li>📌 <strong>Skill Extraction</strong> — Detects technical skills from your resume</li>
            <li>📊 <strong>Skill Matching</strong> — Compares your skills with the job requirements</li>
            <li>🔢 <strong>TF-IDF Similarity</strong> — Measures how similar your text is to the job description</li>
            <li>🏆 <strong>Final Score</strong> — A clear, explainable score out of 100</li>
            <li>💾 <strong>History</strong> — Saves all your past analyses in a local database</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-card" style="text-align:center;">
        <h3>🚀 Quick Start</h3>
        <ol style="text-align:left;">
            <li>Go to <strong>Analyze Resume</strong></li>
            <li>Upload your PDF or DOCX resume</li>
            <li>Paste the job description</li>
            <li>Click <strong>Analyze!</strong></li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # How It Works Section
    st.markdown("## 🔧 How It Works")
    
    how_col1, how_col2, how_col3 = st.columns(3)
    
    with how_col1:
        st.markdown("""
        <div class="result-card" style="text-align:center;">
        <h2>📤</h2>
        <h4>Step 1: Upload</h4>
        <p>Upload your resume as a PDF or DOCX file. The app extracts text from it automatically.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with how_col2:
        st.markdown("""
        <div class="result-card" style="text-align:center;">
        <h2>🧠</h2>
        <h4>Step 2: Analyze</h4>
        <p>Skills are extracted and matched. TF-IDF calculates how similar your resume text is to the job description.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with how_col3:
        st.markdown("""
        <div class="result-card" style="text-align:center;">
        <h2>📈</h2>
        <h4>Step 3: Results</h4>
        <p>View your score, matching skills, missing skills, and personalized suggestions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Scoring Explanation
    st.markdown("## 🏆 How the Score is Calculated")
    st.markdown("""
    <div class="info-box">
    <strong>Final Score Formula:</strong><br><br>
    <code>Final Score = (Skill Match % × 0.60) + (TF-IDF Similarity × 100 × 0.40)</code><br><br>
    <ul>
        <li><strong>60%</strong> of the score comes from <em>Skill Match Percentage</em> (did you have the required skills?)</li>
        <li><strong>40%</strong> of the score comes from <em>TF-IDF Text Similarity</em> (how similar is the overall resume text to the job description?)</li>
    </ul>
    This formula is simple, transparent, and easy to explain!
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 2: RESUME ANALYSIS PAGE
# ═══════════════════════════════════════════════
elif page == "🔍 Analyze Resume":
    st.title("🔍 Resume Analysis")
    st.markdown("Upload your resume and paste the job description below.")
    st.markdown("---")

    # ─── Input Section ───
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📤 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=["pdf", "docx"],
            help="Only PDF and DOCX formats are supported."
        )

    with col_right:
        st.markdown("### 📋 Paste Job Description")
        job_description = st.text_area(
            "Paste the full job description here:",
            height=200,
            placeholder="Example: We are looking for a Python developer with experience in Django, REST APIs, PostgreSQL, and Agile methodologies..."
        )

    st.markdown("---")

    # ─── Analyze Button ───
    analyze_button = st.button("🚀 Analyze My Resume!", type="primary", use_container_width=True)

    if analyze_button:
        # ── Validation: Check if both inputs are provided ──
        if uploaded_file is None:
            st.error("❌ Please upload a resume file (PDF or DOCX).")
            st.stop()

        if not job_description.strip():
            st.error("❌ Please paste a job description before analyzing.")
            st.stop()

        # ── Extract text from the uploaded file ──
        resume_text = ""
        file_name = uploaded_file.name

        if file_name.endswith(".pdf"):
            with st.spinner("📖 Reading your PDF resume..."):
                resume_text = extract_text_from_pdf(uploaded_file)
        elif file_name.endswith(".docx"):
            with st.spinner("📖 Reading your DOCX resume..."):
                resume_text = extract_text_from_docx(uploaded_file)

        # Check if text was successfully extracted
        if not resume_text.strip():
            st.error("❌ Could not extract text from the file. Please check that the file is not empty or image-based.")
            st.stop()

        # ── Run the full analysis ──
        with st.spinner("🧠 Analyzing your resume... please wait..."):
            results = analyze_resume(resume_text, job_description)

        # ── Save to database ──
        save_analysis(
            resume_filename=file_name,
            final_score=results["final_score"],
            skill_match_percentage=results["skill_match_percentage"],
            tfidf_similarity=results["tfidf_similarity"],
            matching_skills=results["matching_skills"],
            missing_skills=results["missing_skills"]
        )

        st.success("✅ Analysis complete! Results saved to history.")
        st.markdown("---")

        # ══════════════════════════════════════
        # RESULTS DASHBOARD
        # ══════════════════════════════════════
        st.markdown("## 📊 Analysis Results")
        st.markdown(f"**Resume:** `{file_name}`")

        # ─── Row 1: Key Metrics ───
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🏆 Final Score", f"{results['final_score']:.1f} / 100")
        with m2:
            st.metric("🎯 Skill Match", f"{results['skill_match_percentage']:.1f}%")
        with m3:
            st.metric("📐 TF-IDF Similarity", f"{results['tfidf_similarity']:.4f}")
        with m4:
            st.metric("✅ Matching Skills", len(results['matching_skills']))

        st.markdown("---")

        # ─── Overall Score Progress Bar ───
        st.markdown("### 🏆 Overall Match Score")
        score = results["final_score"]

        # Determine color label based on score
        if score >= 75:
            score_label = "🟢 Excellent Match"
        elif score >= 50:
            score_label = "🟡 Good Match"
        elif score >= 30:
            score_label = "🟠 Partial Match"
        else:
            score_label = "🔴 Low Match"

        st.progress(int(score) / 100)  # Progress bar (expects 0.0 to 1.0)
        st.markdown(f"**{score_label}** — Score: **{score:.1f} / 100**")

        # ─── Score Breakdown ───
        st.markdown("### 📐 Score Breakdown")
        st.markdown("""
        <div class="info-box">
        <strong>Formula: Final Score = (Skill Match % × 0.60) + (TF-IDF × 100 × 0.40)</strong>
        </div>
        """, unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric(
                "Skill Match Component (60%)",
                f"{results['skill_component']:.2f} pts",
                help=f"Skill Match {results['skill_match_percentage']:.1f}% × 0.60"
            )
        with b2:
            st.metric(
                "TF-IDF Component (40%)",
                f"{results['tfidf_component']:.2f} pts",
                help=f"TF-IDF {results['tfidf_score_percent']:.1f}% × 0.40"
            )
        with b3:
            st.metric("Combined Final Score", f"{results['final_score']:.2f} / 100")

        st.markdown("---")

        # ─── Skills Section ───
        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:
            st.markdown("### ✅ Matching Skills")
            if results["matching_skills"]:
                skills_html = " ".join([
                    f'<span class="skill-pill">{skill}</span>'
                    for skill in sorted(results["matching_skills"])
                ])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No matching skills found. Try updating your resume with relevant keywords.")

        with skill_col2:
            st.markdown("### ❌ Missing Skills (from Job Description)")
            if results["missing_skills"]:
                missing_html = " ".join([
                    f'<span class="skill-pill-missing">{skill}</span>'
                    for skill in sorted(results["missing_skills"])
                ])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("🎉 Great! No missing skills detected.")

        st.markdown("---")

        # ─── Strengths ───
        st.markdown("### 💪 Resume Strengths")
        for strength in results["strengths"]:
            st.markdown(f"- {strength}")

        st.markdown("---")

        # ─── Improvement Suggestions ───
        st.markdown("### 📌 Suggested Improvements")
        for suggestion in results["suggestions"]:
            st.markdown(f"- {suggestion}")

        st.markdown("---")

        # ─── Skills Summary Table ───
        st.markdown("### 📋 Skills Summary Table")
        skills_data = {
            "Category": ["Skills in Resume", "Skills in Job Description", "Matching Skills", "Missing Skills"],
            "Count": [
                len(results["resume_skills"]),
                len(results["job_skills"]),
                len(results["matching_skills"]),
                len(results["missing_skills"])
            ]
        }
        df_skills = pd.DataFrame(skills_data)
        st.dataframe(df_skills, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE 3: ANALYSIS HISTORY PAGE
# ═══════════════════════════════════════════════
elif page == "📊 Analysis History":
    st.title("📊 Analysis History")
    st.markdown("View all your previous resume analyses stored in the local database.")
    st.markdown("---")

    # Load history from database
    history_df = get_all_history()

    if history_df.empty:
        st.info("📭 No analysis history found. Go to **Analyze Resume** to run your first analysis!")
    else:
        # Summary metrics
        total = len(history_df)
        avg_score = history_df["final_score"].mean()
        best_score = history_df["final_score"].max()

        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("📁 Total Analyses", total)
        with h2:
            st.metric("📊 Average Score", f"{avg_score:.1f}")
        with h3:
            st.metric("🏆 Best Score", f"{best_score:.1f}")

        st.markdown("---")

        # ─── History Table ───
        st.markdown("### 📋 All Past Analyses")

        # Select and rename columns for display
        display_df = history_df[[
            "resume_filename", "final_score", "skill_match_percentage",
            "tfidf_similarity", "matching_skills", "missing_skills", "analyzed_at"
        ]].copy()

        display_df.columns = [
            "Resume File", "Final Score", "Skill Match %",
            "TF-IDF Score", "Matching Skills", "Missing Skills", "Analyzed At"
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ─── Score Chart ───
        st.markdown("### 📈 Score Trend")
        chart_data = history_df[["analyzed_at", "final_score"]].copy()
        chart_data = chart_data.rename(columns={"final_score": "Final Score", "analyzed_at": "Date"})
        chart_data = chart_data.set_index("Date")
        st.line_chart(chart_data)

        st.markdown("---")

        # ─── Clear History Button ───
        st.markdown("### 🗑️ Clear History")
        st.warning("⚠️ This will permanently delete all analysis records from the database.")
        if st.button("🗑️ Delete All History", type="secondary"):
            delete_all_history()
            st.success("✅ All history has been cleared.")
            st.rerun()  # Refresh the page to show the empty state
        display_df.columns = [
            "Resume File", "Final Score", "Skill Match %",
            "TF-IDF Score", "Matching Skills", "Missing Skills", "Analyzed At"
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ─── Score Chart ───
        st.markdown("### 📈 Score Trend")
        chart_data = history_df[["analyzed_at", "final_score"]].copy()
        chart_data = chart_data.rename(columns={"final_score": "Final Score", "analyzed_at": "Date"})
        chart_data = chart_data.set_index("Date")
        st.line_chart(chart_data)

        st.markdown("---")

        # ─── Clear History Button ───
        st.markdown("### 🗑️ Clear History")
        st.warning("⚠️ This will permanently delete all analysis records from the database.")
        if st.button("🗑️ Delete All History", type="secondary"):
            delete_all_history()
            st.success("✅ All history has been cleared.")
            st.rerun()  # Refresh the page to show the empty state
