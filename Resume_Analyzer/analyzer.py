# analyzer.py
# This file contains the core logic for analyzing how well a resume matches a job description.
# It calculates:
#   1. Skill match percentage (what % of required skills does the resume have?)
#   2. TF-IDF cosine similarity (how similar are the two texts overall?)
#   3. A final combined score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skill_extractor import extract_skills


def calculate_skill_match(resume_skills, job_skills):
    """
    Compares the skills found in the resume vs skills found in the job description.
    
    Parameters:
        resume_skills: A set of skills extracted from the resume
        job_skills: A set of skills extracted from the job description
    
    Returns:
        A dictionary with:
          - matching_skills: skills that appear in both resume and job description
          - missing_skills: skills in job description but NOT in resume
          - match_percentage: what % of job skills are covered by the resume
    """
    # Find skills present in both (intersection of two sets)
    matching_skills = resume_skills.intersection(job_skills)
    
    # Find skills the job needs but the resume doesn't have
    missing_skills = job_skills.difference(resume_skills)
    
    # Calculate the percentage
    # Example: If job needs 10 skills and resume has 7 → 70%
    if len(job_skills) == 0:
        # Avoid dividing by zero if no skills were found in job description
        match_percentage = 0.0
    else:
        match_percentage = (len(matching_skills) / len(job_skills)) * 100
    
    return {
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "match_percentage": round(match_percentage, 2)
    }


def calculate_tfidf_similarity(resume_text, job_description_text):
    """
    Calculates how similar the resume and job description are as whole documents.
    Uses TF-IDF + Cosine Similarity — a standard technique in information retrieval.
    
    What is TF-IDF?
    - TF = Term Frequency: How often a word appears in a document
    - IDF = Inverse Document Frequency: Penalizes very common words (like "the", "and")
    - TF-IDF gives a score to each word that tells us how "important" that word is
    
    What is Cosine Similarity?
    - After TF-IDF converts text to numbers (vectors), cosine similarity measures
      the angle between those two vectors
    - Score of 1.0 = texts are identical
    - Score of 0.0 = texts have nothing in common
    
    Parameters:
        resume_text: The full text of the resume
        job_description_text: The full text of the job description
    
    Returns:
        A float between 0.0 and 1.0 representing similarity
    """
    # TfidfVectorizer converts text into a numerical matrix
    # stop_words='english' removes common English words like "the", "is", "and"
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # We combine both texts into a list so the vectorizer learns from both
    documents = [resume_text, job_description_text]
    
    try:
        # fit_transform learns vocabulary and converts texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # cosine_similarity compares the two vectors
        # tfidf_matrix[0] = resume vector
        # tfidf_matrix[1] = job description vector
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        
        # similarity_score is a 2D array like [[0.73]], so we extract the number
        return round(float(similarity_score[0][0]), 4)
    
    except Exception as e:
        print(f"Error calculating TF-IDF similarity: {e}")
        return 0.0


def calculate_final_score(skill_match_percentage, tfidf_similarity):
    """
    Combines the skill match score and TF-IDF similarity into one final score.
    
    SCORING FORMULA (easy to explain in interviews!):
    -------------------------------------------------------
    Final Score = (Skill Match % × 0.6) + (TF-IDF Similarity × 100 × 0.4)
    -------------------------------------------------------
    
    Why this weighting?
    - Skill match is more directly important (60% weight)
      → Employers care most about whether you have the required skills
    - TF-IDF similarity adds context from the overall text (40% weight)
      → It captures keywords and experience alignment
    
    Parameters:
        skill_match_percentage: float (0 to 100)
        tfidf_similarity: float (0.0 to 1.0) — we convert it to 0-100 scale
    
    Returns:
        A dictionary with the final score and breakdown
    """
    # Convert TF-IDF similarity from 0-1 scale to 0-100 scale
    tfidf_score_percent = tfidf_similarity * 100
    
    # Apply weights:  60% skill match + 40% TF-IDF
    skill_component = skill_match_percentage * 0.6
    tfidf_component = tfidf_score_percent * 0.4
    
    final_score = skill_component + tfidf_component
    
    return {
        "final_score": round(final_score, 2),
        "skill_component": round(skill_component, 2),
        "tfidf_component": round(tfidf_component, 2),
        "tfidf_score_percent": round(tfidf_score_percent, 2),
    }


def get_resume_strengths(matching_skills, resume_text):
    """
    Generates simple positive feedback about the resume based on matching skills.
    
    Parameters:
        matching_skills: set of skills found in both resume and job description
        resume_text: full resume text (used to estimate length)
    
    Returns:
        A list of strength statements (strings)
    """
    strengths = []
    
    if len(matching_skills) >= 10:
        strengths.append("✅ Strong skill alignment with the job description")
    elif len(matching_skills) >= 5:
        strengths.append("✅ Good number of relevant skills detected")
    elif len(matching_skills) > 0:
        strengths.append("✅ Some matching skills found — there is a foundation to build on")
    
    # Check if resume looks detailed (word count)
    word_count = len(resume_text.split())
    if word_count > 400:
        strengths.append("✅ Resume appears detailed and comprehensive")
    elif word_count > 200:
        strengths.append("✅ Resume has a reasonable amount of content")
    
    # Check for important keywords
    resume_lower = resume_text.lower()
    if "project" in resume_lower or "developed" in resume_lower:
        strengths.append("✅ Resume mentions projects or development experience")
    
    if "internship" in resume_lower or "experience" in resume_lower:
        strengths.append("✅ Resume includes work or internship experience")
    
    if "education" in resume_lower or "bachelor" in resume_lower or "degree" in resume_lower:
        strengths.append("✅ Educational background is mentioned")
    
    # If no strengths detected
    if not strengths:
        strengths.append("ℹ️ Resume is present but limited matching content was found")
    
    return strengths


def get_improvement_suggestions(missing_skills, tfidf_similarity, skill_match_percentage):
    """
    Generates helpful suggestions to improve the resume.
    
    Parameters:
        missing_skills: set of skills the job needs but resume doesn't have
        tfidf_similarity: float (0.0 to 1.0)
        skill_match_percentage: float (0 to 100)
    
    Returns:
        A list of suggestion strings
    """
    suggestions = []
    
    # Suggest adding top missing skills (show up to 5)
    if missing_skills:
        top_missing = list(missing_skills)[:5]
        suggestions.append(
            f"📌 Consider learning or adding these skills: {', '.join(top_missing)}"
        )
    
    if skill_match_percentage < 40:
        suggestions.append(
            "📌 Skill match is low. Tailor your resume to include keywords from the job description."
        )
    
    if tfidf_similarity < 0.2:
        suggestions.append(
            "📌 Overall text similarity is low. Use more relevant industry terms and keywords."
        )
    
    if skill_match_percentage >= 70:
        suggestions.append(
            "🎉 Great match! Focus on quantifying achievements (e.g., 'Improved performance by 30%')."
        )
    
    suggestions.append("📌 Add a professional summary at the top of your resume.")
    suggestions.append("📌 Use action verbs like 'developed', 'designed', 'implemented', 'led'.")
    
    return suggestions


def analyze_resume(resume_text, job_description_text):
    """
    Main function that runs the full analysis pipeline.
    Call this function from app.py to get all results at once.
    
    Parameters:
        resume_text: full text extracted from the resume
        job_description_text: text from the job description input
    
    Returns:
        A dictionary with ALL analysis results
    """
    # Step 1: Extract skills from both texts
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description_text)
    
    # Step 2: Calculate skill match
    skill_results = calculate_skill_match(resume_skills, job_skills)
    
    # Step 3: Calculate TF-IDF similarity
    tfidf_similarity = calculate_tfidf_similarity(resume_text, job_description_text)
    
    # Step 4: Calculate final score
    score_results = calculate_final_score(
        skill_results["match_percentage"],
        tfidf_similarity
    )
    
    # Step 5: Get strengths and suggestions
    strengths = get_resume_strengths(skill_results["matching_skills"], resume_text)
    suggestions = get_improvement_suggestions(
        skill_results["missing_skills"],
        tfidf_similarity,
        skill_results["match_percentage"]
    )
    
    # Combine everything into one result dictionary
    return {
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": skill_results["matching_skills"],
        "missing_skills": skill_results["missing_skills"],
        "skill_match_percentage": skill_results["match_percentage"],
        "tfidf_similarity": tfidf_similarity,
        "final_score": score_results["final_score"],
        "skill_component": score_results["skill_component"],
        "tfidf_component": score_results["tfidf_component"],
        "tfidf_score_percent": score_results["tfidf_score_percent"],
        "strengths": strengths,
        "suggestions": suggestions,
    }
