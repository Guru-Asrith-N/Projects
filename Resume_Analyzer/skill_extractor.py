# skill_extractor.py
# This file handles extracting text from PDF and DOCX files,
# and identifying which skills are present in any given text.

import pdfplumber       # Used for reading PDF files
import docx             # python-docx: used for reading DOCX files
from skills import SKILLS_LIST  # Our predefined list of skills


def extract_text_from_pdf(file):
    """
    Reads a PDF file and returns all the text from it as a single string.
    
    How it works:
    - pdfplumber opens the PDF
    - We loop through each page
    - We extract the text from each page and join it together
    
    Parameters:
        file: A file-like object (the uploaded PDF file from Streamlit)
    
    Returns:
        A string containing all the text in the PDF
    """
    text = ""  # Start with an empty string
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                # page.extract_text() returns the text of that page
                # If a page has no text, it returns None, so we use "or ''"
                page_text = page.extract_text() or ""
                text += page_text + "\n"  # Add a newline between pages
    except Exception as e:
        # If something goes wrong, we return an empty string
        print(f"Error reading PDF: {e}")
        return ""
    return text


def extract_text_from_docx(file):
    """
    Reads a DOCX file and returns all the text from it as a single string.
    
    How it works:
    - python-docx loads the document
    - A DOCX file has paragraphs (like paragraphs in Microsoft Word)
    - We loop through each paragraph and get its text
    
    Parameters:
        file: A file-like object (the uploaded DOCX file from Streamlit)
    
    Returns:
        A string containing all the text in the DOCX file
    """
    text = ""
    try:
        document = docx.Document(file)  # Load the Word document
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"  # Each paragraph on a new line
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""
    return text


def extract_skills(text):
    """
    Finds which skills from our SKILLS_LIST are mentioned in the given text.
    
    How it works:
    - We convert both the text and skill names to lowercase for comparison
    - We check if each skill from our predefined list appears in the text
    - Simple string search: "python" in "I know python and java" → True
    
    Parameters:
        text: Any string (resume text or job description text)
    
    Returns:
        A Python set of skills found in the text (sets avoid duplicates)
    """
    # Convert text to lowercase so "Python" and "python" both match
    text_lower = text.lower()
    
    found_skills = set()  # Use a set so no skill appears twice
    
    for skill in SKILLS_LIST:
        # Check if the skill name appears anywhere in the text
        if skill.lower() in text_lower:
            found_skills.add(skill.lower())  # Store in lowercase
    
    return found_skills
