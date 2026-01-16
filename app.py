import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords

# ==============================
# NLTK
# ==============================
nltk.download("stopwords")

# ==============================
# Skill Whitelist
# ==============================
SKILLS = {
    "python", "java", "javascript", "react", "next", "node", "express",
    "html", "css", "bootstrap", "mysql", "mongodb", "docker",
    "aws", "cloud", "rest", "api", "git", "github",
    "data", "structures", "algorithms", "dsa",
    "machine", "learning", "ai", "nlp", "cnn",
    "selenium", "automation", "sql"
}

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="AI Resume Analyzer (ATS)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer (ATS Analytics)")
st.write("ATS-style Resume Matching Dashboard")

# ==============================
# Helper Functions
# ==============================
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
    except:
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(text):
    sw = set(stopwords.words("english"))
    return " ".join(w for w in text.split() if w not in sw)

def extract_skills(text):
    words = set(text.split())
    return words & SKILLS

def calculate_scores(resume_text, job_text):
    resume_clean = remove_stopwords(clean_text(resume_text))
    job_clean = remove_stopwords(clean_text(job_text))

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, job_clean])
    tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

    # Skill matching
    resume_skills = extract_skills(resume_clean)
    job_skills = extract_skills(job_clean)

    skill_score = (
        (len(resume_skills & job_skills) / len(job_skills)) * 100
        if job_skills else 0
    )

    final_score = (tfidf_score * 0.3) + (skill_score * 0.7)

    return round(final_score, 2), round(tfidf_score, 2), round(skill_score, 2)

# ==============================
# UI
# ==============================
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Resume"):
    if not resume_file or not job_desc.strip():
        st.warning("Please upload resume and paste job description")
    else:
        resume_text = extract_text_from_pdf(resume_file)
        final_score, tfidf_score, skill_score = calculate_scores(resume_text, job_desc)

        st.subheader("📊 ATS Match Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Final ATS Score", f"{final_score}%")
        c2.metric("Text Similarity", f"{tfidf_score}%")
        c3.metric("Skill Match", f"{skill_score}%")

        fig, ax = plt.subplots(figsize=(6, 0.5))
        ax.barh([0], [final_score])
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("ATS Match Percentage")
        ax.set_title("Resume vs Job Description")
        st.pyplot(fig)
