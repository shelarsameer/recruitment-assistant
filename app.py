from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
import spacy
from spacy.matcher import Matcher

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the models
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """Extract all technical terms, tools, languages, and frameworks from the text."""
    doc = nlp(text)
    keywords = set()
    
    # Extract role-specific terms
    role_indicators = ['developer', 'engineer', 'analyst', 'architect', 'manager', 
                      'specialist', 'consultant', 'lead', 'director', 'expert','scientist']
    
    # Extract all technical terms and proper nouns
    for token in doc:
        # Skip stop words, punctuation, and very short words
        if (not token.is_stop and 
            not token.is_punct and 
            len(token.text) > 2):
            
            # Add the word if it's a proper noun or noun
            if token.pos_ in ['NOUN', 'PROPN']:
                keywords.add(token.text.lower())
            
            # Add the word if it's in title case (often indicates technical terms)
            if token.text.istitle():
                keywords.add(token.text.lower())
    
    # Extract noun phrases that might contain technical terms
    for chunk in doc.noun_chunks:
        # Skip very long phrases
        if len(chunk.text.split()) <= 4:
            # Add the phrase if it contains technical-looking terms
            if any(word.istitle() for word in chunk.text.split()):
                keywords.add(chunk.text.lower())
            # Add role-specific phrases
            if any(role in chunk.text.lower() for role in role_indicators):
                keywords.add(chunk.text.lower())
    
    # Extract compound technical terms (e.g., "machine learning", "data analysis")
    for i in range(len(doc) - 1):
        if (doc[i].pos_ in ['NOUN', 'PROPN'] and 
            doc[i + 1].pos_ in ['NOUN', 'PROPN'] and 
            not doc[i].is_stop and 
            not doc[i + 1].is_stop):
            compound = f"{doc[i].text.lower()} {doc[i + 1].text.lower()}"
            keywords.add(compound)
    
    return keywords

def extract_skills(text):
    """Extract specific skills and requirements from the text."""
    doc = nlp(text)
    skills = set()
    
    # Common skill indicators
    skill_indicators = ['proficient', 'knowledge', 'understanding', 'experience', 
                       'skills', 'expertise', 'ability', 'capable', 'familiar']
    
    # Extract skills from bullet points or lists
    for token in doc:
        if token.text.lower() in skill_indicators:
            # Look for the object of the skill indicator
            for child in token.children:
                if child.dep_ in ['dobj', 'attr', 'nsubj']:
                    skills.add(child.text.lower())
    
    # Extract specific technical skills
    tech_skills = ['sql', 'python', 'r', 'qlikview', 'powerbi', 'google analytics', 
                  'firebase', 'bi tools', 'analytics']
    
    # Extract compound skills
    for i in range(len(doc) - 1):
        if doc[i].text.lower() + ' ' + doc[i + 1].text.lower() in tech_skills:
            skills.add(doc[i].text.lower() + ' ' + doc[i + 1].text.lower())
        elif doc[i].text.lower() in tech_skills:
            skills.add(doc[i].text.lower())
    
    return skills

def calculate_keyword_score(resume_text, jd_text):
    """Calculate the keyword matching score between resume and job description."""
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)
    
    # Extract specific skills
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    
    if not jd_keywords and not jd_skills:
        return 0.0
    
    # Calculate intersection of general keywords
    matching_keywords = resume_keywords.intersection(jd_keywords)
    
    # Calculate intersection of specific skills
    matching_skills = resume_skills.intersection(jd_skills)
    
    # Calculate scores with higher weight for specific skills
    keyword_score = len(matching_keywords) / len(jd_keywords) if jd_keywords else 0
    skill_score = len(matching_skills) / len(jd_skills) if jd_skills else 0
    
    # Combine scores with higher weight for specific skills
    total_score = (0.3 * keyword_score + 0.7 * skill_score)
    return min(total_score, 1.0)

def extract_role_keywords(text):
    """Extract role-specific keywords from the text."""
    doc = nlp(text)
    role_keywords = set()
    
    # Common role indicators
    role_indicators = ['developer', 'engineer', 'analyst', 'architect', 'manager', 
                      'specialist', 'consultant', 'lead', 'director', 'expert',
                      'scientist', 'programmer', 'designer', 'administrator', 'coordinator']
    
    # Extract complete role phrases
    for chunk in doc.noun_chunks:
        # Convert to lowercase for comparison
        chunk_text = chunk.text.lower()
        
        # Check if the phrase contains a role indicator
        if any(role in chunk_text for role in role_indicators):
            # Add the complete phrase (e.g., "data analyst", "system engineer")
            role_keywords.add(chunk_text)
    
    # Also extract individual role words for single-word roles
    for token in doc:
        token_text = token.text.lower()
        if token_text in role_indicators:
            role_keywords.add(token_text)
    
    return role_keywords

def calculate_role_score(resume_text, jd_text):
    """Calculate the role matching score between resume and job description."""
    resume_roles = extract_role_keywords(resume_text)
    jd_roles = extract_role_keywords(jd_text)
    
    if not jd_roles:
        return 0.0
    
    # Calculate exact matches of complete role phrases
    exact_matches = resume_roles.intersection(jd_roles)
    
    # Calculate partial matches (for cases where JD has more specific role)
    partial_matches = set()
    for jd_role in jd_roles:
        for resume_role in resume_roles:
            # Check if JD role is more specific than resume role
            # e.g., "data analyst" in JD and "analyst" in resume
            if jd_role in resume_role or resume_role in jd_role:
                partial_matches.add((jd_role, resume_role))
    
    # Calculate scores
    exact_score = len(exact_matches) / len(jd_roles)
    partial_score = len(partial_matches) / (2 * len(jd_roles))  # Partial matches count half
    
    # Return the higher of exact or partial score
    return max(exact_score, partial_score)

def calculate_hybrid_score(resume_text, jd_text, weights=None):
    """Calculate hybrid score combining semantic similarity, keyword matching, and role matching.
    
    Args:
        resume_text (str): The resume text to analyze
        jd_text (str): The job description text to match against
        weights (dict, optional): Custom weights for scoring. Format:
            {
                "semantic_weight_with_role": 0.1,
                "keyword_weight_with_role": 0.7,
                "role_weight": 0.2,
                "semantic_weight_no_role": 0.2,
                "keyword_weight_no_role": 0.8
            }
            If None, default weights will be used.
    """
    # Set default weights if none provided
    if weights is None:
        weights = {
            "semantic_weight_with_role": 0.1,
            "keyword_weight_with_role": 0.7,
            "role_weight": 0.2,
            "semantic_weight_no_role": 0.2,
            "keyword_weight_no_role": 0.8
        }
    
    # Calculate semantic similarity score
    resume_embedding = model.encode([resume_text])[0]
    jd_embedding = model.encode([jd_text])[0]
    semantic_score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    
    # Calculate keyword and skill matching score
    keyword_score = calculate_keyword_score(resume_text, jd_text)
    
    # Calculate role matching score only if roles are mentioned in JD
    jd_roles = extract_role_keywords(jd_text)
    if jd_roles:
        role_score = calculate_role_score(resume_text, jd_text)
        # Use role score in final calculation with custom weights
        final_score = (
            weights["semantic_weight_with_role"] * semantic_score + 
            weights["keyword_weight_with_role"] * keyword_score + 
            weights["role_weight"] * role_score
        )
    else:
        role_score = 0.0
        # Skip role score in final calculation
        final_score = (
            weights["semantic_weight_no_role"] * semantic_score + 
            weights["keyword_weight_no_role"] * keyword_score
        )
    
    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "role_score": role_score,
        "has_role_requirement": bool(jd_roles)
    }

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        "message": "Resume Matching API",
        "endpoints": {
            "/health": "GET - Health check endpoint",
            "/match": "POST - Match resumes against job description"
        },
        "example_request": {
            "url": "/match",
            "method": "POST",
            "body": {
                "jd": "Job description text",
                "resumes": ["Resume 1 text", "Resume 2 text"]
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/match', methods=['POST'])
def match():
    """
    Match resumes against a job description using hybrid scoring
    """
    try:
        data = request.get_json()
        jd_text = data.get("jd")
        resumes = data.get("resumes", [])

        if not jd_text or not resumes:
            return jsonify({"error": "Missing required fields: jd and resumes"}), 400

        # Extract roles from JD once
        jd_roles = extract_role_keywords(jd_text)
        has_role_requirement = bool(jd_roles)

        results = []
        for i, resume in enumerate(resumes):
            resume_text = resume.get("text", "")
            scores = calculate_hybrid_score(resume_text, jd_text)
            
            results.append({
                "index": i,
                "similarity": float(scores["final_score"]),
                "semantic_score": float(scores["semantic_score"]),
                "keyword_score": float(scores["keyword_score"]),
                "role_score": float(scores["role_score"]),
                "id": resume.get("id"),
                "name": resume.get("name")
            })

        # Sort results by final similarity score
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return jsonify({
            "matches": results,
            "total_resumes": len(resumes),
            "has_role_requirement": has_role_requirement
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting server on port {port}...")
    print("API Documentation available at: http://localhost:5000/")
    app.run(host='0.0.0.0', port=port, debug=False) 