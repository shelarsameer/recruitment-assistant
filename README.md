# Resume to JD Similarity Matcher

This project provides an AI-driven resume matching system that compares job descriptions (JDs) with resumes and ranks them based on their similarity. It uses a hybrid scoring approach combining semantic similarity, keyword matching, and role matching to provide comprehensive results.

## Features

### Core Features
- **Hybrid Scoring System**: Combines semantic similarity, keyword matching, and role-specific matching
- **PDF Processing**: Automatically extracts text from PDF files
- **Customizable Weights**: Adjust the importance of different matching factors
- **User-friendly Interface**: Clean Streamlit frontend for easy interaction
- **Bulk Processing**: Process multiple resumes at once
- **Batch Download**: Download all top resumes as a ZIP file

### Technical Features
- **Semantic Analysis**: Uses Sentence-Transformers for embedding generation
- **NLP Processing**: Extracts keywords, skills, and role requirements using spaCy
- **REST API**: Backend API for programmatic access and integration
- **Error Handling**: Robust error handling and input validation

## Components

### Backend (app.py)
- Flask-based REST API
- Sentence-Transformer model for semantic similarity
- spaCy for natural language processing
- Keyword and role extraction algorithms

### Frontend (frontend.py)
- Streamlit-based user interface
- PDF file upload and processing
- Interactive results display
- Customizable scoring weights
- Download options for results

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model (if not already installed):
```bash
python -m spacy download en_core_web_sm
```

3. Create a `.env` file (optional):
```
PORT=5000
```

## Running the Application

### Backend API Only
```bash
python app.py
```
This will start the Flask API server on port 5000 (or the port specified in your .env file).

### Frontend Application
```bash
python -m streamlit run frontend.py
```
This will start the Streamlit application, typically on http://localhost:8501.

## How to Use

### Using the Streamlit Frontend

1. **Input Job Description**:
   - Either paste the job description text directly, or
   - Upload a PDF file containing the job description

2. **Upload Resumes**:
   - Upload multiple PDF resumes for comparison

3. **Customize Settings** (Optional):
   - Adjust the number of top results to display
   - Customize the scoring weights for different matching factors

4. **Process and Match**:
   - Click the "Process and Match" button to analyze and rank the resumes

5. **View Results**:
   - See ranked results with detailed scores
   - Download individual resumes or all top matches as a ZIP file

### Using the API

#### Health Check
- **GET** `/health`
- Returns the health status of the service

#### Resume Matching
- **POST** `/match`
- Request body:
```json
{
    "jd": "Job description text",
    "resumes": [
        {"text": "Resume 1 text", "id": "resume1", "name": "John Doe"},
        {"text": "Resume 2 text", "id": "resume2", "name": "Jane Smith"}
    ]
}
```
- Response:
```json
{
    "matches": [
        {
            "index": 0,
            "similarity": 0.85,
            "semantic_score": 0.82,
            "keyword_score": 0.90,
            "role_score": 0.75,
            "id": "resume1",
            "name": "John Doe"
        },
        {
            "index": 1,
            "similarity": 0.75,
            "semantic_score": 0.70,
            "keyword_score": 0.80,
            "role_score": 0.65,
            "id": "resume2",
            "name": "Jane Smith"
        }
    ],
    "total_resumes": 2,
    "has_role_requirement": true
}
```

## Scoring System

The application uses a hybrid scoring approach that combines:

1. **Semantic Similarity** (0.1-0.2 weight by default):
   - Uses sentence embeddings to capture the overall contextual similarity

2. **Keyword Matching** (0.6-0.8 weight by default):
   - Extracts technical terms, tools, languages, and frameworks
   - Identifies specific skills mentioned in both JD and resumes

3. **Role Matching** (0.2 weight by default, when applicable):
   - Identifies specific roles mentioned in the JD
   - Matches against roles mentioned in resumes
   - Considers both exact and partial role matches

The weights for these components can be customized in the frontend interface.

## n8n Integration

1. Set up a Google Drive trigger to watch for new job descriptions
2. Use Google Drive node to list and read resumes
3. Use HTTP node to call this API:
   - Method: POST
   - URL: `http://your-api-url/match`
   - Body: 
   ```json
   {
     "jd": "{{ $json.text_jd }}",
     "resumes": {{ $json.resume_texts }}
   }
   ```
4. Use Google Sheets node to write results
5. Optionally use Email node to send notifications

## Deployment Options

- Local development
- Render
- Railway
- Hugging Face Spaces

## Notes

- The API returns top 10 matches by default
- Make sure to handle PDF/DOCX text extraction before sending to the API 
