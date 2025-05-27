import streamlit as st
import os
import fitz  # PyMuPDF for PDF processing
import tempfile
import zipfile
import io
from app import calculate_hybrid_score
import pandas as pd
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Resume to JD Similarity Matcher",
    page_icon="📋",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* General styling */
    .stFileUploader {
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        text-decoration: none;
    }
    .download-btn:hover {
        background-color: #45a049;
    }
    .jd-input-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .jd-input-container h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* Clean up spacing */
    .main-section {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF file."""
    try:
        # Create PDF document from bytes
        doc = fitz.open("pdf", pdf_bytes)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def parse_jd(jd_input, is_text=True):
    """Parse job description input."""
    if is_text:
        return jd_input
    elif isinstance(jd_input, bytes):
        try:
            return extract_text_from_pdf(jd_input)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    return ""

def process_resumes(resume_files):
    """Process resume files."""
    resume_texts = {}
    for file in resume_files:
        try:
            text = extract_text_from_pdf(file.getvalue())
            if text:
                resume_texts[file.name] = text
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    return resume_texts

def main():
    st.title("Resume to JD Similarity Matcher")
    
    # Sidebar for display options
    with st.sidebar:
        st.header("Settings")
        display_options = st.selectbox(
            "Number of resumes to display",
            ["Top 5", "Top 10", "All"]
        )
        
        # JD input type toggle
        jd_type = st.radio(
            "Job Description Input Type",
            ["Text", "PDF"],
            horizontal=True
        )
        
        # Scoring weights customization - collapsible
        st.header("Scoring Weights")
        customize_weights = st.checkbox("Customize scoring weights", value=False)
        
        # Default weights
        semantic_weight_with_role = 0.1
        keyword_weight_with_role = 0.7
        role_weight = 0.2
        semantic_weight_no_role = 0.2
        keyword_weight_no_role = 0.8
        
        if customize_weights:
            st.write("Customize the importance of different matching factors")
            
            # Weights when role requirements exist
            st.subheader("With Role Requirements")
            semantic_weight_with_role = st.slider(
                "Semantic Similarity Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.1,
                step=0.05,
                help="Weight for semantic similarity score when role requirements exist"
            )
            
            keyword_weight_with_role = st.slider(
                "Keyword Matching Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7,
                step=0.05,
                help="Weight for keyword matching score when role requirements exist"
            )
            
            role_weight = st.slider(
                "Role Matching Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2,
                step=0.05,
                help="Weight for role matching score"
            )
            
            # Ensure weights sum to 1.0
            total_with_role = semantic_weight_with_role + keyword_weight_with_role + role_weight
            if total_with_role != 1.0:
                st.warning(f"Weights with role sum to {total_with_role:.2f}, not 1.0. Results will be normalized.")
                # Normalize weights
                semantic_weight_with_role /= total_with_role
                keyword_weight_with_role /= total_with_role
                role_weight /= total_with_role
            
            # Weights when no role requirements exist
            st.subheader("Without Role Requirements")
            semantic_weight_no_role = st.slider(
                "Semantic Similarity Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2,
                step=0.05,
                help="Weight for semantic similarity score when no role requirements exist"
            )
            
            keyword_weight_no_role = st.slider(
                "Keyword Matching Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.8,
                step=0.05,
                help="Weight for keyword matching score when no role requirements exist"
            )
            
            # Ensure weights sum to 1.0
            total_no_role = semantic_weight_no_role + keyword_weight_no_role
            if total_no_role != 1.0:
                st.warning(f"Weights without role sum to {total_no_role:.2f}, not 1.0. Results will be normalized.")
                # Normalize weights
                semantic_weight_no_role /= total_no_role
                keyword_weight_no_role /= total_no_role
        
    # Main content - side by side layout with more gap
    col1, col2 = st.columns(2)
    
    # Job Description section
    with col1:
        st.header("Job Description")
        
        # Show appropriate JD input based on selection
        if jd_type == "Text":
            jd_text = st.text_area("Paste job description here", height=300)
            jd_file = None
        else:
            jd_file = st.file_uploader("Upload job description PDF", type=['pdf'], key="jd_uploader")
            jd_text = ""
            
            # Display parsed JD text if PDF is uploaded
            if jd_file is not None:
                try:
                    parsed_text = extract_text_from_pdf(jd_file.getvalue())
                    st.subheader("Parsed Job Description")
                    st.text_area("Parsed text", value=parsed_text, height=200, disabled=True)                    
                except Exception as e:
                    st.error(f"Error parsing PDF: {str(e)}")
    
    # Resume Upload Section
    with col2:
        st.header("Resumes")
        
        resume_files = st.file_uploader(
            "Upload resumes",
            type=['pdf'],
            accept_multiple_files=True,
            key="resume_uploader"
        )
    
    # Process button
    if st.button("Process and Match"):
        if not jd_text and not jd_file:
            st.error("Please provide a job description either by pasting text or uploading a PDF")
            return
            
        if not resume_files:
            st.error("Please upload at least one resume")
            return
            
        # Get job description text
        jd_content = parse_jd(jd_text, is_text=True) if jd_text else parse_jd(jd_file.getvalue(), is_text=False)
        if not jd_content:
            st.error("Failed to extract job description text. Please ensure the file is a valid PDF or text is not empty.")
            return
            
        # Process resumes
        resume_texts = process_resumes(resume_files)
        if not resume_texts:
            st.error("No valid resumes found")
            return
            
        # Prepare weights dictionary from sidebar inputs
        weights = {
            "semantic_weight_with_role": semantic_weight_with_role,
            "keyword_weight_with_role": keyword_weight_with_role,
            "role_weight": role_weight,
            "semantic_weight_no_role": semantic_weight_no_role,
            "keyword_weight_no_role": keyword_weight_no_role
        }
        
        # Calculate scores
        results = []
        for filename, resume_text in resume_texts.items():
            scores = calculate_hybrid_score(resume_text, jd_content, weights=weights)
            results.append({
                "filename": filename,
                "final_score": scores["final_score"],
                "semantic_score": scores["semantic_score"],
                "keyword_score": scores["keyword_score"],
                "role_score": scores["role_score"]
            })
            
        # Sort results by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Filter results based on display option
        if display_options == "Top 5":
            results = results[:5]
        elif display_options == "Top 10":
            results = results[:10]
            
        # Display results
        st.header("Matching Results")
        if results:
            # Create a custom table with download buttons
            st.markdown("""
            <style>
            .download-btn {
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
                text-decoration: none;
                font-size: 12px;
                cursor: pointer;
            }
            .results-table th, .results-table td {
                padding: 8px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .results-table th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .results-table tr:hover {background-color: #f5f5f5;}
            .stButton button {height: 36px; line-height: 1;}
            </style>
            """, unsafe_allow_html=True)
            
            # Prepare file data for download buttons
            file_data = {}
            for result in results:
                original_file = next((f for f in resume_files if f.name == result['filename']), None)
                if original_file:
                    file_data[result['filename']] = original_file.getvalue()
            
            # Option to download all top resumes as a zip file
            if file_data:
                # Create a zip file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, file_content in file_data.items():
                        zip_file.writestr(filename, file_content)
                
                # Create download button for the zip file
                st.download_button(
                    label=f"Download All {len(results)} Resumes as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="top_resumes.zip",
                    mime="application/zip",
                    key="download_all_zip"
                )
            
            # Display results in columns with metrics and download buttons
            # Use a dataframe for better alignment
            df_data = []
            for result in results:
                df_data.append({
                    "Filename": result["filename"],
                    "Final Score": f"{result['final_score']:.2%}",
                    "Semantic Score": f"{result['semantic_score']:.2%}",
                    "Keyword Score": f"{result['keyword_score']:.2%}",
                    "Role Score": f"{result['role_score']:.2%}",
                })
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            st.dataframe(df)
            
            # Add download column to the dataframe
            # Create a new dataframe with download buttons
            st.markdown("### Download Options")
            st.markdown("**Individual Resume Downloads:**")
            
            # Create a grid layout for download buttons - 3 columns
            button_cols = st.columns(3)
            for i, result in enumerate(results):
                col_idx = i % 3
                with button_cols[col_idx]:
                    if result['filename'] in file_data:
                        st.download_button(
                            label=f"Download {result['filename']}",
                            data=file_data[result['filename']],
                            file_name=result["filename"],
                            mime="application/pdf",
                            key=f"download_{result['filename']}",
                            use_container_width=True
                        )
        else:
            st.info("No matching resumes found")

if __name__ == "__main__":
    main()
