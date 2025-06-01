from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for response
class MatchResponse(BaseModel):
    match_score: float

# Utility: Extract text from uploaded PDF/TXT file
def extract_text_from_file(file: UploadFile) -> str:
    try:
        if file.filename.lower().endswith(".pdf"):
            with pdfplumber.open(file.file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                file.file.seek(0)
                return text
        elif file.filename.lower().endswith(".txt"):
            content = file.file.read().decode("utf-8")
            file.file.seek(0)
            return content
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

# Utility: Extract JSON safely from Gemini output
def extract_json_from_text(text: str) -> dict:
    try:
        json_str = re.search(r"\{.*?\}", text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to extract valid JSON from Gemini response")

# Main logic: Send prompt to Gemini and extract match score
def query_gemini_for_match(resume_text: str, jd_text: str) -> MatchResponse:
    prompt = f"""
You are an expert career coach and hiring analyst.

Evaluate how well the candidate matches the following job description based on:
- Required skills and technologies
- Years of experience with each skill
- Relevant job titles held
- Industry alignment
- Educational background

Return your answer strictly in the following JSON format:

{{
  "match_score": float (0 to 95)
}}

Only return the JSON. No explanations or commentary.

Job Description:
{jd_text}

Resume:
{resume_text}
"""

    try:
        response = model.generate_content(prompt)
        print("Gemini response:\n", response.text)  # Optional for debugging
        result = extract_json_from_text(response.text)
        return MatchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Gemini: {str(e)}")

# API endpoint
@app.post("/match")
async def match_resume_jd(
    resume: UploadFile = File(...),
    jd: UploadFile = File(...)
):
    try:
        resume_text = extract_text_from_file(resume)
        jd_text = extract_text_from_file(jd)
        match_result = query_gemini_for_match(resume_text, jd_text)
        return match_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
