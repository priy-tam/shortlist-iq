from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io
import pypdf
import docx
from sentence_transformers import SentenceTransformer, util
from typing import List
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI Semantic Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Direct Groq API Key Setup
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    text = ""
    try:
        if filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
    return text.strip()

@app.get("/")
def read_root():
    return {"message": "Welcome to ShortListIQ AI Backend!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "system": "operational"}

@app.post("/api/parse-resume")
async def parse_resume(
    job_description: str = Form(...),
    file: UploadFile = File(...)
):
    contents = await file.read()
    raw_text = extract_text_from_file(contents, file.filename)
    if not raw_text:
        raw_text = "Empty document."

    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_embedding = model.encode(raw_text, convert_to_tensor=True)
    
    similarity_score = util.cos_sim(jd_embedding, resume_embedding).item()
    match_score = round(max(0.0, min(100.0, similarity_score * 100)), 1)

    tech_stack = ["Python", "FastAPI", "React", "Next.js", "TypeScript", "Tailwind CSS", "Docker", "SQL", "MongoDB", "PyTorch", "LangChain", "Vector Databases", "Node.js", "AWS"]
    detected_skills = [skill for skill in tech_stack if skill.lower() in raw_text.lower()]
    required_skills = [skill for skill in tech_stack if skill.lower() in job_description.lower()]
    missing_skills = [skill for skill in required_skills if skill not in detected_skills]

    if match_score >= 70:
        recommendation = "Strong Match - Highly Recommended for Technical Interview"
    elif match_score >= 45:
        recommendation = "Moderate Match - Shortlist for Initial Screening"
    else:
        recommendation = "Low Match - Skill Gap Detected"

    return {
        "filename": file.filename,
        "status": "success",
        "match_score": match_score,
        "skills_detected": detected_skills if detected_skills else ["General Candidate Skills"],
        "missing_skills": missing_skills if missing_skills else ["None"],
        "recommendation": recommendation,
        "raw_text_preview": raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
    }

@app.post("/api/batch-parse")
async def batch_parse(
    job_description: str = Form(...),
    files: List[UploadFile] = File(...)
):
    results = []
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    tech_stack = ["Python", "FastAPI", "React", "Next.js", "TypeScript", "Tailwind CSS", "Docker", "SQL", "MongoDB", "PyTorch", "LangChain", "Vector Databases", "Node.js", "AWS"]

    for file in files:
        contents = await file.read()
        raw_text = extract_text_from_file(contents, file.filename)
        if not raw_text:
            continue

        resume_embedding = model.encode(raw_text, convert_to_tensor=True)
        similarity_score = util.cos_sim(jd_embedding, resume_embedding).item()
        match_score = round(max(0.0, min(100.0, similarity_score * 100)), 1)

        detected_skills = [skill for skill in tech_stack if skill.lower() in raw_text.lower()]

        results.append({
            "filename": file.filename,
            "match_score": match_score,
            "skills": detected_skills if detected_skills else ["General Skills"],
            "status": "Shortlisted" if match_score >= 50 else "Reviewed"
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return {"total": len(results), "candidates": results}

# --- GROQ REAL AI VOICE ENGINE ---

@app.get("/api/generate-questions")
async def generate_questions():
    initial_greeting = "Hello! Welcome to your technical interview session. Please introduce yourself and mention the primary tech stack you have worked with."
    return {"question": initial_greeting}

@app.post("/api/interview-step")
async def interview_step(
    answer: str = Form(...),
    history: str = Form("[]")
):
    user_input = answer.strip()

    system_prompt = (
        "You are an expert, highly intelligent human-like technical interviewer conducting a live voice interview. "
        "CRITICAL INSTRUCTIONS:\n"
        "1. Read the user's latest answer carefully. NEVER repeat any previous questions.\n"
        "2. Directly pick up tech terms or project details mentioned in their answer (e.g. FastAPI, React, SQL, async operations) and cross-question them deeply.\n"
        "3. If their answer is vague or brief, challenge them to explain edge cases or real implementation details.\n"
        "4. Understand Hindi, English, and Hinglish naturally.\n"
        "5. Keep responses concise (maximum 2 to 3 sentences) so it sounds natural when spoken aloud over voice."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Candidate's latest answer: '{user_input}'"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        ai_response = completion.choices[0].message.content
        return {"next_question": ai_response, "is_finished": False}
    except Exception as e:
        print("Groq API Execution Error:", e)
        return {
            "next_question": f"Got it. Based on what you said about '{user_input[:30]}...', how do you handle performance optimization and scaling in that architecture?",
            "is_finished": False
        }