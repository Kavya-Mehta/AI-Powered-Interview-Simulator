from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from database import InterviewDatabase
from dotenv import load_dotenv
import os
import re
import jwt
import datetime
import traceback

load_dotenv()

app = FastAPI(title="HireReady API", root_path="/api")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "traceback": traceback.format_exc()
        }
    )

try:
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"CRITICAL: Failed to initialize ChatOpenAI: {e}")
    llm = None

try:
    db = InterviewDatabase()
except Exception as e:
    print(f"CRITICAL: Failed to initialize Database: {e}")
    db = None

security = HTTPBearer()

SECRET_KEY = os.getenv("SECRET_KEY", "hireready-secret-key-2026")

class StripAPIPrefixMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and scope["path"].startswith("/api"):
            scope["path"] = scope["path"][4:]
            if not scope["path"]:
                scope["path"] = "/"
        await self.app(scope, receive, send)

app.add_middleware(StripAPIPrefixMiddleware)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Models ───────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    username: str
    password: str

class StartSessionRequest(BaseModel):
    interview_type: str
    difficulty: str
    num_questions: int
    resume_text: str = ""
    job_description: str = ""

class ChatRequest(BaseModel):
    session_id: int
    messages: list
    interview_type: str
    difficulty: str
    num_questions: int
    resume_text: str = ""
    job_description: str = ""

class UpdateStatusRequest(BaseModel):
    status: str

class UpdateUsernameRequest(BaseModel):
    new_username: str

class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class ResumeUploadRequest(BaseModel):
    filename: str
    content: str

# ─── JWT Helpers ───────────────────────────────────────────────────────────────
def create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ─── Auth Routes ───────────────────────────────────────────────────────────────
@app.post("/auth/signup")
def signup(req: AuthRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Database connection is not initialized. Please verify DATABASE_URL and ensure it uses IPv4 pooling on Vercel.")
    user_id, success, message = db.create_user(req.username, req.password)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    token = create_token(user_id, req.username)
    return {"token": token, "user_id": user_id, "username": req.username}

@app.post("/auth/login")
def login(req: AuthRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Database connection is not initialized. Please verify DATABASE_URL and ensure it uses IPv4 pooling on Vercel.")
    user_id, success, message = db.authenticate_user(req.username, req.password)
    if not success:
        raise HTTPException(status_code=401, detail=message)
    token = create_token(user_id, req.username)
    return {"token": token, "user_id": user_id, "username": req.username}

# ─── Interview Routes ──────────────────────────────────────────────────────────
@app.post("/interview/start")
def start_interview(req: StartSessionRequest, user=Depends(verify_token)):
    system_prompt = build_system_prompt(
        req.interview_type, req.difficulty, req.num_questions, req.resume_text, req.job_description
    )
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized properly. Check API keys.")
    messages = [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    ai_msg = response.content
    
    dict_messages = [{"role": "system", "content": system_prompt}]
    
    session_id = db.create_session(
        user["user_id"], req.interview_type, req.difficulty, req.num_questions
    )
    db.save_message(session_id, "assistant", ai_msg)
    return {
        "session_id": session_id,
        "message": ai_msg,
        "messages": dict_messages + [{"role": "assistant", "content": ai_msg}]
    }

@app.post("/interview/chat")
def chat(req: ChatRequest, user=Depends(verify_token)):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized properly. Check API keys.")
    # Save the latest user message if present
    if req.messages and req.messages[-1]["role"] == "user":
        db.save_message(req.session_id, "user", req.messages[-1]["content"])
    
    langchain_messages = []
    for msg in req.messages:
        if msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
            
    response = llm.invoke(langchain_messages)
    ai_reply = response.content
    db.save_message(req.session_id, "assistant", ai_reply)
    
    # Check for completion: contains evaluation keywords OR message count suggests final eval
    reply_lower = ai_reply.lower()
    eval_keywords = ["evaluation", "overall", "assessment", "final feedback", "feedback", "summary", "conclusion"]
    has_eval_keyword = any(kw in reply_lower for kw in eval_keywords)
    
    # Also check if we've collected enough messages (user answers)
    user_message_count = sum(1 for msg in req.messages if msg["role"] == "user")
    is_complete = has_eval_keyword or user_message_count >= req.num_questions
    
    if is_complete:
        db.update_session_status(req.session_id, "completed")
    return {"message": ai_reply, "completed": is_complete}

@app.post("/interview/message")
def save_user_message(session_id: int, content: str, user=Depends(verify_token)):
    db.save_message(session_id, "user", content)
    return {"success": True}

@app.patch("/interview/session/{session_id}/status")
def update_status(session_id: int, req: UpdateStatusRequest, user=Depends(verify_token)):
    db.update_session_status(session_id, req.status)
    return {"success": True}

TECHNICAL_AREAS = [
    {
        "name": "Algorithms / DSA",
        "keywords": ["algorithm", "data structure", "complexity", "time complexity", "space complexity", "optimization", "o(n)", "binary search", "sorting", "tree", "graph", "dynamic programming"],
        "recommendation": "Practice 3–4 timed algorithm problems per week focusing on complexity analysis and edge cases."
    },
    {
        "name": "Coding Quality",
        "keywords": ["code", "coding", "implementation", "syntax", "edge case", "bug", "test case", "clean code", "readable", "modular"],
        "recommendation": "Write clean, modular code from the start and think through edge cases before coding."
    },
    {
        "name": "System Design",
        "keywords": ["system design", "scalability", "architecture", "throughput", "latency", "database", "cache", "api", "distributed", "load balancer"],
        "recommendation": "Solve one system-design prompt weekly and practice discussing trade-offs + scaling strategies."
    },
    {
        "name": "Problem Solving",
        "keywords": ["approach", "problem solving", "break down", "reasoning", "hypothesis", "strategy", "clarifying questions", "assumptions"],
        "recommendation": "Spend 5 minutes framing your approach before coding and verify with small test cases."
    },
    {
        "name": "Communication",
        "keywords": ["communicat", "clarity", "explain", "walk through", "thought process", "articulate", "justify"],
        "recommendation": "Talk through your thought process clearly: state assumptions, explain approach, and justify decisions."
    },
]

BEHAVIORAL_AREAS = [
    {
        "name": "STAR Framework",
        "keywords": ["star", "situation", "task", "action", "result", "specific", "measurable", "example"],
        "recommendation": "Structure every answer using STAR: Situation → Task → Action → Result with measurable outcomes."
    },
    {
        "name": "Leadership",
        "keywords": ["leadership", "lead", "mentor", "influence", "initiative", "ownership", "decision", "responsibility"],
        "recommendation": "Prepare 2–3 stories showing ownership, initiative, and influence without direct authority."
    },
    {
        "name": "Teamwork",
        "keywords": ["collaboration", "team", "stakeholder", "cross-functional", "communication", "conflict", "feedback"],
        "recommendation": "Highlight examples of navigating team dynamics, resolving conflicts, and collaborating effectively."
    },
    {
        "name": "Impact",
        "keywords": ["impact", "results", "improvement", "metrics", "outcome", "value", "measurable", "business"],
        "recommendation": "Quantify your impact with specific metrics (e.g., '30% faster', 'saved $50K', '10K users')."
    },
    {
        "name": "Problem Solving",
        "keywords": ["problem", "challenge", "obstacle", "solution", "approach", "overcome", "analytical"],
        "recommendation": "Show structured problem-solving: how you identified root cause, explored options, and decided."
    },
]

MIXED_AREAS = [
    {
        "name": "Technical Skills",
        "keywords": ["algorithm", "code", "system design", "architecture", "complexity", "optimization", "implementation"],
        "recommendation": "Balance coding practice with system design discussions weekly."
    },
    {
        "name": "Behavioral Skills",
        "keywords": ["star", "leadership", "collaboration", "impact", "ownership", "conflict", "team"],
        "recommendation": "Prepare 3–5 STAR stories covering leadership, conflict, and cross-functional collaboration."
    },
    {
        "name": "Communication",
        "keywords": ["communicat", "clarity", "explain", "articulate", "structure", "justify"],
        "recommendation": "Practice explaining both technical concepts and behavioral examples clearly and concisely."
    },
    {
        "name": "Problem Solving",
        "keywords": ["approach", "problem solving", "strategy", "reasoning", "break down", "analytical"],
        "recommendation": "Demonstrate structured problem-solving in both technical and situational contexts."
    },
]

POSITIVE_WORDS = ["strong", "good", "great", "clear", "excellent", "solid", "well", "effective", "confident"]
NEGATIVE_WORDS = ["improve", "weak", "lacking", "struggle", "unclear", "missed", "incorrect", "incomplete", "needs work"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _area_score(text: str, keywords: list[str]) -> int:
    keyword_hits = sum(text.count(k) for k in keywords)
    positive_hits = sum(text.count(w) for w in POSITIVE_WORDS)
    negative_hits = sum(text.count(w) for w in NEGATIVE_WORDS)
    score = 45 + min(keyword_hits, 10) * 4 + min(positive_hits, 8) * 2 - min(negative_hits, 8) * 3
    return max(20, min(95, score))


class AreaAnalysis(BaseModel):
    area: str = Field(description="Exact area name")
    score: int = Field(description="Performance score 0-100 based on answer quality and feedback")
    evidence: str = Field(description="Brief explanation of what was tested and how they performed")

class InterviewAnalysis(BaseModel):
    covered_areas: list[AreaAnalysis] = Field(description="List of areas actually covered in the interview")

def analyze_qa_pairs_with_ai(qa_pairs: list[dict], area_config: list[dict]) -> dict:
    """Use AI to analyze Q&A pairs and determine coverage and performance for each area."""
    if not qa_pairs or not llm:
        return {}
    
    area_names = [cfg["name"] for cfg in area_config]
    area_descriptions = {cfg["name"]: ", ".join(cfg["keywords"][:5]) for cfg in area_config}
    
    # Build analysis prompt
    qa_text = ""
    for idx, qa in enumerate(qa_pairs[:10], 1):  # Limit to last 10 Q&As to avoid token limits
        qa_text += f"\n\nQ{idx}: {qa['question']}\nA{idx}: {qa['answer']}\n"
        if qa.get('feedback'):
            qa_text += f"Feedback: {qa['feedback']}\n"
    
    analysis_prompt = f"""You are analyzing a technical interview. Below are the questions asked, candidate answers, and interviewer feedback.

Available assessment areas:
{chr(10).join(f"- {name}: {area_descriptions[name]}" for name in area_names)}

For each area that was ACTUALLY COVERED in the interview questions:
1. Assign a performance score 0-100 based on answer quality and feedback
2. Note specific strengths or weaknesses

Only include areas that were actually asked about. If coding wasn't tested, don't include it.

Interview Q&A:
{qa_text}"""

    try:
        structured_llm = llm.with_structured_output(InterviewAnalysis)
        result = structured_llm.invoke([HumanMessage(content=analysis_prompt)])
        return result.model_dump() if hasattr(result, "model_dump") else result.dict()
    except Exception as e:
        print(f"AI analysis error: {e}")
        return {}


def build_dashboard_payload(completed_sessions: list[dict]) -> dict:
    if not completed_sessions:
        return {
            "has_data": False,
            "areas": [],
            "strengths": [],
            "improvements": [],
            "recommendations": [],
            "summary": "Complete an interview to unlock personalized review and recommendations.",
            "source_sessions": 0,
            "interview_context": "Mixed"
        }

    # Detect most common interview type from recent completed sessions
    type_counts = {}
    for session in completed_sessions:
        itype = session.get("interview_type", "Mixed")
        type_counts[itype] = type_counts.get(itype, 0) + 1
    
    dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Mixed"
    
    # Select area config based on interview type
    if dominant_type == "Technical":
        area_config = TECHNICAL_AREAS
    elif dominant_type == "Behavioral":
        area_config = BEHAVIORAL_AREAS
    else:
        area_config = MIXED_AREAS

    # Extract Q&A pairs from the most recent session
    qa_pairs = []
    for session in completed_sessions[:3]:  # Analyze last 3 sessions
        messages = session.get("messages", [])
        current_question = None
        current_answer = None
        current_feedback = None
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "assistant":
                # Check if this is feedback (short) or a new question
                if current_answer and len(content) < 500:
                    # Likely feedback for previous answer
                    current_feedback = content
                    if current_question and current_answer:
                        qa_pairs.append({
                            "question": current_question,
                            "answer": current_answer,
                            "feedback": current_feedback
                        })
                    current_question = None
                    current_answer = None
                    current_feedback = None
                else:
                    # New question
                    if current_question and current_answer:
                        qa_pairs.append({
                            "question": current_question,
                            "answer": current_answer,
                            "feedback": current_feedback
                        })
                    current_question = content
                    current_answer = None
                    current_feedback = None
            elif role == "user":
                current_answer = content
        
        # Add final Q&A if exists
        if current_question and current_answer:
            qa_pairs.append({
                "question": current_question,
                "answer": current_answer,
                "feedback": current_feedback
            })

    # Use AI to analyze Q&A pairs
    ai_analysis = analyze_qa_pairs_with_ai(qa_pairs, area_config)
    
    covered_areas_data = ai_analysis.get("covered_areas", [])
    
    if not covered_areas_data:
        # Fallback to old keyword-based method
        assistant_text_chunks = []
        for session in completed_sessions:
            for message in session.get("messages", []):
                if message.get("role") == "assistant":
                    assistant_text_chunks.append(message.get("content", ""))
        
        corpus = _normalize("\n".join(assistant_text_chunks))
        areas = [{"name": cfg["name"], "score": _area_score(corpus, cfg["keywords"])} for cfg in area_config]
    else:
        # Use AI analysis results
        areas = [{"name": item["area"], "score": item["score"]} for item in covered_areas_data]
    
    if not areas:
        return {
            "has_data": False,
            "areas": [],
            "strengths": [],
            "improvements": [],
            "recommendations": [],
            "summary": "Not enough data to analyze. Complete more interview questions.",
            "source_sessions": len(completed_sessions),
            "interview_context": dominant_type
        }
    
    ranked = sorted(areas, key=lambda a: a["score"], reverse=True)
    
    top_areas = ranked[:min(2, len(ranked))]
    low_areas = list(reversed(ranked[-min(2, len(ranked)):]))
    
    # Get recommendations for low-scoring areas
    rec_map = {cfg["name"]: cfg["recommendation"] for cfg in area_config}
    recommendations = []
    for a in low_areas:
        if a["name"] in rec_map:
            recommendations.append(rec_map[a["name"]])
    
    strengths = [f"{a['name']} ({a['score']}/100)" for a in top_areas]
    improvements = [f"{a['name']} ({a['score']}/100)" for a in low_areas]
    
    interview_label = f"{dominant_type} interview" if dominant_type != "Mixed" else "interviews"
    
    if top_areas and low_areas:
        summary = (
            f"Based on your last {len(completed_sessions)} completed {interview_label}"
            f"{'s' if len(completed_sessions) > 1 else ''}, your strongest area is {top_areas[0]['name']}. "
            f"Focus next on {low_areas[0]['name']} to improve performance."
        )
    else:
        summary = f"Analysis based on {len(completed_sessions)} completed {interview_label}{'s' if len(completed_sessions) > 1 else ''}."

    return {
        "has_data": True,
        "areas": areas,
        "strengths": strengths,
        "improvements": improvements,
        "recommendations": recommendations,
        "summary": summary,
        "source_sessions": len(completed_sessions),
        "interview_context": dominant_type
    }


# ─── History Routes ────────────────────────────────────────────────────────────
@app.get("/history/sessions")
def get_sessions(user=Depends(verify_token)):
    sessions = db.get_user_sessions(user["user_id"], limit=20)
    for s in sessions:
        for k, v in s.items():
            if hasattr(v, 'isoformat'):
                s[k] = v.isoformat()
    return {"sessions": sessions}

@app.get("/history/session/{session_id}")
def get_session_details(session_id: int, user=Depends(verify_token)):
    details = db.get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail="Session not found")
    for k, v in details.items():
        if hasattr(v, 'isoformat'):
            details[k] = v.isoformat()
    for msg in details.get('messages', []):
        for k, v in msg.items():
            if hasattr(v, 'isoformat'):
                msg[k] = v.isoformat()
    return details

@app.get("/history/stats")
def get_stats(user=Depends(verify_token)):
    return db.get_user_stats(user["user_id"])

@app.get("/history/dashboard")
def get_dashboard(user=Depends(verify_token)):
    completed = db.get_completed_sessions_with_messages(user["user_id"], limit=30)
    # Convert any datetime objects to ISO format
    for session in completed:
        for key in ["started_at", "completed_at"]:
            if key in session and hasattr(session[key], "isoformat"):
                session[key] = session[key].isoformat()
        for msg in session.get("messages", []):
            if "timestamp" in msg and hasattr(msg["timestamp"], "isoformat"):
                msg["timestamp"] = msg["timestamp"].isoformat()
    return build_dashboard_payload(completed)

# ─── Profile Routes ────────────────────────────────────────────────────────────
@app.get("/profile")
def get_profile(user=Depends(verify_token)):
    profile = db.get_user_profile(user["user_id"])
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    if profile.get("created_at") and hasattr(profile["created_at"], "isoformat"):
        profile["created_at"] = profile["created_at"].isoformat()
    return profile

@app.put("/profile/username")
def update_username(req: UpdateUsernameRequest, user=Depends(verify_token)):
    success, message = db.update_username(user["user_id"], req.new_username)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    new_token = create_token(user["user_id"], req.new_username)
    return {"message": message, "token": new_token, "username": req.new_username}

@app.put("/profile/password")
def update_password(req: UpdatePasswordRequest, user=Depends(verify_token)):
    success, message = db.update_password(user["user_id"], req.current_password, req.new_password)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@app.delete("/profile")
def delete_account(user=Depends(verify_token)):
    success, message = db.delete_user(user["user_id"])
    if not success:
        raise HTTPException(status_code=500, detail=message)
    return {"message": message}

# ─── Resume Routes ─────────────────────────────────────────────────────────────

@app.get("/profile/resumes")
def get_resumes(user=Depends(verify_token)):
    resumes = db.get_user_resumes(user["user_id"])
    for r in resumes:
        if hasattr(r['uploaded_at'], 'isoformat'):
            r['uploaded_at'] = r['uploaded_at'].isoformat()
    return {"resumes": resumes}

@app.post("/profile/resumes")
def upload_resume(req: ResumeUploadRequest, user=Depends(verify_token)):
    resume_id, success, message = db.upload_resume(user["user_id"], req.filename, req.content)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "resume_id": resume_id}

@app.get("/profile/resumes/{resume_id}")
def get_resume(resume_id: int, user=Depends(verify_token)):
    resume = db.get_resume(user["user_id"], resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    if hasattr(resume['uploaded_at'], 'isoformat'):
        resume['uploaded_at'] = resume['uploaded_at'].isoformat()
    return resume

@app.delete("/profile/resumes/{resume_id}")
def delete_resume(resume_id: int, user=Depends(verify_token)):
    success, message = db.delete_resume(user["user_id"], resume_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

# ─── System Prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(interview_type, difficulty, num_questions, resume_text="", job_description=""):
    base_prompt = f"""You are an expert technical interviewer conducting a {difficulty} {interview_type} interview. Your job is to:
1. Ask one question at a time.
2. Wait for the candidate's response before proceeding.
3. After each answer, provide brief, constructive feedback (2-3 sentences).
4. Then ask the next question.
5. After {num_questions} questions, provide a final overall evaluation with strengths and areas for improvement.

Interview type guidance:
- Technical: Focus on coding problems, system design, algorithms, and domain knowledge.
- Behavioral: Use the STAR method (Situation, Task, Action, Result) for responses.
- Mixed: Alternate between technical and behavioral questions.

Start by greeting the candidate and asking the first question. Be professional, encouraging, and constructive."""

    if resume_text or job_description:
        base_prompt += "\n\nAdditional Context:\n"
        if resume_text:
            base_prompt += f"--- Candidate's Resume ---\n{resume_text}\n\n"
        if job_description:
            base_prompt += f"--- Job Description ---\n{job_description}\n\n"
        base_prompt += "Ensure your questions are tailored to the candidate's experience in their resume and the specific requirements mentioned in the job description."

    return base_prompt