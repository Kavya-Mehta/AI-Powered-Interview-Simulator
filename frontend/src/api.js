import axios from "axios";

const API = axios.create({ baseURL: "http://localhost:8000" });

// Auto-attach JWT token to every request
API.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ─── Auth ──────────────────────────────────────────────────────────────────────
export const login = (username, password) =>
  API.post("/auth/login", { username, password });

export const signup = (username, password) =>
  API.post("/auth/signup", { username, password });

// ─── Interview ─────────────────────────────────────────────────────────────────
export const startInterview = (
  interview_type,
  difficulty,
  num_questions,
  resume_text,
  job_description,
) =>
  API.post("/interview/start", {
    interview_type,
    difficulty,
    num_questions,
    resume_text,
    job_description,
  });

export const sendMessage = (
  session_id,
  messages,
  interview_type,
  difficulty,
  num_questions,
  resume_text,
  job_description,
) =>
  API.post("/interview/chat", {
    session_id,
    messages,
    interview_type,
    difficulty,
    num_questions,
    resume_text,
    job_description,
  });

export const updateSessionStatus = (session_id, status) =>
  API.patch(`/interview/session/${session_id}/status`, { status });

// ─── Profile ───────────────────────────────────────────────────────────────────
export const getProfile = () => API.get("/profile");

export const updateUsername = (new_username) =>
  API.put("/profile/username", { new_username });

export const updatePassword = (current_password, new_password) =>
  API.put("/profile/password", { current_password, new_password });

export const deleteAccount = () => API.delete("/profile");

// ─── History ───────────────────────────────────────────────────────────────────
export const getSessions = () => API.get("/history/sessions");

export const getSessionDetails = (session_id) =>
  API.get(`/history/session/${session_id}`);

export const getStats = () => API.get("/history/stats");

export const getDashboardInsights = () => API.get("/history/dashboard");

// ─── Resumes ───────────────────────────────────────────────────────────────────
export const getResume = (resume_id) =>
  API.get(`/profile/resumes/${resume_id}`);

export const getResumes = () => API.get("/profile/resumes");

export const uploadResume = (filename, content) =>
  API.post("/profile/resumes", { filename, content });

export const deleteResume = (resume_id) =>
  API.delete(`/profile/resumes/${resume_id}`);
