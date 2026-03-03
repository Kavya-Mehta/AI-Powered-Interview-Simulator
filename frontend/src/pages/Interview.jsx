import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  startInterview,
  sendMessage,
  updateSessionStatus,
  getResumes,
  uploadResume,
  getResume,
} from "../api";
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import "pdfjs-dist/build/pdf.worker.mjs";

const TYPES = ["Technical", "Behavioral", "Mixed"];
const DIFFICULTIES = ["Entry Level", "Mid Level", "Senior Level"];

export default function Interview() {
  const [interviewType, setInterviewType] = useState(TYPES[0]);
  const [difficulty, setDifficulty] = useState(DIFFICULTIES[0]);
  const [numQuestions, setNumQuestions] = useState(5);
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeText, setResumeText] = useState("");
  const [userResumes, setUserResumes] = useState([]);
  const [selectedResumeId, setSelectedResumeId] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sessionConfig, setSessionConfig] = useState(null);
  const [started, setStarted] = useState(false);
  const [completed, setCompleted] = useState(false);
  const bottomRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchResumes = async () => {
      try {
        const res = await getResumes();
        setUserResumes(res.data.resumes || []);
      } catch (err) {
        console.error("Failed to load resumes", err);
      }
    };
    fetchResumes();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-detect difficulty from Job Description
  useEffect(() => {
    const jd = jobDescription.toLowerCase();
    if (
      jd.includes("senior") ||
      jd.includes("expert") ||
      jd.includes("lead") ||
      jd.includes("staff")
    ) {
      setDifficulty("Senior Level");
    } else if (
      jd.includes("mid") ||
      jd.includes("intermediate") ||
      jd.includes("experienced")
    ) {
      setDifficulty("Mid Level");
    } else if (
      jd.includes("entry") ||
      jd.includes("junior") ||
      jd.includes("intern") ||
      jd.includes("fresher") ||
      jd.includes("new grad")
    ) {
      setDifficulty("Entry Level");
    }
  }, [jobDescription]);

  const handleResumeUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setResumeFile(file.name);

    if (file.type === "application/pdf") {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        let text = "";
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map((item) => item.str).join(" ") + "\n";
        }
        setResumeText(text);
        await uploadResume(file.name, text);
        const res = await getResumes();
        setUserResumes(res.data.resumes || []);
        // Automatically select the newly uploaded one if possible
        if (res.data.resumes && res.data.resumes.length > 0) {
          setSelectedResumeId(res.data.resumes[0].resume_id);
        }
      } catch (error) {
        console.error("Error reading PDF:", error);
        alert(
          "Failed to read PDF. Please try another file or a plain text document.",
        );
      }
    } else {
      const text = await file.text();
      setResumeText(text);
      await uploadResume(file.name, text);
      const res = await getResumes();
      setUserResumes(res.data.resumes || []);
      if (res.data.resumes && res.data.resumes.length > 0) {
        setSelectedResumeId(res.data.resumes[0].resume_id);
      }
    }
  };

  const handleResumeSelect = (e) => {
    const val = e.target.value;
    setSelectedResumeId(val);
    if (val === "new") {
      setResumeText("");
      setResumeFile(null);
    } else if (val) {
      const selected = userResumes.find((r) => String(r.resume_id) === val);
      if (selected) {
        setResumeFile(selected.filename);
        // The GET /profile/resumes endpoint doesn't return content to save bandwidth,
        // but for simplicity we rely on the backend fetching it, or we already get it.
        // Wait, the backend doesn't fetch it, we send resumeText to /start.
        // We must fetch the actual content if we only got metadata.
        // *Since our get_user_resumes currently does NOT return content, this approach forces us to fetch the full resume or update get_user_resumes to return content. Let me update the startInterview backend endpoint.*
        // Actually, let's keep it simple: the prompt builder expects resumeText.
      }
    }
  };

  const handleStart = async () => {
    let finalResumeText = resumeText;

    if (selectedResumeId && selectedResumeId !== "new" && !finalResumeText) {
      setLoading(true);
      try {
        const res = await getResume(selectedResumeId);
        finalResumeText = res.data.content;
        setResumeText(finalResumeText);
      } catch (err) {
        console.error(err);
        alert("Failed to load the selected resume content.");
        setLoading(false);
        return;
      }
    }

    if (!finalResumeText && selectedResumeId === "new") {
      alert("Please upload your resume to start the interview.");
      return;
    }
    setLoading(true);
    try {
      const res = await startInterview(
        interviewType,
        difficulty,
        numQuestions,
        finalResumeText,
        jobDescription,
      );
      setSessionId(res.data.session_id);
      setSessionConfig({
        interviewType,
        difficulty,
        numQuestions,
        resumeText: finalResumeText,
        jobDescription,
      });
      setMessages(res.data.messages.filter((m) => m.role !== "system"));
      setStarted(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (sessionId) await updateSessionStatus(sessionId, "abandoned");
    setMessages([]);
    setStarted(false);
    setCompleted(false);
    setSessionId(null);
    setSessionConfig(null);
    setResumeFile(null);
    setResumeText("");
    setJobDescription("");
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { role: "user", content: input };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setLoading(true);
    try {
      const allMessages = [
        {
          role: "system",
          content: `You are an expert technical interviewer conducting a ${sessionConfig.difficulty} ${sessionConfig.interviewType} interview. Ask one question at a time and give a final evaluation after ${sessionConfig.numQuestions} questions.`,
        },
        ...updated,
      ];
      const res = await sendMessage(
        sessionId,
        allMessages,
        sessionConfig.interviewType,
        sessionConfig.difficulty,
        sessionConfig.numQuestions,
        sessionConfig.resumeText,
        sessionConfig.jobDescription,
      );
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.data.message },
      ]);
      if (res.data.completed) {
        await updateSessionStatus(sessionId, "completed");
        setCompleted(true);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const selectClass =
    "w-full bg-white border border-[#e0d5c8] rounded-xl px-4 py-3 text-sm text-[#1a1007] outline-none focus:border-[#1a1007] focus:ring-2 focus:ring-[#1a1007]/10 transition-all font-medium appearance-none cursor-pointer";

  return (
    <div className="bg-[#fdf8f3]">
      {/* ─── Config Section (shown before start) ─── */}
      {!started && (
        <div className="max-w-5xl mx-auto px-6 py-16">
          {/* Page Heading */}
          <div className="mb-12">
            <p className="text-[#c2a882] text-xs font-bold uppercase tracking-widest mb-3">
              Interview Practice
            </p>
            <h1
              className="text-5xl lg:text-6xl text-[#1a1007] leading-tight mb-4"
              style={{ fontFamily: "'DM Serif Display', serif" }}
            >
              Ready to
              <br />
              <em className="not-italic text-[#c84b2f]">practice?</em>
            </h1>
            <p className="text-[#8a7060] text-base max-w-md">
              Configure your session below and click Start Interview to begin
              your AI-powered mock interview.
            </p>
          </div>

          {/* Config Card */}
          <div className="grid md:grid-cols-2 gap-8 items-start">
            <div className="space-y-6 bg-white border border-[#e8ddd3] rounded-2xl p-8 shadow-sm">
              <h2 className="text-xs font-bold text-[#8a7060] uppercase tracking-widest border-b border-[#f0e8e0] pb-4">
                Resume & Job Description
              </h2>

              <div>
                <label className="block text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-2 flex justify-between">
                  <span>
                    Select Resume <span className="text-[#c84b2f]">*</span>
                  </span>
                </label>
                <select
                  value={selectedResumeId}
                  onChange={handleResumeSelect}
                  className={selectClass + " mb-4"}
                >
                  <option value="">-- Choose a Saved Resume --</option>
                  {userResumes.map((r) => (
                    <option key={r.resume_id} value={r.resume_id}>
                      {r.filename} (
                      {new Date(r.uploaded_at).toLocaleDateString()})
                    </option>
                  ))}
                  <option value="new">+ Upload New Resume</option>
                </select>

                {selectedResumeId === "new" && (
                  <div className="relative">
                    <input
                      type="file"
                      accept=".pdf,.txt"
                      onChange={handleResumeUpload}
                      className="hidden"
                      id="resume-upload"
                    />
                    <label
                      htmlFor="resume-upload"
                      className="w-full flex items-center justify-center gap-2 bg-[#fdf8f3] border border-dashed border-[#d4c5b5] hover:border-[#1a1007] hover:bg-[#f0e8e0] rounded-xl px-4 py-4 text-sm text-[#1a1007] cursor-pointer transition-all font-medium"
                    >
                      <svg
                        className="w-5 h-5 text-[#8a7060]"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                        />
                      </svg>
                      {resumeFile
                        ? "Change Resume"
                        : "Upload New Resume (PDF/TXT)"}
                    </label>
                    {resumeFile && (
                      <p className="text-xs text-[#8a7060] text-center mt-2">
                        Selected: {resumeFile}
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div>
                <label className="block text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-2">
                  Job Description{" "}
                  <span className="text-gray-400 font-normal normal-case">
                    (Optional)
                  </span>
                </label>
                <textarea
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  placeholder="Paste the job description here to tailor the interview and auto-detect difficulty..."
                  className="w-full bg-white border border-[#e0d5c8] rounded-xl px-4 py-3 text-sm text-[#1a1007] placeholder-[#c2b5a6] outline-none focus:border-[#1a1007] focus:ring-2 focus:ring-[#1a1007]/10 transition-all font-medium resize-none min-h-[100px]"
                />
              </div>

              <h2 className="text-xs font-bold text-[#8a7060] uppercase tracking-widest border-b border-[#f0e8e0] pb-4 mt-8">
                Session Settings
              </h2>

              <div>
                <label className="block text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-2">
                  Interview Type
                </label>
                <select
                  value={interviewType}
                  onChange={(e) => setInterviewType(e.target.value)}
                  className={selectClass}
                >
                  {TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-2">
                  Difficulty
                </label>
                <select
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  className={selectClass}
                >
                  {DIFFICULTIES.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-2 flex justify-between">
                  <span>Questions</span>
                  <span className="text-[#c84b2f] font-bold">
                    {numQuestions}
                  </span>
                </label>
                <input
                  type="range"
                  min="3"
                  max="10"
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(Number(e.target.value))}
                  className="w-full mt-2 cursor-pointer accent-[#c84b2f]"
                />
                <div className="flex justify-between text-xs text-[#c2b5a6] mt-1">
                  <span>3</span>
                  <span>10</span>
                </div>
              </div>

              <button
                onClick={handleStart}
                disabled={
                  loading ||
                  (!resumeText && selectedResumeId === "new") ||
                  !selectedResumeId
                }
                className="w-full bg-[#1a1007] hover:bg-[#2e1e10] disabled:opacity-50 disabled:bg-[#e0d5c8] disabled:text-[#8a7060] text-[#fdf8f3] font-bold py-4 rounded-xl transition-all text-sm mt-2 flex items-center justify-center gap-2"
              >
                {loading
                  ? "Starting…"
                  : !selectedResumeId
                    ? "Select a Resume to Start"
                    : "Start Interview"}
              </button>
            </div>

            {/* Info pane */}
            <div className="bg-[#1a1007] rounded-2xl p-8 text-[#fdf8f3]">
              <img
                src="/logo.png"
                alt="HireReady"
                className="w-16 h-16 object-contain mb-6 brightness-200 opacity-80"
              />
              <h3
                className="text-2xl mb-3"
                style={{ fontFamily: "'DM Serif Display', serif" }}
              >
                How it works
              </h3>
              <div className="space-y-4 text-sm text-[#8a7060]">
                {[
                  "Configure your type and difficulty above.",
                  "Click Start — the AI interviewer asks one question at a time.",
                  "Answer in the text box and press Send (or Enter).",
                  "After all questions, you get a detailed final evaluation.",
                ].map((step, i) => (
                  <div key={i} className="flex gap-3">
                    <span className="text-[#f5a94f] font-bold shrink-0">
                      {i + 1}.
                    </span>
                    <span>{step}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── Active Interview Chat ─── */}
      {started && (
        <div className="max-w-4xl mx-auto px-6 py-8">
          {/* Session Tag */}
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-[#e8ddd3]">
            <div>
              <p className="text-xs text-[#8a7060] font-bold uppercase tracking-widest mb-1">
                Active Session
              </p>
              <h2
                className="text-xl font-bold text-[#1a1007]"
                style={{ fontFamily: "'DM Serif Display', serif" }}
              >
                {sessionConfig.interviewType} Interview
              </h2>
              <span className="text-xs text-[#c2a882]">
                {sessionConfig.difficulty} · {sessionConfig.numQuestions}{" "}
                Questions
              </span>
            </div>
            {!completed && (
              <button
                onClick={handleReset}
                className="text-xs font-semibold text-[#c84b2f] hover:text-[#a33a20] bg-[#ffeee9] hover:bg-[#ffddd5] px-4 py-2 rounded-lg border border-[#f9c5b8] transition-all"
              >
                End Session
              </button>
            )}
            {completed && (
              <span className="bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-bold px-3 py-1.5 rounded-lg">
                ✅ Complete!
              </span>
            )}
          </div>

          {/* Messages */}
          <div className="space-y-5 mb-8">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fade-up`}
                style={{ animationDelay: `${i * 0.03}s` }}
              >
                {msg.role === "assistant" && (
                  <div className="w-7 h-7 rounded-full bg-[#f0e8e0] flex items-center justify-center mr-3 mt-1 shrink-0">
                    <img
                      src="/logo.png"
                      alt=""
                      className="w-4 h-4 object-contain"
                    />
                  </div>
                )}
                <div
                  className={`max-w-2xl px-5 py-4 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === "user"
                      ? "bg-[#1a1007] text-[#fdf8f3] rounded-tr-sm"
                      : "bg-white text-[#1a1007] border border-[#e8ddd3] rounded-tl-sm shadow-sm"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start animate-fade-in">
                <div className="w-7 h-7 rounded-full bg-[#f0e8e0] flex items-center justify-center mr-3 mt-1 shrink-0">
                  <img
                    src="/logo.png"
                    alt=""
                    className="w-4 h-4 object-contain"
                  />
                </div>
                <div className="bg-white border border-[#e8ddd3] px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm">
                  <div className="flex items-center gap-1.5">
                    {[0, 150, 300].map((d) => (
                      <span
                        key={d}
                        className="w-2 h-2 bg-[#c2a882] rounded-full animate-bounce"
                        style={{ animationDelay: `${d}ms` }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input box */}
          {!completed && (
            <div className="sticky bottom-6 bg-[#fdf8f3]/95 backdrop-blur-sm pt-4 border-t border-[#e8ddd3]">
              <div className="flex gap-3">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Type your answer… (Enter to send, Shift+Enter for new line)"
                  className="flex-1 bg-white border border-[#e0d5c8] rounded-xl px-4 py-3 text-sm text-[#1a1007] placeholder-[#c2b5a6] outline-none focus:border-[#1a1007] focus:ring-2 focus:ring-[#1a1007]/10 transition-all resize-none min-h-[52px] max-h-40"
                  rows={1}
                />
                <button
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                  className="bg-[#1a1007] hover:bg-[#2e1e10] disabled:bg-[#e0d5c8] disabled:text-[#c2b5a6] text-[#fdf8f3] font-bold px-6 rounded-xl transition-all text-sm shrink-0"
                >
                  Send
                </button>
              </div>
            </div>
          )}

          {/* Completion actions */}
          {completed && (
            <div className="flex flex-wrap gap-4 mt-4">
              <button
                onClick={handleReset}
                className="bg-[#1a1007] hover:bg-[#2e1e10] text-[#fdf8f3] font-bold px-6 py-3 rounded-xl text-sm transition-all"
              >
                🔄 New Interview
              </button>
              <button
                onClick={() => navigate("/history")}
                className="bg-white hover:bg-[#f0e8e0] text-[#1a1007] font-bold border border-[#e0d5c8] px-6 py-3 rounded-xl text-sm transition-all"
              >
                📚 View History
              </button>
              <button
                onClick={() => navigate("/results")}
                className="bg-white hover:bg-[#f0e8e0] text-[#1a1007] font-bold border border-[#e0d5c8] px-6 py-3 rounded-xl text-sm transition-all"
              >
                📊 View Results
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
