import { useLocation, useNavigate } from "react-router-dom";

export default function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const username = localStorage.getItem("username");

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    navigate("/login");
  };

  const navLinks = [
    { path: "/", label: "Interview" },
    { path: "/history", label: "History" },
    { path: "/results", label: "Results" },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#fdf8f3]">
      {/* ─── Sticky Top Navigation ─── */}
      <header className="sticky top-0 z-50 bg-[#fdf8f3]/95 backdrop-blur-sm border-b border-[#e8ddd3]">
        <nav className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-8">
          {/* Logo */}
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2.5 shrink-0 group"
          >
            <img
              src="/logo.png"
              alt="HireReady"
              className="w-8 h-8 object-contain group-hover:scale-105 transition-transform"
            />
            <span className="text-lg font-bold text-[#1a1007] tracking-tight font-serif-display">
              HireReady
            </span>
          </button>

          {/* Center Nav Links */}
          <div className="flex items-center gap-1">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.path;
              return (
                <button
                  key={link.path}
                  onClick={() => navigate(link.path)}
                  className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                    isActive
                      ? "bg-[#1a1007] text-[#fdf8f3]"
                      : "text-[#6b584a] hover:text-[#1a1007] hover:bg-[#f0e8e0]"
                  }`}
                >
                  {link.label}
                </button>
              );
            })}
          </div>

          {/* Right: User + Logout */}
          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() => navigate("/profile")}
              className="flex items-center gap-2 text-sm font-semibold text-[#6b584a] hover:text-[#1a1007] transition-colors"
            >
              {localStorage.getItem("profilePhoto") ? (
                <img
                  src={localStorage.getItem("profilePhoto")}
                  alt={username}
                  className="w-8 h-8 rounded-full object-cover border-2 border-[#e8ddd3] hover:border-[#1a1007] transition-all"
                />
              ) : (
                <div className="w-8 h-8 rounded-full bg-[#1a1007] text-[#fdf8f3] flex items-center justify-center font-bold text-sm">
                  {username ? username[0].toUpperCase() : "U"}
                </div>
              )}
              <span className="hidden sm:inline">{username}</span>
            </button>
            <button
              onClick={handleLogout}
              className="text-xs font-semibold text-[#c84b2f] hover:text-[#a33a20] bg-[#ffeee9] hover:bg-[#ffddd5] px-3 py-1.5 rounded-lg transition-all border border-[#f9c5b8]"
            >
              Logout
            </button>
          </div>
        </nav>
      </header>

      {/* ─── Page Content ─── */}
      <main className="flex-1">{children}</main>
    </div>
  );
}
