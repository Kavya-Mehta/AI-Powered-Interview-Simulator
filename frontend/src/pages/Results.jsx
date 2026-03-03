import { useEffect, useMemo, useState } from "react";
import { getDashboardInsights } from "../api";

function RadarChart({ areas }) {
  const size = 420;
  const center = size / 2;
  const radius = 110;
  const levels = [20, 40, 60, 80, 100];
  const baseline = 60;

  const splitLabel = (label) => {
    const parts = label.split(" ");
    if (parts.length <= 1) return [label];
    if (parts.length === 2) return parts;
    return [parts.slice(0, 2).join(" "), parts.slice(2).join(" ")];
  };

  const pointsFor = (values) => {
    const count = values.length;
    return values
      .map((value, idx) => {
        const angle = (Math.PI * 2 * idx) / count - Math.PI / 2;
        const r = (Math.max(0, Math.min(100, value)) / 100) * radius;
        const x = center + r * Math.cos(angle);
        const y = center + r * Math.sin(angle);
        return `${x},${y}`;
      })
      .join(" ");
  };

  const baselineValues = useMemo(() => areas.map(() => baseline), [areas]);

  return (
    <div className="bg-white border border-[#e8ddd3] rounded-2xl p-6 shadow-sm">
      <p className="text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-4">
        Relative performance (% from baseline)
      </p>
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="w-full max-w-[420px] mx-auto"
      >
        {levels.map((level) => (
          <polygon
            key={level}
            points={pointsFor(areas.map(() => level))}
            fill="none"
            stroke="#e8ddd3"
            strokeWidth="1"
          />
        ))}

        {areas.map((_, idx) => {
          const angle = (Math.PI * 2 * idx) / areas.length - Math.PI / 2;
          const x = center + radius * Math.cos(angle);
          const y = center + radius * Math.sin(angle);
          return (
            <line
              key={idx}
              x1={center}
              y1={center}
              x2={x}
              y2={y}
              stroke="#e8ddd3"
              strokeWidth="1"
            />
          );
        })}

        <polygon
          points={pointsFor(baselineValues)}
          fill="rgba(37,99,235,0.12)"
          stroke="#2563eb"
          strokeWidth="2"
        />

        <polygon
          points={pointsFor(areas.map((a) => a.score))}
          fill="rgba(16,185,129,0.20)"
          stroke="#10b981"
          strokeWidth="2"
        />

        {areas.map((area, idx) => {
          const angle = (Math.PI * 2 * idx) / areas.length - Math.PI / 2;
          const labelRadius = radius + 38;
          const x = center + labelRadius * Math.cos(angle);
          const y = center + labelRadius * Math.sin(angle);
          const textAnchor =
            x < center - 10 ? "end" : x > center + 10 ? "start" : "middle";
          const labelLines = splitLabel(area.name);

          return (
            <text
              key={area.name}
              x={x}
              y={y}
              textAnchor={textAnchor}
              fontSize="12"
              fontWeight="700"
              fill="#6b584a"
            >
              {labelLines.map((line, lineIndex) => (
                <tspan
                  key={`${area.name}-${lineIndex}`}
                  x={x}
                  dy={lineIndex === 0 ? "0" : "1.1em"}
                >
                  {line}
                </tspan>
              ))}
            </text>
          );
        })}
      </svg>

      <div className="flex items-center justify-center gap-6 mt-4 text-xs text-[#6b584a]">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-[#2563eb]/30 border border-[#2563eb]" />
          Baseline
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-[#10b981]/30 border border-[#10b981]" />
          Your performance
        </div>
      </div>
    </div>
  );
}

export default function Results() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        const res = await getDashboardInsights();
        setData(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboard();
  }, []);

  const topArea = useMemo(() => {
    if (!data?.areas?.length) return null;
    return [...data.areas].sort((a, b) => b.score - a.score)[0];
  }, [data]);

  const lowArea = useMemo(() => {
    if (!data?.areas?.length) return null;
    return [...data.areas].sort((a, b) => a.score - b.score)[0];
  }, [data]);

  return (
    <div className="bg-[#fdf8f3] min-h-screen">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="mb-10">
          <p className="text-[#c2a882] text-xs font-bold uppercase tracking-widest mb-3">
            Post Interview
          </p>
          <h1
            className="text-5xl lg:text-6xl text-[#1a1007] leading-tight mb-4"
            style={{ fontFamily: "'DM Serif Display', serif" }}
          >
            Results &<br />
            <em className="not-italic text-[#c84b2f]">Dashboard.</em>
          </h1>
          <p className="text-[#8a7060] text-base max-w-3xl">
            Review what you are good at and where to focus next (LeetCode,
            Behavioral Questions, System Design and more).
          </p>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-24">
            <div className="flex gap-1.5">
              {[0, 150, 300].map((d) => (
                <span
                  key={d}
                  className="w-2.5 h-2.5 bg-[#c2a882] rounded-full animate-bounce"
                  style={{ animationDelay: `${d}ms` }}
                />
              ))}
            </div>
          </div>
        )}

        {!loading && (!data || !data.has_data) && (
          <div className="bg-white border border-[#e8ddd3] rounded-2xl p-14 text-center shadow-sm">
            <div className="text-6xl mb-4">📊</div>
            <h2
              className="text-2xl font-bold text-[#1a1007] mb-2"
              style={{ fontFamily: "'DM Serif Display', serif" }}
            >
              No completed interview yet.
            </h2>
            <p className="text-[#8a7060]">
              Complete an interview and come back here to see personalized
              review and recommendations.
            </p>
          </div>
        )}

        {!loading && data?.has_data && (
          <div className="grid lg:grid-cols-[1.2fr_1fr] gap-6 animate-fade-up">
            <RadarChart areas={data.areas} />

            <div className="space-y-4">
              <div className="bg-white border border-[#e8ddd3] rounded-2xl p-5 shadow-sm">
                <div className="text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-2">
                  Quick read
                </div>
                <p className="text-sm text-[#6b584a] leading-relaxed">
                  {data.summary}
                </p>
                <div className="grid grid-cols-2 gap-3 mt-4">
                  <div className="bg-[#f8f3ee] border border-[#e8ddd3] rounded-xl p-3">
                    <div className="text-[11px] uppercase tracking-widest text-[#8a7060] font-bold">
                      Good at
                    </div>
                    <div className="text-sm font-bold text-emerald-700 mt-1">
                      {topArea?.name || "-"}
                    </div>
                  </div>
                  <div className="bg-[#f8f3ee] border border-[#e8ddd3] rounded-xl p-3">
                    <div className="text-[11px] uppercase tracking-widest text-[#8a7060] font-bold">
                      Needs focus
                    </div>
                    <div className="text-sm font-bold text-[#c84b2f] mt-1">
                      {lowArea?.name || "-"}
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-[#e8ddd3] rounded-2xl p-5 shadow-sm">
                <div className="text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-3">
                  Recommendations
                </div>
                <ul className="space-y-2">
                  {data.recommendations.map((item, idx) => (
                    <li
                      key={idx}
                      className="text-sm text-[#6b584a] leading-relaxed bg-[#fdf8f3] border border-[#f0e8e0] rounded-lg px-3 py-2"
                    >
                      {item}
                    </li>
                  ))}
                </ul>
                <div className="mt-4 text-xs text-[#8a7060] leading-relaxed bg-[#fdf8f3] border border-[#f0e8e0] rounded-lg px-3 py-2">
                  Scores are estimated from your completed interview feedback
                  text by matching strength/improvement keywords for each area
                  (LeetCode, Behavioral, System Design, Communication, Problem
                  Solving). They are directional guidance, not exact grading.
                </div>
              </div>
            </div>

            <div className="bg-white border border-[#e8ddd3] rounded-2xl p-5 shadow-sm lg:col-span-2">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <div className="text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-3">
                    What you are good at
                  </div>
                  <ul className="space-y-2">
                    {data.strengths.map((item, idx) => (
                      <li
                        key={idx}
                        className="text-sm text-emerald-700 bg-emerald-50 border border-emerald-100 rounded-lg px-3 py-2 font-medium"
                      >
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <div className="text-xs font-bold text-[#8a7060] uppercase tracking-widest mb-3">
                    Where to improve
                  </div>
                  <ul className="space-y-2">
                    {data.improvements.map((item, idx) => (
                      <li
                        key={idx}
                        className="text-sm text-[#c84b2f] bg-[#ffeee9] border border-[#f9c5b8] rounded-lg px-3 py-2 font-medium"
                      >
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
