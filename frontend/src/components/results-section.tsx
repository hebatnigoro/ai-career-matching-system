"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CBtn } from "./cbtn";
import type { AnalyzeResponse } from "@/lib/types";

function pct(score: number | null | undefined) {
  if (score === null || score === undefined) return 0;
  return Math.round(score * 100);
}

function scoreColor(p: number) {
  return p >= 70 ? "#4ECDC4" : p >= 40 ? "#f0b86e" : "#e07a7a";
}

const sectionLabel: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 800,
  color: "var(--pd-text)",
  marginBottom: 10,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const pillBase: React.CSSProperties = {
  padding: "6px 12px",
  border: "2px solid #1e1a3a",
  borderRadius: 100,
  fontSize: 12,
  fontWeight: 700,
  color: "#1e1a3a",
  boxShadow: "2px 2px 0 #1e1a3a",
};

type Props = {
  result: AnalyzeResponse;
  onReset: () => void;
};

export function ResultsSection({ result, onReset }: Props) {
  const score = pct(result.target.score);
  const altScore = pct(result.best_alternative.score);
  const matched = result.skill_gap.matched_skills;
  const missing = result.skill_gap.missing_skills;
  const summary =
    result.transition_context?.summary ||
    result.rationale ||
    `Skor kecocokan untuk posisi ${result.target.title} adalah ${score}%.`;

  return (
    <div style={{ animation: "pd-fadeUp 0.45s ease both" }}>
      {/* Score circle */}
      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 140,
            height: 140,
            borderRadius: "50%",
            background: scoreColor(score),
            border: "2.5px solid #1e1a3a",
            boxShadow: "5px 5px 0 #1e1a3a",
            marginBottom: 14,
          }}
        >
          <div>
            <div style={{ fontSize: 36, fontWeight: 900, color: "#1e1a3a", lineHeight: 1 }}>{score}%</div>
            <div style={{ fontSize: 10, fontWeight: 800, color: "#1e1a3a", textTransform: "uppercase", letterSpacing: "0.08em", marginTop: 4 }}>
              Match
            </div>
          </div>
        </div>
        <div style={{ fontSize: 16, fontWeight: 900, color: "var(--pd-text)" }}>
          {result.target.title}
          {result.target.field && (
            <span style={{ marginLeft: 8, fontSize: 11, fontWeight: 800, color: "#0EA5E9", background: "var(--pd-card)", border: "2px solid #1e1a3a", borderRadius: 100, padding: "2px 10px", boxShadow: "2px 2px 0 #1e1a3a", verticalAlign: "middle" }}>
              {result.target.field}
            </span>
          )}
        </div>
        <div style={{ marginTop: 6, fontSize: 12, fontWeight: 700, color: "#0EA5E9", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Status: {result.status}
        </div>
        {result.target.resolved_from && (
          <div style={{ marginTop: 6, fontSize: 11, color: "var(--pd-text-faint)" }}>
            Input <code style={{ background: "#E0F2FE", padding: "1px 6px", borderRadius: 4, color: "#1e1a3a" }}>{result.target.resolved_from}</code> →{" "}
            <code style={{ background: "#E0F2FE", padding: "1px 6px", borderRadius: 4, color: "#1e1a3a" }}>{result.target.career_id}</code>
          </div>
        )}
      </div>

      {/* Summary */}
      <div style={{ padding: 16, background: "#E0F2FE", border: "2.5px solid #1e1a3a", borderRadius: 12, boxShadow: "3px 3px 0 #1e1a3a", marginBottom: 20, fontSize: 14, color: "#1e1a3a", lineHeight: 1.6 }}>
        {summary}
      </div>

      {/* Best alternative */}
      {result.best_alternative.career_id && result.best_alternative.career_id !== result.target.career_id && (
        <div style={{ padding: 14, background: "#FFF3CD", border: "2.5px solid #FFD93D", borderRadius: 12, boxShadow: "3px 3px 0 #FFD93D", marginBottom: 20 }}>
          <div style={{ fontSize: 11, fontWeight: 800, color: "#1e1a3a", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
            🌟 Alternatif terbaik
          </div>
          <div style={{ fontSize: 15, fontWeight: 900, color: "#1e1a3a" }}>
            {result.best_alternative.title}
            {result.best_alternative.field && (
              <span style={{ marginLeft: 8, fontSize: 11, fontWeight: 800, color: "#0EA5E9" }}>[{result.best_alternative.field}]</span>
            )}
            <span style={{ marginLeft: 10, fontSize: 13, fontWeight: 800, color: "#1e1a3a" }}>· {altScore}%</span>
          </div>
        </div>
      )}

      {/* Matched skills */}
      {matched.length > 0 && (
        <div style={{ marginBottom: 18 }}>
          <div style={sectionLabel}>✅ Skills yang sudah cocok ({matched.length})</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {matched.map((s) => (
              <span key={s.skill} style={{ ...pillBase, background: "#D8F5E5" }}>
                {s.skill}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Missing skills */}
      {missing.length > 0 && (
        <div style={{ marginBottom: 18 }}>
          <div style={sectionLabel}>⚠️ Skill gap ({missing.length})</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {missing.map((s) => (
              <span key={s.skill} style={{ ...pillBase, background: "#ffe4d6" }}>
                {s.skill}
                {s.type === "upgrade" && " (upgrade)"}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Top 5 rankings as roadmap-style cards */}
      {result.rankings.length > 0 && (
        <div style={{ marginBottom: 22 }}>
          <div style={sectionLabel}>🏆 Top {result.rankings.length} Karier yang Cocok</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {result.rankings.map((r) => (
              <div
                key={r.career_id}
                style={{
                  display: "flex",
                  gap: 12,
                  padding: 14,
                  background: r.career_id === result.target.career_id ? "#E0F2FE" : "var(--pd-bg-soft)",
                  border: "2.5px solid #1e1a3a",
                  borderRadius: 12,
                  boxShadow: "3px 3px 0 #1e1a3a",
                  alignItems: "center",
                }}
              >
                <div
                  style={{
                    flexShrink: 0,
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    background: "#0EA5E9",
                    color: "#fff",
                    border: "2.5px solid #1e1a3a",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 900,
                    fontSize: 13,
                  }}
                >
                  {r.rank}
                </div>
                <div style={{ flex: 1 }}>
                  {/* Top-row card uses light blue background (E0F2FE) so its
                      text stays #1e1a3a; alternates pick up the themed
                      pd-bg-soft surface and themed text. */}
                  <div style={{ fontSize: 14, fontWeight: 900, color: r.career_id === result.target.career_id ? "#1e1a3a" : "var(--pd-text)" }}>
                    {r.title}
                    {r.field && (
                      <span style={{ marginLeft: 8, fontSize: 10, fontWeight: 800, color: "#0EA5E9" }}>[{r.field}]</span>
                    )}
                  </div>
                </div>
                <div style={{ fontSize: 13, fontFamily: "ui-monospace, monospace", fontWeight: 800, color: r.career_id === result.target.career_id ? "#1e1a3a" : "var(--pd-text)" }}>
                  {pct(r.score)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Plan */}
      {result.ai_plan && (
        <div style={{ marginBottom: 22, padding: 20, background: "var(--pd-card)", border: "2.5px solid #1e1a3a", borderRadius: 16, boxShadow: "5px 5px 0 #1e1a3a" }}>
          <div style={{ fontSize: 13, fontWeight: 900, color: "var(--pd-text)", marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 18 }}>✨</span> AI Career Plan {"text" in result.ai_plan && (
              <span style={{ fontSize: 10, fontWeight: 700, color: "var(--pd-text-faint)", marginLeft: 4 }}>· {result.ai_plan.model}</span>
            )}
          </div>

          {"error" in result.ai_plan ? (
            <div
              style={{
                padding: "10px 14px",
                background: "#ffe4e4",
                border: "2.5px solid #d04848",
                borderRadius: 10,
                color: "#a02020",
                fontSize: 13,
                fontWeight: 700,
                boxShadow: "2px 2px 0 #1e1a3a",
              }}
            >
              {result.ai_plan.error}
            </div>
          ) : (
            <>
              <div className="pd-md" style={{ fontSize: 14, color: "var(--pd-text)" }}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.ai_plan.text}</ReactMarkdown>
              </div>

              {result.ai_plan.sources && result.ai_plan.sources.length > 0 && (
                <div style={{ marginTop: 16, paddingTop: 14, borderTop: "2px dashed #FFD6A5" }}>
                  <div style={{ ...sectionLabel, marginBottom: 8 }}>🔗 Sumber (Google Search)</div>
                  <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
                    {result.ai_plan.sources.map((s, i) => (
                      <li key={i}>
                        <a
                          href={s.uri}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ color: "#0EA5E9", textDecoration: "underline", fontSize: 12, wordBreak: "break-all" }}
                        >
                          {s.title || s.uri}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      )}

      <div style={{ textAlign: "center" }}>
        <CBtn onClick={onReset}>🔄 Analisis Ulang</CBtn>
      </div>
    </div>
  );
}
