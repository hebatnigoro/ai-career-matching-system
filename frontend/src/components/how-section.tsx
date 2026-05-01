"use client";

import { Businessman } from "./businessman";
import { CBtn } from "./cbtn";
import type { PageId } from "./navbar";

type Mouse = { x: number; y: number };

const steps: { icon: string; color: string; border: string; title: string; sub: string; desc: string; num: string }[] = [
  { icon: "📄", color: "#E0F2FE", border: "#0EA5E9", title: "Upload Your CV", sub: "Unggah CV kamu", desc: "Upload PDF or DOCX. Bilingual CVs (EN & ID) are fully supported via multilingual-e5 embeddings.", num: "01" },
  { icon: "🎯", color: "#E5FFF8", border: "#4ECDC4", title: "Choose Target Career", sub: "Pilih karier target", desc: "Browse career profiles grouped by field. The backend resolver also accepts loose input like 'data analyst' or 'frontend dev'.", num: "02" },
  { icon: "🤖", color: "#FEF3C7", border: "#F59E0B", title: "Deep Analysis", sub: "Analisis mendalam", desc: "BERT multilingual embeddings compute semantic similarity. Drift detection scores your alignment across all careers.", num: "03" },
  { icon: "🗺️", color: "#FFF3CD", border: "#FFD93D", title: "Follow Your Roadmap", sub: "Ikuti rencana belajar", desc: "Get a personalized AI learning plan with skill gap breakdown, interview tips, and curated resources via Gemini.", num: "04" },
];

const stack: [string, string, string][] = [
  ["🧠", "BERT Embeddings", "multilingual-e5-base semantic CV analysis"],
  ["📐", "Cosine Similarity", "Vector-based career matching & drift detection"],
  ["✨", "Gemini AI", "Personalized learning plans with real sources"],
  ["⚡", "FastAPI", "Python backend with cached career embeddings"],
];

export function HowSection({ navigate, mouse }: { navigate: (to: PageId) => void; mouse: Mouse }) {
  return (
    <div style={{ minHeight: "100vh", padding: "100px 24px 80px", position: "relative" }}>
      <div style={{ maxWidth: 820, margin: "0 auto", position: "relative", zIndex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 24, flexWrap: "wrap", marginBottom: 48 }}>
          <div style={{ filter: "var(--pd-character-shadow-sm)" }}>
            <Businessman mouseX={mouse.x} mouseY={mouse.y} expression="thinking" scale={0.75} />
          </div>
          <div style={{ textAlign: "center", maxWidth: 480 }}>
            <div style={{ display: "inline-block", background: "var(--pd-card)", border: "2.5px solid #1e1a3a", borderRadius: 100, padding: "5px 18px", fontWeight: 800, fontSize: 11, color: "#0EA5E9", boxShadow: "3px 3px 0 #1e1a3a", marginBottom: 16, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Simple Process · Proses Mudah
            </div>
            <h2 style={{ fontSize: 40, fontWeight: 900, letterSpacing: "-0.03em", marginBottom: 8, color: "var(--pd-text)" }}>
              How It <span style={{ color: "#0EA5E9" }}>Works</span>
            </h2>
            <p style={{ fontSize: 15, color: "var(--pd-text-muted)", margin: "0 auto" }}>
              4 steps from CV to career clarity · 4 langkah menuju karier impian
            </p>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18, marginBottom: 48 }}>
          {steps.map((s, i) => (
            <div key={i} style={{ display: "flex", gap: 24, alignItems: "flex-start", background: s.color, border: `2.5px solid ${s.border}`, borderRadius: 20, padding: "26px 28px", boxShadow: `5px 5px 0 ${s.border}`, animation: `pd-stepSlide 0.45s ${i * 0.1}s both` }}>
              <div style={{ textAlign: "center", flexShrink: 0, width: 60 }}>
                <div style={{ fontSize: 40, marginBottom: 6 }}>{s.icon}</div>
                <div style={{ fontSize: 12, fontWeight: 900, color: s.border, background: "#fff", border: `2px solid ${s.border}`, borderRadius: 7, padding: "2px 7px", display: "inline-block", boxShadow: `2px 2px 0 ${s.border}` }}>{s.num}</div>
              </div>
              {/* Step cards keep their pastel-on-light backgrounds in both
                  themes — readability of #1e1a3a text on #E0F2FE / #FEF3C7 is
                  fine, and converting them would lose the cartoon feel. */}
              <div style={{ flex: 1, color: "#1e1a3a" }}>
                <h3 style={{ fontSize: 19, fontWeight: 900, marginBottom: 3 }}>{s.title}</h3>
                <p style={{ fontSize: 13, fontWeight: 700, color: s.border, marginBottom: 8 }}>{s.sub}</p>
                <p style={{ fontSize: 14, color: "#334155", lineHeight: 1.7, margin: 0 }}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <div style={{ background: "var(--pd-card)", border: "2.5px solid #1e1a3a", borderRadius: 20, padding: "26px 30px", boxShadow: "5px 5px 0 #1e1a3a", marginBottom: 40 }}>
          <h3 style={{ fontSize: 17, fontWeight: 900, marginBottom: 16, color: "var(--pd-text)" }}>🛠️ Under the Hood</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(168px,1fr))", gap: 12 }}>
            {stack.map(([ic, t, d], i) => (
              <div key={i} style={{ background: "var(--pd-card-soft)", border: "2px solid #1e1a3a", borderRadius: 12, padding: "14px", boxShadow: "2px 2px 0 #1e1a3a" }}>
                <div style={{ fontSize: 22, marginBottom: 5 }}>{ic}</div>
                <div style={{ fontWeight: 800, fontSize: 13, marginBottom: 4, color: "var(--pd-text)" }}>{t}</div>
                <div style={{ fontSize: 12, color: "var(--pd-text-faint)", lineHeight: 1.5 }}>{d}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ textAlign: "center" }}>
          <CBtn big onClick={() => navigate("analyzer")}>🚀 Start Analyzing Now →</CBtn>
        </div>
      </div>
    </div>
  );
}
