"use client";

import { BgBlobs } from "./bg-blobs";
import { CBtn } from "./cbtn";
import type { PageId } from "./navbar";

const steps: { icon: string; color: string; border: string; title: string; sub: string; desc: string; num: string }[] = [
  { icon: "📄", color: "#ede9ff", border: "#7c6fe0", title: "Upload Your CV", sub: "Unggah CV kamu", desc: "Upload PDF or DOCX. Bilingual CVs (EN & ID) are fully supported via multilingual-e5 embeddings.", num: "01" },
  { icon: "🎯", color: "#e6faf7", border: "#6dd5c0", title: "Choose Target Career", sub: "Pilih karier target", desc: "Browse career profiles grouped by field. The backend resolver also accepts loose input like 'data analyst' or 'frontend dev'.", num: "02" },
  { icon: "🤖", color: "#fff0f4", border: "#ff9bb5", title: "Deep Analysis", sub: "Analisis mendalam", desc: "BERT multilingual embeddings compute semantic similarity. Drift detection scores your alignment across all careers.", num: "03" },
  { icon: "🗺️", color: "#fff8e6", border: "#ffd166", title: "Follow Your Roadmap", sub: "Ikuti rencana belajar", desc: "Get a personalized AI learning plan with skill gap breakdown, interview tips, and curated resources via Gemini.", num: "04" },
];

const stack: [string, string, string][] = [
  ["🧠", "BERT Embeddings", "multilingual-e5-base semantic CV analysis"],
  ["📐", "Cosine Similarity", "Vector-based career matching & drift detection"],
  ["✨", "Gemini AI", "Personalized learning plans with real sources"],
  ["⚡", "FastAPI", "Python backend with cached career embeddings"],
];

export function HowSection({ navigate }: { navigate: (to: PageId) => void }) {
  return (
    <div style={{ minHeight: "100vh", padding: "100px 24px 80px", position: "relative" }}>
      <BgBlobs />
      <div style={{ maxWidth: 820, margin: "0 auto", position: "relative", zIndex: 1 }}>
        <div style={{ textAlign: "center", marginBottom: 48 }}>
          <div style={{ display: "inline-block", background: "#fff", border: "2.5px solid #1e1a3a", borderRadius: 100, padding: "5px 18px", fontWeight: 800, fontSize: 11, color: "#7c6fe0", boxShadow: "3px 3px 0 #1e1a3a", marginBottom: 16, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Simple Process · Proses Mudah
          </div>
          <h2 style={{ fontSize: 40, fontWeight: 900, letterSpacing: "-0.03em", marginBottom: 8 }}>
            How It <span style={{ color: "#7c6fe0" }}>Works</span>
          </h2>
          <p style={{ fontSize: 15, color: "#7a789a", maxWidth: 460, margin: "0 auto" }}>
            4 steps from CV to career clarity · 4 langkah menuju karier impian
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18, marginBottom: 48 }}>
          {steps.map((s, i) => (
            <div key={i} style={{ display: "flex", gap: 24, alignItems: "flex-start", background: s.color, border: `2.5px solid ${s.border}`, borderRadius: 20, padding: "26px 28px", boxShadow: `5px 5px 0 ${s.border}`, animation: `pd-stepSlide 0.45s ${i * 0.1}s both` }}>
              <div style={{ textAlign: "center", flexShrink: 0, width: 60 }}>
                <div style={{ fontSize: 40, marginBottom: 6 }}>{s.icon}</div>
                <div style={{ fontSize: 12, fontWeight: 900, color: s.border, background: "#fff", border: `2px solid ${s.border}`, borderRadius: 7, padding: "2px 7px", display: "inline-block", boxShadow: `2px 2px 0 ${s.border}` }}>{s.num}</div>
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: 19, fontWeight: 900, marginBottom: 3 }}>{s.title}</h3>
                <p style={{ fontSize: 13, fontWeight: 700, color: s.border, marginBottom: 8 }}>{s.sub}</p>
                <p style={{ fontSize: 14, color: "#5a587a", lineHeight: 1.7, margin: 0 }}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <div style={{ background: "#fff", border: "2.5px solid #1e1a3a", borderRadius: 20, padding: "26px 30px", boxShadow: "5px 5px 0 #1e1a3a", marginBottom: 40 }}>
          <h3 style={{ fontSize: 17, fontWeight: 900, marginBottom: 16 }}>🛠️ Under the Hood</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(168px,1fr))", gap: 12 }}>
            {stack.map(([ic, t, d], i) => (
              <div key={i} style={{ background: "#f0ecff", border: "2px solid #1e1a3a", borderRadius: 12, padding: "14px", boxShadow: "2px 2px 0 #1e1a3a" }}>
                <div style={{ fontSize: 22, marginBottom: 5 }}>{ic}</div>
                <div style={{ fontWeight: 800, fontSize: 13, marginBottom: 4 }}>{t}</div>
                <div style={{ fontSize: 12, color: "#9896b8", lineHeight: 1.5 }}>{d}</div>
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
