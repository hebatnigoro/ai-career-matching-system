"use client";

import { useEffect, useState } from "react";
import { Businessman } from "./businessman";
import { BgBlobs } from "./bg-blobs";
import { CBtn } from "./cbtn";
import type { PageId } from "./navbar";

type Props = {
  navigate: (to: PageId) => void;
  mouse: { x: number; y: number };
};

export function HomeSection({ navigate, mouse }: Props) {
  const [in_, setIn_] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setIn_(true), 60);
    return () => clearTimeout(t);
  }, []);
  const T = (delay: number): React.CSSProperties => ({
    opacity: in_ ? 1 : 0,
    transform: in_ ? "none" : "translateY(16px)",
    transition: `all 0.55s ease ${delay}ms`,
  });

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", position: "relative", padding: "100px 24px 60px" }}>
      <BgBlobs />
      <div style={{ maxWidth: 920, width: "100%", display: "flex", alignItems: "center", gap: 56, flexWrap: "wrap", justifyContent: "center", position: "relative", zIndex: 1 }}>
        <div style={{ opacity: in_ ? 1 : 0, transform: in_ ? "none" : "scale(0.8)", transition: "all 0.65s cubic-bezier(0.34,1.4,0.64,1) 0.1s", filter: "drop-shadow(5px 8px 0px #b8b0e0)" }}>
          <Businessman mouseX={mouse.x} mouseY={mouse.y} />
        </div>
        <div style={{ flex: 1, minWidth: 270, maxWidth: 460 }}>
          <div style={{ ...T(200), display: "inline-flex", alignItems: "center", gap: 8, background: "#fff", border: "2.5px solid #1e1a3a", borderRadius: 100, padding: "5px 16px", fontSize: 11, fontWeight: 800, color: "#7c6fe0", boxShadow: "3px 3px 0 #1e1a3a", marginBottom: 20, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#6dd5c0", display: "inline-block" }} />
            AI-Powered Career Analysis
          </div>
          <h1 style={{ ...T(320), fontSize: "clamp(36px,5vw,56px)", fontWeight: 900, lineHeight: 1.1, letterSpacing: "-0.03em", marginBottom: 16, color: "#1e1a3a" }}>
            Find Your<br />
            <span style={{ color: "#7c6fe0" }}>Perfect Career</span><br />
            Path
          </h1>
          <p style={{ ...T(420), fontSize: 15, color: "#7a789a", lineHeight: 1.75, marginBottom: 6 }}>
            Upload your CV, pick a target career, and get an instant AI-powered match score, skill gap breakdown, and learning roadmap.
          </p>
          <p style={{ ...T(460), fontSize: 13, color: "#b0aece", fontWeight: 600, marginBottom: 34 }}>
            Unggah CV kamu dan temukan jalur karier terbaik.
          </p>
          <div style={{ ...T(540), display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 36 }}>
            <CBtn big onClick={() => navigate("analyzer")}>🚀 Analyze My CV</CBtn>
            <CBtn big onClick={() => navigate("how")} accent="#fff" textColor="#1e1a3a">📖 How It Works</CBtn>
          </div>
          <div style={{ ...T(640), display: "flex", gap: 12, flexWrap: "wrap" }}>
            {([["50+", "Career Profiles", "💼"], ["BERT", "Embeddings", "🤖"], ["Gemini", "AI Plans", "✨"]] as const).map(([v, l, ic], i) => (
              <div key={i} style={{ background: "#fff", border: "2.5px solid #1e1a3a", borderRadius: 14, padding: "10px 16px", boxShadow: "3px 3px 0 #1e1a3a", textAlign: "center", minWidth: 84 }}>
                <div style={{ fontSize: 18, marginBottom: 2 }}>{ic}</div>
                <div style={{ fontWeight: 900, fontSize: 16, color: "#7c6fe0" }}>{v}</div>
                <div style={{ fontSize: 10, color: "#9896b8", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.04em" }}>{l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
