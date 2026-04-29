"use client";

import { useEffect, useRef, useState } from "react";
import { Navbar, type PageId } from "@/components/navbar";
import { HomeSection } from "@/components/home-section";
import { HowSection } from "@/components/how-section";
import { AnalyzerSection } from "@/components/analyzer-section";

export default function App() {
  const [page, setPage] = useState<PageId>("home");
  const [display, setDisplay] = useState<PageId>("home");
  const [anim, setAnim] = useState<string>("");
  const busy = useRef(false);
  const [mouse, setMouse] = useState({ x: 0, y: 0 });

  useEffect(() => {
    setMouse({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
    const fn = (e: MouseEvent) => setMouse({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", fn);
    return () => window.removeEventListener("mousemove", fn);
  }, []);

  function navigate(to: PageId) {
    if (to === page || busy.current) return;
    busy.current = true;
    setAnim("pd-page-exit");
    setTimeout(() => {
      setDisplay(to);
      setPage(to);
      setAnim("pd-page-enter");
      setTimeout(() => {
        busy.current = false;
      }, 450);
    }, 250);
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
      <Navbar page={page} navigate={navigate} />
      <div className={anim} key={display} style={{ flex: 1 }}>
        {display === "home" && <HomeSection navigate={navigate} mouse={mouse} />}
        {display === "how" && <HowSection navigate={navigate} mouse={mouse} />}
        {display === "analyzer" && <AnalyzerSection mouse={mouse} />}
      </div>
      <footer
        style={{
          borderTop: "2.5px solid #1e1a3a",
          padding: "26px 24px",
          textAlign: "center",
          background: "#fff",
          boxShadow: "inset 0 4px 0 #F1F5F9",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 6 }}>
          <span style={{ fontSize: 18 }}>🧭</span>
          <span style={{ fontWeight: 900, fontSize: 15 }}>
            Path<span style={{ color: "#0EA5E9" }}>Drift</span>
          </span>
          <span style={{ color: "#ccc" }}>·</span>
          <span style={{ color: "#64748B", fontWeight: 600, fontSize: 13 }}>AI Career Matching System</span>
        </div>
        <p style={{ color: "#94A3B8", fontSize: 12, fontWeight: 600 }}>
          Built for thesis research · BERT Embeddings · FastAPI · Gemini AI
        </p>
      </footer>
    </div>
  );
}
