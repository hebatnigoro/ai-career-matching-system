"use client";

import { useEffect, useState } from "react";
import { Businessman } from "./businessman";
import { AnalyzerCard } from "./analyzer-card";
import { ResultsSection } from "./results-section";
import { analyzeCV, fetchHealth, API_BASE, type Health } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

type Mouse = { x: number; y: number };

const errBox: React.CSSProperties = {
  padding: "12px 16px",
  background: "#ffe4e4",
  border: "2.5px solid #d04848",
  borderRadius: 10,
  color: "#a02020",
  fontSize: 13,
  fontWeight: 700,
  boxShadow: "2px 2px 0 #1e1a3a",
  marginBottom: 16,
};

type ConnState =
  | { kind: "checking" }
  | { kind: "ok"; health: Health }
  | { kind: "error"; message: string };

function ConnectionBadge({ state, onRetry }: { state: ConnState; onRetry: () => void }) {
  const base: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 14px",
    borderRadius: 100,
    border: "2.5px solid #1e1a3a",
    boxShadow: "3px 3px 0 #1e1a3a",
    fontSize: 12,
    fontWeight: 800,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  };

  if (state.kind === "checking") {
    return (
      <div style={{ ...base, background: "var(--pd-card)", color: "#0EA5E9" }}>
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: "#94A3B8",
            display: "inline-block",
            animation: "pd-spin 0.9s linear infinite",
          }}
        />
        Cek koneksi backend…
      </div>
    );
  }
  if (state.kind === "ok") {
    return (
      <div style={{ ...base, background: "#D8F5E5", color: "#1e1a3a" }}>
        <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#4ECDC4", display: "inline-block" }} />
        Backend connected · {state.health.careers_loaded} karier dimuat
      </div>
    );
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
      <div style={{ ...base, background: "#ffe4e4", color: "#a02020" }}>
        <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#d04848", display: "inline-block" }} />
        Backend tidak terhubung
      </div>
      <div style={{ fontSize: 12, color: "#475569", textAlign: "center", maxWidth: 480, lineHeight: 1.6 }}>
        Tidak bisa menjangkau <code style={{ background: "#E0F2FE", padding: "1px 5px", borderRadius: 4 }}>{API_BASE}</code>. Pastikan FastAPI berjalan:
        <br />
        <code style={{ background: "#1e1a3a", color: "#fff", padding: "2px 8px", borderRadius: 4, display: "inline-block", marginTop: 6 }}>
          uvicorn api.user_api:app --reload
        </code>
        <br />
        <span style={{ fontSize: 11, color: "#64748B" }}>Detail: {state.message}</span>
        <br />
        <button
          onClick={onRetry}
          style={{
            marginTop: 8,
            padding: "5px 12px",
            background: "var(--pd-card)",
            border: "2px solid #1e1a3a",
            borderRadius: 8,
            fontSize: 11,
            fontWeight: 800,
            cursor: "pointer",
            fontFamily: "inherit",
            color: "var(--pd-text)",
            boxShadow: "2px 2px 0 #1e1a3a",
          }}
        >
          🔄 Coba lagi
        </button>
      </div>
    </div>
  );
}

export function AnalyzerSection({ mouse }: { mouse: Mouse }) {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [conn, setConn] = useState<ConnState>({ kind: "checking" });

  async function checkHealth() {
    setConn({ kind: "checking" });
    try {
      const health = await fetchHealth();
      setConn({ kind: "ok", health });
    } catch (e) {
      setConn({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  useEffect(() => {
    checkHealth();
  }, []);

  async function handleAnalyze(params: { file: File; targetCareerId: string; includeAiPlan: boolean }) {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await analyzeCV(params);
      setResult(res);
      setTimeout(() => window.scrollTo({ top: 0, behavior: "smooth" }), 100);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      // also re-check health in case the backend dropped
      checkHealth();
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setResult(null);
    setError("");
  }

  return (
    <div style={{ minHeight: "100vh", padding: "100px 24px 80px", position: "relative" }}>
      <div style={{ maxWidth: 680, margin: "0 auto", position: "relative", zIndex: 1 }}>
        <div style={{ textAlign: "center", marginBottom: 18 }}>
          <ConnectionBadge state={conn} onRetry={checkHealth} />
        </div>

        <div style={{ display: "flex", justifyContent: "center", marginBottom: 8, filter: "var(--pd-character-shadow-sm)" }}>
          <Businessman mouseX={mouse.x} mouseY={mouse.y} expression="focused" scale={0.6} />
        </div>

        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div
            style={{
              display: "inline-block",
              background: "var(--pd-card)",
              border: "2.5px solid #1e1a3a",
              borderRadius: 100,
              padding: "5px 18px",
              fontWeight: 800,
              fontSize: 11,
              color: "#0EA5E9",
              boxShadow: "3px 3px 0 #1e1a3a",
              marginBottom: 14,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            {result ? "🎉 Results Ready!" : "🔬 Career Analyzer"}
          </div>
          <h2 style={{ fontSize: 34, fontWeight: 900, letterSpacing: "-0.02em", marginBottom: 6, color: "var(--pd-text)" }}>
            {result ? "Your Career Analysis" : "Analyze Your CV"}
          </h2>
          <p style={{ fontSize: 14, color: "var(--pd-text-muted)" }}>
            {result
              ? "Hasil analisis karier kamu di bawah ini"
              : "Upload CV dan tulis karier target · Instant AI analysis"}
          </p>
        </div>

        {error && <div style={errBox}>{error}</div>}

        <div style={{ background: "var(--pd-card)", color: "var(--pd-text)", border: "2.5px solid #1e1a3a", borderRadius: 24, boxShadow: "6px 6px 0 #1e1a3a", padding: "30px" }}>
          {!result ? (
            <AnalyzerCard loading={loading} onAnalyze={handleAnalyze} />
          ) : (
            <ResultsSection result={result} onReset={handleReset} />
          )}
        </div>
      </div>
    </div>
  );
}
