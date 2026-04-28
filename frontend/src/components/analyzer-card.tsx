"use client";

import { useState } from "react";
import { CBtn } from "./cbtn";

type Params = {
  file: File;
  targetCareerId: string;
  includeAiPlan: boolean;
};

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: 12,
  fontWeight: 800,
  color: "#1e1a3a",
  marginBottom: 8,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "12px 14px",
  borderRadius: 12,
  border: "2.5px solid #1e1a3a",
  background: "#faf8ff",
  fontFamily: "inherit",
  fontSize: 14,
  color: "#1e1a3a",
  outline: "none",
  boxShadow: "3px 3px 0 #1e1a3a",
};

const errBox: React.CSSProperties = {
  padding: "10px 14px",
  background: "#ffe4e4",
  border: "2.5px solid #d04848",
  borderRadius: 10,
  color: "#a02020",
  fontSize: 13,
  fontWeight: 700,
  boxShadow: "2px 2px 0 #1e1a3a",
};

type Props = {
  loading: boolean;
  onAnalyze: (params: Params) => void;
};

export function AnalyzerCard({ loading, onAnalyze }: Props) {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [targetCareerId, setTargetCareerId] = useState("");
  const [includeAiPlan, setIncludeAiPlan] = useState(true);
  const [error, setError] = useState("");

  function run() {
    if (!file) {
      setError("Silakan upload file CV (PDF/DOCX) terlebih dahulu.");
      return;
    }
    if (!targetCareerId.trim()) {
      setError("Silakan tulis nama karier target.");
      return;
    }
    setError("");
    onAnalyze({ file, targetCareerId: targetCareerId.trim(), includeAiPlan });
  }

  return (
    <div style={{ animation: "pd-fadeUp 0.4s ease both" }}>
      {/* Stepper */}
      <div style={{ display: "flex", gap: 8, marginBottom: 24, justifyContent: "center" }}>
        {[1, 2, 3].map((n) => (
          <div
            key={n}
            style={{
              width: 32,
              height: 32,
              borderRadius: "50%",
              border: "2.5px solid #1e1a3a",
              background: step >= n ? "#7c6fe0" : "#fff",
              color: step >= n ? "#fff" : "#1e1a3a",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 900,
              fontSize: 13,
              boxShadow: "2px 2px 0 #1e1a3a",
            }}
          >
            {n}
          </div>
        ))}
      </div>

      {/* STEP 1 — CV file upload */}
      {step === 1 && (
        <div style={{ animation: "pd-stepSlide 0.35s ease both" }}>
          <label style={labelStyle}>📄 Upload CV (.pdf / .docx)</label>
          <label style={{ display: "block", cursor: "pointer", marginBottom: 14 }}>
            <input
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(e) => {
                const f = e.target.files?.[0] ?? null;
                setFile(f);
                setError("");
              }}
              style={{ display: "none" }}
            />
            <div
              style={{
                ...inputStyle,
                textAlign: "center",
                padding: "22px 14px",
                background: "#ede9ff",
                fontWeight: 700,
              }}
            >
              {file ? `✅ ${file.name} · ${(file.size / 1024).toFixed(1)} KB` : "📎 Klik untuk pilih file"}
            </div>
          </label>

          <p style={{ fontSize: 12, color: "#9896b8", marginBottom: 18, lineHeight: 1.5 }}>
            CV diekstrak server-side via PyMuPDF / python-docx. CV bilingual EN+ID didukung.
          </p>

          {error && <div style={errBox}>{error}</div>}

          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 22 }}>
            <CBtn
              onClick={() => {
                if (!file) {
                  setError("Silakan upload file CV (PDF/DOCX) terlebih dahulu.");
                  return;
                }
                setError("");
                setStep(2);
              }}
            >
              Lanjut →
            </CBtn>
          </div>
        </div>
      )}

      {/* STEP 2 — Target career (text input) */}
      {step === 2 && (
        <div style={{ animation: "pd-stepSlide 0.35s ease both" }}>
          <label style={labelStyle} htmlFor="career-input">🎯 Tulis Karier Target</label>
          <input
            id="career-input"
            type="text"
            value={targetCareerId}
            onChange={(e) => setTargetCareerId(e.target.value)}
            placeholder="Contoh: data analyst, frontend dev, ML engineer"
            style={inputStyle}
            autoComplete="off"
          />
          <p style={{ fontSize: 12, color: "#9896b8", marginTop: 8, lineHeight: 1.5 }}>
            Tidak perlu format slug — backend punya fuzzy resolver. <code style={{ background: "#ede9ff", padding: "1px 5px", borderRadius: 4 }}>Data Analyst</code>, <code style={{ background: "#ede9ff", padding: "1px 5px", borderRadius: 4 }}>data analyst</code>, dan <code style={{ background: "#ede9ff", padding: "1px 5px", borderRadius: 4 }}>data-analyst</code> semua diterima.
          </p>

          {targetCareerId.trim() && (
            <div style={{ marginTop: 18, padding: 14, background: "#ede9ff", border: "2.5px solid #1e1a3a", borderRadius: 12, boxShadow: "3px 3px 0 #1e1a3a" }}>
              <div style={{ fontSize: 11, fontWeight: 800, color: "#7c6fe0", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
                Target
              </div>
              <div style={{ fontSize: 16, fontWeight: 900, color: "#1e1a3a" }}>{targetCareerId.trim()}</div>
            </div>
          )}

          {error && <div style={{ ...errBox, marginTop: 14 }}>{error}</div>}

          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 22 }}>
            <CBtn onClick={() => setStep(1)} accent="#fff" textColor="#1e1a3a">
              ← Kembali
            </CBtn>
            <CBtn
              onClick={() => {
                if (!targetCareerId.trim()) {
                  setError("Silakan tulis nama karier target.");
                  return;
                }
                setError("");
                setStep(3);
              }}
            >
              Lanjut →
            </CBtn>
          </div>
        </div>
      )}

      {/* STEP 3 — Confirm + analyze */}
      {step === 3 && (
        <div style={{ animation: "pd-stepSlide 0.35s ease both" }}>
          <div style={{ padding: 16, background: "#faf8ff", border: "2.5px solid #1e1a3a", borderRadius: 12, boxShadow: "3px 3px 0 #1e1a3a", marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: "#7c6fe0", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
              Ringkasan
            </div>
            <div style={{ fontSize: 13, color: "#1e1a3a", marginBottom: 4 }}>
              <b>CV:</b> {file?.name ?? "-"}
            </div>
            <div style={{ fontSize: 13, color: "#1e1a3a" }}>
              <b>Target karier:</b> {targetCareerId.trim()}
            </div>
          </div>

          <label style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, padding: "10px 12px", background: "#fff8e6", border: "2.5px solid #1e1a3a", borderRadius: 10, boxShadow: "2px 2px 0 #1e1a3a", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={includeAiPlan}
              onChange={(e) => setIncludeAiPlan(e.target.checked)}
              style={{ width: 16, height: 16, accentColor: "#7c6fe0" }}
            />
            <span style={{ fontSize: 13, fontWeight: 700, color: "#1e1a3a" }}>
              ✨ Generate AI Plan (Gemini) — interview &amp; learning plan, +10–15 detik
            </span>
          </label>

          <p style={{ fontSize: 13, color: "#7a789a", marginBottom: 18, lineHeight: 1.6 }}>
            Klik tombol di bawah untuk menjalankan analisis. Sistem akan membandingkan skills di CV kamu dengan kebutuhan karier target dan memberi roadmap pengembangan.
          </p>

          {error && <div style={{ ...errBox, marginBottom: 14 }}>{error}</div>}

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <CBtn onClick={() => setStep(2)} accent="#fff" textColor="#1e1a3a" disabled={loading}>
              ← Kembali
            </CBtn>
            {loading ? (
              <div style={{ display: "flex", alignItems: "center", gap: 10, fontWeight: 800, color: "#7c6fe0" }}>
                <div style={{ width: 18, height: 18, border: "3px solid #ede9ff", borderTopColor: "#7c6fe0", borderRadius: "50%", animation: "pd-spin 0.7s linear infinite" }} />
                Menganalisis...
              </div>
            ) : (
              <CBtn big onClick={run}>
                🔬 Analisis Sekarang
              </CBtn>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
