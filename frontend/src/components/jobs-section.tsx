"use client";

import { useState } from "react";
import { CBtn } from "./cbtn";
import { matchJobs, refreshJobs } from "@/lib/api";
import type { JobMatchResult, JobScores, MatchJobsResponse } from "@/lib/types";

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: 11,
  fontWeight: 800,
  color: "var(--pd-text)",
  marginBottom: 6,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "10px 12px",
  borderRadius: 10,
  border: "2.5px solid #1e1a3a",
  background: "var(--pd-bg-soft)",
  fontFamily: "inherit",
  fontSize: 13,
  color: "var(--pd-text)",
  outline: "none",
  boxShadow: "2px 2px 0 #1e1a3a",
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
  marginBottom: 12,
};

const SCORE_COLORS: Record<keyof JobScores, string> = {
  semantic:   "#0EA5E9",
  skill:      "#10B981",
  experience: "#F59E0B",
  location:   "#A855F7",
  final:      "#1e1a3a",
};

const SCORE_LABELS: Record<keyof JobScores, string> = {
  semantic:   "Semantic",
  skill:      "Skills",
  experience: "Experience",
  location:   "Location",
  final:      "Final",
};

function ScoreBar({ name, value, color }: { name: string; value: number; color: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12 }}>
      <div style={{ width: 78, fontWeight: 800, color: "var(--pd-text-muted)", textTransform: "uppercase", letterSpacing: "0.04em", fontSize: 10 }}>
        {name}
      </div>
      <div
        style={{
          flex: 1,
          height: 12,
          background: "#F1F5F9",
          borderRadius: 999,
          border: "2px solid #1e1a3a",
          overflow: "hidden",
          position: "relative",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            transition: "width 0.4s",
          }}
        />
      </div>
      <div style={{ width: 38, textAlign: "right", fontWeight: 800, color: "#1e1a3a" }}>
        {(value * 100).toFixed(0)}%
      </div>
    </div>
  );
}

function Pill({ children, color, bg }: { children: React.ReactNode; color: string; bg: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "3px 9px",
        borderRadius: 999,
        background: bg,
        color,
        border: "2px solid #1e1a3a",
        fontSize: 11,
        fontWeight: 800,
        marginRight: 6,
        marginBottom: 6,
      }}
    >
      {children}
    </span>
  );
}

function JobCard({ result }: { result: JobMatchResult }) {
  const [open, setOpen] = useState(false);
  const j = result.job;
  const s = result.scores;
  return (
    <div
      style={{
        background: "var(--pd-card)",
        color: "var(--pd-text)",
        border: "2.5px solid #1e1a3a",
        borderRadius: 18,
        boxShadow: "4px 4px 0 #1e1a3a",
        padding: "20px 22px",
        marginBottom: 16,
        opacity: result.eligible ? 1 : 0.65,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 220 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span
              style={{
                fontSize: 10,
                fontWeight: 800,
                color: "#fff",
                background: "#1e1a3a",
                padding: "2px 8px",
                borderRadius: 6,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              {j.source}
            </span>
            {!result.eligible && (
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 800,
                  color: "#a02020",
                  background: "#ffe4e4",
                  padding: "2px 8px",
                  borderRadius: 6,
                  border: "1.5px solid #d04848",
                }}
              >
                INELIGIBLE
              </span>
            )}
          </div>
          <h3 style={{ fontSize: 17, fontWeight: 900, color: "var(--pd-text)", marginBottom: 4, lineHeight: 1.3 }}>
            {j.title}
          </h3>
          <div style={{ fontSize: 13, color: "var(--pd-text-muted)", fontWeight: 600 }}>
            {j.company}
            {j.location ? ` · ${j.location}` : ""}
            {j.workplace_type && j.workplace_type !== "unspecified"
              ? ` · ${j.workplace_type}`
              : ""}
            {j.employment_type ? ` · ${j.employment_type}` : ""}
          </div>
        </div>
        <div
          style={{
            background: "#0EA5E9",
            color: "#fff",
            border: "2.5px solid #1e1a3a",
            borderRadius: 14,
            padding: "8px 14px",
            fontWeight: 900,
            fontSize: 22,
            boxShadow: "3px 3px 0 #1e1a3a",
            minWidth: 76,
            textAlign: "center",
          }}
        >
          {(s.final * 100).toFixed(0)}
          <span style={{ fontSize: 12, fontWeight: 800 }}>%</span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 14 }}>
        <ScoreBar name={SCORE_LABELS.semantic}   value={s.semantic}   color={SCORE_COLORS.semantic} />
        <ScoreBar name={SCORE_LABELS.skill}      value={s.skill}      color={SCORE_COLORS.skill} />
        <ScoreBar name={SCORE_LABELS.experience} value={s.experience} color={SCORE_COLORS.experience} />
        <ScoreBar name={SCORE_LABELS.location}   value={s.location}   color={SCORE_COLORS.location} />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginTop: 14, flexWrap: "wrap" }}>
        <button
          onClick={() => setOpen((v) => !v)}
          style={{
            padding: "6px 12px",
            border: "2px solid #1e1a3a",
            borderRadius: 10,
            background: "var(--pd-card)",
            fontSize: 12,
            fontWeight: 800,
            color: "var(--pd-text)",
            cursor: "pointer",
            fontFamily: "inherit",
            boxShadow: "2px 2px 0 #1e1a3a",
          }}
        >
          {open ? "▲ Hide breakdown" : "▼ Why this match?"}
        </button>
        {j.url && (
          <a
            href={j.url}
            target="_blank"
            rel="noreferrer"
            style={{
              padding: "6px 14px",
              border: "2px solid #1e1a3a",
              borderRadius: 10,
              background: "#0EA5E9",
              color: "#fff",
              fontSize: 12,
              fontWeight: 800,
              textDecoration: "none",
              boxShadow: "2px 2px 0 #1e1a3a",
            }}
          >
            View posting →
          </a>
        )}
      </div>

      {open && (
        <div style={{ marginTop: 16, padding: "14px 16px", background: "var(--pd-bg-soft)", borderRadius: 12, border: "2px dashed #94A3B8" }}>
          <div style={{ marginBottom: 12 }}>
            <div style={{ ...labelStyle, marginBottom: 6 }}>Matched skills ({result.skill_match.matched.length}/{result.skill_match.required_skills.length})</div>
            {result.skill_match.matched.length === 0 && (
              <div style={{ fontSize: 12, color: "var(--pd-text-muted)" }}>No required skills matched.</div>
            )}
            {result.skill_match.matched.map((m) => (
              <Pill key={m.skill} color="#1e1a3a" bg="#D8F5E5">
                {m.skill} <span style={{ opacity: 0.6, fontWeight: 700 }}>({m.source})</span>
              </Pill>
            ))}
          </div>

          {result.skill_match.missing.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ ...labelStyle, marginBottom: 6 }}>Missing skills ({result.skill_match.missing.length})</div>
              {result.skill_match.missing.slice(0, 12).map((s) => (
                <Pill key={s} color="#a02020" bg="#ffe4e4">{s}</Pill>
              ))}
            </div>
          )}

          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, fontSize: 12, color: "var(--pd-text-muted)" }}>
            <div>
              <strong>Experience:</strong> {result.experience_match.reason}
              {result.experience_match.cv !== undefined && result.experience_match.required && (
                <> · CV {result.experience_match.cv}y vs required {result.experience_match.required[0]}–{result.experience_match.required[1]}y</>
              )}
            </div>
            <div>
              <strong>Location:</strong> {result.location_match.reason}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Employment-type values exposed by all three ATS adapters (Greenhouse,
// Lever, Ashby). Free-text input was a UX trap — recruiters use a fixed
// vocabulary, so we match that.
const EMPLOYMENT_TYPES: { value: string; label: string }[] = [
  { value: "",          label: "Any" },
  { value: "FullTime",  label: "Full-time" },
  { value: "PartTime",  label: "Part-time" },
  { value: "Contract",  label: "Contract" },
  { value: "Intern",    label: "Internship" },
  { value: "Temporary", label: "Temporary" },
];

function FilterRow({
  filters,
  setFilters,
}: {
  filters: { location: string; remote: boolean; employment_type: string };
  setFilters: (f: typeof filters) => void;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
      <div>
        <label style={labelStyle}>Location</label>
        <input
          style={inputStyle}
          placeholder="e.g. Jakarta"
          value={filters.location}
          onChange={(e) => setFilters({ ...filters, location: e.target.value })}
        />
      </div>
      <div>
        <label style={labelStyle}>Employment</label>
        <select
          style={{ ...inputStyle, cursor: "pointer" }}
          value={filters.employment_type}
          onChange={(e) => setFilters({ ...filters, employment_type: e.target.value })}
        >
          {EMPLOYMENT_TYPES.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      <div style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end", paddingBottom: 4 }}>
        <label style={{ ...labelStyle, marginBottom: 8 }}>Remote only</label>
        <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13, fontWeight: 700, color: "var(--pd-text-muted)" }}>
          <input
            type="checkbox"
            checked={filters.remote}
            onChange={(e) => setFilters({ ...filters, remote: e.target.checked })}
            style={{ width: 18, height: 18, cursor: "pointer" }}
          />
          Remote only
        </label>
      </div>
    </div>
  );
}

export function JobsSection() {
  const [file, setFile] = useState<File | null>(null);
  const [filters, setFilters] = useState({ location: "", remote: false, employment_type: "" });
  const [result, setResult] = useState<MatchJobsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");

  async function handleSubmit() {
    if (!file) {
      setError("Silakan upload CV (PDF/DOCX) terlebih dahulu.");
      return;
    }
    setError("");
    setInfo("");
    setLoading(true);
    setResult(null);
    try {
      const r = await matchJobs({
        file,
        filters: {
          location: filters.location || undefined,
          remote: filters.remote || undefined,
          employment_type: filters.employment_type || undefined,
        },
        topk: 20,
      });
      setResult(r);
      setTimeout(() => window.scrollTo({ top: 0, behavior: "smooth" }), 80);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleRefresh() {
    setRefreshing(true);
    setError("");
    setInfo("");
    try {
      const r = await refreshJobs();
      setInfo(
        `Fetched ${r.total_jobs} jobs from ${r.source_count} sources` +
          (r.enrichment.enriched ? ` · enriched ${r.enrichment.jobs_with_skills ?? 0} with inferred skills` : "") +
          (r.errors.length ? ` · ${r.errors.length} source(s) failed` : ""),
      );
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <div style={{ minHeight: "100vh", padding: "100px 24px 80px", position: "relative" }}>
      <div style={{ maxWidth: 880, margin: "0 auto", position: "relative", zIndex: 1 }}>
        <div style={{ textAlign: "center", marginBottom: 24 }}>
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
            💼 Live Job Matcher
          </div>
          <h2 style={{ fontSize: 34, fontWeight: 900, letterSpacing: "-0.02em", marginBottom: 6, color: "var(--pd-text)" }}>
            Find Jobs That Fit Your CV
          </h2>
          <p style={{ fontSize: 14, color: "var(--pd-text-muted)" }}>
            Real postings from Greenhouse, Lever &amp; Ashby. Ranked by skill, experience, and location fit.
          </p>
        </div>

        {error && <div style={errBox}>{error}</div>}
        {info && (
          <div
            style={{
              padding: "10px 14px",
              background: "#D8F5E5",
              border: "2.5px solid #1e1a3a",
              borderRadius: 10,
              color: "#1e1a3a",
              fontSize: 13,
              fontWeight: 700,
              boxShadow: "2px 2px 0 #1e1a3a",
              marginBottom: 12,
            }}
          >
            {info}
          </div>
        )}

        <div style={{ background: "var(--pd-card)", color: "var(--pd-text)", border: "2.5px solid #1e1a3a", borderRadius: 24, boxShadow: "6px 6px 0 #1e1a3a", padding: 26, marginBottom: 24 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 12, alignItems: "end", marginBottom: 16 }}>
            <div>
              <label style={labelStyle}>CV file (PDF / DOCX)</label>
              <input
                type="file"
                accept=".pdf,.docx"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                style={{ ...inputStyle, padding: "8px 10px" }}
              />
              {file && (
                <div style={{ fontSize: 11, color: "var(--pd-text-muted)", marginTop: 4 }}>
                  ✓ {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </div>
              )}
            </div>
            <CBtn onClick={handleRefresh} accent="#fff" textColor="#1e1a3a" disabled={refreshing}>
              {refreshing ? "Refreshing…" : "🔄 Refresh jobs"}
            </CBtn>
          </div>

          <FilterRow filters={filters} setFilters={setFilters} />

          <div style={{ textAlign: "center" }}>
            <CBtn onClick={handleSubmit} disabled={loading || !file} big>
              {loading ? "Matching…" : "🚀 Match jobs to my CV"}
            </CBtn>
          </div>
        </div>

        {result && (
          <div>
            <div
              style={{
                background: "var(--pd-card)",
                border: "2.5px solid #1e1a3a",
                borderRadius: 18,
                padding: "16px 20px",
                boxShadow: "4px 4px 0 #1e1a3a",
                marginBottom: 18,
                fontSize: 13,
                color: "var(--pd-text-muted)",
              }}
            >
              <div style={{ marginBottom: 6, fontWeight: 800, color: "var(--pd-text)" }}>
                Detected from your CV
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 14 }}>
                <span>
                  <strong>Experience:</strong>{" "}
                  {result.cv_profile.experience_years !== null
                    ? `${result.cv_profile.experience_years} years (${result.cv_profile.experience_method})`
                    : "—"}
                </span>
                <span>
                  <strong>Location:</strong>{" "}
                  {result.cv_profile.location_city
                    ? `${result.cv_profile.location_city}${result.cv_profile.location_country ? `, ${result.cv_profile.location_country}` : ""}`
                    : "—"}
                </span>
                <span>
                  <strong>Remote:</strong>{" "}
                  {result.cv_profile.remote_preference ? "open to remote" : "not specified"}
                </span>
              </div>
              <div style={{ marginTop: 10, fontSize: 12 }}>
                {result.eligible_count} of {result.candidates_considered} candidates eligible
                {result.filtered_out_by_pre_filter > 0 ? ` · ${result.filtered_out_by_pre_filter} filtered out` : ""}
              </div>
            </div>

            {result.ranked.length === 0 ? (
              <div style={{ textAlign: "center", color: "var(--pd-text-muted)", padding: 30 }}>
                No jobs match the current filters. Try loosening them or refresh the cache.
              </div>
            ) : (
              result.ranked.map((r) => <JobCard key={r.job.id} result={r} />)
            )}
          </div>
        )}
      </div>
    </div>
  );
}
