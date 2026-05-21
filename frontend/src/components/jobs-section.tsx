"use client";

import { useState } from "react";
import { CBtn } from "./cbtn";
import { matchJobs, refreshJobs } from "@/lib/api";
import type { JobMatchResult, MatchJobsResponse } from "@/lib/types";

type SortDir = "desc" | "asc";

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

function skillMatchCount(r: JobMatchResult): number {
  return r.skill_match.matched.length;
}

function skillRatio(r: JobMatchResult): number {
  // Use the backend's pre-computed ratio when available — it already
  // guards against divide-by-zero for jobs with no inferred skills.
  return typeof r.skill_match.match_ratio === "number"
    ? r.skill_match.match_ratio
    : 0;
}

function sortBySkill(results: JobMatchResult[], dir: SortDir): JobMatchResult[] {
  // Stable sort by (ratio, absolute count). The absolute count is the
  // tie-breaker so "8 / 12" outranks "1 / 1" when the user is looking
  // for the strongest skill match, not the most lopsided ratio.
  const sign = dir === "desc" ? -1 : 1;
  return [...results].sort((a, b) => {
    const dr = sign * (skillRatio(a) - skillRatio(b));
    if (dr !== 0) return dr;
    return sign * (skillMatchCount(a) - skillMatchCount(b));
  });
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
  const [showMissing, setShowMissing] = useState(false);
  const j = result.job;
  const matched = result.skill_match.matched;
  const missing = result.skill_match.missing;
  const requiredCount = result.skill_match.required_skills.length;
  const matchedCount = matched.length;

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
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
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
                title={(result.ineligible_reasons || []).join(" · ") || "Did not meet eligibility filters"}
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
            {result.low_signal && (
              <span
                title="Job posting has too little detail to score reliably"
                style={{
                  fontSize: 10,
                  fontWeight: 800,
                  color: "#8a5a00",
                  background: "#fff3d6",
                  padding: "2px 8px",
                  borderRadius: 6,
                  border: "1.5px solid #d4a020",
                }}
              >
                LOW SIGNAL
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
            background: "#10B981",
            color: "#fff",
            border: "2.5px solid #1e1a3a",
            borderRadius: 14,
            padding: "8px 14px",
            fontWeight: 900,
            fontSize: 22,
            boxShadow: "3px 3px 0 #1e1a3a",
            minWidth: 96,
            textAlign: "center",
            lineHeight: 1.1,
          }}
        >
          {matchedCount}
          <span style={{ fontSize: 14, fontWeight: 800, opacity: 0.85 }}>
            {" / "}{requiredCount || "?"}
          </span>
          <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.06em", marginTop: 2 }}>
            SKILLS
          </div>
        </div>
      </div>

      <div style={{ marginTop: 14 }}>
        <div style={{ ...labelStyle, marginBottom: 6 }}>
          Your matching skills ({matchedCount}
          {requiredCount ? `/${requiredCount}` : ""})
        </div>
        {matchedCount === 0 ? (
          <div style={{ fontSize: 12, color: "var(--pd-text-muted)", fontStyle: "italic" }}>
            No required skills detected in your CV for this role.
          </div>
        ) : (
          <div>
            {matched.map((m) => (
              <Pill key={m.skill} color="#1e1a3a" bg="#D8F5E5">
                {m.skill}
              </Pill>
            ))}
          </div>
        )}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginTop: 14, flexWrap: "wrap" }}>
        {missing.length > 0 ? (
          <button
            onClick={() => setShowMissing((v) => !v)}
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
            {showMissing
              ? `▲ Hide missing (${missing.length})`
              : `▼ Show missing skills (${missing.length})`}
          </button>
        ) : (
          <span style={{ fontSize: 12, color: "var(--pd-text-muted)" }}>
            No missing required skills.
          </span>
        )}
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

      {showMissing && missing.length > 0 && (
        <div style={{ marginTop: 12, padding: "10px 12px", background: "var(--pd-bg-soft)", borderRadius: 10, border: "2px dashed #94A3B8" }}>
          <div style={{ ...labelStyle, marginBottom: 6 }}>Missing skills</div>
          {missing.map((s) => (
            <Pill key={s} color="#a02020" bg="#ffe4e4">{s}</Pill>
          ))}
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
  // Default to descending — most users want the best skill match on top.
  const [sortDir, setSortDir] = useState<SortDir>("desc");

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
            Real postings from Greenhouse, Lever &amp; Ashby. Ranked by how many required skills your CV already has.
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
              <>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 12,
                    marginBottom: 12,
                    flexWrap: "wrap",
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 800, color: "var(--pd-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    Sorted by skills matched
                  </div>
                  <button
                    onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
                    title={sortDir === "desc" ? "Showing best match first" : "Showing worst match first"}
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
                    {sortDir === "desc" ? "↓ Most skills first" : "↑ Fewest skills first"}
                  </button>
                </div>
                {sortBySkill(result.ranked, sortDir).map((r) => (
                  <JobCard key={r.job.id} result={r} />
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
