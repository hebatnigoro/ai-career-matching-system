import type { AnalyzeResponse, Career, MatchJobsResponse } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export type Health = {
  status: string;
  careers_loaded: number;
  embeddings_cached: boolean;
  jobs_loaded?: number;
  jobs_fetched_at?: string | null;
};

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchCareers(): Promise<Career[]> {
  const res = await fetch(`${API_BASE}/careers`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Gagal memuat daftar karier (HTTP ${res.status})`);
  return res.json();
}

export async function analyzeCV(opts: {
  file: File;
  targetCareerId: string;
  topk?: number;
  minSim?: number;
  includeAiPlan?: boolean;
}): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("file", opts.file);
  fd.append("target_career_id", opts.targetCareerId);
  fd.append("topk", String(opts.topk ?? 5));
  fd.append("min_sim", String(opts.minSim ?? 0.55));
  fd.append("include_ai_plan", String(opts.includeAiPlan ?? true));

  const res = await fetch(`${API_BASE}/analyze_cv_file`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail);
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}


// ---------- Job matching ----------

export type MatchJobsParams = {
  file: File;
  filters?: {
    location?: string;
    remote?: boolean;
    employment_type?: string;
    company?: string;
  };
  weights?: {
    semantic?: number;
    skill?: number;
    experience?: number;
    location?: number;
  };
  topk?: number;
};

export async function matchJobs(opts: MatchJobsParams): Promise<MatchJobsResponse> {
  const fd = new FormData();
  fd.append("file", opts.file);
  if (opts.filters?.location) fd.append("location", opts.filters.location);
  if (opts.filters?.remote !== undefined) fd.append("remote", String(opts.filters.remote));
  if (opts.filters?.employment_type) fd.append("employment_type", opts.filters.employment_type);
  if (opts.filters?.company) fd.append("company", opts.filters.company);
  if (opts.weights?.semantic !== undefined) fd.append("w_semantic", String(opts.weights.semantic));
  if (opts.weights?.skill !== undefined) fd.append("w_skill", String(opts.weights.skill));
  if (opts.weights?.experience !== undefined) fd.append("w_experience", String(opts.weights.experience));
  if (opts.weights?.location !== undefined) fd.append("w_location", String(opts.weights.location));
  if (opts.topk !== undefined) fd.append("topk", String(opts.topk));

  const res = await fetch(`${API_BASE}/match_jobs_file`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail);
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export type JobsRefreshResult = {
  fetched_at: string;
  total_jobs: number;
  source_count: number;
  errors: { source: unknown; error: string }[];
  enrichment: {
    enriched: boolean;
    model?: string;
    jobs_with_skills?: number;
    jobs_with_exp_range?: number;
    error?: string;
  };
};

export async function refreshJobs(): Promise<JobsRefreshResult> {
  const res = await fetch(`${API_BASE}/jobs/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
  }
  return res.json();
}
