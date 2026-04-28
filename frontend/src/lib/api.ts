import type { AnalyzeResponse, Career } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export type Health = {
  status: string;
  careers_loaded: number;
  embeddings_cached: boolean;
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
