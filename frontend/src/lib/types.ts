export type Career = {
  id: string;
  title: string;
  field: string | null;
};

export type SkillEntry = {
  skill: string;
  similarity?: number;
  type?: "upgrade" | "new";
  upgrade_from?: string[];
};

export type SkillGap = {
  matched_skills: SkillEntry[];
  missing_skills: SkillEntry[];
  match_ratio: number;
  outdated_in_cv?: { skill: string; modern_alternatives: string[] }[];
};

export type Ranking = {
  rank: number;
  career_id: string;
  title: string;
  field: string | null;
  score: number;
};

export type Recommendation = {
  career_id: string;
  title: string;
  field: string | null;
  score: number;
};

export type AiPlanSuccess = {
  text: string;
  sources: { title: string; uri: string }[];
  model: string;
};

export type AiPlanError = { error: string };

export type AiPlan = AiPlanSuccess | AiPlanError;

export type TransitionContext = {
  from_career_id: string;
  from_career_title: string;
  from_career_field: string | null;
  from_career_score: number;
  to_career_id: string;
  to_career_title: string;
  to_career_field: string | null;
  to_career_score: number;
  score_gap: number;
  skill_match: {
    matched: number;
    missing: number;
    total: number;
    upgrade: number;
    new: number;
    match_ratio: number;
  };
  summary: string;
};

// ---------- Job matching ----------

export type JobSummary = {
  id: string;
  source: "greenhouse" | "lever" | "ashby";
  company: string;
  title: string;
  department: string | null;
  team: string | null;
  location: string | null;
  country: string | null;
  remote: boolean;
  workplace_type: "remote" | "hybrid" | "onsite" | "unspecified";
  employment_type: string | null;
  url: string | null;
  apply_url: string | null;
  posted_at: string | null;
  compensation: string | null;
};

export type JobScores = {
  semantic: number;
  skill: number;
  experience: number;
  location: number;
  final: number;
};

export type SkillMatchEntry = {
  skill: string;
  source: "lexical" | "fuzzy" | "semantic";
  confidence: number;
  evidence: string;
};

export type JobMatchResult = {
  job: JobSummary;
  scores: JobScores;
  eligible: boolean;
  skill_match: {
    matched: SkillMatchEntry[];
    missing: string[];
    match_ratio: number;
    required_skills: string[];
  };
  experience_match: {
    score: number;
    reason: string;
    cv?: number;
    required?: [number, number];
    gap?: number;
  };
  location_match: {
    score: number;
    reason: string;
    cv_city?: string;
    job_location?: string;
  };
};

export type CVProfile = {
  experience_years: number | null;
  experience_method: string | null;
  location_city: string | null;
  location_country: string | null;
  remote_preference: boolean | null;
};

export type MatchJobsResponse = {
  weights: Record<string, number>;
  thresholds: Record<string, number>;
  cv_profile: CVProfile;
  candidates_considered: number;
  filtered_out_by_pre_filter: number;
  eligible_count: number;
  ranked: JobMatchResult[];
  fetched_at: string | null;
  model: string;
};

export type AnalyzeResponse = {
  target: {
    career_id: string;
    title: string;
    field: string | null;
    score: number;
    resolved_from?: string;
  };
  best_alternative: {
    career_id: string;
    title: string | null;
    field: string | null;
    score: number | null;
    advantage: number | null;
  };
  status: string;
  rationale: string;
  transition_context: TransitionContext | null;
  skill_gap: SkillGap;
  rankings: Ranking[];
  recommendations: Recommendation[];
  thresholds: Record<string, number>;
  model: string;
  ai_plan?: AiPlan;
};
