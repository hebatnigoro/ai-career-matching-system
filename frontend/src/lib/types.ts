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
