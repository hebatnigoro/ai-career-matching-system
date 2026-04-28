"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import type { AnalyzeResponse } from "@/lib/types";
import { AiPlanDisplay } from "./ai-plan-display";

function pct(score: number | null | undefined) {
  if (score === null || score === undefined) return "—";
  return `${Math.round(score * 100)}%`;
}

function statusVariant(status: string): "default" | "secondary" | "destructive" {
  if (status?.toLowerCase().includes("major")) return "destructive";
  if (status?.toLowerCase().includes("minor")) return "secondary";
  return "default";
}

export function ResultsDisplay({ result }: { result: AnalyzeResponse }) {
  return (
    <div className="space-y-6">
      {/* Target & status */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                {result.target.title}
                {result.target.field && (
                  <Badge variant="outline">{result.target.field}</Badge>
                )}
              </CardTitle>
              <CardDescription>
                Skor kecocokan: {pct(result.target.score)}
                {result.target.resolved_from && (
                  <>
                    {" · "}input <code>{result.target.resolved_from}</code> →{" "}
                    <code>{result.target.career_id}</code>
                  </>
                )}
              </CardDescription>
            </div>
            <Badge variant={statusVariant(result.status)}>{result.status}</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{result.rationale}</p>
        </CardContent>
      </Card>

      {/* Best alternative */}
      {result.best_alternative.career_id && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Alternatif Terbaik</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="flex items-center gap-2 text-sm">
              <span className="font-medium">{result.best_alternative.title}</span>
              {result.best_alternative.field && (
                <Badge variant="outline">{result.best_alternative.field}</Badge>
              )}
              <span className="text-muted-foreground">
                · {pct(result.best_alternative.score)}
              </span>
            </div>
            {result.best_alternative.advantage !== null && (
              <p className="text-xs text-muted-foreground">
                Keunggulan dibanding target: +
                {((result.best_alternative.advantage ?? 0) * 100).toFixed(1)}%
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Skill match */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Skill Match</CardTitle>
          <CardDescription>
            {result.skill_gap.matched_skills.length} cocok ·{" "}
            {result.skill_gap.missing_skills.length} kurang
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Progress value={result.skill_gap.match_ratio * 100} />
          {result.skill_gap.matched_skills.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">
                Skill yang cocok
              </p>
              <div className="flex flex-wrap gap-1">
                {result.skill_gap.matched_skills.map((s) => (
                  <Badge key={s.skill} variant="secondary">
                    {s.skill}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          {result.skill_gap.missing_skills.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">
                Skill yang perlu ditingkatkan
              </p>
              <div className="flex flex-wrap gap-1">
                {result.skill_gap.missing_skills.map((s) => (
                  <Badge key={s.skill} variant="destructive">
                    {s.skill}
                    {s.type === "upgrade" && " (upgrade)"}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Transition context */}
      {result.transition_context && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Konteks Transisi</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{result.transition_context.summary}</p>
          </CardContent>
        </Card>
      )}

      {/* Top rankings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Top {result.rankings.length} Karier yang Cocok
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="space-y-2">
            {result.rankings.map((r) => (
              <li
                key={r.career_id}
                className="flex items-center justify-between text-sm py-1"
              >
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground w-6">#{r.rank}</span>
                  <span className="font-medium">{r.title}</span>
                  {r.field && <Badge variant="outline">{r.field}</Badge>}
                </div>
                <span className="font-mono text-xs">{pct(r.score)}</span>
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>

      <Separator />

      {/* AI plan */}
      {result.ai_plan && <AiPlanDisplay plan={result.ai_plan} />}
    </div>
  );
}
