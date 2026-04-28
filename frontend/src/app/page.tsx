"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AnalyzeForm } from "@/components/analyze-form";
import { ResultsDisplay } from "@/components/results-display";
import { analyzeCV } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze(params: {
    file: File;
    targetCareerId: string;
    includeAiPlan: boolean;
  }) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await analyzeCV(params);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-3xl px-4 py-10 space-y-8">
        <header className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight">
            Career Path Drift Analyzer
          </h1>
          <p className="text-sm text-muted-foreground">
            Upload CV → pilih target karier → dapatkan analisis kecocokan, skill
            gap, dan rencana belajar dari AI.
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Mulai Analisis</CardTitle>
            <CardDescription>
              CV diproses oleh server lokal. Embedding pakai
              multilingual-e5-base; AI plan oleh Gemini 2.5 Flash dengan Google
              Search grounding.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AnalyzeForm onSubmit={handleAnalyze} loading={loading} />
          </CardContent>
        </Card>

        {loading && (
          <Card>
            <CardContent className="py-8">
              <p className="text-sm text-muted-foreground text-center">
                Menganalisis CV… (10–30 detik, lebih lama bila AI plan aktif)
              </p>
            </CardContent>
          </Card>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Analisis gagal</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && <ResultsDisplay result={result} />}
      </div>
    </main>
  );
}
