"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { AiPlan } from "@/lib/types";

export function AiPlanDisplay({ plan }: { plan: AiPlan }) {
  if ("error" in plan) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>AI Plan</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertTitle>Gagal generate AI plan</AlertTitle>
            <AlertDescription>{plan.error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Career Plan</CardTitle>
        <CardDescription>
          Dibuat oleh {plan.model} berdasarkan hasil analisis CV.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div
          className="
            text-sm leading-relaxed
            [&_h1]:text-xl [&_h1]:font-bold [&_h1]:mt-2
            [&_h2]:text-lg [&_h2]:font-semibold [&_h2]:mt-6 [&_h2]:mb-2
            [&_h3]:text-base [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-1
            [&_p]:my-2
            [&_ul]:list-disc [&_ul]:pl-6 [&_ul]:my-2
            [&_ol]:list-decimal [&_ol]:pl-6 [&_ol]:my-2
            [&_li]:my-1
            [&_strong]:font-semibold
            [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs
            [&_a]:text-blue-600 [&_a]:underline
          "
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{plan.text}</ReactMarkdown>
        </div>

        {plan.sources && plan.sources.length > 0 && (
          <div className="border-t pt-4">
            <h3 className="text-sm font-semibold mb-2">
              Sumber (Google Search)
            </h3>
            <ul className="space-y-1 text-xs">
              {plan.sources.map((s, i) => (
                <li key={i}>
                  <a
                    href={s.uri}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 underline break-all"
                  >
                    {s.title || s.uri}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
