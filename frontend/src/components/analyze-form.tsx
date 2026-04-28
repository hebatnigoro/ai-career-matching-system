"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { Career } from "@/lib/types";
import { fetchCareers } from "@/lib/api";

type Props = {
  onSubmit: (params: {
    file: File;
    targetCareerId: string;
    includeAiPlan: boolean;
  }) => void;
  loading: boolean;
};

export function AnalyzeForm({ onSubmit, loading }: Props) {
  const [careers, setCareers] = useState<Career[]>([]);
  const [careersError, setCareersError] = useState<string | null>(null);
  const [targetCareerId, setTargetCareerId] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [includeAiPlan, setIncludeAiPlan] = useState(true);

  useEffect(() => {
    fetchCareers()
      .then(setCareers)
      .catch((e) => setCareersError(e.message ?? "Unknown error"));
  }, []);

  const groupedByField = useMemo(() => {
    const groups = new Map<string, Career[]>();
    for (const c of careers) {
      const key = c.field ?? "Lain-lain";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(c);
    }
    return Array.from(groups.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [careers]);

  const canSubmit = !!file && !!targetCareerId && !loading;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || !file) return;
    onSubmit({ file, targetCareerId, includeAiPlan });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {careersError && (
        <Alert variant="destructive">
          <AlertTitle>Tidak bisa memuat daftar karier</AlertTitle>
          <AlertDescription>
            {careersError}. Pastikan FastAPI berjalan di{" "}
            <code>http://127.0.0.1:8000</code> dan CORS sudah aktif.
          </AlertDescription>
        </Alert>
      )}

      <div className="space-y-2">
        <Label htmlFor="career">Target Karier</Label>
        <Select
          value={targetCareerId}
          onValueChange={setTargetCareerId}
          disabled={loading || careers.length === 0}
        >
          <SelectTrigger id="career" className="w-full">
            <SelectValue placeholder="Pilih bidang dan posisi…" />
          </SelectTrigger>
          <SelectContent>
            {groupedByField.map(([field, items]) => (
              <SelectGroup key={field}>
                <SelectLabel>{field}</SelectLabel>
                {items.map((c) => (
                  <SelectItem key={c.id} value={c.id}>
                    {c.title}
                  </SelectItem>
                ))}
              </SelectGroup>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Karier dikelompokkan per bidang. {careers.length} posisi tersedia.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="cv">CV (PDF / DOCX)</Label>
        <Input
          id="cv"
          type="file"
          accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          disabled={loading}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file && (
          <p className="text-xs text-muted-foreground">
            {file.name} · {(file.size / 1024).toFixed(1)} KB
          </p>
        )}
      </div>

      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={includeAiPlan}
          onChange={(e) => setIncludeAiPlan(e.target.checked)}
          disabled={loading}
          className="h-4 w-4"
        />
        <span>
          Generate AI Plan (Gemini) — interview & learning plan, +10–15 detik
        </span>
      </label>

      <Button type="submit" disabled={!canSubmit} className="w-full">
        {loading ? "Menganalisis…" : "Analisis CV"}
      </Button>
    </form>
  );
}
