"""Gemini-backed career plan generator.

Reads GEMINI_API_KEY from environment. Uses Google Search grounding so the
learning plan can cite up-to-date resources. Returns None when the key is not
set so the analysis pipeline can degrade gracefully.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List


def _format_rankings(rankings: List[Dict[str, Any]], limit: int = 5) -> str:
    lines = []
    for i, r in enumerate(rankings[:limit], 1):
        field = f" [{r.get('field')}]" if r.get('field') else ""
        lines.append(f"  {i}. {r.get('title')}{field} — skor {r.get('score')}")
    return "\n".join(lines) if lines else "  (kosong)"


def _build_prompt(analysis: Dict[str, Any]) -> str:
    target = analysis.get("target") or {}
    best_alt = analysis.get("best_alternative") or {}
    rankings = analysis.get("rankings") or []
    recommendations = analysis.get("recommendations") or []
    skill_gap = analysis.get("skill_gap") or {}

    matched = [s.get("skill") for s in skill_gap.get("matched_skills", []) if s.get("skill")]
    missing = skill_gap.get("missing_skills", [])
    missing_upgrade = [s for s in missing if s.get("type") == "upgrade"]
    missing_new = [s for s in missing if s.get("type") != "upgrade"]

    target_field = f" (bidang: {target.get('field')})" if target.get("field") else ""
    alt_field = f" (bidang: {best_alt.get('field')})" if best_alt.get("field") else ""

    return f"""Anda adalah career coach. Berdasarkan hasil analisis CV di bawah ini, buat dua dokumen untuk pengguna. 
            Tolong buat responsenya to the point dan jangan mengulangi perintah dalam response.

=== HASIL ANALISIS SISTEM ===
Target karier        : {target.get('title')}{target_field} — skor kecocokan {target.get('score')}
Alternatif terbaik   : {best_alt.get('title')}{alt_field} — skor {best_alt.get('score')}
Status drift         : {analysis.get('status')}
Rationale            : {analysis.get('rationale')}

Top karier yang cocok dengan CV:
{_format_rankings(rankings)}

Rekomendasi sistem:
{_format_rankings(recommendations)}

Skill yang sudah dimiliki: {', '.join(matched) if matched else '(tidak terdeteksi)'}
Skill yang perlu di-upgrade: {', '.join(s.get('skill','') for s in missing_upgrade) or '(tidak ada)'}
Skill yang perlu dipelajari dari nol: {', '.join(s.get('skill','') for s in missing_new) or '(tidak ada)'}

=== TUGAS ===
Tulis dua bagian dengan heading yang jelas, dalam Bahasa Indonesia:

## INTERVIEW PLAN
5–7 area pertanyaan untuk posisi {target.get('title')}: campuran teknis dan behavioral. Untuk setiap area, beri 1 contoh pertanyaan dan tips menjawab singkat. Sesuaikan dengan skill yang sudah/belum dimiliki kandidat.

## LEARNING PLAN
Rencana belajar terurut prioritas untuk menutup skill gap. Untuk setiap skill, gunakan Google Search untuk menemukan resource terkini (kursus, dokumentasi, atau tutorial yang masih aktif tahun ini), sebutkan nama resource dan estimasi waktu belajar. Prioritaskan skill yang muncul di kategori 'perlu di-upgrade' karena sudah ada pondasinya."""


def _extract_sources(response: Any) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    try:
        for cand in getattr(response, "candidates", []) or []:
            meta = getattr(cand, "grounding_metadata", None)
            chunks = getattr(meta, "grounding_chunks", None) if meta else None
            for chunk in chunks or []:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    sources.append({
                        "title": getattr(web, "title", "") or "",
                        "uri": web.uri,
                    })
    except Exception:
        pass
    return sources


def generate_career_plan(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interview + learning plans from an analysis result.

    Always returns a dict so the caller can surface the result — `text` and
    `sources` on success, or `error` on any failure (missing key, missing
    package, API error).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY belum diset di environment uvicorn. Set lalu restart server."}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"error": "google-genai belum terpasang. Jalankan: pip install google-genai"}

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=_build_prompt(analysis),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return {
            "text": getattr(response, "text", "") or "",
            "sources": _extract_sources(response),
            "model": "gemini-2.5-flash",
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
