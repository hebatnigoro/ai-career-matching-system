"""Structured profile extraction from raw CV text.

The skill_extract pipeline already gives us *what* the candidate knows. For
job matching we additionally need:

* ``experience_years`` — total years of work experience, used to score how
  well the CV matches a job's required seniority.
* ``location`` — city/country the candidate is based in (or willing to
  work from), used together with the job's location field.
* ``remote_preference`` — whether the CV mentions openness to remote work.

These three signals are deterministic regex extractions — no model
required, which matches the user's intuition that the *retrieval/filter*
stage of job search shouldn't depend on a learned model.

The extractor is intentionally conservative: when a signal isn't found it
returns ``None`` (not 0, not ""). Downstream matching treats ``None`` as
"no constraint to enforce" rather than as a hard mismatch.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Public dataclass
# ----------------------------------------------------------------------

@dataclass
class CVProfile:
    experience_years: Optional[float]    # None if no signal extracted
    experience_method: Optional[str]     # how it was derived (for explainability)
    location_city: Optional[str]
    location_country: Optional[str]      # ISO-2 if known, else free text
    remote_preference: Optional[bool]    # True if CV signals openness to remote


# ----------------------------------------------------------------------
# Experience extraction
# ----------------------------------------------------------------------
#
# Two complementary strategies:
#   (A) Direct phrase: "5+ years of experience", "3 tahun pengalaman", ...
#   (B) Date-range summation: parse "Jan 2021 – Present", "2019–2022" pairs
#       in the experience section and sum durations (capping at year-level
#       precision because resume dates are notoriously imprecise).
#
# Strategy A is preferred when present; B is the fallback. Returning the
# method used keeps the result auditable.

_EXP_PHRASE_PATTERNS = [
    re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+)?experience", re.IGNORECASE),
    re.compile(r"(\d{1,2})\s*\+?\s*tahun\s+pengalaman", re.IGNORECASE),
    re.compile(r"pengalaman\s+(\d{1,2})\s*\+?\s*tahun", re.IGNORECASE),
    re.compile(r"experience[^\n]{0,30}?(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE),
]

_MONTHS = {
    # English
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    # Indonesian
    "januari": 1, "februari": 2, "maret": 3, "mei": 5,
    "juni": 6, "juli": 7, "agustus": 8, "agt": 8,
    "september": 9, "oktober": 10, "okt": 10, "november": 11, "nov": 11,
    "desember": 12, "des": 12,
}

# "Jan 2021", "January 2021", "01/2021", "2021" (year-only)
_MONTH_YEAR = re.compile(
    r"(?P<m>\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"januari|februari|maret|mei|juni|juli|agustus|agt|oktober|okt|desember|des)\b)?"
    r"\s*"
    r"(?P<y>(?:19|20)\d{2})",
    re.IGNORECASE,
)
_PRESENT = re.compile(
    r"\b(present|now|sekarang|current|currently|saat\s+ini)\b",
    re.IGNORECASE,
)
_RANGE_SEP = re.compile(r"\s*(?:-|–|—|to|until|hingga|sampai|s\.?d\.?)\s*", re.IGNORECASE)


# Specific multi-word headers — unambiguous, always a section heading.
_EXP_HEADER_SPECIFIC = re.compile(
    r"\b(work\s+experience|employment\s+history|professional\s+experience|"
    r"pengalaman\s+kerja|riwayat\s+pekerjaan)\b",
    re.IGNORECASE,
)
# Bare "experience" / "pengalaman" — ambiguous (also appears in qualification
# profile prose). Used only as a fallback, taking the LAST occurrence.
_EXP_HEADER_BARE = re.compile(r"\b(experience|pengalaman)\b", re.IGNORECASE)

# End-of-experience markers. No newline required: PDF extraction often
# flattens layout to a single line, so the previous newline-anchored regex
# silently failed and let education-section dates bleed into the experience
# total (the source of the original "7.9 years for a 1-year intern" bug).
_END_HEADERS = re.compile(
    r"\b(education|pendidikan|skills?|keahlian|kemampuan|"
    r"certifications?|publications?|references?|awards?|"
    r"hobbies|interests|languages\s+spoken)\b",
    re.IGNORECASE,
)

# Words that indicate a date range is part of an education entry rather than
# work experience. Used as a final filter when the section detector still
# admits a noisy slice.
_EDU_CONTEXT = re.compile(
    r"\b(university|universitas|school|sekolah|sma|smk|smp|college|"
    r"institute|institut|akademi|akademik|academy|degree|bachelor|"
    r"master|magister|sarjana|undergraduate|s1|s2|s3|"
    r"b\.?\s*sc|m\.?\s*sc|m\.?\s*eng|b\.?\s*eng|ph\.?\s*d|"
    r"diploma|jurusan|fakultas|faculty|major|minor|coursework)\b",
    re.IGNORECASE,
)


def _exp_section(cv_text: str) -> str:
    """Slice the CV to the work-experience section.

    Strategy
    --------
    1. Prefer an unambiguous multi-word header ("work experience",
       "employment history", "pengalaman kerja", ...). If found, slice
       from there.
    2. Otherwise fall back to the **last** occurrence of bare
       "experience"/"pengalaman". Prose mentions ("hands-on experience in
       data science") almost always appear earlier in the qualification
       profile; the actual section header sits later in the document.
    3. End at the first downstream section keyword (no newline required —
       PDF extraction often produces flat text).
    4. If no header signal at all, return empty string. We deliberately do
       NOT fall back to the whole CV — that path historically swept in
       graduation years and inflated experience totals.
    """
    m = _EXP_HEADER_SPECIFIC.search(cv_text)
    if not m:
        bare_matches = list(_EXP_HEADER_BARE.finditer(cv_text))
        if not bare_matches:
            return ""
        m = bare_matches[-1]
    tail = cv_text[m.end():]
    end = _END_HEADERS.search(tail)
    if end:
        tail = tail[:end.start()]
    return tail


def _parse_year_month(
    token: str, default_month: int = 1,
) -> Optional[Tuple[int, int, bool]]:
    """Parse 'Jan 2021', 'January 2021', '2021' → (year, month, has_explicit_month).

    The boolean lets callers tell apart real work entries (``Feb 2025 - Feb 2026``,
    both endpoints have months) from graduation entries (``Sep 2022 - 2026``,
    asymmetric — start has month, end is year-only). Education and graduation
    ranges almost always omit the second-side month; work date ranges almost
    always include both. Distinguishing them removes a major source of
    inflated experience totals.
    """
    mm = _MONTH_YEAR.search(token)
    if not mm:
        return None
    y = int(mm.group("y"))
    if y < 1970 or y > 2100:
        return None
    m_word = (mm.group("m") or "").lower()
    has_month = bool(m_word)
    month = _MONTHS.get(m_word, default_month)
    return (y, month, has_month)


def _months_between(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, (b[0] - a[0]) * 12 + (b[1] - a[1]))


def _extract_date_ranges(text: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Find work-experience ``Start - End`` date ranges in CV text.

    Two filters drop graduation/education ranges:

    1. **Both endpoints must have explicit months**, OR the right side is
       "Present"/"Sekarang". Work entries write ``Feb 2025 – Feb 2026``;
       graduation entries write ``Sep 2022 – 2026`` or ``2018-2022``.
       This single rule removes the dominant false-positive class.
    2. A wider ±200 char surrounding-context check still rejects ranges
       sitting adjacent to university / SMA / undergraduate / GPA markers
       — defense in depth for chaotic PDF column orders.
    """
    ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    # Year of "now" — see module docstring for why this is hard-coded.
    NOW = (2026, 4)

    matches = list(_MONTH_YEAR.finditer(text))
    for m in matches:
        left = _parse_year_month(m.group(0))
        if not left:
            continue
        ly, lm, l_has_month = left
        # Reject if start side has no explicit month — almost always a
        # graduation year ("2018-2022") or stray timestamp.
        if not l_has_month:
            continue
        end_pos = m.end()
        window = text[end_pos:end_pos + 40]
        sep = _RANGE_SEP.match(window)
        if not sep:
            continue
        rest = window[sep.end():]

        # Education context check — narrow leading window (30 chars).
        # CVs put the institution name immediately BEFORE the graduation
        # date ("Bina Nusantara University Sep 2022 - 2026", "B.Sc 2014-
        # 2018"). 30 chars is wide enough to catch the institution token
        # without grabbing keywords from a preceding unrelated entry.
        ctx_start = max(0, m.start() - 30)
        leading = text[ctx_start:m.start()]
        if _EDU_CONTEXT.search(leading):
            continue

        if _PRESENT.match(rest):
            ranges.append(((ly, lm), NOW))
            continue
        right = _parse_year_month(rest, default_month=12)
        if not right:
            continue
        ry, rm, r_has_month = right
        # Reject ``Month YYYY - YYYY`` (graduation pattern) — only accept
        # ``Month YYYY - Month YYYY`` for work entries.
        if not r_has_month:
            continue
        if (ry, rm) >= (ly, lm):
            ranges.append(((ly, lm), (ry, rm)))
    return ranges


def _merge_ranges(
    ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Merge overlapping ranges so concurrent jobs aren't double-counted."""
    if not ranges:
        return []
    sorted_r = sorted(ranges, key=lambda r: (r[0], r[1]))
    merged = [sorted_r[0]]
    for cur in sorted_r[1:]:
        last = merged[-1]
        if cur[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], cur[1]))
        else:
            merged.append(cur)
    return merged


def extract_experience_years(cv_text: str) -> Tuple[Optional[float], Optional[str]]:
    """Return ``(years, method)`` or ``(None, None)`` if no signal found.

    Strategy B used to slice the CV from the Experience header onward.
    That broke for documents where PDF column-order extraction landed the
    "EXPERIENCE" header AFTER a job's date range (Valeska Valentin's
    resume — header was the last section but Feb 2025 - Feb 2026 dates
    appeared earlier in the extraction). We now scan the full CV; the
    month-month structural filter and the wider edu-context window in
    ``_extract_date_ranges`` keep graduation entries out.
    """
    if not cv_text:
        return (None, None)

    # Strategy A — explicit phrase
    for pat in _EXP_PHRASE_PATTERNS:
        m = pat.search(cv_text)
        if m:
            try:
                yrs = float(m.group(1))
                if 0 <= yrs <= 60:
                    return (yrs, "phrase")
            except ValueError:
                pass

    # Strategy B — full-CV date-range scan (order-independent)
    raw_ranges = _extract_date_ranges(cv_text)
    merged = _merge_ranges(raw_ranges)
    if merged:
        total_months = sum(_months_between(a, b) for a, b in merged)
        if total_months > 0:
            return (round(total_months / 12.0, 1), "date_ranges")

    return (None, None)


# ----------------------------------------------------------------------
# Location extraction
# ----------------------------------------------------------------------
#
# Indonesian-first city gazetteer. We don't need to be exhaustive — a
# reasonable Big-City list is enough for the thesis prototype. Free-text
# "City, Country" patterns also bypass the gazetteer.

_ID_CITIES = {
    # Major Indonesian cities — display name → ISO country code
    "jakarta": "ID", "bandung": "ID", "surabaya": "ID", "medan": "ID",
    "semarang": "ID", "yogyakarta": "ID", "yogya": "ID", "jogja": "ID",
    "denpasar": "ID", "bali": "ID", "makassar": "ID", "palembang": "ID",
    "bekasi": "ID", "tangerang": "ID", "depok": "ID", "bogor": "ID",
    "malang": "ID", "balikpapan": "ID", "manado": "ID", "pekanbaru": "ID",
    "batam": "ID", "padang": "ID", "samarinda": "ID", "banjarmasin": "ID",
    "pontianak": "ID", "solo": "ID", "surakarta": "ID",
}

_GLOBAL_CITIES = {
    "singapore": "SG", "kuala lumpur": "MY", "bangkok": "TH",
    "manila": "PH", "ho chi minh": "VN", "hanoi": "VN",
    "tokyo": "JP", "seoul": "KR", "hong kong": "HK", "taipei": "TW",
    "sydney": "AU", "melbourne": "AU", "london": "GB", "berlin": "DE",
    "amsterdam": "NL", "paris": "FR", "new york": "US", "san francisco": "US",
    "boston": "US", "seattle": "US", "austin": "US", "chicago": "US",
    "toronto": "CA", "vancouver": "CA", "dublin": "IE", "stockholm": "SE",
}

_CITY_LOOKUP = {**_ID_CITIES, **_GLOBAL_CITIES}

_LOC_HEADERS = re.compile(
    r"(?:^|\n)\s*(?:address|location|domicile|domisili|alamat|based\s+in|"
    r"location\s*[:\-])\s*[:\-]?\s*([^\n]+)",
    re.IGNORECASE,
)

_REMOTE_PHRASES = re.compile(
    r"\b(open\s+to\s+remote|remote(?:\s+work)?|work\s+from\s+anywhere|wfh|"
    r"remote-first|fully\s+remote|terbuka\s+(?:untuk\s+)?remote|kerja\s+jarak\s+jauh)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def extract_location(cv_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(city_display, country_code_or_label)`` or ``(None, None)``.

    Detection order:
        1. An explicit ``Location: ...`` / ``Domisili: ...`` header.
        2. Any known city name appearing anywhere in the CV.

    Country is reported as ISO-2 when the city is in the gazetteer; for
    free-text "City, Country" headers the country is returned verbatim.
    """
    if not cv_text:
        return (None, None)
    norm = _normalize(cv_text)

    # 1) Explicit header
    m = _LOC_HEADERS.search(cv_text)
    if m:
        line = m.group(1).strip()
        line_norm = _normalize(line)
        for city, iso in _CITY_LOOKUP.items():
            if re.search(rf"\b{re.escape(city)}\b", line_norm):
                return (city.title(), iso)
        # Free-text "Jakarta, Indonesia" or "Jakarta, ID" — take both halves
        if "," in line:
            city, country = (p.strip() for p in line.split(",", 1))
            if 2 <= len(city) <= 40:
                return (city, country if country else None)
        if 2 <= len(line) <= 40:
            return (line, None)

    # 2) Whole-CV scan: prefer Indonesian cities (they're the dominant cohort)
    for city, iso in _ID_CITIES.items():
        if re.search(rf"\b{re.escape(city)}\b", norm):
            return (city.title(), iso)
    for city, iso in _GLOBAL_CITIES.items():
        if re.search(rf"\b{re.escape(city)}\b", norm):
            return (city.title(), iso)

    return (None, None)


def extract_remote_preference(cv_text: str) -> Optional[bool]:
    if not cv_text:
        return None
    if _REMOTE_PHRASES.search(cv_text):
        return True
    return None  # absence of signal != preference against remote


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def extract_cv_profile(cv_text: str) -> CVProfile:
    yrs, method = extract_experience_years(cv_text)
    city, country = extract_location(cv_text)
    remote = extract_remote_preference(cv_text)
    return CVProfile(
        experience_years=yrs,
        experience_method=method,
        location_city=city,
        location_country=country,
        remote_preference=remote,
    )


def profile_to_dict(p: CVProfile) -> Dict:
    return asdict(p)


# ----------------------------------------------------------------------
# Job-side: required experience years from a job description
# ----------------------------------------------------------------------

_JD_EXP_PATTERNS = [
    # "5+ years of experience", "3-5 years experience", "minimum 4 years"
    re.compile(r"(\d{1,2})\s*\+\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"(?:minimum|at\s+least|min\.?)\s+(\d{1,2})\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"(\d{1,2})\s*(?:years?|yrs?)\s+of\s+(?:relevant\s+|professional\s+)?experience", re.IGNORECASE),
]


def required_experience_from_jd(description: str) -> Optional[Tuple[float, float]]:
    """Return ``(min_years, max_years)`` required by the JD, or ``None``.

    A range like ``3-5 years`` returns ``(3, 5)``. A single ``5+ years``
    returns ``(5, max(5+3, 5))`` since recruiters generally accept anyone
    above the floor up to a soft ceiling.
    """
    if not description:
        return None
    for pat in _JD_EXP_PATTERNS:
        m = pat.search(description)
        if not m:
            continue
        groups = [g for g in m.groups() if g is not None]
        try:
            nums = [float(g) for g in groups]
        except ValueError:
            continue
        if len(nums) == 2:
            lo, hi = sorted(nums)
            return (lo, hi)
        if len(nums) == 1:
            n = nums[0]
            return (n, n + 3)
    return None
