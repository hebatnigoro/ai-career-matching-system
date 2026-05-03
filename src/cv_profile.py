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
from datetime import datetime
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
    # Use the actual current month so an open-ended "2024 - present" reflects
    # how long the role has been running *now*, not at the time the matcher
    # was deployed.
    today = datetime.now()
    NOW = (today.year, today.month)

    matches = list(_MONTH_YEAR.finditer(text))
    for m in matches:
        left = _parse_year_month(m.group(0))
        if not left:
            continue
        ly, lm, l_has_month = left
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

        # ``YYYY - Present`` is a valid ongoing-role pattern (side hustles,
        # bare-year shorthand). Accept it — start defaults to January.
        if _PRESENT.match(rest):
            ranges.append(((ly, lm), NOW))
            continue

        # For closed ranges, reject if start side has no explicit month —
        # ``2018 - 2022`` is the canonical graduation/education pattern.
        if not l_has_month:
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
# Backed by the GeoNames database (via the offline ``geonamescache``
# package): ~32k cities with population ≥15k and 252 countries, each with
# ISO-2 codes and alternate-name lists. No external API calls — the data
# ships with the package and is loaded lazily on first use.
#
# A CV's location is resolved in three stages:
#   1. Explicit header (``Location: ...`` / ``Domisili: ...``) — exact
#      lookup, then fuzzy lookup for typo tolerance.
#   2. Whole-CV n-gram scan against the gazetteer; ranked by population
#      and weighted toward cities matching any country mentioned in the
#      CV body.
#   3. Country-only fallback (e.g. CV says ``Indonesia`` but no city) —
#      returns ``(None, ISO)`` instead of giving up.

from functools import lru_cache

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

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_ALT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z\s'\-]+$")
_FUZZY_CUTOFF = 90
_MAX_NGRAM = 3
_MIN_UNIGRAM_LEN = 4   # single-token matches must be ≥4 chars to avoid stop-word hits
_MIN_PHRASE_LEN = 5    # safety net for multi-token matches
# Common English / Indonesian stop-words and prose tokens that overlap with
# obscure alt-names in the GeoNames data ("and" → Anderson alt, etc.).
_STOPWORDS = frozenset({
    "and", "the", "for", "with", "from", "into", "over", "this", "that",
    "what", "when", "where", "while", "have", "been", "more", "than", "into",
    "your", "their", "they", "them", "also", "any", "all", "are", "but",
    "dan", "atau", "yang", "untuk", "saya", "kami", "kita", "dari", "pada",
    "ini", "itu", "ada", "akan", "tidak", "sudah", "dengan", "adalah",
    "career", "profile", "resume", "summary", "skills", "experience",
    "education", "project", "projects", "company", "engineer", "developer",
    "manager", "designer", "intern", "student", "university", "college",
    "bachelor", "master", "degree", "language", "english", "indonesian",
})
# Common country aliases not present in GeoNames' canonical names.
_COUNTRY_ALIASES = {
    "usa": "US", "us": "US",
    "uk": "GB", "great britain": "GB", "britain": "GB",
    "south korea": "KR", "korea": "KR",
    "russia": "RU", "vietnam": "VN", "uae": "AE",
}


def _normalize(text: str) -> str:
    """NFKC + lowercase + strip diacritics. Used for case/accent-insensitive
    lookups against the gazetteer."""
    if not text:
        return ""
    n = unicodedata.normalize("NFKD", text).lower()
    return "".join(ch for ch in n if not unicodedata.combining(ch))


@lru_cache(maxsize=1)
def _gazetteer() -> Tuple[Dict[str, Tuple[str, str, int]], Dict[str, str], Dict[str, str]]:
    """Build the city + country lookup tables from GeoNames data.

    Returned dicts:
        cities          name_norm → (display_name, iso2, population)
        country_names   name_norm → iso2  (full names + multi-char aliases — safe for body scans)
        country_codes   code_norm → iso2  (iso2 / iso3 — for explicit-header lookup only)

    Cities with shared names (e.g. 'San Jose' in CR/US/PH) are
    deduplicated by population — the largest wins. Alternate names
    are included only if they are ASCII (skipping CJK/Cyrillic
    transliterations to keep the table compact and avoid common-word
    collisions in latin-script CVs).
    """
    import geonamescache  # local import — keeps test-only modules cheap
    gc = geonamescache.GeonamesCache()
    raw_cities = list(gc.get_cities().values())

    # Two-pass build:
    #   Pass 1 — primary canonical names. These are authoritative; they
    #            must never be overwritten by an alternate-name collision.
    #   Pass 2 — ASCII alternate names, skipped when the key is already
    #            owned by a primary or by a stop word. Without this split,
    #            obscure alt names ("Jayapura" listed under Jaipur, "And"
    #            under Anderson) would shadow the real primaries.
    cities_by_name: Dict[str, Tuple[str, str, int]] = {}
    primary_keys: set = set()
    for c in raw_cities:
        pop = int(c.get("population") or 0)
        iso = c.get("countrycode") or ""
        if not iso:
            continue
        key = _normalize(c["name"])
        if not key or len(key) < 3 or key in _STOPWORDS:
            continue
        existing = cities_by_name.get(key)
        if existing is None or pop > existing[2]:
            cities_by_name[key] = (c["name"], iso, pop)
        primary_keys.add(key)

    for c in raw_cities:
        pop = int(c.get("population") or 0)
        iso = c.get("countrycode") or ""
        if not iso:
            continue
        for alt in c.get("alternatenames") or []:
            if not (3 <= len(alt) <= 30) or not _ALT_NAME_RE.match(alt):
                continue
            key = _normalize(alt)
            if not key or len(key) < 3 or key in _STOPWORDS or key in primary_keys:
                continue
            existing = cities_by_name.get(key)
            if existing is None or pop > existing[2]:
                cities_by_name[key] = (c["name"], iso, pop)

    # Country tables are split: names go into the body-safe dict (long
    # enough to use with \b regex without matching prose tokens), codes
    # go into a separate dict used only when parsing explicit ``Location:``
    # headers — never for whole-document scans, where 2-char codes like
    # AD/AT/KM would collide with random word fragments.
    country_names: Dict[str, str] = {}
    country_codes: Dict[str, str] = {}
    for iso, info in gc.get_countries().items():
        country_names[_normalize(info["name"])] = iso
        country_codes[iso.lower()] = iso
        if info.get("iso3"):
            country_codes[info["iso3"].lower()] = iso
    country_names.update({_normalize(k): v for k, v in _COUNTRY_ALIASES.items()})
    country_names.pop("", None)
    country_codes.pop("", None)

    return cities_by_name, country_names, country_codes


def _detect_country_isos(text_norm: str, country_names: Dict[str, str]) -> set:
    """Return ISO codes mentioned anywhere in normalized CV text. Only
    full country names (≥4 chars) are considered — ISO-2 codes like
    ``AD`` collide with prose tokens and would yield false positives."""
    found = set()
    for name, iso in country_names.items():
        if len(name) < 4:
            continue
        if re.search(rf"\b{re.escape(name)}\b", text_norm):
            found.add(iso)
    return found


def _lookup_city(
    text_norm: str,
    cities: Dict[str, Tuple[str, str, int]],
    *,
    fuzzy: bool,
) -> Optional[Tuple[str, str]]:
    """Match a single normalized phrase against the gazetteer. Tries exact
    hit, then optionally fuzzy (rapidfuzz token-set) for typo tolerance."""
    hit = cities.get(text_norm)
    if hit:
        return hit[0], hit[1]
    if fuzzy and 5 <= len(text_norm) <= 30:
        # ``QRatio`` compares full strings (no substring inflation), so a
        # 7-char query can't ride a 3-char prefix to a 90-score match the
        # way ``WRatio`` would. Combined with the ``_FUZZY_CUTOFF`` of 90,
        # this rejects most spurious hits while still tolerating typos.
        from rapidfuzz import process, fuzz
        match = process.extractOne(
            text_norm, cities.keys(), scorer=fuzz.QRatio, score_cutoff=_FUZZY_CUTOFF,
        )
        if match:
            display, iso, _pop = cities[match[0]]
            return display, iso
    return None


def _scan_for_city(
    cv_text: str,
    cities: Dict[str, Tuple[str, str, int]],
    country_names: Dict[str, str],
) -> Optional[Tuple[str, str]]:
    """Whole-CV n-gram scan. Picks the gazetteer match with the highest
    population, with a multiplier when the city's country is also
    mentioned somewhere in the CV (helps disambiguate ``San Francisco``
    when the CV clearly says ``Indonesia``)."""
    norm_text = _normalize(cv_text)
    declared_isos = _detect_country_isos(norm_text, country_names)
    tokens = _TOKEN_RE.findall(norm_text)
    if not tokens:
        return None

    best: Optional[Tuple[float, str, str]] = None  # (score, display, iso)
    seen: set = set()
    for i in range(len(tokens)):
        for n in range(1, _MAX_NGRAM + 1):
            if i + n > len(tokens):
                break
            phrase = " ".join(tokens[i : i + n])
            if phrase in seen:
                continue
            seen.add(phrase)
            min_len = _MIN_UNIGRAM_LEN if n == 1 else _MIN_PHRASE_LEN
            if len(phrase) < min_len or phrase in _STOPWORDS:
                continue
            entry = cities.get(phrase)
            if not entry:
                continue
            display, iso, pop = entry
            # When the CV explicitly names a country, treat that as a hard
            # filter rather than a soft bias — otherwise large foreign
            # cities (Jaipur, Tokyo) outrank the candidate's smaller home
            # city even when ``Indonesia`` is written one line above.
            if declared_isos and iso not in declared_isos:
                continue
            score = float(pop)
            if best is None or score > best[0]:
                best = (score, display, iso)

    if best:
        return best[1], best[2]
    return None


def _resolve_header_line(
    line: str,
    cities: Dict[str, Tuple[str, str, int]],
    country_names: Dict[str, str],
    country_codes: Dict[str, str],
) -> Optional[Tuple[str, str]]:
    """Resolve an explicit ``Location: ...`` line. Tries the line as a
    whole, then each comma-separated segment, with fuzzy fallback. ISO-2
    codes are honored here (``Jakarta, ID``) because the structured
    header context makes them unambiguous."""
    line_norm = _normalize(line)
    segments = [s.strip() for s in line_norm.split(",") if s.strip()]
    header_country = {**country_names, **country_codes}

    for candidate in [line_norm] + segments:
        hit = _lookup_city(candidate, cities, fuzzy=False)
        if hit:
            country_in_line = next(
                (header_country[s] for s in segments
                 if s in header_country and header_country[s] != hit[1]),
                None,
            )
            return hit[0], country_in_line or hit[1]

    # Fuzzy fallback — only on segments shorter than ~30 chars to avoid
    # matching whole sentences.
    for candidate in [s for s in segments if 3 <= len(s) <= 30]:
        hit = _lookup_city(candidate, cities, fuzzy=True)
        if hit:
            return hit
    return None


def extract_location(cv_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(city_display, country_iso2)`` or ``(None, None)``.

    Resolution order:
        1. Explicit ``Location:`` / ``Domisili:`` header — exact then fuzzy
           gazetteer lookup.
        2. Whole-CV n-gram scan, ranked by population and biased toward
           any country the CV explicitly names.
        3. Country-only mention (e.g. ``Indonesia`` without a city).

    Country code is always ISO-2 when known. Free-text headers that
    reference an unknown city are returned verbatim with country=None
    so downstream code can still display "what the CV said".
    """
    if not cv_text:
        return (None, None)
    cities, country_names, country_codes = _gazetteer()

    # 1) Explicit header
    m = _LOC_HEADERS.search(cv_text)
    if m:
        line = m.group(1).strip()
        resolved = _resolve_header_line(line, cities, country_names, country_codes)
        if resolved:
            return resolved
        # Fallback to free-text "City, Country" so the UI still has
        # something to display, even when neither side is in the gazetteer.
        if "," in line:
            city_part, country_part = (p.strip() for p in line.split(",", 1))
            if 2 <= len(city_part) <= 40:
                country_part_norm = _normalize(country_part)
                country_iso = (
                    country_names.get(country_part_norm)
                    or country_codes.get(country_part_norm)
                )
                return (city_part, country_iso or (country_part or None))
        if 2 <= len(line) <= 40:
            return (line, None)

    # 2) Whole-CV scan
    scanned = _scan_for_city(cv_text, cities, country_names)
    if scanned:
        return scanned

    # 3) Country-only mention — better than nothing for the matcher.
    declared = _detect_country_isos(_normalize(cv_text), country_names)
    if len(declared) == 1:
        return (None, next(iter(declared)))

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
