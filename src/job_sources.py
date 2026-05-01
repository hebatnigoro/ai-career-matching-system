"""Public job-board adapters: Greenhouse, Lever, Ashby.

Why these three?
----------------
All three expose public, no-auth REST endpoints that return JSON. Together
they cover thousands of real companies hiring right now, which is what
makes the matching feature meaningful for a thesis prototype rather than a
toy seed dataset.

Each adapter normalizes its source-specific schema into the same ``Job``
dict shape used by the rest of the pipeline. This is the only place in the
system that knows about source-specific fields.

Documented endpoints (verified against live APIs):

* Greenhouse — ``GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true``
  Returns ``{ jobs: [ { id, title, location.name, content (HTML),
  departments[], offices[], absolute_url, ... } ] }``.

* Lever — ``GET https://api.lever.co/v0/postings/{site}?mode=json``
  Returns a JSON array; each job has ``id, text (title), categories
  (location/team/department/commitment), workplaceType, descriptionPlain,
  lists, hostedUrl, applyUrl, country, salaryRange``.

* Ashby — ``GET https://api.ashbyhq.com/posting-api/job-board/{board}?includeCompensation=true``
  Returns ``{ apiVersion, jobs: [ { id, title, location, isRemote,
  workplaceType, employmentType, descriptionPlain, descriptionHtml, jobUrl,
  applyUrl, department, team, compensation } ] }``. A non-default
  User-Agent is required (the default ``Python-urllib/...`` is blocked).
"""

from __future__ import annotations

import html as _html
import json
import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import Dict, List, Optional


# ----------------------------------------------------------------------
# HTTP helper
# ----------------------------------------------------------------------

USER_AGENT = "CareerMatchBot/1.0 (+thesis-prototype)"
REQUEST_TIMEOUT = 20  # seconds


def _http_get_json(url: str) -> object:
    """Fetch JSON with a real User-Agent. Raises ``RuntimeError`` on failure."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error for {url}: {e.reason}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Non-JSON response from {url}: {e}") from e


# ----------------------------------------------------------------------
# HTML → plain text (Greenhouse returns HTML in `content`)
# ----------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data):
        self._parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "li", "div", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "li", "div", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def _html_to_text(content: Optional[str]) -> str:
    """Unescape HTML entities (Greenhouse double-encodes) then strip tags."""
    if not content:
        return ""
    # Greenhouse returns ``&lt;div&gt;...&lt;/div&gt;`` so we unescape twice
    # to normalize both single- and double-encoded payloads.
    text = _html.unescape(_html.unescape(content))
    parser = _HTMLStripper()
    try:
        parser.feed(text)
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)
    return parser.text()


# ----------------------------------------------------------------------
# Unified Job dict shape
# ----------------------------------------------------------------------
# {
#   "id":              "<source>:<board>:<job_id>",   # globally unique
#   "source":          "greenhouse" | "lever" | "ashby",
#   "company":         str,                            # board / org label
#   "title":           str,
#   "description":     str,                            # plain text
#   "department":      Optional[str],
#   "team":            Optional[str],
#   "location":        Optional[str],                  # display string
#   "country":         Optional[str],                  # ISO-2 if known
#   "remote":          bool,
#   "workplace_type":  "remote" | "hybrid" | "onsite" | "unspecified",
#   "employment_type": Optional[str],                  # FullTime / PartTime / Contract / Intern / ...
#   "url":             Optional[str],                  # public posting page
#   "apply_url":       Optional[str],
#   "posted_at":       Optional[str],                  # ISO date
#   "compensation":    Optional[str],
# }


# ----------------------------------------------------------------------
# Workplace-type normalization
# ----------------------------------------------------------------------

def _norm_workplace(value: Optional[str], is_remote: Optional[bool] = None) -> str:
    if is_remote is True:
        return "remote"
    if not value:
        return "unspecified"
    v = value.strip().lower().replace("-", "").replace("_", "")
    if v in {"remote"}:
        return "remote"
    if v in {"hybrid"}:
        return "hybrid"
    if v in {"onsite", "inoffice", "in-office"}:
        return "onsite"
    return "unspecified"


# ----------------------------------------------------------------------
# Greenhouse adapter
# ----------------------------------------------------------------------

GREENHOUSE_BASE = "https://boards-api.greenhouse.io/v1/boards"


def fetch_greenhouse(board_token: str, company_label: Optional[str] = None) -> List[Dict]:
    """Fetch all published jobs for a Greenhouse board.

    ``board_token`` is the segment after ``/boards/`` in the public board URL
    (e.g., for ``boards.greenhouse.io/airbnb`` the token is ``airbnb``).
    """
    url = f"{GREENHOUSE_BASE}/{board_token}/jobs?content=true"
    data = _http_get_json(url)
    if not isinstance(data, dict) or "jobs" not in data:
        return []
    company = company_label or data.get("meta", {}).get("total") and board_token or board_token
    out: List[Dict] = []
    for j in data["jobs"]:
        loc = (j.get("location") or {}).get("name")
        offices = [o.get("location") or o.get("name") for o in (j.get("offices") or []) if o]
        country = None
        # Best-effort country guess from offices' location strings (after last comma)
        if offices and offices[0]:
            tail = offices[0].rsplit(",", 1)[-1].strip()
            if 2 <= len(tail) <= 40:
                country = tail
        text = _html_to_text(j.get("content") or "")
        # Greenhouse exposes department(s) and offices arrays
        depts = [d.get("name") for d in (j.get("departments") or []) if d.get("name")]
        out.append({
            "id": f"greenhouse:{board_token}:{j.get('id')}",
            "source": "greenhouse",
            "company": j.get("company_name") or board_token,
            "title": j.get("title") or "",
            "description": text,
            "department": depts[0] if depts else None,
            "team": None,
            "location": loc,
            "country": country,
            "remote": bool(loc and "remote" in loc.lower()),
            "workplace_type": _norm_workplace(None, is_remote=bool(loc and "remote" in loc.lower())),
            "employment_type": None,  # Greenhouse public board API doesn't expose this
            "url": j.get("absolute_url"),
            "apply_url": j.get("absolute_url"),
            "posted_at": (j.get("updated_at") or j.get("first_published") or "")[:10] or None,
            "compensation": None,
        })
    return out


# ----------------------------------------------------------------------
# Lever adapter
# ----------------------------------------------------------------------

LEVER_BASE = "https://api.lever.co/v0/postings"


def fetch_lever(site: str, company_label: Optional[str] = None) -> List[Dict]:
    """Fetch all published postings for a Lever site (e.g., ``leverdemo``)."""
    url = f"{LEVER_BASE}/{site}?mode=json"
    data = _http_get_json(url)
    if not isinstance(data, list):
        return []
    out: List[Dict] = []
    for j in data:
        cats = j.get("categories") or {}
        location = cats.get("location")
        text = j.get("descriptionPlain") or _html_to_text(j.get("description"))
        # Lever's `lists` carry requirements/benefits as separate sections.
        if isinstance(j.get("lists"), list):
            for section in j["lists"]:
                title = section.get("text") or ""
                content = _html_to_text(section.get("content") or "")
                if content:
                    text = f"{text}\n\n{title}\n{content}".strip()
        salary = j.get("salaryRange") or {}
        comp = None
        if salary.get("min") and salary.get("max"):
            comp = f"{salary.get('currency','')} {salary['min']}-{salary['max']} / {salary.get('interval','')}".strip()
        created = j.get("createdAt")
        posted_iso = None
        if isinstance(created, (int, float)):
            posted_iso = time.strftime("%Y-%m-%d", time.gmtime(created / 1000.0))
        out.append({
            "id": f"lever:{site}:{j.get('id')}",
            "source": "lever",
            "company": company_label or site,
            "title": j.get("text") or "",
            "description": text,
            "department": cats.get("department"),
            "team": cats.get("team"),
            "location": location,
            "country": j.get("country"),
            "remote": _norm_workplace(j.get("workplaceType")) == "remote",
            "workplace_type": _norm_workplace(j.get("workplaceType")),
            "employment_type": cats.get("commitment"),
            "url": j.get("hostedUrl"),
            "apply_url": j.get("applyUrl"),
            "posted_at": posted_iso,
            "compensation": comp,
        })
    return out


# ----------------------------------------------------------------------
# Ashby adapter
# ----------------------------------------------------------------------

ASHBY_BASE = "https://api.ashbyhq.com/posting-api/job-board"


def fetch_ashby(board: str, company_label: Optional[str] = None) -> List[Dict]:
    """Fetch published postings for an Ashby job board."""
    url = f"{ASHBY_BASE}/{board}?includeCompensation=true"
    data = _http_get_json(url)
    if not isinstance(data, dict) or "jobs" not in data:
        return []
    out: List[Dict] = []
    for j in data["jobs"]:
        if j.get("isListed") is False:
            continue
        loc = j.get("location")
        addr = j.get("address") or {}
        postal = addr.get("postalAddress") or {}
        country = postal.get("addressCountry")
        comp_summary = j.get("compensationTierSummary") or None
        if not comp_summary and isinstance(j.get("compensation"), dict):
            comp_summary = (j["compensation"] or {}).get("compensationTierSummary")
        out.append({
            "id": f"ashby:{board}:{j.get('id')}",
            "source": "ashby",
            "company": company_label or board,
            "title": j.get("title") or "",
            "description": j.get("descriptionPlain") or _html_to_text(j.get("descriptionHtml")),
            "department": j.get("department"),
            "team": j.get("team"),
            "location": loc,
            "country": country,
            "remote": bool(j.get("isRemote")),
            "workplace_type": _norm_workplace(j.get("workplaceType"), is_remote=j.get("isRemote")),
            "employment_type": j.get("employmentType"),
            "url": j.get("jobUrl"),
            "apply_url": j.get("applyUrl") or j.get("jobUrl"),
            "posted_at": (j.get("publishedAt") or "")[:10] or None,
            "compensation": comp_summary,
        })
    return out


# ----------------------------------------------------------------------
# Multi-source orchestration
# ----------------------------------------------------------------------

_FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
}


def fetch_all(sources: List[Dict]) -> Dict:
    """Fetch jobs from multiple sources defined in a config list.

    Each item: ``{ "type": "greenhouse"|"lever"|"ashby", "board": "...",
    "company": "..." }``. Failures are reported per-source rather than
    aborting the whole batch — useful when one ATS flakes during a refresh.
    """
    jobs: List[Dict] = []
    errors: List[Dict] = []
    for src in sources:
        kind = (src.get("type") or "").lower()
        board = src.get("board") or src.get("token") or src.get("site")
        label = src.get("company") or board
        if not board or kind not in _FETCHERS:
            errors.append({"source": src, "error": "missing or unknown type/board"})
            continue
        try:
            fetched = _FETCHERS[kind](board, label)
            jobs.extend(fetched)
        except Exception as e:
            errors.append({"source": src, "error": str(e)})
    # Dedup by id (defensive — companies sometimes publish on multiple ATS)
    seen: set = set()
    unique: List[Dict] = []
    for j in jobs:
        if j["id"] in seen:
            continue
        seen.add(j["id"])
        unique.append(j)
    return {
        "jobs": unique,
        "errors": errors,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
