"""Multi-layer evidence-based skill extraction from CV text.

Design (thesis rationale)
-------------------------
Naive skill matching uses a single signal — usually embedding similarity —
to decide whether a skill is present in a CV. This is brittle:

* Pure embedding matching conflates atomic proper nouns. "React" and "Redux"
  live in similar semantic neighborhoods, so cosine similarity often cannot
  distinguish them. Tech names are defined by lexical identity, not context.
* Pure regex / keyword matching misses paraphrases ("containerization"
  → Docker, "version control" → Git) and Indonesian-language descriptions
  of technical concepts.
* Pure LLM extraction is opaque, non-reproducible, and costly at inference
  time. For a research artifact, deterministic pipelines are preferable.

This module implements a layered ensemble. Each layer has a precondition,
a calibrated confidence range, and produces auditable evidence (the CV
span that triggered the match plus surrounding context).

    Layer 1 (lexical) : word-boundary regex over auto-generated variants.
                        Confidence band 0.90-1.0. Best for proper nouns.
    Layer 2 (fuzzy)   : rapidfuzz token-set ratio for typos / morphology.
                        Confidence band 0.60-0.90. Only runs on skills L1
                        did not catch.
    Layer 3 (semantic): E5 embedding similarity over CV segments.
                        Confidence band 0.50-0.80. Only runs on skills
                        L1/L2 did not catch. Reserved for paraphrases.

Each match is annotated with the CV section it came from (skills_list,
experience, other) — enabling a "mentioned vs demonstrated" distinction
that downstream analysis can exploit.

The pipeline is ablation-ready: each layer can be toggled independently
to support comparative experiments.
"""

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import embed_texts
from src.preprocess import preprocess_text


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class SkillEvidence:
    """One audited evidence record that a skill is present in the CV."""
    canonical: str           # canonical skill name (matches careers.json form)
    source: str              # "lexical" | "fuzzy" | "semantic"
    confidence: float        # calibrated [0, 1]
    matched_text: str        # exact CV substring that triggered the match
    cv_span: Tuple[int, int] # (start, end) char offsets in original CV
    context: str             # surrounding sentence for explanation
    section: str = "other"   # "skills_list" | "experience" | "other"


@dataclass
class SkillRegistry:
    """Catalog of canonical skills with surface-form variants for L1 matching."""
    canonical_to_variants: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def canonical_skills(self) -> List[str]:
        return list(self.canonical_to_variants.keys())

    def variants_for(self, canonical: str) -> Set[str]:
        return self.canonical_to_variants.get(canonical, set())

    def add(self, canonical: str) -> None:
        if canonical and canonical not in self.canonical_to_variants:
            self.canonical_to_variants[canonical] = _generate_variants(canonical)


# ----------------------------------------------------------------------
# Variant generation (Layer 1 prerequisite)
# ----------------------------------------------------------------------

_DOT_LIB_SUFFIX = re.compile(r"\.(js|io|ai|net|py|ts)$", re.IGNORECASE)
_PUNCT_CLEAN = re.compile(r"[\(\)\[\]\{\}\"',;:]")
_AMBIGUOUS_STEMS = {
    # Stems too generic to match safely on their own (would over-trigger).
    "react", "node", "ember", "knockout", "backbone", "next", "nuxt",
    "vue", "angular", "express",
}


def _normalize_form(s: str) -> str:
    """NFKC + lowercase + strip stray punctuation + collapse whitespace."""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = _PUNCT_CLEAN.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _generate_variants(canonical: str) -> Set[str]:
    """Auto-generate plausible surface forms for a canonical skill name.

    Examples
    --------
    >>> sorted(_generate_variants("React.js"))
    ['react js', 'react.js', 'reactjs']
    >>> sorted(_generate_variants("REST API"))
    ['rest api', 'rest-api', 'restapi']
    >>> sorted(_generate_variants("CI/CD"))
    ['ci cd', 'ci-cd', 'ci/cd', 'cicd']

    Note: bare stems of well-known frameworks (react, vue, node, ...) are
    NOT emitted as variants on their own. They are too ambiguous in CVs
    where a sentence may say "react to feedback" or "node of the network".
    The lexical layer relies on the punctuated/joined forms instead.
    """
    variants: Set[str] = set()
    base = _normalize_form(canonical)
    if not base:
        return variants
    variants.add(base)

    # Acronym shortcuts
    special = {"c++": {"cpp"}, "c#": {"csharp"}, ".net": {"dotnet"}}
    if base in special:
        variants.update(special[base])

    # ".js" / ".io" / ".net" / ".py" libraries
    stripped = _DOT_LIB_SUFFIX.sub("", base)
    if stripped != base and len(stripped) >= 3 and stripped not in _AMBIGUOUS_STEMS:
        variants.add(stripped)

    # "." separator → joined / spaced / hyphenated
    if "." in base:
        variants.add(base.replace(".", ""))
        variants.add(base.replace(".", " "))

    # "/" separator
    if "/" in base:
        variants.add(base.replace("/", ""))
        variants.add(base.replace("/", " "))
        variants.add(base.replace("/", "-"))

    # Whitespace ↔ hyphen ↔ joined
    if " " in base:
        variants.add(base.replace(" ", "-"))
        joined = base.replace(" ", "")
        if len(joined) >= 4:  # avoid too-short joinings
            variants.add(joined)

    if "-" in base:
        variants.add(base.replace("-", " "))
        joined = base.replace("-", "")
        if len(joined) >= 4:
            variants.add(joined)

    # Remove anything too short or ambiguous
    variants = {_normalize_form(v) for v in variants}
    variants = {v for v in variants if len(v) >= 2 and v not in _AMBIGUOUS_STEMS}
    return variants


def build_skill_registry(careers: List[Dict]) -> SkillRegistry:
    """Construct a SkillRegistry from a list of career dicts.

    Each career dict is expected to have a "skills": [str, ...] field.
    Skills appearing in multiple careers are deduplicated (canonical form).
    """
    registry = SkillRegistry()
    for c in careers or []:
        for skill in (c.get("skills") or []):
            if isinstance(skill, str) and skill.strip():
                registry.add(skill.strip())
    return registry


# ----------------------------------------------------------------------
# Section detection (mentioned vs demonstrated)
# ----------------------------------------------------------------------

_SECTION_HEADERS = {
    "skills_list": [
        r"\bskills?\b", r"\btechnical\s+skills?\b", r"\btech\s+stack\b",
        r"\bkemampuan\b", r"\bkeahlian\b", r"\bkompetensi\b",
    ],
    "experience": [
        r"\bexperience\b", r"\bwork\s+experience\b", r"\bemployment\b",
        r"\bprojects?\b", r"\bportfolio\b",
        r"\bpengalaman\b", r"\bproyek\b", r"\bpekerjaan\b",
    ],
}


def _classify_position_section(cv_text: str, position: int) -> str:
    """Find which CV section a character offset belongs to.

    Heuristic: scan backwards for the most recent section header and
    return its label. Defaults to "other" if no header is found upstream.
    """
    window = cv_text[:position].lower()
    last_section = "other"
    last_pos = -1
    for section, patterns in _SECTION_HEADERS.items():
        for pat in patterns:
            for m in re.finditer(pat, window):
                if m.start() > last_pos:
                    last_pos = m.start()
                    last_section = section
    return last_section


def _surrounding_context(cv_text: str, start: int, end: int, window: int = 80) -> str:
    """Extract a small surrounding window for human-readable explanation."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(cv_text), end + window)
    snippet = cv_text[ctx_start:ctx_end].replace("\n", " ").strip()
    return re.sub(r"\s+", " ", snippet)


# ----------------------------------------------------------------------
# Layer 1: lexical (regex with word boundaries)
# ----------------------------------------------------------------------

# Word-boundary templates are intentionally asymmetric:
#   * Left boundary  — keeps "." in the inclusion class so we do NOT match
#                      "node.js" inside a dotted path like "X.node.js".
#   * Right boundary — drops "." from the inclusion class so a trailing
#                      sentence period ("Kubernetes.") doesn't suppress a
#                      legitimate match.
# Letters, digits, "+", "#", "/" are treated as continuation chars on
# both sides so atomic tokens like "C++" or "CI/CD" match correctly.
_WORD_BOUNDARY_LEFT = r"(?:^|[^a-zA-Z0-9.+#/-])"
_WORD_BOUNDARY_RIGHT = r"(?=$|[^a-zA-Z0-9+#/-])"


def _build_lexical_pattern(variant: str) -> re.Pattern:
    variant = _normalize_form(variant)
    escaped = re.escape(variant)
    return re.compile(
        f"{_WORD_BOUNDARY_LEFT}({escaped}){_WORD_BOUNDARY_RIGHT}",
        re.IGNORECASE,
    )

def _extract_lexical(
    cv_text: str,
    registry: SkillRegistry,
) -> Dict[str, SkillEvidence]:
    """Layer 1 — exact boundary-aware matches over canonical+variant forms.

    Strategy: for each canonical skill, try variants in length-descending
    order. The first variant that matches wins (longer = more specific).
    """
    found: Dict[str, SkillEvidence] = {}
    for canonical, variants in registry.canonical_to_variants.items():
        for variant in sorted(variants, key=len, reverse=True):
            pattern = _build_lexical_pattern(variant)
            match = pattern.search(cv_text)
            if not match:
                continue
            start, end = match.span(1)
            section = _classify_position_section(cv_text, start)
            # Confidence: full canonical form > derived variant
            base_conf = 0.95 if variant == _normalize_form(canonical) else 0.90
            # Demonstrated > listed
            if section == "experience":
                base_conf = min(1.0, base_conf + 0.03)
            found[canonical] = SkillEvidence(
                canonical=canonical,
                source="lexical",
                confidence=round(base_conf, 4),
                matched_text=match.group(1),
                cv_span=(start, end),
                context=_surrounding_context(cv_text, start, end),
                section=section,
            )
            break
    return found


# ----------------------------------------------------------------------
# Layer 2: fuzzy (rapidfuzz, optional dependency)
# ----------------------------------------------------------------------

def _extract_fuzzy(
    cv_text: str,
    registry: SkillRegistry,
    already_found: Set[str],
    threshold: float = 0.92,
) -> Dict[str, SkillEvidence]:
    """Layer 2 — fuzzy match for skills not caught by Layer 1.

    Uses rapidfuzz token_set_ratio over sliding token windows. The window
    size matches the number of tokens in the longest variant (a 1-token
    target uses a 1-token window, a "rest api" target uses 2-3).

    Threshold default 0.92 is intentionally tight to avoid false positives
    on near-neighbour skills (e.g., "redux" vs "react" score ~0.6, far
    below threshold).
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return {}

    found: Dict[str, SkillEvidence] = {}
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", cv_text):
        tokens.append((m.group(), m.start(), m.end()))
    if not tokens:
        return found

    for canonical, variants in registry.canonical_to_variants.items():
        if canonical in already_found:
            continue
        # Use the longest variant as the fuzzy target (most specific)
        target = max(variants, key=len) if variants else _normalize_form(canonical)
        if len(target) < 4:
            # Targets shorter than 4 chars are unsafe for fuzzy matching
            continue
        n_words = max(1, target.count(" ") + 1)
        best_score = 0.0
        best_span: Optional[Tuple[int, int]] = None
        best_text = ""
        for w in (n_words, n_words + 1):
            if w > len(tokens):
                continue
            for i in range(len(tokens) - w + 1):
                start = tokens[i][1]
                end = tokens[i + w - 1][2]
                window_text = cv_text[start:end]
                if n_words == 1:
                    score = fuzz.ratio(target.lower(), window_text.lower()) / 100.0
                else:
                    score = fuzz.token_set_ratio(target.lower(), window_text.lower()) / 100.0
                if score > best_score:
                    best_score = score
                    best_span = (start, end)
                    best_text = window_text
        effective_threshold = threshold if n_words > 1 else min(threshold, 0.90)
        if best_score >= effective_threshold and best_span is not None:
            section = _classify_position_section(cv_text, best_span[0])
            # Calibrate: linearly map [threshold, 1.0] → [0.6, 0.9]
            conf = 0.6 + 0.3 * (best_score - threshold) / max(1e-6, 1 - threshold)
            found[canonical] = SkillEvidence(
                canonical=canonical,
                source="fuzzy",
                confidence=round(conf, 4),
                matched_text=best_text,
                cv_span=best_span,
                context=_surrounding_context(cv_text, best_span[0], best_span[1]),
                section=section,
            )
    return found

# ----------------------------------------------------------------------
# Layer 3: semantic (E5 embedding fallback)
# ----------------------------------------------------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?;])\s+|\n+')


def _segment_cv(cv_text: str, min_length: int = 10) -> List[Tuple[str, int, int]]:
    """Split CV into sentence-like segments while preserving char offsets."""
    segments: List[Tuple[str, int, int]] = []
    pos = 0
    for m in _SENT_SPLIT.finditer(cv_text):
        seg = cv_text[pos:m.start()]
        if len(seg.strip()) >= min_length:
            segments.append((seg, pos, m.start()))
        pos = m.end()
    tail = cv_text[pos:]
    if len(tail.strip()) >= min_length:
        segments.append((tail, pos, len(cv_text)))
    if not segments and cv_text.strip():
        segments = [(cv_text, 0, len(cv_text))]
    return segments


def _extract_semantic(
    cv_text: str,
    registry: SkillRegistry,
    model: SentenceTransformer,
    already_found: Set[str],
    threshold: float = 0.78,
) -> Dict[str, SkillEvidence]:
    """Layer 3 — embedding fallback for skills L1 and L2 did not catch.

    The threshold is deliberately stricter than the previous flat
    pipeline (0.6 → 0.78). Proper nouns are now handled by Layer 1, so
    semantic similarity only needs to clear the bar for genuine paraphrase
    matches, not for atomic name matches it would do poorly on anyway.
    """
    remaining = [c for c in registry.canonical_skills if c not in already_found]
    if not remaining:
        return {}

    segments = _segment_cv(cv_text)
    if not segments:
        return {}

    seg_texts = [preprocess_text(s[0]) for s in segments]
    skill_emb = embed_texts(model, remaining, is_passage=False)
    seg_emb = embed_texts(model, seg_texts, is_passage=True)
    sim = skill_emb @ seg_emb.T  # (n_skills, n_segments)

    found: Dict[str, SkillEvidence] = {}
    for i, canonical in enumerate(remaining):
        best_j = int(np.argmax(sim[i]))
        score = float(sim[i, best_j])
        if score < threshold:
            continue
        seg_text, start, end = segments[best_j]
        section = _classify_position_section(cv_text, start)
        # Calibrate: [threshold, 1.0] → [0.5, 0.8]
        conf = 0.5 + 0.3 * (score - threshold) / max(1e-6, 1 - threshold)
        found[canonical] = SkillEvidence(
            canonical=canonical,
            source="semantic",
            confidence=round(conf, 4),
            matched_text=seg_text.strip()[:80],
            cv_span=(start, end),
            context=seg_text.strip(),
            section=section,
        )
    return found


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def extract_cv_skills(
    cv_text: str,
    registry: SkillRegistry,
    model: SentenceTransformer,
    fuzzy_threshold: float = 0.92,
    semantic_threshold: float = 0.78,
    enable_lexical: bool = True,
    enable_fuzzy: bool = True,
    enable_semantic: bool = True,
) -> Dict[str, SkillEvidence]:
    """Run the full layered pipeline against a CV.

    Parameters
    ----------
    cv_text : str
        Raw CV text. The pipeline assumes preprocessing was already applied
        upstream where appropriate; section detection still uses raw text
        so original headings stay visible.
    registry : SkillRegistry
        Catalog of skills the pipeline should look for. Build via
        ``build_skill_registry`` from your career list.
    model : SentenceTransformer
        Embedding model used by the semantic layer.
    fuzzy_threshold : float
        rapidfuzz token_set_ratio cutoff in [0, 1]. Default 0.92.
    semantic_threshold : float
        Cosine similarity cutoff for the semantic layer. Default 0.78.
    enable_lexical, enable_fuzzy, enable_semantic : bool
        Toggles for each layer. Useful for ablation studies.

    Returns
    -------
    Dict[str, SkillEvidence]
        Map canonical skill name → SkillEvidence (best evidence per skill).
    """
    if not cv_text or not cv_text.strip() or not registry.canonical_skills:
        return {}

    found: Dict[str, SkillEvidence] = {}

    if enable_lexical:
        found.update(_extract_lexical(cv_text, registry))

    if enable_fuzzy:
        fuzzy_hits = _extract_fuzzy(
            cv_text, registry,
            already_found=set(found.keys()),
            threshold=fuzzy_threshold,
        )
        found.update(fuzzy_hits)

    if enable_semantic:
        semantic_hits = _extract_semantic(
            cv_text, registry, model,
            already_found=set(found.keys()),
            threshold=semantic_threshold,
        )
        found.update(semantic_hits)

    return found


def evidence_to_dict(ev: SkillEvidence) -> Dict:
    """Render an evidence record as a JSON-serializable dict."""
    d = asdict(ev)
    d["cv_span"] = list(d["cv_span"])  # tuple → list for JSON
    return d
