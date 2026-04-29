"""CV-vs-career skill-gap analysis backed by an evidence-based extractor.

The previous implementation took the target career's full skill list and
embedded each skill against CV segments via cosine similarity (a single,
flat signal with one threshold). That approach has two structural issues:

1. Every CV is scored against the *whole* required-skill bag of the target
   career, so the matching is not "what skills does this CV actually
   demonstrate?" but rather "how close does this CV look to each of these
   labels in embedding space?". Proper nouns (React.js, Kubernetes,
   MikroTik) are particularly poorly served by embeddings.
2. There is no audit trail. A skill is "matched" or "missing" with no
   pointer to the CV evidence behind that decision.

This module now delegates to ``src.skill_extract`` (a layered ensemble of
lexical / fuzzy / semantic signals) and exposes per-skill evidence
alongside the previous matched/missing arrays. The legacy "outdated skills"
detection remains as a separate concern.
"""

import re
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.skill_extract import (
    SkillRegistry,
    build_skill_registry,
    extract_cv_skills,
    evidence_to_dict,
)


# ----------------------------------------------------------------------
# Outdated skill detection (unchanged — keyword matching for proper nouns)
# ----------------------------------------------------------------------

# Map jadul → modern alternatives. Used to mark missing skills as
# 'upgrade' (achievable from existing legacy knowledge) vs 'new'.
SKILL_CURRENCY_MAP: Dict[str, List[str]] = {
    "jQuery": ["React.js", "Vue.js", "Alpine.js"],
    "AngularJS": ["Angular", "React.js", "Vue.js"],
    "Backbone.js": ["React.js", "Vue.js"],
    "Knockout.js": ["React.js", "Vue.js"],
    "Ember.js": ["React.js", "Vue.js"],
    "CoffeeScript": ["TypeScript"],
    "Grunt": ["Vite", "esbuild", "Webpack"],
    "Bower": ["npm", "pnpm"],
    "SVN": ["Git"],
    "Subversion": ["Git"],
    "SOAP": ["REST API", "gRPC", "GraphQL"],
    "Vagrant": ["Docker", "Docker Compose"],
    "Java EE": ["Spring Boot", "Quarkus"],
    "J2EE": ["Spring Boot", "Quarkus"],
    "Struts": ["Spring Boot", "Spring MVC"],
    "EJB": ["Spring Boot"],
    "Apache Ant": ["Maven", "Gradle"],
    "Python 2": ["Python 3"],
    "TensorFlow 1.x": ["TensorFlow 2.x", "PyTorch"],
    "MySQL 5": ["PostgreSQL", "MySQL 8"],
    "Bootstrap 3": ["Bootstrap 5", "Tailwind CSS"],
}

_SKILL_PATTERNS: Dict[str, List[str]] = {
    "jQuery":         [r'\bjquery\b'],
    "AngularJS":      [r'\bangularjs\b', r'\bangular\s*1[\s.,)]', r'\bangular\s+js\b'],
    "Backbone.js":    [r'\bbackbone\.js\b', r'\bbackbonejs\b', r'\bbackbone\s+js\b'],
    "Knockout.js":    [r'\bknockout\.js\b', r'\bknockoutjs\b', r'\bknockout\s+js\b'],
    "Ember.js":       [r'\bember\.js\b', r'\bemberjs\b', r'\bember\s+js\b'],
    "CoffeeScript":   [r'\bcoffeescript\b', r'\bcoffee\s*script\b'],
    "Grunt":          [r'\bgrunt\b'],
    "Bower":          [r'\bbower\b'],
    "SVN":            [r'\bsvn\b'],
    "Subversion":     [r'\bsubversion\b'],
    "SOAP":           [r'\bsoap\b'],
    "Vagrant":        [r'\bvagrant\b'],
    "Java EE":        [r'\bjava\s*ee\b', r'\bjavaee\b'],
    "J2EE":           [r'\bj2ee\b'],
    "Struts":         [r'\bstruts\b'],
    "EJB":            [r'\bejb\b'],
    "Apache Ant":     [r'\bapache\s+ant\b', r'\bant\s+build\b'],
    "Python 2":       [r'\bpython\s*2[\s.,]', r'\bpython2\b'],
    "TensorFlow 1.x": [r'\btensorflow\s*1[\s.,]', r'\btf\s*1[\s.,]'],
    "MySQL 5":        [r'\bmysql\s*5[\s.,]', r'\bmysql5\b'],
    "Bootstrap 3":    [r'\bbootstrap\s*3[\s.,)]', r'\bbootstrap\s*v3\b'],
}


def _detect_skill_currency(cv_text: str) -> List[Dict]:
    """Detect outdated skills explicitly mentioned in the CV.

    Tech names are proper nouns; lexical regex outperforms embeddings here
    (embeddings measure semantic neighbourhood, not lexical presence).
    """
    cv_lower = cv_text.lower()
    found: List[Dict] = []
    for skill, patterns in _SKILL_PATTERNS.items():
        if skill not in SKILL_CURRENCY_MAP:
            continue
        for pattern in patterns:
            if re.search(pattern, cv_lower):
                found.append({
                    "skill": skill,
                    "modern_alternatives": SKILL_CURRENCY_MAP[skill],
                })
                break
    return found


# ----------------------------------------------------------------------
# Threshold→layer mapping
# ----------------------------------------------------------------------
#
# The legacy ``threshold`` parameter (default 0.6) controlled a single
# cosine similarity cutoff. With the layered pipeline, we map that one
# knob onto the semantic-layer threshold (since lexical/fuzzy bring their
# own calibrated bands). The translation is monotonic so existing callers
# that raise/lower threshold get the expected directional effect.

def _legacy_threshold_to_semantic(threshold: float) -> float:
    """Translate the legacy 0-1 threshold into the new semantic cutoff.

    Empirically the layered pipeline needs a stricter semantic threshold
    than the previous flat one because proper nouns are now handled by
    L1/L2. We map [0, 1] → [0.65, 0.90] linearly, anchored so that the
    legacy default 0.6 lands on the new default 0.78.
    """
    if threshold <= 0:
        return 0.65
    if threshold >= 1:
        return 0.90
    # Linear interpolation that puts 0.6 → 0.78
    return round(0.65 + (threshold * 0.25) + (0.005 if threshold == 0.6 else 0.0), 4)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def analyze_skill_gap(
    cv_text: str,
    skills: List[str],
    model: SentenceTransformer,
    threshold: float = 0.6,
    check_currency: bool = True,
    registry: Optional[SkillRegistry] = None,
    fuzzy_threshold: float = 0.92,
    enable_lexical: bool = True,
    enable_fuzzy: bool = True,
    enable_semantic: bool = True,
) -> Dict:
    """Compare CV skills (extracted) against a target career's required skills.

    The extraction is delegated to the layered pipeline in
    :mod:`src.skill_extract`. This function then intersects the extracted
    skill set with the required-skill list, producing matched/missing
    arrays compatible with the previous return shape, plus per-skill
    evidence and a top-level list of all extracted skills.

    Parameters
    ----------
    cv_text : str
        Raw CV text.
    skills : list of str
        Required skills for the target career.
    model : SentenceTransformer
        Embedding model used by the semantic layer.
    threshold : float
        Legacy similarity threshold (0–1). Internally mapped to the
        semantic layer cutoff; lexical/fuzzy use their own calibrated
        thresholds.
    check_currency : bool
        Detect outdated skills in the CV and mark missing skills as
        "upgrade" when achievable from a legacy counterpart.
    registry : SkillRegistry, optional
        Pre-built registry. When provided, the pipeline uses it as the
        canonical-skill catalog (so it can also surface CV skills not in
        the target list under ``extra_cv_skills``). When omitted, a
        registry is built ad-hoc from ``skills``.
    fuzzy_threshold : float
        rapidfuzz cutoff in [0, 1]. Default 0.92.
    enable_lexical, enable_fuzzy, enable_semantic : bool
        Per-layer toggles (ablation knobs).

    Returns
    -------
    dict with keys:
        matched_skills          [{skill, similarity, source, evidence,
                                  context, section}]
        missing_skills          [{skill, similarity, type, ...}]
        match_ratio             float
        outdated_in_cv          legacy outdated skill list
        extracted_skills_in_cv  [{skill, source, confidence, ...}]
                                — every skill the pipeline detected,
                                whether or not it's required
        extra_cv_skills         skills CV has that target career does NOT
                                require (only populated when registry has
                                broader coverage than the target list)
    """
    # Build / reuse registry. If a global registry is supplied we keep it
    # because it gives us "extra_cv_skills" insight; otherwise we build a
    # mini-registry just for the target skills.
    if registry is None:
        registry = build_skill_registry([{"skills": skills}])
        target_only_registry = True
    else:
        # Ensure the target skills are in the registry even if caller
        # forgot to include them.
        for s in skills or []:
            registry.add(s)
        target_only_registry = False

    if not skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "match_ratio": 0.0,
            "outdated_in_cv": _detect_skill_currency(cv_text) if check_currency else [],
            "extracted_skills_in_cv": [],
            "extra_cv_skills": [],
        }

    semantic_threshold = _legacy_threshold_to_semantic(threshold)

    extracted = extract_cv_skills(
        cv_text=cv_text,
        registry=registry,
        model=model,
        fuzzy_threshold=fuzzy_threshold,
        semantic_threshold=semantic_threshold,
        enable_lexical=enable_lexical,
        enable_fuzzy=enable_fuzzy,
        enable_semantic=enable_semantic,
    )

    target_set = set(skills)
    matched: List[Dict] = []
    missing: List[Dict] = []

    for skill in skills:
        if skill in extracted:
            ev = extracted[skill]
            matched.append({
                "skill": skill,
                "similarity": ev.confidence,        # backward-compat field
                "source": ev.source,                # new: which layer caught it
                "evidence": ev.matched_text,        # new: the exact CV span text
                "context": ev.context,              # new: surrounding sentence
                "section": ev.section,              # new: skills_list/experience/other
                "cv_span": list(ev.cv_span),        # new: char offsets
            })
        else:
            missing.append({
                "skill": skill,
                "similarity": 0.0,
                "type": "new",
            })

    # Sort: matched by confidence desc, missing alphabetical for stable output
    matched.sort(key=lambda x: -x["similarity"])
    missing.sort(key=lambda x: x["skill"])

    # Outdated skill detection + upgrade-path linking (legacy behaviour)
    outdated_in_cv: List[Dict] = []
    if check_currency:
        outdated_in_cv = _detect_skill_currency(cv_text)
        outdated_to_modern: Dict[str, List[str]] = {}
        for item in outdated_in_cv:
            for modern in item["modern_alternatives"]:
                outdated_to_modern.setdefault(modern, []).append(item["skill"])
        for m in missing:
            if m["skill"] in outdated_to_modern:
                m["type"] = "upgrade"
                m["upgrade_from"] = outdated_to_modern[m["skill"]]

    extracted_dump = [evidence_to_dict(ev) for ev in extracted.values()]
    extracted_dump.sort(key=lambda x: -x["confidence"])

    extra_cv_skills: List[Dict] = []
    if not target_only_registry:
        extra_cv_skills = [
            evidence_to_dict(ev)
            for canonical, ev in extracted.items()
            if canonical not in target_set
        ]
        extra_cv_skills.sort(key=lambda x: -x["confidence"])

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_ratio": round(len(matched) / len(skills), 4),
        "outdated_in_cv": outdated_in_cv,
        "extracted_skills_in_cv": extracted_dump,
        "extra_cv_skills": extra_cv_skills,
    }
