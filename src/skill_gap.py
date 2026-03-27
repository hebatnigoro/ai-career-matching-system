import re
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import embed_texts
from src.preprocess import preprocess_text


_SENT_SPLIT = re.compile(r'(?<=[.!?;])\s+|\n+')

# Peta skill jadul → alternatif modern yang relevan di pasar saat ini.
SKILL_CURRENCY_MAP: Dict[str, List[str]] = {
    # JavaScript framework lama
    "jQuery": ["React.js", "Vue.js", "Alpine.js"],
    "AngularJS": ["Angular", "React.js", "Vue.js"],
    "Backbone.js": ["React.js", "Vue.js"],
    "Knockout.js": ["React.js", "Vue.js"],
    "Ember.js": ["React.js", "Vue.js"],
    "CoffeeScript": ["TypeScript"],
    # Build tools lama
    "Grunt": ["Vite", "esbuild", "Webpack"],
    "Bower": ["npm", "pnpm"],
    # Version control lama
    "SVN": ["Git"],
    "Subversion": ["Git"],
    # Protokol lama
    "SOAP": ["REST API", "gRPC", "GraphQL"],
    # Infrastruktur lama
    "Vagrant": ["Docker", "Docker Compose"],
    # Java legacy stack
    "Java EE": ["Spring Boot", "Quarkus"],
    "J2EE": ["Spring Boot", "Quarkus"],
    "Struts": ["Spring Boot", "Spring MVC"],
    "EJB": ["Spring Boot"],
    "Apache Ant": ["Maven", "Gradle"],
    # Versi Python/ML lama
    "Python 2": ["Python 3"],
    "TensorFlow 1.x": ["TensorFlow 2.x", "PyTorch"],
    # Versi database lama
    "MySQL 5": ["PostgreSQL", "MySQL 8"],
    # CSS/UI lama
    "Bootstrap 3": ["Bootstrap 5", "Tailwind CSS"],
}

# Pola regex (lowercase) untuk mencari skill jadul di teks CV.
# Nama teknologi adalah proper noun: lebih akurat dicari secara leksikal, bukan embedding.
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


def _split_to_segments(text: str, min_length: int = 10) -> List[str]:
    """Pecah teks CV menjadi segmen kalimat untuk pencocokan skill granular."""
    parts = _SENT_SPLIT.split(text)
    segments = []
    for p in parts:
        clean = preprocess_text(p)
        if len(clean) >= min_length:
            segments.append(clean)
    if not segments:
        full = preprocess_text(text)
        if full:
            segments = [full]
    return segments


def _detect_skill_currency(cv_text: str) -> List[Dict]:
    """Deteksi skill jadul yang EKSPLISIT disebutkan di teks CV menggunakan keyword matching.

    Nama teknologi adalah proper noun — lebih akurat dideteksi secara leksikal
    daripada embedding (embedding mengukur kemiripan semantik, bukan keberadaan kata).

    Returns
    -------
    list of {skill, modern_alternatives}
    """
    cv_lower = cv_text.lower()
    found = []
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


def analyze_skill_gap(
    cv_text: str,
    skills: List[str],
    model: SentenceTransformer,
    threshold: float = 0.6,
    check_currency: bool = True,
) -> Dict:
    """Analisis skill gap antara CV dan skill yang dibutuhkan karier target.

    CV dipecah menjadi segmen-segmen kalimat, lalu setiap skill dicocokkan
    terhadap segmen dengan similarity tertinggi (max-pooling per skill).

    Jika check_currency=True, juga mendeteksi skill jadul yang eksplisit ada di CV
    dan menandai missing skill sebagai 'upgrade' atau 'new'.

    Parameters
    ----------
    cv_text : str
        Teks CV mentah.
    skills : list of str
        Daftar skill yang dibutuhkan karier target.
    model : SentenceTransformer
        Model yang sama dengan yang digunakan untuk embedding CV/karier.
    threshold : float
        Batas similarity untuk menentukan skill matched vs missing.
    check_currency : bool
        Jika True, deteksi skill jadul di CV dan beri rekomendasi upgrade.

    Returns
    -------
    dict with keys:
        matched_skills  : skill yang sudah dimiliki (similarity >= threshold)
        missing_skills  : skill yang kurang, dengan type='upgrade'|'new'
                          dan upgrade_from jika type=='upgrade'
        match_ratio     : rasio matched/total
        outdated_in_cv  : skill jadul yang ditemukan di CV + alternatif modernnya
    """
    outdated_in_cv: List[Dict] = []

    if not skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "match_ratio": 0.0,
            "outdated_in_cv": outdated_in_cv,
        }

    segments = _split_to_segments(cv_text)
    if not segments:
        return {
            "matched_skills": [],
            "missing_skills": [{"skill": s, "similarity": 0.0, "type": "new"} for s in skills],
            "match_ratio": 0.0,
            "outdated_in_cv": outdated_in_cv,
        }

    # Skills sebagai query, segmen CV sebagai passage (E5 convention)
    skill_emb = embed_texts(model, skills, is_passage=False)
    segment_emb = embed_texts(model, segments, is_passage=True)

    sim_matrix = skill_emb @ segment_emb.T
    max_sims = np.max(sim_matrix, axis=1)

    matched = []
    missing = []

    for skill, sim in zip(skills, max_sims):
        if sim >= threshold:
            matched.append({"skill": skill, "similarity": round(float(sim), 4)})
        else:
            missing.append({"skill": skill, "similarity": round(float(sim), 4), "type": "new"})

    matched.sort(key=lambda x: -x["similarity"])
    missing.sort(key=lambda x: -x["similarity"])

    # Deteksi skill jadul menggunakan keyword matching (bukan embedding)
    if check_currency:
        outdated_in_cv = _detect_skill_currency(cv_text)

        # Petakan skill jadul → skill modern yang bisa jadi penggantinya
        outdated_to_modern: Dict[str, List[str]] = {}
        for item in outdated_in_cv:
            for modern in item["modern_alternatives"]:
                outdated_to_modern.setdefault(modern, []).append(item["skill"])

        # Tandai missing skill sebagai 'upgrade' jika bisa dicapai dari skill jadul yang ada
        for m in missing:
            if m["skill"] in outdated_to_modern:
                m["type"] = "upgrade"
                m["upgrade_from"] = outdated_to_modern[m["skill"]]

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_ratio": round(len(matched) / len(skills), 4),
        "outdated_in_cv": outdated_in_cv,
    }
