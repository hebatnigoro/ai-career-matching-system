from typing import Dict, List, Optional
import numpy as np

from src.similarity import normalize_scores_minmax


def analyze_drift(
    student_vector: np.ndarray,
    career_vectors: np.ndarray,
    career_ids: List[str],
    declared_interest: Optional[str],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    """
    Compute drift analysis using per-student min-max normalized scores.

    Raw cosine similarities from E5 models tend to cluster in a narrow range
    (e.g. 0.76-0.82), making absolute thresholds unreliable. Min-max normalization
    rescales scores to [0, 1] relative to each student's own score distribution,
    enabling meaningful comparison.

    Categorization (on normalized scores, range 0-1):
      0. Exploration Needed:  sim_range < 0.02 (model cannot discriminate)
      1. Aligned:             declared_norm >= tau_high AND norm_advantage <= delta_minor
      2. Minor Drift:         declared_norm >= tau_mid  AND norm_advantage > delta_minor
      3. Major Drift:         declared_norm <  tau_mid  AND best_alt_norm >= tau_mid
      4. Exploration Needed:  declared_norm <  tau_mid  AND best_alt_norm < tau_mid

    Default thresholds (calibrated for normalized 0-1 range):
      tau_high = 0.70, tau_mid = 0.40, delta_minor = 0.10
    """
    tau_high = thresholds.get('tau_high', 0.70)
    tau_mid = thresholds.get('tau_mid', 0.40)
    delta_minor = thresholds.get('delta_minor', 0.10)

    # Compute raw cosine similarities
    s_norm = student_vector / max(np.linalg.norm(student_vector), 1e-12)
    c_norms = np.clip(np.linalg.norm(career_vectors, axis=1, keepdims=True), 1e-12, None)
    raw_sims = s_norm @ (career_vectors / c_norms).T

    # Per-student min-max normalization
    norm_sims = normalize_scores_minmax(raw_sims)

    best_idx = int(np.argmax(raw_sims))
    best_id = career_ids[best_idx]
    best_raw = float(raw_sims[best_idx])
    best_norm = float(norm_sims[best_idx])

    if declared_interest is not None and declared_interest in career_ids:
        decl_idx = career_ids.index(declared_interest)
        declared_raw = float(raw_sims[decl_idx])
        declared_norm = float(norm_sims[decl_idx])

        # Find best alternative excluding declared interest
        alt_candidates = [
            (i, float(raw_sims[i]), float(norm_sims[i]))
            for i in range(len(raw_sims)) if i != decl_idx
        ]
        best_alt_idx, best_alt_raw, best_alt_norm = max(alt_candidates, key=lambda x: x[2])
        best_alt_id = career_ids[best_alt_idx]
        raw_advantage = best_alt_raw - declared_raw
        norm_advantage = best_alt_norm - declared_norm

        # Check if model can discriminate at all (raw similarity range)
        sim_range = float(np.max(raw_sims) - np.min(raw_sims))

        # Categorization using normalized scores
        if sim_range < 0.02:
            # Model cannot discriminate: all careers have nearly equal similarity
            status = 'Exploration Needed'
            rationale = (
                f"Model tidak dapat membedakan kesesuaian CV terhadap karier manapun "
                f"(range similarity={sim_range:.4f} < 0.02). "
                f"Disarankan eksplorasi karier lebih lanjut atau perkaya isi CV."
            )
        elif declared_norm >= tau_high and norm_advantage <= delta_minor:
            status = 'Aligned'
            rationale = (
                f"Minat yang dideklarasikan sangat sesuai "
                f"(relative_score={declared_norm:.3f} >= {tau_high}). "
                f"Tidak ada alternatif dengan keunggulan signifikan "
                f"(norm_adv={norm_advantage:.3f} <= {delta_minor})."
            )
        elif declared_norm >= tau_mid and norm_advantage > delta_minor:
            status = 'Minor Drift'
            rationale = (
                f"Minat yang dideklarasikan cukup sesuai "
                f"(relative_score={declared_norm:.3f} >= {tau_mid}), "
                f"namun alternatif '{best_alt_id}' lebih cocok "
                f"(norm_adv={norm_advantage:.3f} > {delta_minor})."
            )
        elif declared_norm < tau_mid and best_alt_norm >= tau_mid:
            status = 'Major Drift'
            rationale = (
                f"Minat yang dideklarasikan kurang sesuai "
                f"(relative_score={declared_norm:.3f} < {tau_mid}), "
                f"sementara alternatif '{best_alt_id}' jauh lebih cocok "
                f"(relative_score={best_alt_norm:.3f} >= {tau_mid})."
            )
        else:
            # declared_norm < tau_mid AND best_alt_norm < tau_mid
            status = 'Exploration Needed'
            rationale = (
                f"Minat yang dideklarasikan kurang sesuai "
                f"(relative_score={declared_norm:.3f} < {tau_mid}) "
                f"dan tidak ada alternatif yang kuat "
                f"(best alt relative_score={best_alt_norm:.3f} < {tau_mid}). "
                f"Disarankan eksplorasi karier lebih lanjut atau perkaya isi CV."
            )

        return {
            'status': status,
            'declared_similarity': round(declared_raw, 4),
            'declared_relative_score': round(declared_norm, 4),
            'best_alt_id': best_alt_id,
            'best_alt_similarity': round(best_alt_raw, 4),
            'best_alt_relative_score': round(best_alt_norm, 4),
            'raw_advantage': round(raw_advantage, 4),
            'relative_advantage': round(norm_advantage, 4),
            'rationale': rationale,
        }
    else:
        # No declared interest or invalid career id
        if best_norm >= tau_high:
            status = 'Strong Fit (No Declared Interest)'
            rationale = (
                f"Karier terbaik sangat sesuai "
                f"(relative_score={best_norm:.3f} >= {tau_high})."
            )
        elif best_norm >= tau_mid:
            status = 'Moderate Fit (No Declared Interest)'
            rationale = (
                f"Karier terbaik cukup sesuai "
                f"(relative_score={best_norm:.3f} >= {tau_mid})."
            )
        else:
            status = 'Exploration Needed (No Declared Interest)'
            rationale = (
                f"Tidak ada karier dengan kesesuaian tinggi "
                f"(best relative_score={best_norm:.3f} < {tau_mid})."
            )

        return {
            'status': status,
            'declared_similarity': None,
            'declared_relative_score': None,
            'best_alt_id': best_id,
            'best_alt_similarity': round(best_raw, 4),
            'best_alt_relative_score': round(best_norm, 4),
            'raw_advantage': None,
            'relative_advantage': None,
            'rationale': rationale,
        }
