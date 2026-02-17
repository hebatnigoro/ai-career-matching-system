from typing import Dict, List, Optional
import numpy as np


def analyze_drift(
    student_vector: np.ndarray,
    career_vectors: np.ndarray,
    career_ids: List[str],
    declared_interest: Optional[str],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    """
    Compute drift analysis for a student.
    If declared_interest is provided:
      - declared_similarity = cos(student, declared_career)
      - best_alt = argmax over all careers (excluding declared_interest)
      - advantage = best_alt_sim - declared_similarity
      - Categorize using thresholds:
          Aligned: declared_similarity >= tau_high AND advantage <= delta_minor
          Minor Drift: declared_similarity >= tau_mid AND advantage > delta_minor
          Major Drift: declared_similarity < tau_mid AND best_alt_sim >= tau_high
          Else: Moderate Fit (borderline)
    If not provided:
      - Use best_alt_sim alone to categorize suitability.
    """
    tau_high = thresholds.get('tau_high', 0.70)
    tau_mid = thresholds.get('tau_mid', 0.60)
    delta_minor = thresholds.get('delta_minor', 0.08)

    # career vectors already normalized upstream
    # normalize student vector for safety
    s_norm = student_vector / max(np.linalg.norm(student_vector), 1e-12)

    sims = s_norm @ (career_vectors / np.clip(np.linalg.norm(career_vectors, axis=1, keepdims=True), 1e-12, None)).T

    best_idx = int(np.argmax(sims))
    best_id = career_ids[best_idx]
    best_sim = float(sims[best_idx])

    declared_similarity = None
    advantage = None

    if declared_interest is not None and declared_interest in career_ids:
        decl_idx = career_ids.index(declared_interest)
        declared_similarity = float(sims[decl_idx])
        # Exclude declared when finding alternative
        mask = np.ones_like(sims, dtype=bool)
        mask[decl_idx] = False
        alt_idx = int(np.argmax(sims[mask]))
        # Map masked index back to original index
        alt_candidates = [i for i in range(len(sims)) if i != decl_idx]
        best_alt_idx = alt_candidates[alt_idx]
        best_alt_id = career_ids[best_alt_idx]
        best_alt_sim = float(sims[best_alt_idx])
        advantage = best_alt_sim - declared_similarity

        if declared_similarity >= tau_high and advantage <= delta_minor:
            status = 'Aligned'
            rationale = (
                f"Declared interest is strongly suitable (sim={declared_similarity:.3f}). "
                f"No alternative with significant advantage (adv={advantage:.3f} <= {delta_minor})."
            )
        elif declared_similarity >= tau_mid and advantage > delta_minor:
            status = 'Minor Drift'
            rationale = (
                f"Declared interest is moderately suitable (sim={declared_similarity:.3f}), "
                f"but an alternative is notably better (adv={advantage:.3f} > {delta_minor})."
            )
        elif declared_similarity < tau_mid and best_alt_sim >= tau_high:
            status = 'Major Drift'
            rationale = (
                f"Declared interest has low suitability (sim={declared_similarity:.3f} < {tau_mid}), "
                f"while an alternative is highly suitable (sim={best_alt_sim:.3f} >= {tau_high})."
            )
        else:
            status = 'Moderate Fit'
            rationale = (
                f"Borderline case: declared sim={declared_similarity:.3f}, best alt sim={best_alt_sim:.3f}."
            )

        return {
            'status': status,
            'declared_similarity': round(declared_similarity, 4),
            'best_alt_id': best_alt_id,
            'best_alt_similarity': round(best_alt_sim, 4),
            'advantage': round(advantage, 4),
            'rationale': rationale,
        }
    else:
        # No declared interest, use overall suitability
        if best_sim >= tau_high:
            status = 'Aligned (No Declared Interest)'
            rationale = f"Best career is highly suitable (sim={best_sim:.3f})."
        elif best_sim >= tau_mid:
            status = 'Potential Fit (No Declared Interest)'
            rationale = f"Best career is moderately suitable (sim={best_sim:.3f})."
        else:
            status = 'Exploration Needed (No Declared Interest)'
            rationale = f"No career above moderate suitability (best sim={best_sim:.3f})."
        return {
            'status': status,
            'declared_similarity': None,
            'best_alt_id': best_id,
            'best_alt_similarity': round(best_sim, 4),
            'advantage': None,
            'rationale': rationale,
        }
