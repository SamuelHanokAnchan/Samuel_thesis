"""
Features for System 3 — the learned re-ranker.

The design principle: **give the model the signals the embedding provably throws away.**

Production collapses a profile into one 768-d vector built from a text paragraph that
lists language *names* but never *proficiency levels* (see ``src/text.py``). So "German
A1" and "German C2" look nearly identical to Stage 2, even though that distinction decides
a German apprenticeship. The same is true of dates, cities, and how much experience someone
actually has: all are either discarded or reduced to fuzzy word-similarity.

The hard filters *do* know these things — but they use them as **gates** (pass/fail).
The real data says recruiters hired across those gates 95% of the time. So here the same
signals are handed to the model as **graded features**, and the thesis gets to answer:
should these be hard gates, or learned soft signals?

Both similarity scores (cosine and TF-IDF) are included as features, so the re-ranker
starts from everything the earlier systems knew and can only add information, never lose
it. That makes "S3 beats S2" a fair claim rather than an artefact of a different input set.
"""
from __future__ import annotations

from src.loaders import PROFICIENCY_TO_CEFR, CEFR_ORDER

FEATURE_NAMES = [
    "cosine_similarity",      # what System 2 knows
    "tfidf_similarity",       # what System 1 knows
    "language_gap_min",       # ← invisible to the embedding
    "meets_all_languages",    # ← the production gate, as a feature
    "same_city",              # ← the production gate, as a feature
    "start_gap_months",       # ← the production gate, as a feature
    "category_match",         # ← the production gate, as a feature
    "skill_overlap",
    "n_work_experience",
    "n_languages",
]


def _months(date_str: str | None) -> int | None:
    """'2026-09' -> 24321. Absolute month index, so two dates can be subtracted."""
    if not date_str or "-" not in str(date_str):
        return None
    try:
        year, month = str(date_str).split("-")[:2]
        return int(year) * 12 + int(month)
    except (ValueError, TypeError):
        return None


def language_gap_min(student: dict, job: dict) -> float:
    """
    How far the student's weakest *required* language sits above (+) or below (-) the level
    the job asks for, on the CEFR ladder.

    -2 means they are two levels short of what the role requires; +1 means one level above.
    Jobs with no required language return 0.0 (nothing to be short of).

    This is the headline feature: Stage 2 cannot see it at all.
    """
    required = [r for r in (job.get("language_requirements") or []) if r.get("kind") == "REQUIRED"]
    if not required:
        return 0.0

    held = {
        lang.get("language"): PROFICIENCY_TO_CEFR.get(lang.get("proficiency"), 0)
        for lang in (student.get("languages") or [])
    }

    gaps = []
    for req in required:
        needed = CEFR_ORDER.get(req.get("level"), 0)
        has = held.get(req.get("language"), 0)  # not listed at all -> 0, a large negative gap
        gaps.append(has - needed)
    return float(min(gaps))


def skill_overlap(student: dict, job: dict) -> float:
    """How many of the job's required skills appear anywhere in the student's own words.

    Crude token containment, deliberately: this is the *lexical* signal, the thing TF-IDF
    is good at. The semantic version is already covered by ``cosine_similarity``.
    """
    required = [str(s).lower() for s in (job.get("required_skills") or []) if s]
    if not required:
        return 0.0

    blob = " ".join(
        [str(i) for i in (student.get("career_fields") or [])]
        + [str(i) for i in (student.get("top_job_interests") or [])]
        + [str(w.get("job_title", "")) + " " + str(w.get("description", ""))
           for w in (student.get("work_experience") or [])]
        + [str(a.get("name", "")) + " " + str(a.get("description", ""))
           for a in (student.get("activities") or [])]
    ).lower()

    return float(sum(1 for skill in required if skill in blob))


def build_features(
    student: dict,
    job: dict,
    cosine_score: float,
    tfidf_score: float,
) -> list[float]:
    """One feature row for a (student, job) pair, in ``FEATURE_NAMES`` order."""
    s_start, j_start = _months(student.get("start_date")), _months(job.get("start_date"))
    # Positive = the job starts after the student is free (good). Negative = it starts
    # before they are available. 0 when either date is missing, so a missing date is
    # neutral rather than silently looking like a perfect match.
    start_gap = float(j_start - s_start) if (s_start is not None and j_start is not None) else 0.0

    gap = language_gap_min(student, job)
    student_category = (student.get("matching_category") or "").strip()
    category_match = (
        1.0
        if student_category not in ("VOCATIONAL", "DUAL_STUDY")  # NOT_SURE fits anything
        else float((job.get("matching_category") or "").strip() == student_category)
    )

    return [
        float(cosine_score),
        float(tfidf_score),
        gap,
        float(gap >= 0),  # meets every required language
        float(student.get("city") == job.get("city")),
        start_gap,
        category_match,
        skill_overlap(student, job),
        float(len(student.get("work_experience") or [])),
        float(len(student.get("languages") or [])),
    ]
