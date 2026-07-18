"""
Offline replica of Stage 1 — the hard eligibility filters.

Mirrors ``flai/backend/jobs/services/matching.py`` (``filter_published_jobs_for_student_profile``
and ``filter_jobs_by_required_languages``). Production runs these as SQL; here they run in
Python over the exported snapshot.

All three systems (TF-IDF baseline, embedding, embedding + re-ranker) are scored on the
**same** filtered candidate pool. If the baseline were allowed to rank jobs the student is
not even eligible for, the comparison would be measuring the filters rather than the
ranking, and the thesis claim would be void.
"""
from __future__ import annotations

from src.loaders import PROFICIENCY_TO_CEFR

# Production tiers (jobs/services/matching.py):
#   B1 -> the student must merely list the language, at any proficiency
#   B2 -> NATIVE, FLUENT or INTERMEDIATE
#   C1 -> NATIVE or FLUENT
_MEETS_B2 = {"NATIVE", "FLUENT", "INTERMEDIATE"}
_MEETS_C1 = {"NATIVE", "FLUENT"}


def student_meets_language_requirement(student: dict, requirement: dict) -> bool:
    """True when the student satisfies one REQUIRED language requirement of a job."""
    if requirement.get("kind") != "REQUIRED":
        return True  # optional requirements never exclude anyone

    wanted = requirement.get("language")
    level = requirement.get("level")

    held = {
        lang.get("language"): lang.get("proficiency")
        for lang in (student.get("languages") or [])
    }
    if wanted not in held:
        return False

    proficiency = held[wanted]
    if level == "B1":
        return True  # merely listing the language is enough
    if level == "B2":
        return proficiency in _MEETS_B2
    if level == "C1":
        return proficiency in _MEETS_C1
    # A1/A2/C2 are not tiers production gates on; treat "listed" as sufficient rather
    # than inventing a rule the live system does not have.
    return True


def meets_all_language_requirements(student: dict, job: dict) -> bool:
    return all(
        student_meets_language_requirement(student, req)
        for req in (job.get("language_requirements") or [])
    )


def category_compatible(student: dict, job: dict) -> bool:
    """
    Production filters on ``matching_category`` only when the student picked a concrete
    track. ``NOT_SURE`` students are eligible for everything.
    """
    student_category = (student.get("matching_category") or "").strip()
    if student_category not in ("VOCATIONAL", "DUAL_STUDY"):
        return True  # NOT_SURE / unset -> no restriction
    return (job.get("matching_category") or "").strip() == student_category


def start_date_compatible(student: dict, job: dict) -> bool:
    """
    The job must start on or after the student is available.

    Dates are ``YYYY-MM`` strings, which compare correctly as plain strings because the
    format is zero-padded and fixed-width. Missing either date makes the pair ineligible,
    matching production, which returns nothing when the profile has no ``start_date``.
    """
    s_start, j_start = student.get("start_date"), job.get("start_date")
    if not s_start or not j_start:
        return False
    return j_start >= s_start


def is_eligible(student: dict, job: dict) -> bool:
    """All of Stage 1: city, timing, track, and required languages."""
    if not student.get("city") or not job.get("city"):
        return False
    if student["city"] != job["city"]:
        return False
    if not start_date_compatible(student, job):
        return False
    if not category_compatible(student, job):
        return False
    return meets_all_language_requirements(student, job)


def eligible_jobs(student: dict, jobs: list[dict]) -> list[dict]:
    """The candidate pool for one student. Career field is deliberately NOT filtered —
    production leaves that to semantic similarity so students can still discover adjacent
    roles when few close matches exist."""
    return [j for j in jobs if is_eligible(student, j)]
