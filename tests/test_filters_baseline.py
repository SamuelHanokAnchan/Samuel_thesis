"""
Tests for the offline hard filters (Stage 1) and the TF-IDF baseline (System 1).

The filter tests matter because these rules are a hand-port of production SQL. If the
offline replica gates differently from the live system, every number in the thesis
describes a system that does not exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import TfidfBaseline, job_keyword_text, student_keyword_text  # noqa: E402
from src.filters import (  # noqa: E402
    category_compatible,
    eligible_jobs,
    is_eligible,
    meets_all_language_requirements,
    start_date_compatible,
)


def a_student(**over):
    base = {
        "student_id": "s_1",
        "matching_category": "VOCATIONAL",
        "city": "Berlin",
        "start_date": "2026-09",
        "career_fields": ["Automotive"],
        "top_job_interests": ["Motor vehicle mechanic"],
        "work_experience": [{"job_title": "Workshop assistant"}],
        "languages": [{"language": "German", "proficiency": "INTERMEDIATE"}],
    }
    base.update(over)
    return base


def a_job(**over):
    base = {
        "job_id": "j_1",
        "title": "Kfz-Mechatroniker",
        "matching_category": "VOCATIONAL",
        "city": "Berlin",
        "start_date": "2026-09",
        "required_skills": ["Basic mechanics"],
        "language_requirements": [{"language": "German", "level": "B2", "kind": "REQUIRED"}],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- language gate
def test_intermediate_german_satisfies_b2():
    # Production: B2 is met by NATIVE, FLUENT or INTERMEDIATE.
    assert meets_all_language_requirements(a_student(), a_job())


def test_intermediate_german_does_not_satisfy_c1():
    # C1 requires FLUENT or NATIVE. This is the exact signal the embedding throws away.
    job = a_job(language_requirements=[
        {"language": "German", "level": "C1", "kind": "REQUIRED"}
    ])
    assert not meets_all_language_requirements(a_student(), job)


def test_any_proficiency_satisfies_b1():
    student = a_student(languages=[{"language": "German", "proficiency": "BASIC"}])
    job = a_job(language_requirements=[
        {"language": "German", "level": "B1", "kind": "REQUIRED"}
    ])
    assert meets_all_language_requirements(student, job)


def test_missing_the_language_entirely_fails():
    student = a_student(languages=[{"language": "English", "proficiency": "NATIVE"}])
    assert not meets_all_language_requirements(student, a_job())


def test_optional_requirements_never_exclude():
    job = a_job(language_requirements=[
        {"language": "French", "level": "C1", "kind": "OPTIONAL"}
    ])
    assert meets_all_language_requirements(a_student(), job)


# ---------------------------------------------------------------- other gates
def test_job_must_not_start_before_the_student_is_available():
    assert not start_date_compatible(a_student(start_date="2026-09"), a_job(start_date="2026-08"))
    assert start_date_compatible(a_student(start_date="2026-09"), a_job(start_date="2026-09"))
    assert start_date_compatible(a_student(start_date="2026-09"), a_job(start_date="2027-01"))


def test_not_sure_students_are_eligible_for_any_track():
    student = a_student(matching_category="NOT_SURE")
    assert category_compatible(student, a_job(matching_category="DUAL_STUDY"))
    assert category_compatible(student, a_job(matching_category="VOCATIONAL"))


def test_vocational_student_is_not_shown_dual_study_roles():
    assert not category_compatible(a_student(), a_job(matching_category="DUAL_STUDY"))


def test_different_city_is_ineligible():
    assert not is_eligible(a_student(), a_job(city="Hamburg"))


def test_eligible_jobs_returns_only_passing_jobs():
    jobs = [
        a_job(job_id="ok"),
        a_job(job_id="wrong_city", city="Hamburg"),
        a_job(job_id="too_early", start_date="2026-01"),
        a_job(job_id="needs_c1", language_requirements=[
            {"language": "German", "level": "C1", "kind": "REQUIRED"}
        ]),
    ]
    assert [j["job_id"] for j in eligible_jobs(a_student(), jobs)] == ["ok"]


# ---------------------------------------------------------------- baseline
def test_baseline_text_excludes_free_text_descriptions():
    # Descriptions must not leak in, or the "keyword" control quietly becomes semantic.
    student = a_student(work_experience=[
        {"job_title": "Workshop assistant", "description": "Repaired brake systems daily."}
    ])
    text = student_keyword_text(student)
    assert "Workshop assistant" in text
    assert "brake" not in text.lower()


def test_baseline_ranks_lexical_overlap_first():
    jobs = [
        a_job(job_id="j_office", title="Büromanagement", required_skills=["MS Office"]),
        a_job(job_id="j_auto", title="Automotive mechanic", required_skills=["Automotive"]),
    ]
    baseline = TfidfBaseline(jobs)
    ranked = baseline.rank(a_student(), ["j_office", "j_auto"])
    assert ranked[0] == "j_auto"  # shares the word "Automotive"


def test_baseline_is_blind_to_synonyms():
    # THE point of the baseline. The student wants a vehicle mechanic role; the German
    # advert says "Kfz-Mechatroniker". No shared token, so TF-IDF scores it zero, while a
    # semantic system should recognise them as the same job. This is the failure the
    # thesis is built to demonstrate.
    jobs = [
        a_job(job_id="j_kfz", title="Kfz-Mechatroniker", required_skills=["Teamwork"]),
        a_job(job_id="j_bake", title="Automotive baker", required_skills=["Automotive"]),
    ]
    baseline = TfidfBaseline(jobs)
    scores = baseline.score(a_student(), ["j_kfz", "j_bake"])
    assert scores["j_kfz"] == 0.0          # the correct job scores nothing
    assert scores["j_bake"] > scores["j_kfz"]  # a nonsense job wins on a shared token


def test_baseline_scores_every_candidate_so_systems_rank_the_same_set():
    baseline = TfidfBaseline([a_job(job_id="j_1")])
    scores = baseline.score(a_student(), ["j_1", "j_unknown"])
    assert set(scores) == {"j_1", "j_unknown"}


def test_baseline_ranking_is_deterministic_on_ties():
    jobs = [a_job(job_id="j_b", title="Zzz"), a_job(job_id="j_a", title="Zzz")]
    baseline = TfidfBaseline(jobs)
    assert baseline.rank(a_student(), ["j_b", "j_a"]) == ["j_a", "j_b"]  # tie -> id order


def test_job_keyword_text_includes_title_and_skills():
    text = job_keyword_text(a_job(optional_skills=["Driving licence"]))
    assert "Kfz-Mechatroniker" in text
    assert "Basic mechanics" in text
    assert "Driving licence" in text
