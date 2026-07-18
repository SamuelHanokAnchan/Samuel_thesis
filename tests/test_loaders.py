"""
Tests for the export loader/validator.

The point of these is that the validator *catches* bad exports. A loader that silently
accepts a broken file is worse than no loader, because the corruption surfaces as a
plausible-looking number in the results table.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loaders import (  # noqa: E402
    PROFICIENCY_TO_CEFR,
    gold_labels,
    load_dataset,
    outcome_counts,
)


def write(tmp_path: Path, students, jobs, applications) -> Path:
    (tmp_path / "students.json").write_text(json.dumps(students), encoding="utf-8")
    (tmp_path / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (tmp_path / "applications.json").write_text(json.dumps(applications), encoding="utf-8")
    return tmp_path


def a_student(**over):
    base = {
        "student_id": "s_1",
        "matching_category": "VOCATIONAL",
        "city": "Berlin",
        "start_date": "2026-09",
        "top_job_interests": ["7231.1"],
        "career_fields": ["Automotive"],
        "languages": [{"language": "German", "proficiency": "INTERMEDIATE"}],
        "work_experience": [{"job_title": "Workshop assistant", "description": "Servicing."}],
    }
    base.update(over)
    return base


def a_job(**over):
    base = {
        "job_id": "j_1",
        "company_id": "c_1",
        "title": "Ausbildung Kfz-Mechatroniker/in",
        "matching_category": "VOCATIONAL",
        "city": "Berlin",
        "start_date": "2026-09",
        "required_skills": ["Teamwork"],
        "language_requirements": [
            {"language": "German", "level": "B2", "kind": "REQUIRED"}
        ],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- happy path
def test_clean_export_has_no_problems(tmp_path):
    d = write(tmp_path, [a_student()], [a_job()],
              [{"student_id": "s_1", "job_id": "j_1", "status": "ACCEPTED"}])
    ds = load_dataset(d)
    assert ds.problems == []
    assert len(ds.students) == 1
    assert ds.students_by_id["s_1"]["city"] == "Berlin"


# ---------------------------------------------------------------- privacy
def test_export_containing_a_name_is_flagged(tmp_path):
    d = write(tmp_path, [a_student(name="Samuel Anchan")], [a_job()], [])
    ds = load_dataset(d)
    assert any(p.startswith("PRIVACY") and "name" in p for p in ds.problems)


def test_export_containing_an_email_anywhere_is_flagged(tmp_path):
    # Buried in free text, not in an obviously-named field.
    s = a_student(work_experience=[
        {"job_title": "Assistant", "description": "Report to chef@garage-berlin.de daily."}
    ])
    ds = load_dataset(write(tmp_path, [s], [a_job()], []))
    assert any("email" in p for p in ds.problems)


# ---------------------------------------------------------------- integrity
def test_dangling_application_reference_is_flagged(tmp_path):
    # The three files were exported at different times; they cannot be joined.
    d = write(tmp_path, [a_student()], [a_job()],
              [{"student_id": "s_GHOST", "job_id": "j_1", "status": "ACCEPTED"}])
    ds = load_dataset(d)
    assert any("not in students.json" in p for p in ds.problems)


def test_duplicate_student_id_is_flagged(tmp_path):
    ds = load_dataset(write(tmp_path, [a_student(), a_student()], [a_job()], []))
    assert any("duplicate student_id" in p for p in ds.problems)


def test_unknown_proficiency_value_is_flagged(tmp_path):
    s = a_student(languages=[{"language": "German", "proficiency": "B2"}])  # wrong enum
    ds = load_dataset(write(tmp_path, [s], [a_job()], []))
    assert any("unknown proficiency" in p for p in ds.problems)


def test_unknown_cefr_level_is_flagged(tmp_path):
    j = a_job(language_requirements=[
        {"language": "German", "level": "FLUENT", "kind": "REQUIRED"}  # enum confusion
    ])
    ds = load_dataset(write(tmp_path, [a_student()], [j], []))
    assert any("unknown CEFR level" in p for p in ds.problems)


def test_student_with_nothing_to_embed_is_flagged(tmp_path):
    s = a_student(top_job_interests=[], career_fields=[], work_experience=[], activities=[])
    ds = load_dataset(write(tmp_path, [s], [a_job()], []))
    assert any("no interests" in p for p in ds.problems)


def test_missing_required_field_is_flagged(tmp_path):
    s = a_student()
    del s["city"]
    ds = load_dataset(write(tmp_path, [s], [a_job()], []))
    assert any("missing required field 'city'" in p for p in ds.problems)


# ---------------------------------------------------------------- gold labels
def test_accept_is_positive_and_company_rejection_is_negative(tmp_path):
    apps = [
        {"student_id": "s_1", "job_id": "j_1", "status": "ACCEPTED"},
        {"student_id": "s_1", "job_id": "j_2", "status": "REJECTED_BY_COMPANY"},
    ]
    labels = gold_labels(apps)
    assert labels[("s_1", "j_1")] == 1
    assert labels[("s_1", "j_2")] == 0


def test_advancing_to_interview_counts_as_a_positive_signal():
    apps = [{"student_id": "s_1", "job_id": "j_1", "status": "AI_INTERVIEW_INVITED"}]
    assert gold_labels(apps)[("s_1", "j_1")] == 1


def test_undecided_and_student_withdrawal_produce_no_label():
    # PENDING_COMPANY = the company has not judged yet.
    # REJECTED_BY_STUDENT = the student walked away; that is not the company calling it a
    # bad match. Scoring either as 0 would invent a negative that nobody expressed.
    apps = [
        {"student_id": "s_1", "job_id": "j_1", "status": "PENDING_COMPANY"},
        {"student_id": "s_2", "job_id": "j_1", "status": "REJECTED_BY_STUDENT"},
    ]
    assert gold_labels(apps) == {}
    assert outcome_counts(apps) == {}


def test_proficiency_ladder_orders_correctly():
    p = PROFICIENCY_TO_CEFR
    assert p["BASIC"] < p["INTERMEDIATE"] < p["FLUENT"] < p["NATIVE"]
