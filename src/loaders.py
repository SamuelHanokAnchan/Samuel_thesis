"""
Load and validate the exported JSON snapshot (see ``DATA_REQUEST.md``).

The validator is deliberately **loud**. A silently-missing field here does not crash
anything downstream — it quietly degrades into a null feature and corrupts a number that
ends up in the thesis. So every record is checked on load, and anything unexpected is
reported before a single metric is computed.

It is also a **privacy check**: if the export still contains names, emails or phone
numbers, the exporter did not apply the pseudonymisation rules and the file must be
rejected and re-requested, not cleaned up locally.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# CEFR ladder used to turn a language requirement into a comparable number.
CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

# The platform's proficiency enum, mapped onto the same ladder so student level and job
# requirement can be subtracted. Mirrors the tiers in jobs/services/matching.py:
# B1 needs the language listed at all; B2 needs INTERMEDIATE+; C1 needs FLUENT+.
PROFICIENCY_TO_CEFR = {
    "BASIC": 2,          # A2 - satisfies a B1 "just list it" requirement, nothing higher
    "INTERMEDIATE": 4,   # B2
    "FLUENT": 5,         # C1
    "NATIVE": 6,         # C2
}

# Fields that must never appear in the export. Their presence means the privacy rules
# were not applied at source.
FORBIDDEN_FIELDS = {
    "name", "first_name", "last_name", "full_name",
    "email", "phone", "phone_number", "date_of_birth", "dob",
    "address", "street", "photo", "profile_picture", "cv", "cv_file", "resume",
    "iban", "stripe_customer_id",
}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9,}")


@dataclass
class Dataset:
    students: list[dict]
    jobs: list[dict]
    applications: list[dict]
    problems: list[str] = field(default_factory=list)

    @property
    def students_by_id(self) -> dict[str, dict]:
        return {s["student_id"]: s for s in self.students}

    @property
    def jobs_by_id(self) -> dict[str, dict]:
        return {j["job_id"]: j for j in self.jobs}

    def summary(self) -> str:
        lines = [
            "Dataset loaded",
            f"  students     : {len(self.students)}",
            f"  jobs         : {len(self.jobs)}",
            f"  applications : {len(self.applications)}",
        ]
        outcomes = outcome_counts(self.applications)
        if outcomes:
            lines.append("  outcomes (usable as gold labels):")
            for status, n in sorted(outcomes.items(), key=lambda kv: -kv[1]):
                lines.append(f"      {status:<28} {n}")
        if self.problems:
            lines.append(f"\n  ⚠ {len(self.problems)} problem(s):")
            lines += [f"      - {p}" for p in self.problems[:40]]
            if len(self.problems) > 40:
                lines.append(f"      ... and {len(self.problems) - 40} more")
        else:
            lines.append("\n  ✓ no problems found")
        return "\n".join(lines)


def _scan_for_pii(record: dict, where: str, problems: list[str]) -> None:
    """Reject the export if PII survived the pseudonymisation step."""
    for key in record:
        if key.lower() in FORBIDDEN_FIELDS:
            problems.append(f"PRIVACY: {where} contains forbidden field '{key}'")

    blob = json.dumps(record, ensure_ascii=False)
    if _EMAIL_RE.search(blob):
        problems.append(f"PRIVACY: {where} contains something that looks like an email")
    # Long digit runs are checked only in free text, where a phone number would hide.
    for key in ("description", "summary", "motivation_text"):
        val = record.get(key)
        if isinstance(val, str) and _PHONE_RE.search(val):
            problems.append(f"PRIVACY: {where} field '{key}' may contain a phone number")


def _require(record: dict, keys: tuple[str, ...], where: str, problems: list[str]) -> None:
    for k in keys:
        if k not in record:
            problems.append(f"{where}: missing required field '{k}'")
        elif record[k] is None or record[k] == "":
            problems.append(f"{where}: field '{k}' is empty")


def load_students(path: Path, problems: list[str]) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()

    for i, s in enumerate(records):
        where = f"students[{i}]"
        _require(s, ("student_id", "city", "start_date", "matching_category"), where, problems)
        _scan_for_pii(s, where, problems)

        sid = s.get("student_id")
        if sid in seen:
            problems.append(f"{where}: duplicate student_id '{sid}'")
        seen.add(sid)

        for lang in s.get("languages") or []:
            prof = lang.get("proficiency")
            if prof not in PROFICIENCY_TO_CEFR:
                problems.append(
                    f"{where}: unknown proficiency '{prof}' "
                    f"(expected one of {sorted(PROFICIENCY_TO_CEFR)})"
                )

        # A student with no interests, no experience and no career fields has nothing to
        # embed and nothing to rank on. Better to know now than to see a mystery zero later.
        if not any((s.get("top_job_interests"), s.get("career_fields"),
                    s.get("work_experience"), s.get("activities"))):
            problems.append(f"{where}: no interests, career fields, experience or activities")

    return records


def load_jobs(path: Path, problems: list[str]) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()

    for i, j in enumerate(records):
        where = f"jobs[{i}]"
        _require(j, ("job_id", "title", "city", "start_date", "matching_category"), where, problems)
        _scan_for_pii(j, where, problems)

        jid = j.get("job_id")
        if jid in seen:
            problems.append(f"{where}: duplicate job_id '{jid}'")
        seen.add(jid)

        for req in j.get("language_requirements") or []:
            level = req.get("level")
            if level not in CEFR_ORDER:
                problems.append(f"{where}: unknown CEFR level '{level}'")
            if req.get("kind") not in ("REQUIRED", "OPTIONAL"):
                problems.append(f"{where}: language kind must be REQUIRED or OPTIONAL")

    return records


def load_applications(
    path: Path, student_ids: set[str], job_ids: set[str], problems: list[str]
) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))

    for i, a in enumerate(records):
        where = f"applications[{i}]"
        _require(a, ("student_id", "job_id", "status"), where, problems)

        # A dangling reference means the three files were exported at different times and
        # cannot be joined; any evaluation built on them would be quietly wrong.
        if a.get("student_id") not in student_ids:
            problems.append(f"{where}: student_id '{a.get('student_id')}' not in students.json")
        if a.get("job_id") not in job_ids:
            problems.append(f"{where}: job_id '{a.get('job_id')}' not in jobs.json")

    return records


# Company decisions that constitute a real human relevance judgement.
POSITIVE_STATUSES = {
    "ACCEPTED",
    "AI_INTERVIEW_INVITED", "AI_INTERVIEW_IN_PROGRESS", "AI_INTERVIEW_COMPLETED",
    "ONLINE_INTERVIEW_INVITED", "ONLINE_INTERVIEW_COMPLETED",
    "IN_PERSON_INTERVIEW_COMPLETED",
    "CHAT_STARTED", "PENDING_PAYMENT", "PENDING_STUDENT_AI",
}
NEGATIVE_STATUSES = {"REJECTED_BY_COMPANY"}
# Everything else (e.g. PENDING_COMPANY, REJECTED_BY_STUDENT) carries no company verdict:
# the company either has not decided, or the student withdrew. Neither says the match was
# bad, so treating them as negatives would poison the labels.


def outcome_counts(applications: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in applications:
        st = a.get("status")
        if st in POSITIVE_STATUSES or st in NEGATIVE_STATUSES:
            counts[st] = counts.get(st, 0) + 1
    return counts


def gold_labels(applications: list[dict]) -> dict[tuple[str, str], int]:
    """
    Real recruiter verdicts, as binary relevance: 1 = the company advanced or accepted
    the candidate, 0 = the company rejected them.

    Pairs with no company decision are **omitted**, not defaulted to 0 — this is the
    missing-not-at-random caveat declared in the protocol.
    """
    labels: dict[tuple[str, str], int] = {}
    for a in applications:
        key = (a["student_id"], a["job_id"])
        if a["status"] in POSITIVE_STATUSES:
            labels[key] = 1
        elif a["status"] in NEGATIVE_STATUSES:
            labels[key] = 0
    return labels


def load_dataset(data_dir: Path) -> Dataset:
    problems: list[str] = []
    students = load_students(data_dir / "students.json", problems)
    jobs = load_jobs(data_dir / "jobs.json", problems)
    applications = load_applications(
        data_dir / "applications.json",
        {s.get("student_id") for s in students},
        {j.get("job_id") for j in jobs},
        problems,
    )
    return Dataset(students=students, jobs=jobs, applications=applications, problems=problems)


if __name__ == "__main__":
    import sys

    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "data"
    ds = load_dataset(directory)
    print(ds.summary())
    if any(p.startswith("PRIVACY") for p in ds.problems):
        print("\n❌ PRIVACY FAILURE — do not use this export. Ask for it to be re-exported.")
        sys.exit(1)
