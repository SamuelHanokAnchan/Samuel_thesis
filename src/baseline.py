"""
System 1 — the TF-IDF keyword baseline (the exposé's control condition).

Exposé, §3.4: *"Implement Baseline System: keyword overlap matching using TF-IDF vectors
(scikit-learn) on skills and role title fields only."*

That restriction is the point. The baseline is meant to be a **fair but deliberately
shallow** representative of how conventional job portals match: lexical overlap between
what the student says they can do and what the advert asks for. It has no semantics — it
cannot know that "Kfz-Mechatroniker" and "vehicle technician" are the same job — and
demonstrating exactly that failure is the reason it exists.

It is *not* a strawman: it is fitted on the same corpus, ranks the same filtered candidate
pool, and is scored with the same metrics as the other two systems.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def student_keyword_text(student: dict) -> str:
    """Student side of the baseline: interests, career fields and job titles held.

    Skills-and-titles only, per the exposé. Free-text descriptions are excluded — giving
    the baseline the full profile text would quietly turn it into a bag-of-words semantic
    model and stop it being the keyword control the thesis compares against.
    """
    parts: list[str] = []
    parts += [str(i) for i in (student.get("career_fields") or [])]
    parts += [str(i) for i in (student.get("top_job_interests") or [])]
    parts += [
        str(w.get("job_title", ""))
        for w in (student.get("work_experience") or [])
    ]
    return " ".join(p for p in parts if p).strip()


def job_keyword_text(job: dict) -> str:
    """Job side of the baseline: role title plus required and optional skills."""
    parts: list[str] = [str(job.get("title") or "")]
    parts += [str(s) for s in (job.get("required_skills") or [])]
    parts += [str(s) for s in (job.get("optional_skills") or [])]
    return " ".join(p for p in parts if p).strip()


class TfidfBaseline:
    """Rank jobs for a student by TF-IDF cosine similarity over skills + title."""

    def __init__(self, jobs: list[dict]) -> None:
        self.jobs = jobs
        self.job_ids = [j["job_id"] for j in jobs]
        job_texts = [job_keyword_text(j) for j in jobs]

        # Fitted on the job corpus only. IDF must come from one fixed vocabulary; fitting
        # per student would give the same word a different weight for different students
        # and make the scores incomparable across queries.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
        )
        self.job_matrix = self.vectorizer.fit_transform(job_texts)
        self._row_of = {jid: i for i, jid in enumerate(self.job_ids)}

    def score(self, student: dict, candidate_job_ids: list[str]) -> dict[str, float]:
        """Cosine similarity between the student's keyword text and each candidate job."""
        if not candidate_job_ids:
            return {}

        text = student_keyword_text(student)
        if not text:
            # No keywords at all -> the baseline has nothing to go on. Score everything 0
            # rather than crashing; the metrics will (correctly) show it performing badly.
            return {jid: 0.0 for jid in candidate_job_ids}

        student_vec = self.vectorizer.transform([text])
        rows = [self._row_of[jid] for jid in candidate_job_ids if jid in self._row_of]
        if not rows:
            return {jid: 0.0 for jid in candidate_job_ids}

        sims = cosine_similarity(student_vec, self.job_matrix[rows])[0]
        scored = {
            jid: float(sim)
            for jid, sim in zip(
                [j for j in candidate_job_ids if j in self._row_of], sims
            )
        }
        # Any candidate the vectorizer never saw scores 0 rather than vanishing, so every
        # system ranks exactly the same candidate set.
        for jid in candidate_job_ids:
            scored.setdefault(jid, 0.0)
        return scored

    def rank(self, student: dict, candidate_job_ids: list[str]) -> list[str]:
        """Candidate job ids, best first. Ties break on job_id so runs are reproducible."""
        scored = self.score(student, candidate_job_ids)
        return sorted(candidate_job_ids, key=lambda jid: (-scored.get(jid, 0.0), jid))
