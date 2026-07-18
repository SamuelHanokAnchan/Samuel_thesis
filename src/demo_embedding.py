"""
Local, key-free stand-in for the production Stage 2 embedding.

WHY THIS EXISTS
---------------
In production, Stage 2 embeds each profile and vacancy with Google's
``gemini-embedding-001`` (768-d), reached over an API. Reproducing that needs a
paid key and network access, and the exact production embedding text also depends
on the proprietary ESCO occupation catalogue. Neither is shipped in this public
repository.

So that the pipeline walkthrough runs anywhere, offline and without a key, this
module supplies a small deterministic embedder built only on ``scikit-learn``. It
is NOT the model evaluated in the thesis. It stands in for it, and it keeps the
*shape* of the pipeline identical: flatten a profile into one text paragraph,
embed it into a unit vector, and compare by cosine similarity.

ONE PROPERTY IS PRESERVED ON PURPOSE
------------------------------------
The search text below lists language *names* and never *proficiency levels*, exactly
as the production ``get_search_text`` does (see thesis §3.4.2). That blind spot is
the whole reason the re-ranker exists, and keeping it here lets the walkthrough
demonstrate it: "German (Basic)" and "German (Native)" produce nearly identical
vectors, and only Stage 3 can tell them apart.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# A fixed hashing vectoriser over word and character n-grams. Deterministic, no
# vocabulary to fit, no network. 256 dimensions is plenty for a 10-row demo.
_WORD = HashingVectorizer(n_features=256, alternate_sign=False, norm=None,
                          analyzer="word", ngram_range=(1, 2))
_CHAR = HashingVectorizer(n_features=256, alternate_sign=False, norm=None,
                          analyzer="char_wb", ngram_range=(3, 4))


def student_search_text(student: dict) -> str:
    """Flatten a student into one paragraph. Language NAMES only, no proficiency."""
    interests = ", ".join(str(i) for i in (student.get("career_fields") or [])
                          + (student.get("top_job_interests") or [])) or "various fields"
    work = "; ".join(str(w.get("job_title", "")) for w in (student.get("work_experience") or [])) or "none"
    activities = ", ".join(str(a.get("name", "")) for a in (student.get("activities") or [])) or "none"
    languages = ", ".join(l.get("language", "") for l in (student.get("languages") or [])) or "none"
    return (f"A student interested in {interests} with experience in {work}, "
            f"active in {activities}, with languages including {languages}.")


def job_search_text(job: dict) -> str:
    """Flatten a vacancy into one paragraph. Language NAMES only, no CEFR level."""
    parts = [str(job.get("title") or "")]
    parts += [str(job.get("description") or "")]
    parts += [str(r) for r in (job.get("responsibilities") or [])]
    parts += [str(s) for s in (job.get("required_skills") or [])]
    parts += [str(s) for s in (job.get("optional_skills") or [])]
    langs = ", ".join(r.get("language", "") for r in (job.get("language_requirements") or []))
    parts.append(f"Languages: {langs}")
    return " ".join(p for p in parts if p)


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into L2-normalised vectors (rows). Deterministic."""
    w = _WORD.transform(texts).toarray()
    c = _CHAR.transform(texts).toarray()
    return _l2_normalise(np.hstack([w, c]).astype(float))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Vectors are unit length, so this is just the dot product."""
    return float(np.dot(a, b))
