"""
Ranking metrics for the thesis evaluation.

Every number reported in the thesis comes out of this module, so each function is
kept small, pure, and covered by a hand-computed test in ``tests/test_metrics.py``.

Conventions
-----------
Relevance grades are integers **1..5** (the human/LLM rating of a student-job pair).

* **Binary relevance** (Precision@k, MRR): a pair is *relevant* when
  ``grade >= RELEVANT_THRESHOLD``. The threshold is 4, fixed by the approved exposé
  ("threshold: rating >= 4 = relevant match").
* **Graded relevance** (NDCG): grades are converted to *gains*. The default is
  ``gain = grade - 1`` so a grade of 1 ("no match") contributes **zero** gain.
  Using the raw grade would give an irrelevant item a gain of 1, which silently
  rewards a system for ranking bad matches highly.

A "query" is one student. Metrics are computed **per query** and then averaged
across queries, never pooled over all pairs at once — pooling would let a student
with many candidates dominate the mean.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

RELEVANT_THRESHOLD = 4  # exposé: rating >= 4 counts as a relevant match


# --------------------------------------------------------------------------
# gains
# --------------------------------------------------------------------------
def linear_gain(grade: float) -> float:
    """``grade - 1``, floored at 0. Grade 1 -> 0.0, grade 5 -> 4.0."""
    return max(0.0, float(grade) - 1.0)


def exponential_gain(grade: float) -> float:
    """``2**(grade-1) - 1``. Rewards top grades far more steeply than ``linear_gain``."""
    return float(2 ** (float(grade) - 1.0) - 1.0)


# --------------------------------------------------------------------------
# per-query metrics
# --------------------------------------------------------------------------
def precision_at_k(ranked_grades: Sequence[float], k: int) -> float:
    """
    Fraction of the top ``k`` ranked items that are relevant (grade >= 4).

    ``ranked_grades`` is the list of true grades **in the order the system ranked them**.
    The denominator is ``min(k, len(ranked_grades))``, so a query with fewer than ``k``
    candidates is not unfairly penalised for candidates it never had.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranked_grades:
        return 0.0

    top = ranked_grades[:k]
    hits = sum(1 for g in top if g >= RELEVANT_THRESHOLD)
    return hits / len(top)


def dcg_at_k(
    ranked_grades: Sequence[float],
    k: int,
    gain_fn: Callable[[float], float] = linear_gain,
) -> float:
    """Discounted cumulative gain: ``sum(gain(g_i) / log2(i + 2))`` over the top ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(
        gain_fn(g) / math.log2(i + 2)  # i is 0-based, so rank 1 -> log2(2) == 1
        for i, g in enumerate(ranked_grades[:k])
    )


def ndcg_at_k(
    ranked_grades: Sequence[float],
    k: int,
    gain_fn: Callable[[float], float] = linear_gain,
) -> float:
    """
    Normalised DCG: the system's DCG divided by the best DCG achievable for this query.

    The ideal ordering is this query's own grades sorted descending, so NDCG is 1.0
    when the system ranks perfectly and is comparable across queries of differing
    difficulty.

    Returns 0.0 when the query has no positive gain at all (nothing to rank well);
    without this guard the ideal DCG would be 0 and we would divide by zero.
    """
    if not ranked_grades:
        return 0.0

    ideal = sorted(ranked_grades, reverse=True)
    idcg = dcg_at_k(ideal, k, gain_fn)
    if idcg == 0.0:
        return 0.0

    return dcg_at_k(ranked_grades, k, gain_fn) / idcg


def reciprocal_rank(ranked_grades: Sequence[float]) -> float:
    """``1 / rank`` of the first relevant item, or 0.0 if the query has none."""
    for i, g in enumerate(ranked_grades):
        if g >= RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


# --------------------------------------------------------------------------
# aggregation across queries
# --------------------------------------------------------------------------
def evaluate_system(
    ranked_grades_per_query: Sequence[Sequence[float]],
    ks: Sequence[int] = (3, 5),
    gain_fn: Callable[[float], float] = linear_gain,
) -> dict[str, float]:
    """
    Mean metrics across queries (one entry per student).

    Each element of ``ranked_grades_per_query`` is that student's true grades in the
    order the system ranked their candidate jobs. Empty queries are dropped rather
    than scored as 0, so a student with no candidates cannot drag the mean down.
    """
    queries = [q for q in ranked_grades_per_query if q]
    if not queries:
        return {}

    results: dict[str, float] = {}
    for k in ks:
        results[f"precision@{k}"] = _mean(precision_at_k(q, k) for q in queries)
        results[f"ndcg@{k}"] = _mean(ndcg_at_k(q, k, gain_fn) for q in queries)
    results["mrr"] = _mean(reciprocal_rank(q) for q in queries)
    results["n_queries"] = float(len(queries))
    return results


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------
# annotation agreement
# --------------------------------------------------------------------------
def cohens_kappa(a: Sequence[int], b: Sequence[int], weighted: str | None = None) -> float:
    """
    Cohen's kappa between two raters (here: the LLM's draft grade vs. the human's
    verified grade), measuring agreement **above what chance would produce**.

    ``weighted``:
      * ``None``       - exact agreement only; a 4-vs-5 disagreement counts as badly
                         as 1-vs-5. Harsh for ordinal grades.
      * ``"linear"``   - disagreement cost grows with the gap.
      * ``"quadratic"`` - disagreement cost grows with the *square* of the gap; this
                         is the usual choice for ordinal 1-5 scales and is what the
                         thesis reports.

    Returns 1.0 when the raters agree perfectly, 0.0 at chance level, negative below.
    Degenerate case: if both raters gave one identical constant grade to everything,
    expected agreement is 1.0 and kappa is undefined; we return 1.0 (they did agree).
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must be the same length")
    if not a:
        raise ValueError("cannot compute kappa on empty input")

    categories = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(categories)}
    n_cat = len(categories)
    n = len(a)

    observed = [[0.0] * n_cat for _ in range(n_cat)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1.0

    a_marg = [sum(row) / n for row in observed]
    b_marg = [sum(observed[i][j] for i in range(n_cat)) / n for j in range(n_cat)]

    if weighted is None:
        def cost(i: int, j: int) -> float:
            return 0.0 if i == j else 1.0
    elif weighted == "linear":
        def cost(i: int, j: int) -> float:
            return abs(categories[i] - categories[j]) / (categories[-1] - categories[0])
    elif weighted == "quadratic":
        def cost(i: int, j: int) -> float:
            return ((categories[i] - categories[j]) / (categories[-1] - categories[0])) ** 2
    else:
        raise ValueError("weighted must be None, 'linear' or 'quadratic'")

    if n_cat == 1:
        return 1.0  # both raters used a single identical category

    obs_disagree = sum(
        cost(i, j) * observed[i][j] / n for i in range(n_cat) for j in range(n_cat)
    )
    exp_disagree = sum(
        cost(i, j) * a_marg[i] * b_marg[j] for i in range(n_cat) for j in range(n_cat)
    )

    if exp_disagree == 0.0:
        return 1.0
    return 1.0 - obs_disagree / exp_disagree


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------
def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Paired bootstrap over queries: is system B really better than system A, or is the
    gap noise?

    ``scores_a`` / ``scores_b`` are **per-query** scores of the same metric on the same
    queries, in the same order (e.g. NDCG@5 for each of 40 students under each system).

    With ~40 queries a raw improvement of a few points is very often chance, so the
    thesis reports the confidence interval and p-value from this function alongside
    every headline comparison.

    Returns the observed mean difference (B - A), a 95% CI, and a two-sided p-value
    for H0: no difference. ``p < 0.05`` supports a real improvement.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("paired bootstrap needs equal-length, aligned score lists")
    if not scores_a:
        raise ValueError("cannot bootstrap empty input")

    rng = random.Random(seed)
    n = len(scores_a)
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    observed = _mean(diffs)

    # CI: resample the observed differences.
    means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]

    # p-value: centre the differences so the null (mean diff == 0) is true, then ask
    # how often a resample is at least as extreme as what we actually observed.
    centred = [d - observed for d in diffs]
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        sample_mean = _mean([centred[rng.randrange(n)] for _ in range(n)])
        if abs(sample_mean) >= abs(observed):
            at_least_as_extreme += 1
    p = (at_least_as_extreme + 1) / (n_resamples + 1)  # add-one: never report p == 0

    return {
        "mean_diff": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "n_queries": float(n),
    }
