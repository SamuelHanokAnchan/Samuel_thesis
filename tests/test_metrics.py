"""
Hand-computed tests for the ranking metrics.

Expected values here are worked out by hand (arithmetic shown in the comments), not
copied from the implementation's own output — otherwise the tests would only prove
the code agrees with itself.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (  # noqa: E402
    cohens_kappa,
    dcg_at_k,
    evaluate_system,
    linear_gain,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------- precision
def test_precision_counts_only_grades_4_and_5():
    # grades 5,4 are relevant; 3,2,1 are not -> 2 of the top 5
    assert precision_at_k([5, 4, 3, 2, 1], 5) == pytest.approx(2 / 5)


def test_precision_at_3_looks_only_at_top_3():
    # top 3 = [5, 1, 4] -> two relevant. The trailing 5s are ignored.
    assert precision_at_k([5, 1, 4, 5, 5], 3) == pytest.approx(2 / 3)


def test_precision_denominator_shrinks_for_short_queries():
    # Only 2 candidates exist. 1 relevant out of 2 -> 0.5, NOT 1/5.
    # A student the system had few candidates for must not be scored as if it had 5.
    assert precision_at_k([4, 1], 5) == pytest.approx(0.5)


def test_precision_all_irrelevant_is_zero():
    assert precision_at_k([1, 2, 3], 3) == 0.0


# ---------------------------------------------------------------- ndcg
def test_linear_gain_makes_grade_1_worthless():
    # This is the reason we don't use the raw grade as the gain.
    assert linear_gain(1) == 0.0
    assert linear_gain(5) == 4.0


def test_dcg_matches_hand_calculation():
    # grades [5, 1] -> gains [4, 0]
    # DCG = 4/log2(2) + 0/log2(3) = 4/1 + 0 = 4.0
    assert dcg_at_k([5, 1], 2) == pytest.approx(4.0)


def test_ndcg_is_1_when_ranking_is_perfect():
    assert ndcg_at_k([5, 4, 3, 2, 1], 5) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking():
    # ranked [1,5] -> gains [0,4]: DCG = 0/1 + 4/log2(3) = 4/1.58496 = 2.5237
    # ideal  [5,1] -> gains [4,0]: DCG = 4.0
    # NDCG = 2.5237 / 4.0 = 0.6309
    expected = (4 / math.log2(3)) / 4.0
    assert ndcg_at_k([1, 5], 2) == pytest.approx(expected)
    assert ndcg_at_k([1, 5], 2) < 1.0


def test_ndcg_is_zero_when_nothing_is_relevant():
    # All grades 1 -> all gains 0 -> ideal DCG is 0. Must not raise ZeroDivisionError.
    assert ndcg_at_k([1, 1, 1], 3) == 0.0


# ---------------------------------------------------------------- mrr
def test_reciprocal_rank_finds_first_relevant():
    assert reciprocal_rank([1, 2, 4]) == pytest.approx(1 / 3)  # first hit at rank 3
    assert reciprocal_rank([5, 1, 1]) == pytest.approx(1.0)  # rank 1
    assert reciprocal_rank([1, 2, 3]) == 0.0  # no hit


# ---------------------------------------------------------------- aggregation
def test_metrics_are_averaged_per_query_not_pooled():
    # Query A is perfect (P@3 = 1.0). Query B has one relevant of three (P@3 = 1/3),
    # but has many more candidates. Pooling pairs would let B dominate; averaging
    # per query must give exactly (1.0 + 1/3) / 2 = 0.6667.
    q_a = [5, 4, 4]
    q_b = [4, 1, 1, 1, 1, 1, 1, 1, 1]
    res = evaluate_system([q_a, q_b], ks=(3,))
    assert res["precision@3"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert res["n_queries"] == 2.0


def test_empty_queries_are_dropped_not_scored_zero():
    res = evaluate_system([[5, 4, 4], []], ks=(3,))
    assert res["n_queries"] == 1.0
    assert res["precision@3"] == pytest.approx(1.0)


# ---------------------------------------------------------------- kappa
def test_kappa_is_1_on_perfect_agreement():
    a = [1, 2, 3, 4, 5]
    assert cohens_kappa(a, a, weighted="quadratic") == pytest.approx(1.0)


def test_kappa_near_zero_when_agreement_is_chance():
    # Raters are independent and uncorrelated -> kappa should sit near 0.
    a = [1, 2, 1, 2, 1, 2, 1, 2]
    b = [1, 1, 2, 2, 1, 1, 2, 2]
    assert abs(cohens_kappa(a, b)) < 0.35


def test_quadratic_kappa_is_kinder_to_near_misses_than_unweighted():
    # The human nudged every grade by one step. Unweighted kappa treats that as total
    # disagreement; quadratic weighting recognises the raters nearly agreed.
    ai = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    human = [2, 3, 4, 5, 4, 2, 3, 4, 5, 4]
    assert cohens_kappa(ai, human, weighted="quadratic") > cohens_kappa(ai, human)


def test_kappa_handles_both_raters_giving_one_constant_grade():
    # Degenerate: expected disagreement is 0, so kappa is mathematically undefined.
    # They did in fact agree on every item, so we report 1.0 rather than crashing.
    assert cohens_kappa([3, 3, 3], [3, 3, 3]) == 1.0


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_detects_a_real_consistent_improvement():
    # B beats A on every one of 30 queries by a clear margin.
    a = [0.30] * 30
    b = [0.60] * 30
    res = paired_bootstrap(a, b, n_resamples=2000)
    assert res["mean_diff"] == pytest.approx(0.30)
    assert res["p_value"] < 0.05
    assert res["ci_low"] > 0  # CI excludes zero -> improvement is real


def test_bootstrap_reports_noise_as_not_significant():
    # This is the case that protects the thesis: B's mean is higher, but it is driven
    # by noise, so the CI must straddle zero and p must be large. Claiming victory on
    # this data would be a false positive.
    a = [0.5, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]
    b = [0.1, 0.9, 0.2, 0.9, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9]
    res = paired_bootstrap(a, b, n_resamples=2000)
    assert res["p_value"] > 0.05
    assert res["ci_low"] < 0 < res["ci_high"]


def test_bootstrap_p_value_is_never_zero():
    # add-one smoothing: reporting "p = 0.000" would be a false claim of certainty.
    res = paired_bootstrap([0.0] * 20, [1.0] * 20, n_resamples=500)
    assert res["p_value"] > 0.0


def test_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])
