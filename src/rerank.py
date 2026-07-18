"""
System 3 — the learned re-ranker (the thesis contribution).

LightGBM with the ``lambdarank`` objective: a learning-to-rank model that optimises the
*order* of a student's candidate jobs directly, rather than predicting each pair's score in
isolation. That matches what the metrics actually measure.

Why gradient-boosted trees and not a neural ranker: with ~700 labelled pairs a neural model
would memorise the training set. GBDTs are strong in the small-n regime and — critically for
the thesis — expose feature importances, which is what turns a results table into a
Discussion chapter.

**Evaluation is grouped by student, never by pair.** A student's pairs all go into the same
fold. Splitting by pair would put the same student in both train and test, and the model
would score itself on people it had already seen — inflating every number. This is the
single most common way a thesis like this gets quietly invalidated, so it is enforced here
rather than left to convention.
"""
from __future__ import annotations

import numpy as np

from src.features import FEATURE_NAMES


def train_reranker(X: np.ndarray, y: np.ndarray, groups: list[int], seed: int = 42):
    """
    Fit a LambdaRank model.

    ``groups`` is the number of candidate rows per query, in order — LightGBM needs this to
    know which rows compete against each other. Rows from different students must never be
    compared: a job that is "better" for one student says nothing about another.
    """
    import lightgbm as lgb

    dataset = lgb.Dataset(X, label=y, group=groups, feature_name=FEATURE_NAMES)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.05,
        # Deliberately small trees. With a few hundred queries, a deeper forest fits noise
        # and the held-out score collapses.
        "num_leaves": 7,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambdarank_truncation_level": 10,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
    }
    return lgb.train(params, dataset, num_boost_round=120)


def feature_importance(model) -> list[tuple[str, float]]:
    """
    Which signals the model actually leaned on, as a share of total gain.

    This is the interpretability payoff: it is what lets the thesis say *"language gap
    mattered N times more than distance"* — a finding about apprenticeship matching, not
    just about the model.
    """
    gains = model.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    pairs = [(name, float(g) / total) for name, g in zip(FEATURE_NAMES, gains)]
    return sorted(pairs, key=lambda kv: -kv[1])


def group_kfold_by_student(student_ids: list[str], n_folds: int = 5, seed: int = 42):
    """
    Yield (train_idx, test_idx) with every student wholly in one fold.

    Written by hand rather than using sklearn's GroupKFold so the shuffle is seeded and the
    folds are reproducible run to run — a requirement for the thesis to be re-runnable.
    """
    rng = np.random.RandomState(seed)
    uniq = sorted(set(student_ids))
    rng.shuffle(uniq)
    fold_of = {sid: i % n_folds for i, sid in enumerate(uniq)}

    ids = np.array(student_ids)
    for fold in range(n_folds):
        test_mask = np.array([fold_of[s] == fold for s in ids])
        yield np.where(~test_mask)[0], np.where(test_mask)[0]
