"""
Cross-validation and hyperparameter search for the training pipelines.

Every choice is made on the training rows only: which settings a model
gets, and which model ranks first. The held-out test rows are scored
once, after those choices, so the test score stays an honest estimate
even when settings were tuned. The cross-validated score of a tuned
model is slightly optimistic, because it is the score the search picked.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

DEFAULT_FOLDS = 5
MAX_FOLDS = 10

# Candidates tried per model. The spaces below are small on purpose: a
# hosted demo shares a few CPU cores, and a wide search on a small table
# mostly finds settings that fit the folds rather than the problem.
SEARCH_ITERATIONS = 8

RANDOM_STATE = 42

SCORING = {
    "classification": "f1_macro",
    "regression": "neg_root_mean_squared_error",
}

SCORE_LABEL = {
    "classification": "macro F1",
    "regression": "RMSE",
}

TREE_SPACE = {
    "model__max_depth": [None, 3, 5, 8, 12],
    "model__min_samples_leaf": [1, 2, 5, 10, 20],
}

FOREST_SPACE = {
    "model__max_depth": [None, 8, 16],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features": ["sqrt", 0.5, 1.0],
}

SEARCH_SPACES: dict[str, dict[str, dict[str, list[Any]]]] = {
    "classification": {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        },
        "Decision Tree": TREE_SPACE,
        "Random Forest": FOREST_SPACE,
    },
    "regression": {
        # Ordinary least squares has no setting worth searching.
        "Linear Regression": {},
        "Decision Tree": TREE_SPACE,
        "Random Forest": FOREST_SPACE,
    },
}


@dataclass
class FoldPlan:
    """How many folds to use, and why, if that differs from the request."""

    folds: int
    note: str = ""


@dataclass
class Selection:
    """A fitted pipeline with the evidence behind it."""

    pipeline: Pipeline
    cv_mean: float | None = None
    cv_std: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    candidates_tried: int = 0


def plan_folds(
    y_train: pd.Series,
    task: str,
    requested: int,
) -> FoldPlan:
    """
    Choose a fold count the training rows can support.

    Args:
        y_train:
            Target values of the training rows.

        task:
            classification or regression.

        requested:
            Folds asked for. Below 2 switches cross-validation off.

    Returns:
        The fold count, 0 when cross-validation cannot run, with a note
        whenever the result differs from the request.
    """

    if requested < 2:
        return FoldPlan(0, "Cross-validation was switched off.")

    requested = min(int(requested), MAX_FOLDS)

    if task == "classification":
        smallest = int(pd.Series(y_train).value_counts().min())
        folds = min(requested, smallest)

        if folds < 2:
            return FoldPlan(
                0,
                f"Cross-validation was skipped: the rarest class has "
                f"{smallest} training row, and every fold needs one.",
            )

        if folds < requested:
            return FoldPlan(
                folds,
                f"Reduced to {folds} folds, because the rarest class has "
                f"only {smallest} training rows and every fold needs one.",
            )

        return FoldPlan(folds)

    folds = min(requested, len(y_train) // 2)

    if folds < 2:
        return FoldPlan(
            0,
            "Cross-validation was skipped: there are too few training "
            "rows to split into folds.",
        )

    if folds < requested:
        return FoldPlan(
            folds,
            f"Reduced to {folds} folds, so that every fold holds at "
            "least two rows.",
        )

    return FoldPlan(folds)


def splitter(task: str, folds: int) -> KFold | StratifiedKFold:
    """Shuffled folds with a fixed seed, stratified for classification."""

    if task == "classification":
        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def search_space(task: str, model_name: str) -> dict[str, list[Any]]:
    """The settings searched for one model, empty when there are none."""

    return SEARCH_SPACES.get(task, {}).get(model_name, {})


def plain(value: Any) -> Any:
    """Turn numpy scalars into Python values, so the result is JSON-safe."""

    if isinstance(value, np.generic):
        return value.item()

    return value


def fit_and_score(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: str,
    model_name: str,
    folds: int,
    tune: bool,
) -> Selection:
    """
    Cross-validate, optionally tune, then fit on all the training rows.

    Args:
        pipeline:
            Unfitted preprocessing and model pipeline.

        X_train, y_train:
            The training rows. The test rows must never be passed here.

        task:
            classification or regression.

        model_name:
            Leaderboard name, used to look up the search space.

        folds:
            Fold count from plan_folds. 0 skips cross-validation and
            tuning, and the pipeline is simply fitted.

        tune:
            Search the model's settings with randomised search.

    Returns:
        The fitted pipeline, its cross-validated score (RMSE is reported
        as a positive number), and the settings the search chose.
    """

    scoring = SCORING[task]

    # scikit-learn maximises scores, so errors arrive negated.
    sign = -1.0 if scoring.startswith("neg_") else 1.0

    if not folds:
        pipeline.fit(X_train, y_train)

        return Selection(pipeline)

    space = search_space(task, model_name) if tune else {}

    if space:
        size = len(ParameterGrid(space))

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=space,
            n_iter=min(SEARCH_ITERATIONS, size),
            scoring=scoring,
            cv=splitter(task, folds),
            random_state=RANDOM_STATE,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best = search.best_index_
        results = search.cv_results_

        return Selection(
            pipeline=search.best_estimator_,
            cv_mean=sign * float(results["mean_test_score"][best]),
            cv_std=float(results["std_test_score"][best]),
            parameters={
                name.removeprefix("model__"): plain(value)
                for name, value in search.best_params_.items()
            },
            candidates_tried=int(len(results["params"])),
        )

    scores = cross_val_score(
        clone(pipeline),
        X_train,
        y_train,
        scoring=scoring,
        cv=splitter(task, folds),
        error_score="raise",
    )

    pipeline.fit(X_train, y_train)

    return Selection(
        pipeline=pipeline,
        cv_mean=sign * float(np.mean(scores)),
        cv_std=float(np.std(scores)),
    )


def rounded(value: float | None) -> float | None:
    """Round a score for the leaderboard, keeping a missing score missing."""

    return None if value is None else round(float(value), 4)
