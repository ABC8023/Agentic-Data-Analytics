"""
Machine learning: target analysis, training pipelines, metrics and
feature importance, with the agent tools that wrap them.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import ml_export
import ml_tuning
from dataset_store import get_or_load_dataframe
from privacy import model_safe
from profiling import data_time_column


def analyse_tool(name: str,target_column: str)-> dict[str,Any]:
    """
    Inspect a selected target column and recommend whether the
    machine-learning task should be classification or regression.

    Args:
        name:
            Name of the dataset stored in DATAFRAME_CACHE.

        target_column:
            Column that the model should predict.

    Returns:
        A structured report describing the target column and the
        recommended machine-learning task.
    """

    try:
        df= get_or_load_dataframe(name)
    except(ValueError , FileNotFoundError) as error:
        return{
            "success":False,
            "error":str(error)

        }

    if target_column not in df.columns:
        return{
            "success":False,
            "error":(
                f"Column {target_column} not found in dataset {name}"

            )}
    target=df[target_column]

    total_rows=int(len(target))
    missing_values=int(target.isnull().sum())

    non_missing_value=target.dropna()

    if non_missing_value.empty:
        return{
            "success":False,
            "error":(
                f"Target Column {target_column} contains only missing values"
            )
        }
    unique_values=int(non_missing_value.nunique())

    unique_ratio=(unique_values/ len(non_missing_value))

    is_numeric=pd.api.types.is_numeric_dtype(non_missing_value)

    is_boolean=pd.api.types.is_bool_dtype(non_missing_value)

    warnings=[]

    if is_boolean:
        reccomended_task= "Classification"

        reason=(
            "The target holds true and false values, so the model "
            "predicts which of the two it is."
        )
    elif not is_numeric:
        reccomended_task="Classification"

        reason=(
            "The target holds text or category values rather than "
            "numbers, so the model predicts a label."
        )
    elif(
        unique_values <= 20
        and unique_ratio <=0.20
    ):
        reccomended_task="Classification"
        reason=(f"The target is numerical but contains only "
            f"{unique_values} repeated values. These values probably "
            f"represent classes rather than continuous measurements.")
    else:
        reccomended_task="Regression"
        reason = (
            "The target contains many numerical values, which suggests "
            "that the model should predict a continuous quantity."
        )

    # Whether the task type is classification or regression is a separate
    # question from whether this column can actually be trained on. A
    # product code or any other identifier is categorical, so the task
    # type reads as classification, but it has roughly one row per value
    # and there is nothing to generalise from. These checks catch that
    # before the user presses Train and hits a stratification error.
    is_trainable = True
    blocking_reason = ""

    singleton_classes = 0
    looks_like_identifier = False

    # These checks only make sense for a categorical target. A continuous
    # regression target is expected to be almost entirely unique, so a
    # high unique ratio there is normal rather than a warning sign.
    if reccomended_task == "Classification":
        class_counts = non_missing_value.value_counts()
        singleton_classes = int((class_counts < 2).sum())

        looks_like_identifier = (
            unique_values > 20
            and unique_ratio >= 0.5
        )

    if unique_values < 2:
        is_trainable = False
        blocking_reason = (
            f"'{target_column}' holds the same value in every row, so "
            "there is nothing for a model to tell apart. Choose a "
            "column whose values differ between rows."
        )

    elif looks_like_identifier:
        is_trainable = False
        blocking_reason = (
            f"'{target_column}' has {unique_values} distinct values "
            f"across {len(non_missing_value)} rows "
            f"({unique_ratio:.0%} of rows are unique), so it behaves "
            "like an identifier rather than something to predict. "
            "Choose a column with a smaller number of repeated "
            "categories, or a numerical measurement."
        )

    elif singleton_classes:
        is_trainable = False
        blocking_reason = (
            f"{singleton_classes} of the {unique_values} categories in "
            f"'{target_column}' appear in only one row. A model needs "
            "at least two rows per category so each one can appear in "
            "both the training and the testing set. Group the rare "
            "categories together or pick a different column."
        )

    if not is_trainable:
        warnings.append(blocking_reason)

    sample_values=(
        non_missing_value.drop_duplicates().head().tolist()

    )

    cleaned_sample_values=[]

    for value in sample_values:
        if hasattr(value,"item"):
            value=value.item()

        cleaned_sample_values.append(value)

    return {
        "success": True,
        "target_column": target_column,
        "data_type": str(target.dtype),
        "total_rows": total_rows,
        "missing_values": missing_values,
        "missing_percentage": round(
    (
        missing_values / total_rows * 100
        if total_rows > 0
        else 0
    ),
    2),
        "unique_values": unique_values,
        "unique_ratio": round(
            unique_ratio,
            4
        ),
        "sample_values": cleaned_sample_values,
        "recommended_task": reccomended_task,
        "recommendation_reason": reason,
        "is_trainable": is_trainable,
        "blocking_reason": blocking_reason,
        "single_row_categories": singleton_classes,
        "looks_like_identifier": bool(looks_like_identifier),
        "warnings": warnings
    }
@tool
@model_safe
def analyse_target_for_ml(
    name: str,
    target_column: str
) -> dict[str, Any]:
    """
    Analyse a selected target column and recommend whether the
    machine-learning task should be classification or regression.

    Use this before training a machine-learning model.

    Args:
        name:
            Name of the CSV dataset.

        target_column:
            Column the user wants to predict.

    Returns:
        Target statistics, task recommendation, reasoning and warnings.
    """

    return analyse_tool(
        name=name,
        target_column=target_column
    )


def prepare_ml_data(name:str, 
                    target_column:str,
                    max_categorical_levels: int = 50,
                    identifier_unique_ratio: float = 0.5)->dict[str,Any]:
    """
    Prepare features and target data for machine-learning training.

    This function:
    - loads the selected dataset,
    - removes rows with missing target values,
    - separates features from the target,
    - identifies numerical and categorical columns,
    - removes constant feature columns,
    - converts date-like text columns into a single numerical feature,
    - removes identifier-like high-cardinality categorical columns,
    - reports unsupported column types.

    Args:
        name:
            Name of the dataset stored in DATAFRAME_CACHE.

        target_column:
            Column that the model should predict.

        max_categorical_levels:
            Categorical columns with more distinct values than this are
            removed rather than one-hot encoded.

        identifier_unique_ratio:
            Categorical columns whose distinct values make up at least
            this proportion of the rows are treated as identifiers and
            removed.

    Returns:
        A dictionary containing X, y, detected feature groups,
        removed columns and preparation metadata.
    """

    try:
        df = get_or_load_dataframe(name).copy()

    except (FileNotFoundError, ValueError) as error:
        return {
            "success": False,
            "error": str(error)
        }

    if target_column not in df.columns:
        return {
            "success": False,
            "error": (
                f"Target column '{target_column}' was not found "
                f"in dataset '{name}'."
            )
        }
    original_row=len(df)

    df= df.dropna(subset=[target_column]).copy()

    removed_target_rows=original_row-len(df)

    if df.empty:
        return{
            "success":False,
            "error":(
                "No training rows remain after removing rows with missing target values"
            )
        }
    x= df.drop(columns=[target_column]).copy()
    y= df[target_column].copy()

    if x.shape[1]==0:
        return{
            "success":False,
            "error":
            (
                "The dataset contains no feature columns after removing the target column"
            )
        }
    constant_columns=[
        column
        for column in x.columns
        if x[column].nunique(dropna=False)<=1

    ]

    if constant_columns:
        x= x.drop(
            columns=constant_columns
        )
    if x.shape[1]==0:
        return{
        "success": False,
        "error": (
            "All available feature columns are constant "
                "and cannot be used for training.")
        }
    numeric_column=x.select_dtypes(
        include="number"
    ).columns.tolist()

    categorial=x.select_dtypes(
        include=["category","bool","object","str"]
    ).columns.tolist()

    supported_columns=set(numeric_column+categorial)

    unsupported_column=[
        column
        for column in x.columns
        if column not in supported_columns
    ]

    # Date-like text columns would otherwise be one-hot encoded into one
    # column per timestamp. Converting them to elapsed seconds keeps the
    # ordering information in a single numerical feature. This is a
    # row-wise conversion with no fitted statistics, so it does not leak
    # information across the train/test split.
    datetime_converted_columns = [
        column
        for column in data_time_column(x)
        if column in categorial
    ]

    for column in datetime_converted_columns:
        parsed_values = pd.to_datetime(
            x[column],
            errors="coerce"
        )

        x[column] = (
            parsed_values
            - pd.Timestamp("1970-01-01")
        ).dt.total_seconds()

    if datetime_converted_columns:
        categorial = [
            column
            for column in categorial
            if column not in datetime_converted_columns
        ]

        numeric_column = (
            numeric_column
            + datetime_converted_columns
        )

    # Identifier-like columns (customer IDs, free text, order numbers)
    # explode the one-hot matrix and let models memorise rows, so they are
    # removed instead of encoded.
    row_count = len(x)

    high_cardinality_columns = []

    for column in categorial:
        level_count = int(
            x[column].nunique(dropna=True)
        )

        unique_ratio = (
            level_count / row_count
            if row_count
            else 0.0
        )

        if (
            level_count > max_categorical_levels
            or (
                level_count > 20
                and unique_ratio >= identifier_unique_ratio
            )
        ):
            high_cardinality_columns.append(column)

    if high_cardinality_columns:
        categorial = [
            column
            for column in categorial
            if column not in high_cardinality_columns
        ]

    model_column=(numeric_column+categorial)

    x=x[model_column].copy()

    if x.shape[1]==0:

        if high_cardinality_columns:
            return {
                "success": False,
                "error": (
                    "No usable feature columns remain. Every remaining "
                    "column looks like an identifier or free text with "
                    "too many distinct values to encode: "
                    f"{high_cardinality_columns}."
                )
            }

        return {
            "success":False,
            "error":
            ("No supported numerical or categorial feature column were found")
        }

    return {
        "success": True,
        "X": x,
        "y": y,
        "original_rows": int(original_row),
        "training_rows": int(len(df)),
        "removed_target_rows": int(removed_target_rows),
        "numeric_columns": numeric_column,
        "categorical_columns": categorial,
        "constant_columns_removed": constant_columns,
        "unsupported_columns_removed": unsupported_column,
        "datetime_columns_converted": datetime_converted_columns,
        "high_cardinality_columns_removed": high_cardinality_columns
    }

def build_ml_preprocessor(numerical_column: list[str],categorial_column:list[str])-> ColumnTransformer:
    """
    Build a preprocessing pipeline for numerical and categorical features.

    Args:
        numeric_columns:
            Names of numerical feature columns.

        categorical_columns:
            Names of categorical feature columns.

    Returns:
        A ColumnTransformer that prepares the features for modelling.
    """
    transformers=[]

    if numerical_column:
        numerical_pipeline=Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )
        transformers.append(
            (
                "numeric",
                numerical_pipeline,
                numerical_column
            )
        )
    if categorial_column:
            categorial_pipeline=Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            transformers.append((
                "categorical",
                categorial_pipeline,
                categorial_column
            ))

    if not transformers:
        raise ValueError(
                    "No categorial nor numerical columns were found during preprocessing"
        )
    preprocessor=ColumnTransformer(
        transformers=transformers,
        remainder="drop"
            )
    return preprocessor

def cross_validation_summary(
    task: str,
    fold_plan: ml_tuning.FoldPlan,
    requested_folds: int,
    tune: bool
) -> dict[str, Any]:
    """
    Describe how the models were compared, for the page and the model card.

    Args:
        task:
            classification or regression.

        fold_plan:
            The folds actually used.

        requested_folds:
            The folds asked for.

        tune:
            Whether a search was requested.

    Returns:
        Fold counts, the score used, whether a search ran, and a note
        whenever the plan differed from the request.
    """

    tuned = bool(tune and fold_plan.folds)

    note = fold_plan.note

    if tune and not fold_plan.folds:
        note = (
            f"{note} Settings were not tuned, because the search needs "
            "cross-validation."
        ).strip()

    return {
        "requested_folds": int(requested_folds),
        "folds": int(fold_plan.folds),
        "scoring": ml_tuning.SCORE_LABEL[task],
        "tuned": tuned,
        "search_iterations": (
            ml_tuning.SEARCH_ITERATIONS
            if tuned
            else 0
        ),
        "note": note
    }


def train_classification_model(
    name: str,
    target_column: str,
    test_size: float = 0.2,
    cv_folds: int = ml_tuning.DEFAULT_FOLDS,
    tune: bool = False
) -> dict[str, Any]:
    """
    Train and compare several classification models.

    The function:
    - prepares the dataset,
    - creates a stratified train/test split,
    - applies preprocessing separately within each model pipeline,
    - cross-validates, and optionally tunes, each classifier on the
      training rows,
    - scores each classifier once on the held-out test rows,
    - ranks the models on cross-validated macro F1, or on test macro
      F1 when cross-validation cannot run.

    Args:
        name:
            Name of the dataset stored in DATAFRAME_CACHE.

        target_column:
            Column that the models should predict.

        test_size:
            Proportion of rows reserved for testing.

        cv_folds:
            Cross-validation folds on the training rows. Below 2 turns
            cross-validation, and with it tuning, off.

        tune:
            Search each model's settings with randomised search.

    Returns:
        A dictionary containing the leaderboard, confusion matrices,
        trained pipelines, split information and the best model name.
        """
    if test_size <= 0 or test_size >= 0.5:
        return {
            "success": False,
            "error": "test_size must be greater than 0 and below 0.5."
        }

    prepared_data = prepare_ml_data(
        name=name,
        target_column=target_column
    )

    if not prepared_data.get("success"):
        return prepared_data

    X = prepared_data["X"]
    y = prepared_data["y"]

    number_of_classes = int(y.nunique())

    if number_of_classes < 2:
        return {
            "success": False,
            "error": (
                "Classification requires at least two "
                "different target classes."
            )
        }

    class_counts = y.value_counts()

    if int(class_counts.min()) < 2:
        single_row_classes = int((class_counts < 2).sum())

        return {
            "success": False,
            "error": (
                f"{single_row_classes} of the {number_of_classes} "
                f"categories in '{target_column}' appear in only one "
                f"row, out of {len(y)} rows in total. Every category "
                "needs at least two rows so it can appear in both the "
                "training and the testing set. This usually means the "
                "column is an identifier rather than a label. Group the "
                "rare categories together, or choose a column with "
                "fewer repeated values."
            )
        }

    requested_test_rows = int(
        np.ceil(len(y) * test_size)
    )

    # A stratified split needs at least one test row per class, so the
    # requested size is raised when it is too small. The adjustment is
    # reported rather than applied silently.
    test_rows = max(
        requested_test_rows,
        number_of_classes
    )

    test_size_adjusted = test_rows != requested_test_rows

    test_size_adjustment_reason = (
        (
            f"The requested testing size of {test_size:.0%} would have "
            f"reserved only {requested_test_rows} row(s), which cannot "
            f"hold all {number_of_classes} classes. It was raised to "
            f"{test_rows} rows so the split can stay stratified."
        )
        if test_size_adjusted
        else ""
    )

    training_rows = len(y) - test_rows

    if training_rows < number_of_classes:
        return {
            "success": False,
            "error": (
                "The dataset is too small to place every class "
                "in both the training and testing sets."
            )
        }
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_rows,
        random_state=42,
        stratify=y
    )

    base_preprocessor = build_ml_preprocessor(
        numerical_column=prepared_data["numeric_columns"],
        categorial_column=prepared_data["categorical_columns"]
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    class_labels = y.drop_duplicates().tolist()

    fold_plan = ml_tuning.plan_folds(
        y_train,
        "classification",
        cv_folds
    )

    leaderboard = []
    confusion_matrices = {}
    trained_models = {}
    predictions_by_model={}
    tuned_parameters = {}

    for model_name, estimator in candidate_models.items():

        # Each model receives its own fresh preprocessing pipeline.
        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(base_preprocessor)
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        start_time = time.perf_counter()

        # Cross-validation and any search see the training rows only.
        selection = ml_tuning.fit_and_score(
            model_pipeline,
            X_train,
            y_train,
            task="classification",
            model_name=model_name,
            folds=fold_plan.folds,
            tune=tune
        )

        model_pipeline = selection.pipeline

        if selection.parameters:
            tuned_parameters[model_name] = selection.parameters

        training_seconds = (
            time.perf_counter() - start_time
        )

        predictions = model_pipeline.predict(
            X_test
        )

        model_metrics = {
            "model": model_name,

            "accuracy": round(
                float(
                    accuracy_score(
                        y_test,
                        predictions
                    )
                ),
                4
            ),

            "balanced_accuracy": round(
                float(
                    balanced_accuracy_score(
                        y_test,
                        predictions
                    )
                ),
                4
            ),

            "precision_macro": round(
                float(
                    precision_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0
                    )
                ),
                4
            ),

            "recall_macro": round(
                float(
                    recall_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0
                    )
                ),
                4
            ),

            "f1_macro": round(
                float(
                    f1_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0
                    )
                ),
                4
            ),

            "cv_f1_macro": ml_tuning.rounded(selection.cv_mean),

            "cv_f1_macro_std": ml_tuning.rounded(selection.cv_std),

            "training_seconds": round(
                float(training_seconds),
                4
            )
        }

        leaderboard.append(
            model_metrics
        )

        confusion_matrices[model_name] = (
            confusion_matrix(
                y_test,
                predictions,
                labels=class_labels
            ).tolist()
        )

        trained_models[model_name] = model_pipeline
        predictions_by_model[model_name] = (
        predictions.tolist())

    # Rank on the cross-validated score when there is one, so the test
    # rows play no part in choosing the model. Without folds, the test
    # score is the only evidence left.
    ranking_key = (
        "cv_f1_macro"
        if fold_plan.folds
        else "f1_macro"
    )

    leaderboard = sorted(
        leaderboard,
        key=lambda result: result[ranking_key],
        reverse=True
    )

    best_model_name = leaderboard[0]["model"]

    cleaned_class_labels = []

    for label in class_labels:
        if hasattr(label, "item"):
            label = label.item()

        cleaned_class_labels.append(label)

    return {
        "success": True,
        "task": "classification",
        "target_column": target_column,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "requested_test_size": round(float(test_size), 4),
        "actual_test_size": round(
            len(X_test) / len(y),
            4
        ),
        "test_size_adjusted": bool(test_size_adjusted),
        "test_size_adjustment_reason": (
            test_size_adjustment_reason
        ),
        "number_of_classes": number_of_classes,
        "class_labels": cleaned_class_labels,
        "class_distribution": {
            str(label): int(count)
            for label, count in class_counts.items()
        },
        "numeric_columns": prepared_data["numeric_columns"],
        "categorical_columns": prepared_data[
            "categorical_columns"
        ],
        "constant_columns_removed": prepared_data[
            "constant_columns_removed"
        ],
        "unsupported_columns_removed": prepared_data[
            "unsupported_columns_removed"
        ],
        "datetime_columns_converted": prepared_data[
            "datetime_columns_converted"
        ],
        "high_cardinality_columns_removed": prepared_data[
            "high_cardinality_columns_removed"
        ],
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "ranked_by": ranking_key,
        "cross_validation": cross_validation_summary(
            "classification",
            fold_plan,
            cv_folds,
            tune
        ),
        "tuned_parameters": tuned_parameters,
        "training_fingerprint": ml_export.training_fingerprint(
            X_train,
            y_train
        ),
        "confusion_matrices": confusion_matrices,
        "trained_models": trained_models,
        "predictions_by_model": predictions_by_model,
        "X_test": X_test,
        "y_test": y_test
    }
@tool
@model_safe
def train_classification_model_tool(
    name: str,
    target_column: str,
    test_size: float = 0.2
) -> dict[str, Any]:
    """
    Train and compare classification models for a selected target.

    Use this only when:
    - the user has explicitly selected a target column, and
    - the target represents a classification problem, and
    - the user has requested model training.

    The tool compares Logistic Regression, Decision Tree and
    Random Forest using a shared preprocessing pipeline.

    Args:
        name:
            Name of the CSV dataset.

        target_column:
            Column the models should predict.

        test_size:
            Proportion of rows reserved for model testing.

    Returns:
        Model leaderboard, best model, metrics, class distribution,
        confusion matrices and data-preparation information.
    """

    result = train_classification_model(
        name=name,
        target_column=target_column,
        test_size=test_size
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get(
                "error",
                "Classification training failed."
            )
        }

    return {
        "success": True,
        "task": result["task"],
        "target_column": result["target_column"],
        "training_rows": result["training_rows"],
        "testing_rows": result["testing_rows"],
        "requested_test_size": result["requested_test_size"],
        "actual_test_size": result["actual_test_size"],
        "test_size_adjusted": result["test_size_adjusted"],
        "test_size_adjustment_reason": result[
            "test_size_adjustment_reason"
        ],
        "number_of_classes": result["number_of_classes"],
        "class_labels": result["class_labels"],
        "class_distribution": result["class_distribution"],
        "numeric_columns": result["numeric_columns"],
        "categorical_columns": result["categorical_columns"],
        "constant_columns_removed": result[
            "constant_columns_removed"
        ],
        "unsupported_columns_removed": result[
            "unsupported_columns_removed"
        ],
        "datetime_columns_converted": result[
            "datetime_columns_converted"
        ],
        "high_cardinality_columns_removed": result[
            "high_cardinality_columns_removed"
        ],
        "leaderboard": result["leaderboard"],
        "best_model_name": result["best_model_name"],
        "ranked_by": result["ranked_by"],
        "cross_validation": result["cross_validation"],
        "confusion_matrices": result["confusion_matrices"]
    }

def train_regression_model(
    name: str,
    target_column: str,
    test_size: float = 0.2,
    cv_folds: int = ml_tuning.DEFAULT_FOLDS,
    tune: bool = False
) -> dict[str, Any]:
    """
    Train and compare several regression models.

    This function:
    - prepares numerical and categorical features,
    - removes rows with missing target values,
    - creates a train/test split,
    - cross-validates, and optionally tunes, each model on the training
      rows,
    - evaluates each model once on the test rows using MAE, RMSE and R²,
    - ranks the models on cross-validated RMSE, or on test RMSE when
      cross-validation cannot run.

    Args:
        name:
            Name or path of the CSV dataset.

        target_column:
            Numerical column the models should predict.

        test_size:
            Fraction of rows reserved for testing.

        cv_folds:
            Cross-validation folds on the training rows. Below 2 turns
            cross-validation, and with it tuning, off.

        tune:
            Search each model's settings with randomised search.

    Returns:
        Regression leaderboard, best model, predictions,
        fitted pipelines and preparation information.
    """
    if test_size <= 0 or test_size >= 0.5:
        return {
            "success": False,
            "error": "test_size must be greater than 0 and below 0.5."
        }

    prepared_data = prepare_ml_data(
        name=name,
        target_column=target_column
    )

    if not prepared_data.get("success"):
        return prepared_data

    X = prepared_data["X"]
    y = prepared_data["y"]

    if not pd.api.types.is_numeric_dtype(y):
        return {
            "success": False,
            "error": (
                f"Regression requires a numerical target, but "
                f"'{target_column}' has data type '{y.dtype}'."
            )
        }

    if y.nunique(dropna=True) < 2:
        return {
            "success": False,
            "error": (
                "Regression requires at least two different "
                "target values."
            )
        }

    requested_test_rows = int(
        np.ceil(len(y) * test_size)
    )

    # At least two test rows are needed for the error metrics to mean
    # anything, so a very small request is raised and reported.
    test_rows = max(
        2,
        requested_test_rows
    )

    test_size_adjusted = test_rows != requested_test_rows

    test_size_adjustment_reason = (
        (
            f"The requested testing size of {test_size:.0%} would have "
            f"reserved only {requested_test_rows} row(s). It was raised "
            f"to {test_rows} rows so the error metrics can be "
            f"calculated."
        )
        if test_size_adjusted
        else ""
    )

    training_rows = len(y) - test_rows

    if training_rows < 2:
        return {
            "success": False,
            "error": (
                "The dataset do not contain enough row for a valid training and testing sample."
            )
        }
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_rows,
        random_state=42,
        
    )

    base_preprocessor = build_ml_preprocessor(
        numerical_column=prepared_data["numeric_columns"],
        categorial_column=prepared_data["categorical_columns"]
    )

    candidate_models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    

    fold_plan = ml_tuning.plan_folds(
        y_train,
        "regression",
        cv_folds
    )

    leaderboard = []
    predictions_by_model = {}
    trained_models = {}
    tuned_parameters = {}

    for model_name, estimator in candidate_models.items():

        # Each model receives its own fresh preprocessing pipeline.
        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(base_preprocessor)
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        start_time = time.perf_counter()

        # Cross-validation and any search see the training rows only.
        selection = ml_tuning.fit_and_score(
            model_pipeline,
            X_train,
            y_train,
            task="regression",
            model_name=model_name,
            folds=fold_plan.folds,
            tune=tune
        )

        model_pipeline = selection.pipeline

        if selection.parameters:
            tuned_parameters[model_name] = selection.parameters

        training_seconds = (
            time.perf_counter() - start_time
        )

        predictions = model_pipeline.predict(
            X_test
        )
        mae=mean_absolute_error(y_test,predictions)

        mse = mean_squared_error(
            y_test,
            predictions)

        rmse=np.sqrt(mse)

        r2=r2_score(y_test,predictions)

        

        leaderboard.append({
            "model":model_name,
            "mae":round(float(mae),4),
            "rmse":round(float(rmse),4),
            "r2_score":round(float(r2),4),
            "cv_rmse": ml_tuning.rounded(selection.cv_mean),
            "cv_rmse_std": ml_tuning.rounded(selection.cv_std),
            "training_seconds":round(float(training_seconds),4)})

        

        trained_models[model_name] = model_pipeline
        predictions_by_model[model_name]=predictions.tolist()
        

    # Rank on the cross-validated error when there is one, so the test
    # rows play no part in choosing the model.
    ranking_key = (
        "cv_rmse"
        if fold_plan.folds
        else "rmse"
    )

    leaderboard = sorted(
        leaderboard,
        key=lambda result: result[ranking_key],
    )

    best_model_name = leaderboard[0]["model"]

    

    return {
        "success": True,
        "task": "regression",
        "target_column": target_column,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "requested_test_size": round(float(test_size), 4),
        "actual_test_size": round(
            len(X_test) / len(y),
            4
        ),
        "test_size_adjusted": bool(test_size_adjusted),
        "test_size_adjustment_reason": (
            test_size_adjustment_reason
        ),
        "numeric_columns": prepared_data[
            "numeric_columns"
        ],
        "categorical_columns": prepared_data[
            "categorical_columns"
        ],
        "constant_columns_removed": prepared_data[
            "constant_columns_removed"
        ],
        "unsupported_columns_removed": prepared_data[
            "unsupported_columns_removed"
        ],
        "datetime_columns_converted": prepared_data[
            "datetime_columns_converted"
        ],
        "high_cardinality_columns_removed": prepared_data[
            "high_cardinality_columns_removed"
        ],
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "ranked_by": ranking_key,
        "cross_validation": cross_validation_summary(
            "regression",
            fold_plan,
            cv_folds,
            tune
        ),
        "tuned_parameters": tuned_parameters,
        "training_fingerprint": ml_export.training_fingerprint(
            X_train,
            y_train
        ),
        "trained_models": trained_models,
        "predictions_by_model": predictions_by_model,
        "X_test": X_test,
        "y_test": y_test
    }
@tool
@model_safe
def train_regression_model_tool(
    name: str,
    target_column: str,
    test_size: float = 0.2
) -> dict[str, Any]:
    """
    Train and compare regression models for a numerical target.

    Use this tool only when:
    - the user explicitly provides a target column,
    - target analysis recommends regression,
    - the user explicitly asks to train or compare models.

    The tool compares Linear Regression, Decision Tree and
    Random Forest using consistent preprocessing.

    Args:
        name:
            Name or path of the CSV dataset.

        target_column:
            Numerical column the models should predict.

        test_size:
            Fraction of rows reserved for testing.

    Returns:
        Regression leaderboard, best model, evaluation metrics
        and data-preparation details.
    """

    target_report = analyse_tool(
        name=name,
        target_column=target_column
    )

    if not target_report.get("success"):
        return target_report

    recommended_task = (
        target_report["recommended_task"]
        .strip()
        .lower()
    )

    if recommended_task != "regression":
        return {
            "success": False,
            "error": (
                f"The selected target appears to be a "
                f"{recommended_task} target, not a regression target."
            ),
            "recommended_task": recommended_task,
            "recommendation_reason": target_report[
                "recommendation_reason"
            ]
        }

    result = train_regression_model(
        name=name,
        target_column=target_column,
        test_size=test_size
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get(
                "error",
                "Regression model training failed."
            )
        }

    return {
        "success": True,
        "task": result["task"],
        "target_column": result["target_column"],
        "training_rows": result["training_rows"],
        "testing_rows": result["testing_rows"],
        "requested_test_size": result["requested_test_size"],
        "actual_test_size": result["actual_test_size"],
        "test_size_adjusted": result["test_size_adjusted"],
        "test_size_adjustment_reason": result[
            "test_size_adjustment_reason"
        ],
        "numeric_columns": result["numeric_columns"],
        "categorical_columns": result[
            "categorical_columns"
        ],
        "constant_columns_removed": result[
            "constant_columns_removed"
        ],
        "unsupported_columns_removed": result[
            "unsupported_columns_removed"
        ],
        "datetime_columns_converted": result[
            "datetime_columns_converted"
        ],
        "high_cardinality_columns_removed": result[
            "high_cardinality_columns_removed"
        ],
        "leaderboard": result["leaderboard"],
        "best_model_name": result["best_model_name"],
        "ranked_by": result["ranked_by"],
        "cross_validation": result["cross_validation"]
    }

def calculate_feature_importance(
    trained_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    n_repeats: int = 10
) -> dict[str, Any]:
    """
    Calculate permutation feature importance for a fitted model pipeline.

    Args:
        trained_model:
            Fitted Scikit-learn pipeline.

        X_test:
            Original unprocessed test features.

        y_test:
            Actual test target values.

        task:
            Either classification or regression.

        n_repeats:
            Number of times each feature is shuffled.

    Returns:
        Ranked feature-importance results.
    """

    normalised_task = task.strip().lower()

    if normalised_task not in {
        "classification",
        "regression"
    }:
        return {
            "success": False,
            "error": (
                "Task must be either classification "
                "or regression."
            )
        }

    if X_test.empty:
        return {
            "success": False,
            "error": "The test dataset contains no rows."
        }

    if len(X_test) != len(y_test):
        return {
            "success": False,
            "error": (
                "X_test and y_test contain different "
                "numbers of rows."
            )
        }

    if n_repeats < 1 or n_repeats > 50:
        return {
            "success": False,
            "error": (
                "n_repeats must be between 1 and 50."
            )
        }

    scoring = (
        "f1_macro"
        if normalised_task == "classification"
        else "neg_root_mean_squared_error"
    )

    try:
        importance_result = permutation_importance(
            estimator=trained_model,
            X=X_test,
            y=y_test,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

    except Exception as error:
        return {
            "success": False,
            "error": (
                f"Feature importance could not be calculated: "
                f"{error}"
            )
        }

    feature_results = []

    for (
        feature_name,
        importance_mean,
        importance_std
    ) in zip(
        X_test.columns,
        importance_result.importances_mean,
        importance_result.importances_std,
        strict=True
    ):
        feature_results.append({
            "feature": str(feature_name),
            "importance_mean": round(
                float(importance_mean),
                6
            ),
            "importance_std": round(
                float(importance_std),
                6
            )
        })

    feature_results = sorted(
        feature_results,
        key=lambda result: result["importance_mean"],
        reverse=True
    )

    return {
        "success": True,
        "task": normalised_task,
        "scoring": scoring,
        "number_of_features": len(feature_results),
        "feature_importance": feature_results
    }

