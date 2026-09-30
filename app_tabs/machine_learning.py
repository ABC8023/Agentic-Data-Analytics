"""
The machine learning tab.
"""


import pandas as pd
import plotly.express as px
import streamlit as st

from ml import (
    calculate_feature_importance,
    train_classification_model,
    train_regression_model,
)
from ui_cache import (
    cached_target_analysis,
)
from ui_theme import (
    CHART_ROLES,
    CHART_SEQUENTIAL,
    chart,
    lede,
    section,
    styled,
    table,
)


def render(*, data_version, dataframe, dataset_key):

    section("Machine-learning analysis")

    lede(
        "Select the target column that the model should predict."
    )

    target_column = st.selectbox(
        "Target column",
        options=dataframe.columns.tolist(),
        key="ml_target_column"
    )

    target_report = cached_target_analysis(
        dataset_key,
        data_version,
        target_column
    )

    if not target_report.get("success"):

        st.error(
            target_report.get(
                "error",
                "Unable to analyse the target column."
            )
    )

    else:
        target_metric1, target_metric2, target_metric3 = (
            st.columns(3)
        )

        with target_metric1:
            st.metric(
                "Unique target values",
                target_report["unique_values"]
            )

        with target_metric2:
            st.metric(
                "Missing target values",
                target_report["missing_values"]
            )

        with target_metric3:
            st.metric(
                "Recommended task",
                target_report["recommended_task"]
            )

        st.info(
            target_report["recommendation_reason"]
        )

        target_is_trainable = target_report.get(
            "is_trainable",
            True
        )

        if not target_is_trainable:
            st.error(
                target_report.get(
                    "blocking_reason",
                    "This column cannot be used as a target."
                )
            )
        else:
            for warning in target_report.get("warnings", []):
                st.warning(warning)

        recommended_task = (
            target_report["recommended_task"]
            .strip()
            .lower()
        )

        recommended_task_label = (
            "Classification"
            if recommended_task == "classification"
            else "Regression"
        )

        if st.session_state.last_ml_target != target_column:
            st.session_state.ml_task_choice = recommended_task_label
            st.session_state.last_ml_target = target_column

            st.session_state.ml_result = None
            st.session_state.ml_target = None
            st.session_state.ml_task = None
            st.session_state.feature_importance_result = None

        # Streamlit discards widget state for widgets that were not
        # rendered during a run. Any st.rerun() raised from an earlier
        # tab aborts the script before this radio is created, which
        # leaves the stored choice empty on the following run. Reseed it
        # so the radio always has a valid selection.
        if st.session_state.ml_task_choice not in (
            "Classification",
            "Regression"
        ):
            st.session_state.ml_task_choice = recommended_task_label

        selected_task = st.radio(
            "Machine-learning task",
            options=[
                "Classification",
                "Regression"
            ],
            key="ml_task_choice",
            horizontal=True,
            help=(
                "The recommended task is selected automatically, "
                "but you can override it."
            )
        )

        if selected_task.lower() != recommended_task:
            st.warning(
                "You selected a different task from the automatic "
                "recommendation. Training may fail if the target is "
                "not suitable for that task."
            )

        test_size = st.slider(
            "Testing-set percentage",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )

        test_size_fraction = test_size / 100

        if st.button(
            "Train and compare models",
            type="primary",
            key="train_ml_models",
            disabled=not target_is_trainable,
            help=(
                None
                if target_is_trainable
                else "Choose a different target column to enable "
                     "training."
            )
        ):

            with st.spinner(
                f"Training {selected_task.lower()} models..."
            ):

                if selected_task == "Classification":
                    ml_result = train_classification_model(
                        name=dataset_key,
                        target_column=target_column,
                        test_size=test_size_fraction
                    )

                else:
                    ml_result = train_regression_model(
                        name=dataset_key,
                        target_column=target_column,
                        test_size=test_size_fraction
                    )

                st.session_state.ml_result = ml_result
                st.session_state.ml_target = target_column
                st.session_state.ml_task = selected_task
                st.session_state.feature_importance_result = None

        ml_result = st.session_state.ml_result

        result_target = st.session_state.ml_target
        result_task = st.session_state.ml_task

        result_matches_current_selection = (
            ml_result is not None
            and result_target == target_column
            and result_task == selected_task
        )

        if (
            ml_result is not None
            and not result_matches_current_selection
        ):
            st.info(
                "The target or task has changed. Train the models "
                "again for the current selection."
            )

        if result_matches_current_selection:

            if not ml_result.get("success"):
                st.error(
                    ml_result.get(
                        "error",
                        "Model training failed."
                    )
                )

            elif selected_task == "Classification":
                st.success(
                    "Classification models trained successfully."
                )

                section("Model leaderboard")

                classification_leaderboard = pd.DataFrame(
                    ml_result["leaderboard"]
                )

                classification_leaderboard = (
                    classification_leaderboard.rename(
                        columns={
                            "model": "Model",
                            "accuracy": "Accuracy",
                            "balanced_accuracy": (
                                "Balanced accuracy"
                            ),
                            "precision_macro": "Macro precision",
                            "recall_macro": "Macro recall",
                            "f1_macro": "Macro F1",
                            "training_seconds": (
                                "Training time (seconds)"
                            )
                        }
                    )
                )

                table(
                    classification_leaderboard,
                    width="stretch",
                    hide_index=True
                )

                best_model_name = ml_result[
                    "best_model_name"
                ]

                best_result = next(
                    model_result
                    for model_result in ml_result["leaderboard"]
                    if model_result["model"] == best_model_name
                )

                section("Best model")

                best_column1, best_column2, best_column3 = (
                    st.columns(3)
                )

                with best_column1:
                    st.metric(
                        "Model",
                        best_model_name
                    )

                with best_column2:
                    st.metric(
                        "Accuracy",
                        f"{best_result['accuracy']:.2%}"
                    )

                with best_column3:
                    st.metric(
                        "Macro F1",
                        f"{best_result['f1_macro']:.2%}"
                    )


                section("Confusion matrix")

                class_labels = [
                    str(label)
                    for label in ml_result["class_labels"]
                ]

                matrix = ml_result[
                    "confusion_matrices"
                ][best_model_name]

                confusion_figure = px.imshow(
                    matrix,
                    x=class_labels,
                    y=class_labels,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=CHART_SEQUENTIAL,
                    labels={
                        "x": "Predicted class",
                        "y": "Actual class",
                        "color": "Number of rows"
                    },
                    title=(
                        f"Confusion matrix: {best_model_name}"
                    )
                )

                chart(
                    styled(confusion_figure),
                    width="stretch",
                    key="ml_classification_confusion_matrix"
                )

                section("Download classification results")

                classification_predictions = ml_result[
                    "predictions_by_model"
                ][best_model_name]

                classification_predictions_df = (
                    ml_result["X_test"]
                    .reset_index()
                    .rename(
                        columns={
                            "index": "original_row_index"
                        }
                    )
                )

                classification_predictions_df[
                    "actual_target"
                ] = (
                    ml_result["y_test"]
                    .reset_index(drop=True)
                )

                classification_predictions_df[
                    "predicted_target"
                ] = classification_predictions

                classification_predictions_df[
                    "correct_prediction"
                ] = (
                    classification_predictions_df["actual_target"]
                    == classification_predictions_df[
                        "predicted_target"
                    ]
                )

                classification_leaderboard_csv = (
                    classification_leaderboard
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                classification_predictions_csv = (
                    classification_predictions_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                download_column1, download_column2 = st.columns(2)

                with download_column1:
                    st.download_button(
                        label="Download model leaderboard",
                        data=classification_leaderboard_csv,
                        file_name=(
                            f"{target_column}_classification_"
                            f"leaderboard.csv"
                        ),
                        mime="text/csv",
                        key="download_classification_leaderboard"
                    )

                with download_column2:
                    st.download_button(
                        label="Download test predictions",
                        data=classification_predictions_csv,
                        file_name=(
                            f"{target_column}_classification_"
                            f"predictions.csv"
                        ),
                        mime="text/csv",
                        key="download_classification_predictions"
                    )

            else:
                st.success(
                    "Regression models trained successfully."
                )

                section("Model leaderboard")

                regression_leaderboard = pd.DataFrame(
                    ml_result["leaderboard"]
                )

                regression_leaderboard = (
                    regression_leaderboard.rename(
                        columns={
                            "model": "Model",
                            "mae": "MAE",
                            "rmse": "RMSE",
                            "r2_score": "R²",
                            "training_seconds": (
                                "Training time (seconds)"
                            )
                        }
                    )
                )

                table(
                    regression_leaderboard,
                    width="stretch",
                    hide_index=True
                )

                best_model_name = ml_result[
                    "best_model_name"
                ]

                best_result = next(
                    model_result
                    for model_result in ml_result["leaderboard"]
                    if model_result["model"] == best_model_name
                )

                section("Best model")

                best_column1, best_column2, best_column3, best_column4 = (
                    st.columns(4)
                )

                with best_column1:
                    st.metric(
                        "Model",
                        best_model_name
                    )

                with best_column2:
                    st.metric(
                        "MAE",
                        f"{best_result['mae']:.4f}"
                    )

                with best_column3:
                    st.metric(
                        "RMSE",
                        f"{best_result['rmse']:.4f}"
                    )

                with best_column4:
                    st.metric(
                        "R²",
                        f"{best_result['r2_score']:.4f}"
                    )

                actual_values = (
                    ml_result["y_test"]
                    .reset_index(drop=True)
                    .tolist()
                )

                predicted_values = ml_result[
                    "predictions_by_model"
                ][best_model_name]

                prediction_dataframe = pd.DataFrame({
                    "Actual": actual_values,
                    "Predicted": predicted_values
                })

                prediction_figure = px.scatter(
                    prediction_dataframe,
                    x="Actual",
                    y="Predicted",
                    title=(
                        f"Actual versus predicted: "
                        f"{best_model_name}"
                    )
                )

                minimum_value = min(
                    prediction_dataframe[
                        ["Actual", "Predicted"]
                    ].min()
                )

                maximum_value = max(
                    prediction_dataframe[
                        ["Actual", "Predicted"]
                    ].max()
                )

                prediction_figure.add_shape(
                    type="line",
                    x0=minimum_value,
                    y0=minimum_value,
                    x1=maximum_value,
                    y1=maximum_value,
                    line={
                        "dash": "dash",
                        "color": CHART_ROLES["reference"],
                        "width": 1.5
                    }
                )

                chart(
                    styled(prediction_figure),
                    width="stretch",
                    key="ml_regression_actual_predicted"
                )

                section("Download regression results")

                regression_predictions_df = (
                    ml_result["X_test"]
                    .reset_index()
                    .rename(
                        columns={
                            "index": "original_row_index"
                        }
                    )
                )

                regression_predictions_df[
                    "actual_target"
                ] = actual_values

                regression_predictions_df[
                    "predicted_target"
                ] = predicted_values

                regression_predictions_df[
                    "residual"
                ] = (
                    regression_predictions_df["actual_target"]
                    - regression_predictions_df["predicted_target"]
                )

                regression_predictions_df[
                    "absolute_error"
                ] = (
                    regression_predictions_df["residual"]
                    .abs()
                )

                regression_leaderboard_csv = (
                    regression_leaderboard
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                regression_predictions_csv = (
                    regression_predictions_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                download_column1, download_column2 = st.columns(2)

                with download_column1:
                    st.download_button(
                        label="Download model leaderboard",
                        data=regression_leaderboard_csv,
                        file_name=(
                            f"{target_column}_regression_"
                            f"leaderboard.csv"
                        ),
                        mime="text/csv",
                        key="download_regression_leaderboard"
                    )

                with download_column2:
                    st.download_button(
                        label="Download test predictions",
                        data=regression_predictions_csv,
                        file_name=(
                            f"{target_column}_regression_"
                            f"predictions.csv"
                        ),
                        mime="text/csv",
                        key="download_regression_predictions"
                    )

            if ml_result.get("success"):

                if ml_result.get("test_size_adjusted"):
                    st.warning(
                        "Testing size changed from "
                        f"**{ml_result['requested_test_size']:.0%}** "
                        "to "
                        f"**{ml_result['actual_test_size']:.0%}** "
                        f"({ml_result['testing_rows']} of "
                        f"{ml_result['training_rows'] + ml_result['testing_rows']} "
                        "rows). "
                        + ml_result.get(
                            "test_size_adjustment_reason",
                            ""
                        )
                    )
                else:
                    st.caption(
                        "Split: "
                        f"{ml_result['training_rows']} training rows, "
                        f"{ml_result['testing_rows']} testing rows "
                        f"({ml_result['actual_test_size']:.0%} held "
                        "back, as requested)."
                    )

                st.divider()

                section("Feature handling")

                dropped_identifier_columns = ml_result.get(
                    "high_cardinality_columns_removed",
                    []
                )

                converted_date_columns = ml_result.get(
                    "datetime_columns_converted",
                    []
                )

                if dropped_identifier_columns:
                    st.warning(
                        "Removed before training because they look "
                        "like identifiers or free text with too many "
                        "distinct values to encode: "
                        f"**{', '.join(dropped_identifier_columns)}**."
                    )

                if converted_date_columns:
                    st.info(
                        "Converted to a numerical time feature: "
                        f"**{', '.join(converted_date_columns)}**."
                    )

                if ml_result.get("constant_columns_removed"):
                    st.info(
                        "Removed because every row held the same "
                        "value: "
                        f"**{', '.join(ml_result['constant_columns_removed'])}**."
                    )

                if not (
                    dropped_identifier_columns
                    or converted_date_columns
                    or ml_result.get("constant_columns_removed")
                ):
                    st.caption(
                        "Every feature column was used as-is."
                    )

                st.divider()

                section("Feature importance")

                lede(
                    "Permutation importance measures how much model "
                    "performance changes when each feature is shuffled."
                )

                if st.button(
                    "Calculate feature importance",
                    key="calculate_feature_importance"
                ):
                    with st.spinner(
                        "Calculating feature importance..."
                    ):
                        best_model_name = ml_result[
                            "best_model_name"
                        ]

                        best_trained_model = ml_result[
                            "trained_models"
                        ][best_model_name]

                        importance_result = (
                            calculate_feature_importance(
                                trained_model=best_trained_model,
                                X_test=ml_result["X_test"],
                                y_test=ml_result["y_test"],
                                task=ml_result["task"],
                                n_repeats=10
                            )
                        )

                        st.session_state.feature_importance_result = (
                            importance_result
                        )

                importance_result = (
                    st.session_state.feature_importance_result
                )

                if importance_result is not None:

                    if not importance_result.get("success"):
                        st.error(
                            importance_result.get(
                                "error",
                                "Feature importance calculation failed."
                            )
                        )

                    else:
                        importance_dataframe = pd.DataFrame(
                            importance_result[
                                "feature_importance"
                            ]
                        )

                        table(
                            importance_dataframe.rename(
                                columns={
                                    "feature": "Feature",
                                    "importance_mean": (
                                        "Mean importance"
                                    ),
                                    "importance_std": (
                                        "Importance variation"
                                    )
                                }
                            ),
                            width="stretch",
                            hide_index=True
                        )

                        chart_dataframe = (
                            importance_dataframe
                            .sort_values(
                                "importance_mean",
                                ascending=True
                            )
                            .tail(20)
                        )

                        importance_figure = px.bar(
                            chart_dataframe,
                            x="importance_mean",
                            y="feature",
                            orientation="h",
                            error_x="importance_std",
                            title=(
                                f"Feature importance: "
                                f"{ml_result['best_model_name']}"
                            ),
                            labels={
                                "importance_mean": (
                                    "Decrease in model performance"
                                ),
                                "feature": "Feature"
                            }
                        )

                        chart(
                            styled(importance_figure),
                            width="stretch",
                            key="ml_feature_importance_chart"
                        )

                        st.caption(
                            "Larger positive values indicate that the "
                            "model relies more heavily on that feature. "
                            "Values near zero indicate little measured "
                            "effect. Negative values can indicate noise, "
                            "instability or correlated features."
                        )

