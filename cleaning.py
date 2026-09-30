"""
Dataset cleaning: duplicates, dropped columns and missing values.
"""

from typing import Any

from dataset_store import get_or_load_dataframe


def clean_dataset(
    name: str,
    remove_duplicates: bool = True,
    drop_columns: list[str] | None = None,
    numeric_missing_strategy: str = "median",
    categorical_missing_strategy: str = "most_frequent"
) -> dict[str, Any]:
    """
    Create a cleaned copy of a dataset.

    The original cached dataset is not modified by this function.

    Args:
        name:
            Name of the dataset stored in DATAFRAME_CACHE.

        remove_duplicates:
            Whether duplicate rows should be removed.

        drop_columns:
            Columns that should be removed.

        numeric_missing_strategy:
            One of:
            - leave
            - median
            - mean
            - zero
            - drop_rows

        categorical_missing_strategy:
            One of:
            - leave
            - most_frequent
            - missing_label
            - drop_rows

    Returns:
        Cleaned DataFrame and a cleaning summary.
    """

    try:
        dataframe = get_or_load_dataframe(
            name
        ).copy()

    except (FileNotFoundError, ValueError) as error:
        return {
            "success": False,
            "error": str(error)
        }

    if drop_columns is None:
        drop_columns = []

    allowed_numeric_strategies = {
        "leave",
        "median",
        "mean",
        "zero",
        "drop_rows"
    }

    allowed_categorical_strategies = {
        "leave",
        "most_frequent",
        "missing_label",
        "drop_rows"
    }

    if (
        numeric_missing_strategy
        not in allowed_numeric_strategies
    ):
        return {
            "success": False,
            "error": (
                "Invalid numerical missing-value strategy."
            )
        }

    if (
        categorical_missing_strategy
        not in allowed_categorical_strategies
    ):
        return {
            "success": False,
            "error": (
                "Invalid categorical missing-value strategy."
            )
        }

    invalid_drop_columns = [
        column
        for column in drop_columns
        if column not in dataframe.columns
    ]

    if invalid_drop_columns:
        return {
            "success": False,
            "error": (
                "The following columns were not found: "
                f"{invalid_drop_columns}"
            )
        }

    original_rows = int(len(dataframe))
    original_columns = int(dataframe.shape[1])

    original_missing_values = int(
        dataframe.isnull().sum().sum()
    )

    original_duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    warnings = []

    if drop_columns:
        dataframe = dataframe.drop(
            columns=drop_columns
        )

    if dataframe.shape[1] == 0:
        return {
            "success": False,
            "error": (
                "Cleaning removed every column from the dataset."
            )
        }

    duplicates_removed = 0

    if remove_duplicates:
        rows_before_duplicates = len(dataframe)

        dataframe = (
            dataframe
            .drop_duplicates()
            .reset_index(drop=True)
        )

        duplicates_removed = (
            rows_before_duplicates - len(dataframe)
        )

    numerical_columns = (
        dataframe
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = (
        dataframe
        .select_dtypes(
            include=[
                "object",
                "str",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    numerical_missing_columns = [
        column
        for column in numerical_columns
        if dataframe[column].isnull().any()
    ]

    categorical_missing_columns = [
        column
        for column in categorical_columns
        if dataframe[column].isnull().any()
    ]

    rows_removed_for_missing_values = 0

    if (
        numeric_missing_strategy == "drop_rows"
        and numerical_missing_columns
    ):
        rows_before = len(dataframe)

        dataframe = dataframe.dropna(
            subset=numerical_missing_columns
        )

        rows_removed_for_missing_values += (
            rows_before - len(dataframe)
        )

    elif numeric_missing_strategy != "leave":

        for column in numerical_missing_columns:

            non_missing_values = (
                dataframe[column].dropna()
            )

            if non_missing_values.empty:
                warnings.append(
                    f"Numerical column '{column}' contains "
                    "only missing values and could not be "
                    f"filled using {numeric_missing_strategy}."
                )
                continue

            if numeric_missing_strategy == "median":
                replacement_value = (
                    non_missing_values.median()
                )

            elif numeric_missing_strategy == "mean":
                replacement_value = (
                    non_missing_values.mean()
                )

            else:
                replacement_value = 0

            dataframe[column] = (
                dataframe[column]
                .fillna(replacement_value)
            )

    if (
        categorical_missing_strategy == "drop_rows"
        and categorical_missing_columns
    ):
        rows_before = len(dataframe)

        dataframe = dataframe.dropna(
            subset=categorical_missing_columns
        )

        rows_removed_for_missing_values += (
            rows_before - len(dataframe)
        )

    elif categorical_missing_strategy == "most_frequent":

        for column in categorical_missing_columns:

            most_common_values = (
                dataframe[column]
                .mode(dropna=True)
            )

            if most_common_values.empty:
                warnings.append(
                    f"Categorical column '{column}' contains "
                    "only missing values and has no most "
                    "frequent value."
                )
                continue

            replacement_value = (
                most_common_values.iloc[0]
            )

            dataframe[column] = (
                dataframe[column]
                .fillna(replacement_value)
            )

    elif (
        categorical_missing_strategy
        == "missing_label"
    ):
        for column in categorical_missing_columns:

            dataframe[column] = (
                dataframe[column]
                .astype("object")
                .fillna("Missing")
            )

    dataframe = dataframe.reset_index(
        drop=True
    )

    if dataframe.empty:
        return {
            "success": False,
            "error": (
                "No rows remain after applying the "
                "selected cleaning operations."
            )
        }

    final_missing_values = int(
        dataframe.isnull().sum().sum()
    )

    return {
        "success": True,
        "cleaned_dataframe": dataframe,
        "original_rows": original_rows,
        "cleaned_rows": int(len(dataframe)),
        "original_columns": original_columns,
        "cleaned_columns": int(
            dataframe.shape[1]
        ),
        "original_missing_values": (
            original_missing_values
        ),
        "remaining_missing_values": (
            final_missing_values
        ),
        "original_duplicate_rows": (
            original_duplicate_rows
        ),
        "duplicates_removed": int(
            duplicates_removed
        ),
        "rows_removed_for_missing_values": int(
            rows_removed_for_missing_values
        ),
        "columns_removed": drop_columns,
        "numeric_missing_strategy": (
            numeric_missing_strategy
        ),
        "categorical_missing_strategy": (
            categorical_missing_strategy
        ),
        "warnings": warnings
    }

