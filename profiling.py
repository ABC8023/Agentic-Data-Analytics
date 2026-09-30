"""
Dataset profiling for the agent and the interface: quality, schema,
statistics and date detection.
"""

import glob
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

import dataset_store
from dataset_store import get_or_load_dataframe
from privacy import model_safe
from schema import column_schema, describe_column


#identifying the data tool
@tool
@model_safe
def list_csv_file()->list[str] | None:
     """List all CSV file names in the permitted data directory.

    Returns:
        A list containing CSV file names.
        If no CSV files are found, returns None.
    """
     csv_files= glob.glob(os.path.join(dataset_store.DATASET_ROOT, "*.csv"))
     if not csv_files:
          return None
     return [os.path.basename(file) for file in csv_files]




@tool
@model_safe
def get_data_quality_report(name: str)-> dict[str,Any]:
    """
    Inspect a CSV dataset and report common data-quality problems.

    Use this tool before statistical analysis or machine-learning training.

    Args:
        file_name: Name or path of the CSV dataset.

    Returns:
        A structured report containing dataset dimensions, missing values,
        duplicate rows, constant columns and column data types.
    """

    try:
        df= get_or_load_dataframe(name)
    except(FileNotFoundError ,ValueError) as error:
        return{
            "success":False,
            "error":str(error)

        }
    missing_count=df.isnull().sum()
    missing_percent=(
        df.isnull().mean() * 100 ).round(2)
    missing_value={}

    for column in df.columns:
        if missing_count[column]>0:
            missing_value[column]={
                "count": int(missing_count[column]),
                "percentage": float(missing_percent[column])
            }

    constant_coulmn=[
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <=1
    ]

    duplicate_row=int(df.duplicated().sum())
    return {
        "success": True,
        "file_name": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": missing_value,
        "total_missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicate_rows": duplicate_row,
        "constant_columns": constant_coulmn
        }
    

@tool
@model_safe
def preload_dataset(path: list[str])-> dict[str,Any]:
     """
    Loads CSV files into a global cache if not already loaded.
    
    This function helps to efficiently manage datasets by loading them once
    and storing them in memory for future use. Without caching, you would
    waste tokens describing dataset contents repeatedly in agent responses.
    
    Args:
        paths: A list of file paths to CSV files.

    Returns:
        A message summarizing which datasets were loaded or already cached.
    """
     loaded = []
     cached = []
     failed = []

     for dataset_path in path:
        already_cached = dataset_path in dataset_store.DATAFRAME_CACHE

        try:
            get_or_load_dataframe(dataset_path)

        except (FileNotFoundError, ValueError) as error:
            failed.append({
                "file_name": dataset_path,
                "error": str(error)
            })
            continue

        if already_cached:
            cached.append(dataset_path)
        else:
            loaded.append(dataset_path)

     return {
        "success": len(failed) == 0,
        "loaded": loaded,
        "already_cached": cached,
        "failed": failed
    }

@tool
@model_safe
def get_dataset_summary(dataset_paths:list[str])-> list[dict[str,Any]]:
     """
    Analyze multiple CSV files and return metadata summaries for each.

    Args:
        dataset_paths (List[str]): 
            A list of file paths to CSV datasets.

    Returns:
        List[Dict[str, Any]]: 
            A list of summaries, one per dataset, each containing:
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types (as strings).
    """
     
     summaries=[]

     for dataset_path in dataset_paths:
        try:
            df = get_or_load_dataframe(dataset_path)

        except (FileNotFoundError, ValueError) as error:
            summaries.append({
                "success": False,
                "file_name": dataset_path,
                "error": str(error)
            })
            continue

        summary = {
            "success": True,
            "file_name": dataset_path,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        summaries.append(summary)

     return summaries

@tool
@model_safe
def get_column_schema(name: str) -> dict[str, Any]:
    """
    Describe the structure of a dataset without revealing any values.

    Use this instead of asking for example rows. It reports the columns,
    their kinds and data types, how many values are missing, how many are
    distinct, and which columns look like identifiers or are constant.

    Args:
        name:
            Name of the dataset.

    Returns:
        Row and column counts, the column names grouped by kind, and a
        per-column structural description.
    """

    try:
        df = get_or_load_dataframe(name)

    except (FileNotFoundError, ValueError) as error:
        return {
            "success": False,
            "error": str(error)
        }

    report = column_schema(df)
    report["success"] = True
    report["file_name"] = name

    return report


def local_dataset_preview(
    name: str,
    number_rows: int = 5
) -> dict[str, Any]:
    """
    Return the first few rows of a dataset, for local display only.

    This is deliberately not registered as an agent tool. It returns real
    cell values, so it must never be reachable from a model prompt. The
    interface renders row previews directly from the cached DataFrame.

    Args:
        name: Name of the dataset.
        number_rows: Number of rows to return, between 1 and 20.

    Returns:
        A structured preview containing column names and sample records.
    """
    if number_rows<1 or number_rows>20:
        return{
            "success":False,
            "error": " number of rows should be between 1 and 20"

        }
    try:
        df= get_or_load_dataframe(name)

    except (FileNotFoundError,ValueError) as error:
        return {
            "success": False,
            "error": str(error)
        }

    preview_df= df.head(number_rows).copy()

    preview_df = preview_df.astype(object).where(
        pd.notnull(preview_df),
        None
    )
    return {
        "success": True,
        "file_name": name,
        "requested_rows": number_rows,
        "returned_rows": int(len(preview_df)),
        "column_names": preview_df.columns.tolist(),
        "records": preview_df.to_dict(orient="records")
    }
     
def stat_overview(
     name: str,
     n_cate: int = 5,
     mask_identifier_labels: bool = False
) -> dict[str, Any]:
     """
    Generate numerical statistics and categorical frequency summaries.

    Args:
        name:
            Name of the dataset.

        n_cate:
            Number of common values to return for each categorical
            column. Must be between 1 and 10.

        mask_identifier_labels:
            When true, columns that look like identifiers report only how
            many distinct values they hold, never the values themselves.
            The agent tool sets this; the local interface does not, since
            nothing leaves the machine there.

    Returns:
        A structured dictionary containing numerical statistics,
        categorical frequencies, and detected column groups.
    """

     if n_cate <1 or n_cate>10:
         return{
             "success":False,
             "error": "should be bw 1 and 10"
         }

     try:
         df=get_or_load_dataframe(name)
     except(ValueError , FileNotFoundError) as error:
         return{
             "success" : False,
             "error": str(error)
         }
     numeric_column=df.select_dtypes(
         include=np.number
     ).columns.to_list()

     categorial_column= df.select_dtypes(
         include=["object","str","category","bool"]
     ).columns.to_list()

     stats={}

     if numeric_column:
         numeric_des=(
             df[numeric_column].describe().round(3)
         )

         stats= json.loads(
             numeric_des.to_json(orient="index")
         )
     categorial_stats={}
     withheld_columns=[]

     for column in categorial_column:

         # Enumerating the values of an identifier-like column would put
         # names, emails or reference codes into a model prompt. Report
         # the shape of such a column instead of its contents.
         if mask_identifier_labels:
             description = describe_column(df[column], len(df))

             if (
                 description["looks_like_identifier"]
                 or description["is_high_cardinality"]
                 or description["name_suggests_identity"]
             ):
                 categorial_stats[column] = {
                     "unique_values": description["distinct"],
                     "most_common_values": None,
                     "note": (
                         "Values withheld: this column looks like an "
                         "identifier or free text."
                     )
                 }

                 withheld_columns.append(column)
                 continue

         value_count=(
             df[column].value_counts(dropna=False).head(n_cate)
         )

         common_value=[]

         for value,count in value_count.items():
             display_value=(
                 "Missing" 
                 if pd.isna(value)
                 else str(value)
                 )

             percentage = (int(count)/ len(df) ) * 100

             common_value.append({
                 "value":display_value,
                 "percentage":round(percentage),
                 "count":int(count)
             })
         categorial_stats[column]={
                 "unique_values": int(
                     df[column].nunique(dropna=True)
                 ),
                 "most_common_values":common_value
             }

     analysed_column=set(categorial_column+numeric_column)

     other_column=[
                 column
                 for column in df.columns
                 if column not in analysed_column
             ]

     report = {
        "success": True,
        "file_name": name,
        "numeric_columns": numeric_column,
        "categorical_columns": categorial_column,
        "other_columns": other_column,
        "numeric_statistics": stats,
        "categorical_statistics": categorial_stats
    }

     if withheld_columns:
         report["identifier_columns_withheld"] = withheld_columns

     return report


@tool
@model_safe
def get_stat_overview(name: str, n_cate: int = 5) -> dict[str, Any]:
    """
    Summarise a dataset's numerical columns and category frequencies.

    Use this when the user asks about averages, medians, ranges,
    distributions, variation or common categorical values.

    Columns that look like identifiers or free text report only how many
    distinct values they hold, because listing their contents would be
    the same as sending rows.

    Args:
        name:
            Name of the dataset.

        n_cate:
            Number of common values per categorical column, 1 to 10.

    Returns:
        Numerical statistics, categorical frequencies and the detected
        column groups.
    """

    return stat_overview(
        name=name,
        n_cate=n_cate,
        mask_identifier_labels=True
    )

def data_time_column(df:pd.DataFrame, minimum_success_rate: float =0.8)->list[str]:
    """
    Detect columns that probably contain dates or timestamps.

    Args:
        df:
            The DataFrame to inspect.

        minimum_success_rate:
            Minimum proportion of non-missing values that must be
            successfully converted to dates.

    Returns:
        A list containing likely date or datetime column names.
    """

    datetime_column=[]

    for column in df.columns:
        series=df[column]


        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_column.append(column)
            continue
        if not (
            pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
            
        ):
            continue

        non_missing_series=series.dropna()

        if non_missing_series.empty:
            continue

        sample=non_missing_series.head(100)

        converted= pd.to_datetime(
            sample,
            errors="coerce"
        )
        conversion_rate=(converted.notna().mean())

        if conversion_rate >= minimum_success_rate:
            datetime_column.append(column)
    return datetime_column

