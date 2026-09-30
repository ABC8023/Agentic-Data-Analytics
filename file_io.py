"""
Validated loading of uploaded tabular files.

This module is the single place that decides whether an upload is
acceptable and how it becomes a DataFrame. It is deliberately free of
Streamlit imports so the rules can be unit tested without a browser
session or a running server.

Supported formats:
    .csv   comma, semicolon, tab or pipe delimited
    .xlsx  Excel workbook, one worksheet at a time
    .xlsm  macro-enabled Excel workbook, macros are never executed
"""

import csv
import io
import os
from typing import Any

import pandas as pd

# Upload ceiling. Large enough for real business extracts, small enough
# that a hosted deployment stays predictable.
MAX_FILE_SIZE_MB = 25

# Row ceiling. Anything longer is truncated and the truncation is
# reported rather than applied silently.
MAX_ROWS = 200_000

CSV_EXTENSIONS = {".csv"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS

# Display order for the uploader label. Alphabetical ordering would put
# the rarer macro format ahead of the common one.
EXTENSION_DISPLAY_ORDER = (".csv", ".xlsx", ".xlsm")

# Delimiters worth sniffing for. Ordered by how common they are in
# exports from business systems.
CANDIDATE_DELIMITERS = ",;\t|"

# Bytes read from the front of a CSV when guessing its delimiter.
SNIFF_SAMPLE_BYTES = 64 * 1024

TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class FileLoadError(Exception):
    """
    Raised when an upload cannot be turned into a usable table.

    The message is written for the person who chose the file, not for a
    log file, so it can be shown directly in the interface.
    """


def file_extension(file_name: str) -> str:
    """
    Return the lowercased extension of a file name, including the dot.

    Args:
        file_name:
            Name of the uploaded file.

    Returns:
        The extension, for example ".csv". An empty string when the name
        carries no extension.
    """

    if not file_name:
        return ""

    return os.path.splitext(file_name)[1].lower()


def is_supported(file_name: str) -> bool:
    """
    Report whether this file name is one of the supported formats.

    Args:
        file_name:
            Name of the uploaded file.

    Returns:
        True when the extension is .csv, .xlsx or .xlsm.
    """

    return file_extension(file_name) in SUPPORTED_EXTENSIONS


def is_excel(file_name: str) -> bool:
    """
    Report whether this file name is an Excel workbook.

    Args:
        file_name:
            Name of the uploaded file.

    Returns:
        True for .xlsx and .xlsm.
    """

    return file_extension(file_name) in EXCEL_EXTENSIONS


def describe_supported_formats() -> str:
    """
    Return a human readable list of the accepted extensions.

    Returns:
        A string such as "CSV, XLSX, XLSM".
    """

    return ", ".join(
        extension.lstrip(".").upper()
        for extension in EXTENSION_DISPLAY_ORDER
        if extension in SUPPORTED_EXTENSIONS
    )


def check_file_size(size_in_bytes: int) -> None:
    """
    Confirm an upload is within the size ceiling.

    Args:
        size_in_bytes:
            Size of the uploaded file.

    Raises:
        FileLoadError:
            If the file is empty or above MAX_FILE_SIZE_MB.
    """

    if size_in_bytes is None:
        return

    if size_in_bytes <= 0:
        raise FileLoadError(
            "That file is empty."
        )

    size_in_mb = size_in_bytes / (1024 ** 2)

    if size_in_mb > MAX_FILE_SIZE_MB:
        raise FileLoadError(
            f"That file is {size_in_mb:.1f} MB. Please upload a file "
            f"smaller than {MAX_FILE_SIZE_MB} MB."
        )


def _read_all_bytes(uploaded_file: Any) -> bytes:
    """
    Read an upload fully into memory and rewind it.

    Streamlit hands over a file-like object that may already have been
    consumed by an earlier read, so the position is reset both before
    and after.

    Args:
        uploaded_file:
            File-like object, or a path to a file on disk.

    Returns:
        The raw contents of the file.
    """

    if isinstance(uploaded_file, (str, os.PathLike)):
        with open(uploaded_file, "rb") as handle:
            return handle.read()

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    payload = uploaded_file.read()

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if isinstance(payload, str):
        return payload.encode("utf-8")

    return payload


def decode_text(payload: bytes) -> str:
    """
    Decode CSV bytes, trying the encodings business exports actually use.

    Args:
        payload:
            Raw file contents.

    Returns:
        The decoded text.

    Raises:
        FileLoadError:
            If none of the candidate encodings can decode the file.
    """

    for encoding in TEXT_ENCODINGS:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise FileLoadError(
        "The character encoding of that file is not supported. Save it "
        "as UTF-8 CSV and try again."
    )


def sniff_delimiter(sample: str) -> str:
    """
    Guess which delimiter a CSV sample uses.

    csv.Sniffer is tried first. When it cannot decide, the delimiter that
    appears most consistently across the first few lines wins, which
    handles files whose quoting confuses the sniffer.

    Args:
        sample:
            Leading text of the file.

    Returns:
        A single delimiter character. Falls back to a comma.
    """

    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=CANDIDATE_DELIMITERS
        )

        if dialect.delimiter in CANDIDATE_DELIMITERS:
            return dialect.delimiter

    except csv.Error:
        pass

    lines = [
        line
        for line in sample.splitlines()[:20]
        if line.strip()
    ]

    if not lines:
        return ","

    best_delimiter = ","
    best_score = 0

    for candidate in CANDIDATE_DELIMITERS:
        counts = [line.count(candidate) for line in lines]

        if not counts or counts[0] == 0:
            continue

        # Reward a delimiter that appears the same number of times on
        # every line, which is what a real column separator does.
        consistent = sum(
            1
            for count in counts
            if count == counts[0]
        )

        score = counts[0] * consistent

        if score > best_score:
            best_delimiter = candidate
            best_score = score

    return best_delimiter


def list_worksheets(uploaded_file: Any) -> list[str]:
    """
    List the worksheet names in an Excel workbook.

    Args:
        uploaded_file:
            Excel file-like object or path.

    Returns:
        Worksheet names in workbook order.

    Raises:
        FileLoadError:
            If the workbook cannot be opened.
    """

    payload = _read_all_bytes(uploaded_file)

    try:
        with pd.ExcelFile(io.BytesIO(payload)) as workbook:
            return [str(name) for name in workbook.sheet_names]

    except ImportError as error:
        raise FileLoadError(
            "Reading Excel files needs the openpyxl package. Install it "
            "with: pip install openpyxl"
        ) from error

    except Exception as error:
        raise FileLoadError(
            f"That workbook could not be opened: {error}"
        ) from error


def _load_csv(payload: bytes) -> pd.DataFrame:
    """
    Parse CSV bytes into a DataFrame.

    Args:
        payload:
            Raw file contents.

    Returns:
        The parsed table.

    Raises:
        FileLoadError:
            If the text cannot be decoded or parsed.
    """

    text = decode_text(payload)

    delimiter = sniff_delimiter(text[:SNIFF_SAMPLE_BYTES])

    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=delimiter
        )

    except pd.errors.EmptyDataError as error:
        raise FileLoadError(
            "That file has no columns to read."
        ) from error

    except pd.errors.ParserError as error:
        raise FileLoadError(
            "That file could not be parsed as a table. Check for "
            f"inconsistent numbers of columns between rows. ({error})"
        ) from error


def _load_excel(
    payload: bytes,
    worksheet: str | None
) -> pd.DataFrame:
    """
    Parse one worksheet of an Excel workbook into a DataFrame.

    Args:
        payload:
            Raw workbook contents.

    worksheet:
            Name of the worksheet to read. The first worksheet is used
            when this is None.

    Returns:
        The parsed table.

    Raises:
        FileLoadError:
            If the workbook or the named worksheet cannot be read.
    """

    try:
        with pd.ExcelFile(io.BytesIO(payload)) as workbook:
            available = [str(name) for name in workbook.sheet_names]

            if not available:
                raise FileLoadError(
                    "That workbook contains no worksheets."
                )

            target = worksheet or available[0]

            if target not in available:
                raise FileLoadError(
                    f"That workbook has no worksheet called "
                    f"'{target}'. Available worksheets: "
                    f"{', '.join(available)}."
                )

            return workbook.parse(target)

    except FileLoadError:
        raise

    except ImportError as error:
        raise FileLoadError(
            "Reading Excel files needs the openpyxl package. Install it "
            "with: pip install openpyxl"
        ) from error

    except Exception as error:
        raise FileLoadError(
            f"That worksheet could not be read: {error}"
        ) from error


def load_tabular_file(
    uploaded_file: Any,
    file_name: str | None = None,
    worksheet: str | None = None,
    size_in_bytes: int | None = None
) -> dict[str, Any]:
    """
    Turn an upload into a DataFrame plus a description of what happened.

    Args:
        uploaded_file:
            File-like object or path to read.

        file_name:
            Name used to pick the parser. Taken from the object's own
            name attribute when omitted.

        worksheet:
            Worksheet to read for Excel uploads.

        size_in_bytes:
            Declared size, used for the size check. Measured from the
            payload when omitted.

    Returns:
        A dictionary holding the DataFrame under "dataframe" together
        with the format, worksheet, worksheet list, original row count,
        whether the table was truncated, and any notes worth showing the
        user.

    Raises:
        FileLoadError:
            If the file is unsupported, oversized, empty or unparseable.
    """

    resolved_name = (
        file_name
        or getattr(uploaded_file, "name", "")
        or ""
    )

    if isinstance(uploaded_file, (str, os.PathLike)):
        resolved_name = file_name or os.path.basename(
            str(uploaded_file)
        )

    if not is_supported(resolved_name):
        raise FileLoadError(
            f"'{resolved_name or 'that file'}' is not a supported "
            f"format. Upload one of: "
            f"{describe_supported_formats()}."
        )

    declared_size = (
        size_in_bytes
        if size_in_bytes is not None
        else getattr(uploaded_file, "size", None)
    )

    if declared_size is not None:
        check_file_size(declared_size)

    payload = _read_all_bytes(uploaded_file)

    if declared_size is None:
        check_file_size(len(payload))

    worksheet_names: list[str] = []
    notes: list[str] = []

    if is_excel(resolved_name):
        source_format = file_extension(resolved_name).lstrip(".")

        with pd.ExcelFile(io.BytesIO(payload)) as workbook:
            worksheet_names = [
                str(name) for name in workbook.sheet_names
            ]

        dataframe = _load_excel(payload, worksheet)
        active_worksheet = worksheet or (
            worksheet_names[0] if worksheet_names else None
        )

        if source_format == "xlsm":
            notes.append(
                "Macros in this workbook were ignored. Only the cell "
                "values were read."
            )

    else:
        source_format = "csv"
        active_worksheet = None
        dataframe = _load_csv(payload)

    if dataframe.columns.empty:
        raise FileLoadError(
            "That file has no columns to read."
        )

    if dataframe.empty:
        raise FileLoadError(
            "That file has column names but no data rows."
        )

    original_rows = int(len(dataframe))
    truncated = original_rows > MAX_ROWS

    if truncated:
        dataframe = dataframe.head(MAX_ROWS).copy()
        notes.append(
            f"Only the first {MAX_ROWS:,} rows were loaded, out of "
            f"{original_rows:,} in the file. Every figure below "
            "describes the loaded rows."
        )

    dataframe = dataframe.reset_index(drop=True)

    return {
        "dataframe": dataframe,
        "file_name": resolved_name,
        "source_format": source_format,
        "worksheet": active_worksheet,
        "worksheet_names": worksheet_names,
        "rows_loaded": int(len(dataframe)),
        "original_rows": original_rows,
        "truncated": truncated,
        "notes": notes
    }
