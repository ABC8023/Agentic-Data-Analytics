"""
Uploaded datasets: session-scoped handles, the cache, and safe loading by name.
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# Uploaded datasets are held here. Keys are opaque per-session handles
# built by make_dataset_key, never bare file names, because a single
# Streamlit process serves every browser session. Two users uploading
# "data.csv" must not share an entry.
DATAFRAME_CACHE={}

SESSION_KEY_SEPARATOR = "/"

# Upper bound on cached datasets so a long-running process cannot grow
# without limit. Sessions that have been idle long enough to be evicted
# are asked to upload again rather than being served another user's data.
MAX_CACHED_DATASETS = 32

# Datasets may only be read from this directory. Uploaded datasets live in
# DATAFRAME_CACHE and never touch the filesystem, so this limit applies only
# to paths the agent or a tool caller supplies by name.
DATASET_ROOT = os.path.realpath(
    os.environ.get("AI_ANALYST_DATA_DIR")
    or os.getcwd()
)


def make_dataset_key(session_token: str, file_name: str) -> str:
    """
    Build a cache handle that is unique to one browser session.

    Args:
        session_token:
            Opaque identifier for the current session.

        file_name:
            Name of the uploaded file.

    Returns:
        A handle of the form "<session_token>/<file_name>".
    """

    return (
        f"{session_token}"
        f"{SESSION_KEY_SEPARATOR}"
        f"{file_name}"
    )


def dataset_display_name(dataset_key: str) -> str:
    """
    Return the original file name carried by a dataset handle.

    Args:
        dataset_key:
            Handle produced by make_dataset_key, or a plain file name.

    Returns:
        The file name without the session token.
    """

    return dataset_key.split(
        SESSION_KEY_SEPARATOR
    )[-1]


def register_dataset(
    dataset_key: str,
    dataframe: pd.DataFrame
) -> None:
    """
    Store a dataset under a session-scoped handle.

    Re-registering an existing handle replaces its contents, which is how
    cleaned datasets are applied.

    Args:
        dataset_key:
            Handle produced by make_dataset_key.

        dataframe:
            Dataset to cache.
    """

    # Reinsert so the handle counts as most recently used for eviction.
    DATAFRAME_CACHE.pop(dataset_key, None)
    DATAFRAME_CACHE[dataset_key] = dataframe

    while len(DATAFRAME_CACHE) > MAX_CACHED_DATASETS:
        oldest_key = next(iter(DATAFRAME_CACHE))
        DATAFRAME_CACHE.pop(oldest_key, None)


def release_session_datasets(session_token: str) -> list[str]:
    """
    Drop every dataset belonging to one session.

    Used when a session uploads a replacement file, so the previous
    dataset does not linger in memory.

    Args:
        session_token:
            Opaque identifier for the session to clear.

    Returns:
        The handles that were removed.
    """

    prefix = (
        f"{session_token}"
        f"{SESSION_KEY_SEPARATOR}"
    )

    released = [
        key
        for key in DATAFRAME_CACHE
        if key.startswith(prefix)
    ]

    for key in released:
        DATAFRAME_CACHE.pop(key, None)

    return released


class DatasetAccessError(ValueError):
    """
    Raised when a requested dataset is outside the permitted directory.

    Subclasses ValueError so existing tool error handling reports it as a
    normal failure instead of raising out of the Streamlit app.
    """


def resolve_dataset_path(name: str) -> str:
    """
    Resolve a dataset name to a real path inside DATASET_ROOT.

    Args:
        name:
            Dataset file name, relative to DATASET_ROOT.

    Returns:
        The absolute path of the dataset.

    Raises:
        DatasetAccessError:
            If the name is empty, is not a CSV file, is an absolute path,
            or resolves to a location outside DATASET_ROOT.
    """

    if not isinstance(name, str) or not name.strip():
        raise DatasetAccessError(
            "A dataset name must be provided."
        )

    candidate = name.strip()

    if not candidate.lower().endswith(".csv"):
        raise DatasetAccessError(
            f"'{candidate}' is not a CSV file. Only .csv datasets "
            "can be opened."
        )

    # Absolute paths, drive letters and UNC paths are rejected before any
    # filesystem access so they cannot point outside DATASET_ROOT.
    if (
        os.path.isabs(candidate)
        or os.path.splitdrive(candidate)[0]
    ):
        raise DatasetAccessError(
            f"'{candidate}' is an absolute path. Datasets must be named "
            "relative to the permitted data directory."
        )

    resolved = os.path.realpath(
        os.path.join(DATASET_ROOT, candidate)
    )

    # Catches parent traversal and symlinks that escape the root.
    try:
        common_prefix = os.path.commonpath([
            resolved,
            DATASET_ROOT
        ])

    except ValueError as error:
        raise DatasetAccessError(
            f"'{candidate}' is outside the permitted data directory."
        ) from error

    if (
        os.path.normcase(common_prefix)
        != os.path.normcase(DATASET_ROOT)
    ):
        raise DatasetAccessError(
            f"'{candidate}' is outside the permitted data directory."
        )

    return resolved


def get_or_load_dataframe(name: str)-> pd.DataFrame:

    """
    Return a DataFrame from the cache or load it from a CSV file.

    Cached datasets, such as those uploaded through the application, are
    returned directly. Anything else must be a CSV file inside
    DATASET_ROOT.

    Args:
        file_name: Name of the CSV file, relative to DATASET_ROOT.

    Returns:
        The loaded Pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file is empty, cannot be parsed, or sits
            outside the permitted data directory.
    """

    if name in DATAFRAME_CACHE:
        return DATAFRAME_CACHE[name]

    dataset_path = resolve_dataset_path(name)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
             f"file {name} not found"
        )

    try:
        dataframe= pd.read_csv(dataset_path)

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f" file {name} is empty"
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f" File {name} is not valid CSV file "
        ) from error

    if dataframe.empty:
        raise ValueError(
            f" {name} file has no rows"
        )
    register_dataset(name, dataframe)

    return dataframe
