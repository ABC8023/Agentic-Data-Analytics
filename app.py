import os
import uuid

import streamlit as st

import app_tabs.ai_analysis
import app_tabs.cleaning_tab
import app_tabs.dashboard
import app_tabs.machine_learning
import app_tabs.overview
import app_tabs.statistics
import app_tabs.time_series
import app_tabs.visualisation
import auth
import llm
import telemetry
from dataset_store import (
    make_dataset_key,
    register_dataset,
    release_session_datasets,
)
from file_io import (
    MAX_FILE_SIZE_MB,
    FileLoadError,
    describe_supported_formats,
    is_excel,
    list_worksheets,
    load_tabular_file,
)
from ui_cache import (
    cached_date_columns,
)
from ui_components import (
    SAMPLE_DATASETS,
    render_sample_picker,
)
from ui_theme import (
    THEME_CSS,
    bar_status,
    brand,
    capabilities,
    dataset_chip,
    hero,
)

SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_data"
)

FAVICON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "favicon.png"
)

st.set_page_config(
    page_title="AI data analyst",
    page_icon=(
        FAVICON_PATH
        if os.path.exists(FAVICON_PATH)
        else None
    ),
    layout="wide"
)


# Rendered on every run: Streamlit rebuilds the page each rerun, and the
# ui_theme module is only imported once per process.
st.markdown(THEME_CSS, unsafe_allow_html=True)


# Nothing below renders, and no dataset is read, until the visitor may use
# the app. Off unless configured: see auth.py.
if not auth.require_login():
    st.stop()


# ──────────────────────────────────────────────────────────────
#  APP
# ──────────────────────────────────────────────────────────────


# One Streamlit process serves every browser session, so each session gets
# its own token. Dataset cache handles are built from it, which keeps two
# users who upload the same file name from reading each other's data.
if "session_token" not in st.session_state:
    st.session_state.session_token = uuid.uuid4().hex

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

if "name" not in st.session_state:
    st.session_state.name = None

# Opaque handle the analysis tools and the agent use to reference the
# active dataset. st.session_state.name stays the plain file name and is
# only used for display and download file names.
if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = None

# Bumped whenever the active dataset's contents change, so memoised
# profiling results are recomputed instead of served stale.
if "data_version" not in st.session_state:
    st.session_state.data_version = 0

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None

# Notes raised while loading the file, such as row truncation or ignored
# Excel macros. Shown under the uploader on every rerun.
if "load_notes" not in st.session_state:
    st.session_state.load_notes = []

if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None
if "ml_result" not in st.session_state:
    st.session_state.ml_result = None

if "ml_target" not in st.session_state:
    st.session_state.ml_target = None

if "ml_task" not in st.session_state:
    st.session_state.ml_task = None
if "feature_importance_result" not in st.session_state:
    st.session_state.feature_importance_result = None
if "agent_chat_messages" not in st.session_state:
    st.session_state.agent_chat_messages = []
if "session_usage" not in st.session_state:
    st.session_state.session_usage = telemetry.SessionUsage()

telemetry.configure_console_logging()
if "original_dataframe" not in st.session_state:
    st.session_state.original_dataframe = None

if "cleaning_result" not in st.session_state:
    st.session_state.cleaning_result = None

if "last_ml_target" not in st.session_state:
    st.session_state.last_ml_target = None

if "ml_task_choice" not in st.session_state:
    st.session_state.ml_task_choice = None

def activate_dataset(load_result, dataset_label, signature):
    """
    Make a loaded table the session's dataset, clearing everything that
    was computed from the previous one.

    Every way in, an upload or a sample, goes through here, so no path can
    forget to reset a stale model, chart or conversation.
    """

    dataframe = load_result["dataframe"]

    # Drop the previous dataset for this session before caching the
    # replacement, so a session never holds two datasets.
    release_session_datasets(st.session_state.session_token)

    st.session_state.dataframe = dataframe
    st.session_state.name = dataset_label
    st.session_state.dataset_key = make_dataset_key(
        st.session_state.session_token,
        dataset_label
    )
    st.session_state.data_version += 1
    st.session_state.original_dataframe = dataframe.copy()
    st.session_state.upload_signature = signature
    st.session_state.load_notes = load_result["notes"]
    st.session_state.recommendation_result = None
    st.session_state.ml_result = None
    st.session_state.ml_target = None
    st.session_state.ml_task = None
    st.session_state.feature_importance_result = None
    st.session_state.agent_chat_messages = []
    st.session_state.cleaning_result = None
    st.session_state.last_ml_target = None
    st.session_state.ml_task_choice = None
    st.session_state.loaded_message = (
        f"Loaded {dataset_label}: "
        f"{load_result['rows_loaded']:,} rows, "
        f"{dataframe.shape[1]} columns."
    )


def open_sample(sample_key):
    """Load one of the bundled sample datasets."""

    file_name = SAMPLE_DATASETS[sample_key]["file"]

    try:
        load_result = load_tabular_file(
            os.path.join(SAMPLE_DATA_DIR, file_name)
        )

    except FileLoadError as error:
        st.error(str(error))
        st.stop()

    activate_dataset(load_result, file_name, ("sample", sample_key))

    st.session_state.upload_generation = (
        st.session_state.get("upload_generation", 0) + 1
    )


def go_home():
    """
    Close the loaded table and return to the start page.

    Runs as a button callback, before the page is redrawn. The link that
    opened a sample is cleared too, or ?sample= would load it straight
    back, and the uploader gets a new key, or a file still held in it
    would be loaded again on the next run.
    """

    release_session_datasets(st.session_state.session_token)

    for key in (
        "dataframe", "original_dataframe", "name", "dataset_key",
        "upload_signature", "recommendation_result", "ml_result",
        "ml_target", "ml_task", "feature_importance_result",
        "cleaning_result", "last_ml_target", "ml_task_choice",
        "loaded_message",
    ):
        st.session_state[key] = None

    st.session_state.load_notes = []
    st.session_state.agent_chat_messages = []
    st.session_state.data_version += 1
    st.session_state.upload_generation = (
        st.session_state.get("upload_generation", 0) + 1
    )
    st.query_params.clear()


# A sample can be opened from the landing page, or from a shared link such
# as ?sample=store_orders. The link only applies while nothing is loaded,
# so it never replaces a file the reader chose. Handled before the top bar
# is drawn, so the bar already shows the sample.
requested_sample = st.query_params.get("sample")

if (
    st.session_state.dataframe is None
    and requested_sample in SAMPLE_DATASETS
):
    open_sample(requested_sample)


# ── top bar ──────────────────────────────────────────────────
# A site header on every page, pinned to the top while the page scrolls:
# the name on the left (a way home once a table is open), drop-down menus
# beside it, and one strong action on the right. Rows rather than
# columns, so the bar stays one line instead of stacking on a phone.
table_loaded = st.session_state.dataframe is not None

with st.container(key="topbar"), st.container(
    horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",
    key="topbar_row",
):
    with st.container(
        horizontal=True, vertical_alignment="center", key="topbar_left"
    ):
        brand(home_action=go_home if table_loaded else None)

    with st.container(
        horizontal=True,
        horizontal_alignment="right",
        vertical_alignment="center",
        key="topbar_actions",
    ):
        if auth.login_mode() != auth.MODE_NONE:
            auth.render_sign_out()

        if table_loaded:
            dataset_chip(
                st.session_state.name,
                *st.session_state.dataframe.shape
            )
            upload_slot = st.popover(
                "Change dataset", type="primary", key="change_dataset"
            )

        else:
            bar_status(model_ready=llm.load_config() is not None)

if not table_loaded:
    hero_column, start_column = st.columns([1.2, 1], gap="large")

    with hero_column:
        hero()

    with start_column:
        st.markdown(
            '<div class="panel-title">Start with a table</div>'
            '<p class="panel-note">Nothing leaves this machine except, '
            'when a key is set, column names and labels sent to the '
            'model planner.</p>',
            unsafe_allow_html=True
        )

    upload_slot = start_column.container(border=True)


with upload_slot:
    uploaded_file = st.file_uploader(
        f"Upload a dataset ({describe_supported_formats()})",
        type=["csv", "xlsx", "xlsm"],
        # A new key after a sample is opened empties the uploader, so a
        # file chosen earlier cannot replace the sample on the next run.
        key=f"dataset_upload_{st.session_state.get('upload_generation', 0)}",
        help=(
            f"CSV files may be comma, semicolon, tab or pipe delimited. "
            f"Excel workbooks let you choose a worksheet. Maximum "
            f"{MAX_FILE_SIZE_MB} MB."
        )
    )

    st.markdown(
        '<p class="panel-note sample-note">Or open a sample</p>',
        unsafe_allow_html=True
    )

    chosen_sample = render_sample_picker()

if chosen_sample:
    open_sample(chosen_sample)
    st.rerun()


if uploaded_file is not None:

    # Excel workbooks often hold several worksheets, and which one the
    # user wants cannot be guessed. Offer the choice before loading, and
    # treat the chosen worksheet as part of the upload identity so
    # switching worksheets reloads the data.
    chosen_worksheet = None

    if is_excel(uploaded_file.name):
        try:
            available_worksheets = list_worksheets(uploaded_file)

        except FileLoadError as error:
            st.error(str(error))
            st.stop()

        if len(available_worksheets) > 1:
            chosen_worksheet = upload_slot.selectbox(
                "Worksheet",
                options=available_worksheets,
                key="worksheet_choice",
                help=(
                    "This workbook has "
                    f"{len(available_worksheets)} worksheets."
                )
            )
        elif available_worksheets:
            chosen_worksheet = available_worksheets[0]

    current_upload_signature = (
        uploaded_file.name,
        uploaded_file.size,
        chosen_worksheet
    )

    if (
        current_upload_signature
        != st.session_state.upload_signature
    ):
        try:
            load_result = load_tabular_file(
                uploaded_file,
                file_name=uploaded_file.name,
                worksheet=chosen_worksheet,
                size_in_bytes=uploaded_file.size
            )

        except FileLoadError as error:
            st.error(str(error))
            st.stop()

        except Exception as error:
            st.error(
                f"Unexpected file-loading error: {error}"
            )
            st.stop()

        # A dataset label that includes the worksheet, so downloads and
        # captions stay unambiguous for multi-sheet workbooks.
        dataset_label = uploaded_file.name

        if load_result["worksheet"] and len(
            load_result["worksheet_names"]
        ) > 1:
            dataset_label = (
                f"{uploaded_file.name} [{load_result['worksheet']}]"
            )

        activate_dataset(load_result, dataset_label, current_upload_signature)

        # Redraw from the top, so the bar names the new table.
        st.rerun()


if st.session_state.get("loaded_message"):
    st.toast(st.session_state.loaded_message)
    st.session_state.loaded_message = None

for note in st.session_state.get("load_notes", []):
    st.warning(note)


if st.session_state.dataframe is None:
    capabilities()


if st.session_state.dataframe is not None:

    dataframe = st.session_state.dataframe

    # name is for display and download file names only. dataset_key is the
    # handle every analysis tool and the agent must use.
    name = st.session_state.name or "dataset.csv"

    if st.session_state.dataset_key is None:
        st.session_state.dataset_key = make_dataset_key(
            st.session_state.session_token,
            name
        )

    dataset_key = st.session_state.dataset_key
    data_version = st.session_state.data_version

    # Register before anything reads the dataset by handle. The tab list
    # itself now depends on the data, so this cannot wait until after the
    # tabs are built.
    register_dataset(dataset_key, dataframe)

    # The time series tab only appears when the data can support one.
    # Offering an empty timeline for a dataset with no dates would be a
    # dead end rather than a feature.
    date_columns = cached_date_columns(dataset_key, data_version)
    time_series_available = bool(date_columns)

    tab_names = []

    # The dashboard leads when the data can support one, because a reader
    # opening a business extract wants the finding before the schema. It
    # needs a timeline: without a date there is no movement to report and
    # the tab would restate the overview.
    if time_series_available:
        tab_names.append("Dashboard")

    tab_names += [
        "Dataset overview",
        "Statistics",
        "Data visualisation",
    ]

    if time_series_available:
        tab_names.append("Time series")

    tab_names += [
        "Machine learning",
        "AI analysis",
        "Data cleaning",
    ]

    rendered_tabs = dict(
        zip(tab_names, st.tabs(tab_names), strict=True)
    )

    dashboard_tab = rendered_tabs.get("Dashboard")
    overview_tab = rendered_tabs["Dataset overview"]
    statistic_tab = rendered_tabs["Statistics"]
    visualisation_tab = rendered_tabs["Data visualisation"]
    time_series_tab = rendered_tabs.get("Time series")
    machine_learning_tab = rendered_tabs["Machine learning"]
    ai_analysis = rendered_tabs["AI analysis"]
    Data_cleaning = rendered_tabs["Data cleaning"]

    if dashboard_tab is not None:

        with dashboard_tab:
            app_tabs.dashboard.render(data_version=data_version, dataset_key=dataset_key, date_columns=date_columns, name=name)

    with overview_tab:
        app_tabs.overview.render(data_version=data_version, dataframe=dataframe, dataset_key=dataset_key)

    with Data_cleaning:
        app_tabs.cleaning_tab.render(data_version=data_version, dataframe=dataframe, dataset_key=dataset_key, name=name)

    with statistic_tab:
        app_tabs.statistics.render(data_version=data_version, dataset_key=dataset_key)

    with visualisation_tab:
        app_tabs.visualisation.render(data_version=data_version, dataframe=dataframe, dataset_key=dataset_key)

    if time_series_tab is not None:

        with time_series_tab:
            app_tabs.time_series.render(data_version=data_version, dataset_key=dataset_key, date_columns=date_columns)

    with machine_learning_tab:
        app_tabs.machine_learning.render(data_version=data_version, dataframe=dataframe, dataset_key=dataset_key)

    with ai_analysis:
        app_tabs.ai_analysis.render(data_version=data_version, dataframe=dataframe, dataset_key=dataset_key, name=name)

