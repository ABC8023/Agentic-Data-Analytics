"""
Rendering helpers shared by the tabs: evidence, answers, the glossary
upload, session usage, the calculation trail and error messages.
"""


import streamlit as st

import glossary
import llm
import nlq_answer
import sql_backend
from agent_data_analysis import (
    get_llm_provider,
)
from audit import (
    describe_trail,
    trail_is_empty,
)
from ui_theme import safe
from ui_theme import table as render_table


def severity_class(severity):
    """A CSS class for a severity, from a fixed set so nothing else leaks in."""

    return {"act": "badge-act", "watch": "badge-watch"}.get(severity, "")


def severity_badge(severity):
    """Name a severity in words, since colour alone is not accessible."""

    return {
        "act": "Needs attention",
        "watch": "Worth watching",
        "info": "For information",
    }.get(severity, "For information")


def render_evidence_ledger(cards, key_prefix):
    """
    Show every finding with the arithmetic that produced it.

    The calculation is not hidden behind a second click. A reader who
    cannot see how a number was reached has to take it on trust, which is
    the failure this whole layer exists to avoid.
    """

    if not cards:
        st.info("No evidence could be calculated for this dataset.")

        return

    # Most severe first, then in the order the evidence layer produced
    # them, so a reader scanning the grid meets what needs acting on first.
    rank = {"act": 0, "watch": 1}
    ordered = sorted(
        enumerate(cards),
        key=lambda item: (rank.get(item[1]["severity"], 2), item[0])
    )

    # Two columns, filled row by row, so findings sit side by side
    # instead of as a single wall of full-width cards.
    columns = st.columns(2, gap="medium")

    for position, (_, card) in enumerate(ordered):
        with columns[position % 2], st.container(border=True):
            st.markdown(
                f"**{safe(card['title'])}** "
                f"<span class='badge {severity_class(card['severity'])}'>"
                f"{severity_badge(card['severity'])}"
                "</span>",
                unsafe_allow_html=True
            )

            st.write(card["finding"])
            st.caption(f"Calculation: {card['calculation']}")


def render_recommendations(recommendations, evidence):
    """
    Show what to look at next, and what each suggestion rests on.

    Every entry names the evidence behind it. A suggestion with no
    traceable basis is indistinguishable from a guess, and this tab would
    then be presenting guesses with the same authority as sums.
    """

    if not recommendations:
        st.success(
            "Nothing in the numbers is asking for attention. No unusual "
            "periods, no large recent move, and no data problem big "
            "enough to undermine the totals."
        )

        return

    titles = {card["id"]: card["title"] for card in evidence}

    for entry in recommendations:
        with st.container(border=True):
            st.markdown(
                f"**{safe(entry['priority'])}. {safe(entry['title'])}** "
                f"<span class='badge {severity_class(entry['severity'])}'>"
                f"{severity_badge(entry['severity'])}</span>",
                unsafe_allow_html=True
            )

            st.write(entry["action"])
            st.caption(f"Why: {entry['why']}")

            resting = [
                titles[identifier]
                for identifier in entry["rests_on"]
                if identifier in titles
            ]

            if resting:
                st.caption(f"Rests on: {', '.join(resting)}")

    st.caption(
        "These are investigations the numbers suggest, not established "
        "causes. A table of totals can show what moved; it cannot show "
        "why."
    )


def planning_badge(planned_by):
    """
    Say in words how an answer's question was interpreted.

    A plan matched by rules and a plan guessed by a model are different
    kinds of claim, and the reader is entitled to know which one they are
    reading before they act on the figure.
    """

    return {
        "rules": "Read locally",
        "model": "Interpreted by AI",
        "agent": "Written by AI",
        "nobody": "Not understood",
    }.get(planned_by, "Unknown")


def equivalent_sql(plan, frame):
    """The plan as parameterised SQL, or None when it has no single query."""

    if plan is None:
        return None

    try:
        return sql_backend.compile_plan(plan, frame).display()

    except sql_backend.UnsupportedPlan:
        return None


def render_answer_steps(steps, key_prefix):
    """
    Show the arithmetic behind a deterministic answer.

    Each step is recorded while the calculation runs rather than described
    afterwards, so this is what happened rather than what was meant to.
    """

    if not steps:
        return

    with st.expander(nlq_answer.describe_trail(steps)):
        for step in steps:
            st.markdown(f"**{step['step']}. {step['action']}**")
            st.caption(step["detail"])

            if step["result"]:
                st.code(step["result"], language=None)


def render_deterministic_answer(message, key_prefix):
    """
    Show an answer that was computed locally from a query plan.

    The plan is shown above the figure on purpose. A right answer to the
    wrong question and a wrong answer to the right question look identical
    until the question is written down.
    """

    planned_by = message.get("planned_by", "rules")

    st.markdown(
        f"<span class='badge'>{planning_badge(planned_by)}</span>",
        unsafe_allow_html=True
    )

    if message.get("plan_text"):
        st.caption(f"Read as: {message['plan_text']}")

    st.markdown(message["content"])

    table = message.get("table")

    if table is not None and len(table):
        render_table(
            table,
            width="stretch",
            hide_index=True
        )

    for note in message.get("notes", []):
        st.caption(note)

    ignored = message.get("ignored") or []

    if ignored:
        st.caption(
            "Not used from the question: "
            + ", ".join(ignored)
            + "."
        )

    render_answer_steps(message.get("steps", []), key_prefix)

    # The same plan as one parameterised query, for a reader who would
    # rather check it against a database than read the pandas steps.
    if message.get("sql"):
        with st.expander("Equivalent SQL"):
            st.code(message["sql"], language="sql")


# Bundled datasets a reader can open without a file of their own. Each one
# shows a different part of the app, so the descriptions say which.
SAMPLE_DATASETS = {
    "store_orders": {
        "file": "store_orders.csv",
        "title": "Store orders",
        "detail": (
            "3,660 dated orders across four regions. Opens on the "
            "dashboard, with a planted anomaly in March 2024."
        ),
    },
    "customers_messy": {
        "file": "customers_messy.csv",
        "title": "Messy customers",
        "detail": "Blanks, duplicates and mixed types to clean.",
    },
    "iris": {
        "file": "iris.csv",
        "title": "Iris",
        "detail": "A small classification target for model comparison.",
    },
    "regression_test": {
        "file": "regression_test.csv",
        "title": "Regression",
        "detail": "A numeric target for regression models.",
    },
}


def render_sample_picker():
    """
    Offer the bundled datasets as a way to start without a file.

    Returns:
        The key of the sample chosen on this run, or None.
    """

    chosen = None

    for key, sample in SAMPLE_DATASETS.items():
        with st.container(key=f"sample_card_{key}"):
            if st.button(
                sample["title"],
                key=f"sample_{key}",
                help=f"Open sample_data/{sample['file']}",
                width="stretch",
            ):
                chosen = key

            st.caption(sample["detail"])

    return chosen


def example_questions(detected):
    """
    Questions the rules can answer on this table, from its own columns.

    Args:
        detected:
            Output of schema.detect_business_schema.

    Returns:
        Up to four questions, most useful first.
    """

    measure = detected.get("measure")
    segment = detected.get("segment")
    date = detected.get("date")
    questions = []

    if measure:
        questions.append(f"total {measure}")

    if measure and segment:
        questions.append(f"{measure} by {segment}")

    if measure and date:
        questions.append(f"{measure} last month compared with the month before")
        questions.append(f"which month had the highest {measure}")

    if len(questions) < 4:
        questions.append("how many rows")

    return questions[:4]


def render_example_questions(questions):
    """
    Offer example questions as one-click prompts.

    Returns:
        The question clicked on this run, or None.
    """

    if not questions:
        return None

    st.caption("Try one of these, or type your own below.")

    chosen = None

    with st.container(horizontal=True, gap="small"):
        for position, question in enumerate(questions):
            if st.button(question, key=f"example_question_{position}"):
                chosen = question

    return chosen


MAX_GLOSSARY_BYTES = 200_000


def render_glossary_upload(model_available):
    """
    Offer an optional glossary of the reader's own business terms.

    The glossary is indexed once per upload and kept in the session. When
    a question needs the model planner, the most relevant definitions go
    with it, so "takings" or "territory" can be mapped onto the columns.

    Returns:
        The glossary.Glossary in use, or None.
    """

    uploaded = st.file_uploader(
        "Business glossary (optional, .md or .txt)",
        type=["md", "txt"],
        key="glossary_upload",
        help=(
            "Definitions of your own terms, such as 'Takings: revenue'. "
            "Relevant entries are sent to the model with questions it "
            "plans. They help it choose columns; figures still come from "
            "the table."
        ),
    )

    if uploaded is None:
        st.session_state.glossary = None
        st.session_state.glossary_signature = None

        return None

    signature = (uploaded.name, uploaded.size)

    if st.session_state.get("glossary_signature") != signature:
        if uploaded.size > MAX_GLOSSARY_BYTES:
            st.warning(
                f"The glossary is larger than {MAX_GLOSSARY_BYTES // 1000} KB "
                "and was not loaded."
            )
            st.session_state.glossary = None
            st.session_state.glossary_signature = None

            return None

        text = uploaded.getvalue().decode("utf-8", errors="replace")
        embedder = None

        if model_available:
            try:
                embedder = llm.build_embedder(llm.load_config())

            except Exception:  # noqa: BLE001
                embedder = None

        try:
            st.session_state.glossary = glossary.Glossary.from_text(text, embedder)

        except Exception:  # noqa: BLE001
            # An embedding call that fails leaves lexical retrieval, which
            # needs no model.
            st.session_state.glossary = glossary.Glossary.from_text(text)

        st.session_state.glossary_signature = signature

    active = st.session_state.glossary
    retrieval = "meaning and wording" if active.embedder else "wording"

    st.caption(
        f"{len(active.chunks)} definition(s) loaded. Retrieved by {retrieval} "
        "for questions the model plans."
    )

    return active


def render_session_usage(usage):
    """
    Show how this session's questions were answered and what the model
    cost, so the claim that most questions never reach it can be checked.
    """

    if not usage.questions:
        return

    with st.expander(
        f"This session: {usage.questions} question(s), "
        f"{usage.model_calls} model call(s)"
    ):
        local_share = usage.answered_locally_share
        latency = usage.median_model_latency_ms

        # A label-and-figure list rather than metric cards: this sits in a
        # narrow side column, where four cards side by side cut off both
        # their labels and their figures.
        rows = [
            (
                "Answered locally",
                f"{local_share:.0%}" if local_share is not None else "n/a",
            ),
            ("Model calls", f"{usage.model_calls:,}"),
            ("Tokens in", f"{usage.input_tokens:,}"),
            ("Tokens out", f"{usage.output_tokens:,}"),
            (
                "Median model wait",
                f"{latency / 1000:.1f} s" if latency is not None else "n/a",
            ),
        ]

        items = "".join(
            f"<dt>{safe(label)}</dt><dd>{safe(value)}</dd>"
            for label, value in rows
        )

        st.markdown(f'<dl class="usage-list">{items}</dl>', unsafe_allow_html=True)

        st.caption(
            "Planned by rules: "
            f"{usage.by_route.get('rules', 0)}, by the model: "
            f"{usage.by_route.get('model', 0)}, not planned: "
            f"{usage.by_route.get('nobody', 0)}. Handed to the AI "
            f"analyst: {usage.handed_to_agent}."
        )


def render_calculation_trail(trail, key_prefix):
    """
    Show the tool calls behind an assistant answer.

    Prose alone gives a reader no way to separate a computed figure from an
    invented one, so every answer carries the work underneath it.

    Args:
        trail:
            Output of audit.extract_calculation_trail.

        key_prefix:
            Unique prefix for the widget keys in this block.
    """

    if trail_is_empty(trail):
        st.caption(
            "No calculation ran for this answer. The model replied from "
            "the conversation alone, so treat any figure in it as "
            "unverified."
        )

        return

    with st.expander(describe_trail(trail)):

        for entry in trail:
            arguments = ", ".join(
                f"{name}={value!r}"
                for name, value in entry["arguments"].items()
            )

            st.markdown(
                f"**{entry['step']}. `{entry['tool']}`**"
                + (f"  \n`{arguments}`" if arguments else "")
            )

            if entry["failed"]:
                st.warning(
                    entry["error"] or "This calculation failed."
                )

            elif entry["highlights"]:
                st.markdown(
                    "\n".join(
                        f"- {line}"
                        for line in entry["highlights"]
                    )
                )

            else:
                st.caption("Returned no figures.")


def describe_agent_error(error):
    """
    Turn a Gemini or network failure into something a user can act on.

    Raw provider exceptions arrive as a JSON blob containing the whole
    error envelope, which tells the reader nothing about what to do next.

    Args:
        error:
            The exception raised while calling the agent.

    Returns:
        A short explanation, with the original message appended only when
        the failure is not one of the recognised cases.
    """

    detail = str(error)
    lowered = detail.lower()

    provider = get_llm_provider()
    model_name = provider[1] if provider else llm.DEFAULT_MODEL

    if (
        "insufficient_quota" in lowered
        or "credit_balance_exhausted" in lowered
        or "no credits remaining" in lowered
        or "exceeded your current quota" in lowered
        or "resource_exhausted" in lowered
    ):
        billing_url = "https://aistudio.google.com/app/plan_information"
        return (
            "**This Gemini account has no quota left.** The API "
            "rejected the request with a quota error, so the AI "
            "analyst cannot answer until the account is topped up or "
            f"its limits reset, at [{billing_url}]({billing_url})."
            "\n\nNothing is wrong with the app or your key. Every other "
            "tab works without the API, so dataset profiling, cleaning, "
            "charts and model training are all still available."
        )

    if (
        "invalid_api_key" in lowered
        or "incorrect api key" in lowered
        or "api_key_invalid" in lowered
        or "api key not valid" in lowered
        or ("401" in lowered and "auth" in lowered)
    ):
        return (
            "**The API key was rejected.** Check `GOOGLE_API_KEY` in "
            "your `.env` file for a typo, a trailing space, or a key "
            "that has been revoked, then restart the app."
        )

    if "rate limit" in lowered or "rate_limit" in lowered:
        return (
            "**Rate limit reached.** The account is sending requests "
            "faster than the plan allows. Wait a few seconds and ask "
            "again."
        )

    if (
        "connection" in lowered
        or "timeout" in lowered
        or "timed out" in lowered
        or "getaddrinfo" in lowered
    ):
        return (
            "**Could not reach the Gemini API.** This looks like a "
            "network problem rather than a problem with the dataset. "
            "Check your connection or any proxy, then try again."
        )

    if "model" in lowered and (
        "does not exist" in lowered
        or "not found" in lowered
        or "do not have access" in lowered
    ):
        return (
            "**This account cannot use the configured model.** The app "
            f"requests `{model_name}`. Either enable it for the account "
            "or set `AI_ANALYST_MODEL` in `.env` to one it can use."
        )

    return (
        "**The AI analyst could not complete the request.**"
        f"\n\n```\n{detail}\n```"
    )


# ──────────────────────────────────────────────────────────────
