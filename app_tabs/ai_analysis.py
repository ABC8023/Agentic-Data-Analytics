"""
The ai analysis tab.
"""

import time

import streamlit as st

import nlq_answer
import nlq_model
import telemetry
from agent_data_analysis import (
    AgentStream,
    api_key_is_configured,
    plan_invoker,
)
from audit import (
    extract_calculation_trail,
)
from privacy import describe_policy as describe_privacy_policy
from ui_cache import (
    cached_business_schema,
)
from ui_components import (
    describe_agent_error,
    equivalent_sql,
    example_questions,
    planning_badge,
    render_calculation_trail,
    render_deterministic_answer,
    render_example_questions,
    render_glossary_upload,
    render_session_usage,
)


def render(*, data_version, dataframe, dataset_key, name):

    agent_is_available = api_key_is_configured()
    session_usage = st.session_state.session_usage

    # The conversation gets the main column. Settings and the fine
    # print sit beside it, so the chat is the first thing on the tab
    # rather than the last.
    chat_column, side_column = st.columns([2.4, 1], gap="large")

    with side_column:
        st.markdown(
            '<div class="panel-title">How answers are made</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Read into a query plan and computed here, with the "
            "working shown. Questions a plan cannot express go to "
            "the AI analyst, and those answers say so."
        )

        if not agent_is_available:
            st.info(
                "No API key. Figures, periods, segments and shares "
                "are still answered here. Open-ended questions need "
                "`GOOGLE_API_KEY` in `.env`."
            )

        active_glossary = render_glossary_upload(agent_is_available)

        with st.expander("What the model can see"):
            st.caption(describe_privacy_policy())

        if st.session_state.agent_chat_messages and st.button(
            "Clear conversation",
            key="clear_agent_conversation",
            type="tertiary"
        ):
            st.session_state.agent_chat_messages = []
            st.rerun()

        # Filled in after the question below is handled, so the
        # totals include it.
        usage_container = st.container()

    chat_column.markdown(
        '<div class="sec sec-first">Ask about this table</div>',
        unsafe_allow_html=True
    )

    # Fixed height only once there is a conversation to scroll; an empty
    # one should not reserve half a screen of blank space.
    chat_container = chat_column.container(
        height=560 if st.session_state.agent_chat_messages else "content",
        border=True
    )

    with chat_container:

        for message_index, message in enumerate(
            st.session_state.agent_chat_messages
        ):

            with st.chat_message(
                message["role"]
            ):
                if message["role"] != "assistant":
                    st.markdown(message["content"])

                elif message.get("planned_by") in {
                    "rules",
                    "model",
                    "nobody",
                }:
                    render_deterministic_answer(
                        message,
                        f"history_{message_index}"
                    )

                else:
                    st.markdown(message["content"])

                    render_calculation_trail(
                        message.get("trail", []),
                        f"history_{message_index}"
                    )

        # An empty conversation offers questions this table can
        # answer, built from its own columns, instead of a blank box.
        examples_slot = st.empty()

        if not st.session_state.agent_chat_messages:
            with examples_slot.container():
                example = render_example_questions(
                    example_questions(
                        cached_business_schema(
                            dataset_key, data_version, None, None, None
                        )
                    )
                )

            if example:
                st.session_state.pending_question = example

    # Never disabled. Most questions are answered locally, so a
    # missing key is a reduced service rather than no service.
    user_question = chat_column.chat_input(
        "Ask a question, e.g. revenue by region last quarter",
        max_chars=nlq_model.MAX_QUESTION_CHARS,
        key="agent_chat_input"
    ) or st.session_state.pop("pending_question", None)

    if user_question:

        # The conversation has started, so the examples make way.
        examples_slot.empty()

        user_message = {
            "role": "user",
            "content": user_question
        }

        st.session_state.agent_chat_messages.append(
            user_message
        )

        with chat_container:

            with st.chat_message("user"):
                st.markdown(user_question)

            # The query plan runs first. Most questions are simple, and
            # asking a model to compute a sum is slower, costlier and
            # no more accurate than doing it here.
            request_id = telemetry.new_request_id()
            question_calls = []
            question_started = time.perf_counter()

            with st.spinner("Working out the question..."):
                planned = nlq_answer.answer_question(
                    dataframe,
                    user_question,
                    (
                        plan_invoker(sink=question_calls)
                        if agent_is_available
                        else None
                    ),
                    glossary=active_glossary,
                )

            if planned.get("glossary_used"):
                planned["notes"].append(
                    "Glossary definitions sent with the question: "
                    + ", ".join(planned["glossary_used"])
                    + "."
                )

            deterministic = planned["answer"]

            question_record = telemetry.question_event(
                planned,
                (time.perf_counter() - question_started) * 1000,
                question_calls,
                request_id,
                question=user_question,
                handed_to_agent=(
                    not deterministic.success
                    and agent_is_available
                    and not planned.get("model_unanswerable")
                ),
            )
            telemetry.emit(question_record)
            session_usage.add_question(question_record)

            if deterministic.success:
                stored_message = {
                    "role": "assistant",
                    "content": deterministic.headline,
                    "sql": equivalent_sql(planned["plan"], dataframe),
                    "planned_by": planned["planned_by"],
                    "plan_text": (
                        planned["plan"].describe()
                        if planned["plan"]
                        else ""
                    ),
                    "steps": deterministic.trail,
                    "table": deterministic.table,
                    "notes": planned["notes"],
                    "ignored": planned["ignored"],
                    "trail": [],
                }

                with st.chat_message("assistant"):
                    render_deterministic_answer(
                        stored_message,
                        "latest"
                    )

                st.session_state.agent_chat_messages.append(
                    stored_message
                )

            elif (
                not agent_is_available
                or planned.get("model_unanswerable")
            ):
                # Nothing else can be tried, or the planner has
                # already judged that the columns cannot answer it,
                # so the reason the plan failed is the answer.
                follow_up = (
                    "It can be reworded to name a column this "
                    "dataset has."
                    if planned.get("model_unanswerable")
                    else (
                        "This question needs a Gemini API key to "
                        "interpret, or it can be reworded to name a "
                        "figure, a column, a period or a condition."
                    )
                )

                stored_message = {
                    "role": "assistant",
                    "content": f"{deterministic.error}\n\n{follow_up}",
                    "planned_by": "nobody",
                    "plan_text": "",
                    "steps": [],
                    "table": None,
                    "notes": planned["notes"],
                    "ignored": planned["ignored"],
                    "trail": [],
                }

                with st.chat_message("assistant"):
                    render_deterministic_answer(
                        stored_message,
                        "latest"
                    )

                st.session_state.agent_chat_messages.append(
                    stored_message
                )

            else:
                # The questions a plan is the wrong shape for:
                # explanations, judgement calls, anything
                # conversational. Labelled as written by AI, because
                # unlike everything above it the figures in the answer
                # were not computed here.
                with st.chat_message("assistant"):

                    st.markdown(
                        "<span class='badge'>"
                        f"{planning_badge('agent')}</span>",
                        unsafe_allow_html=True
                    )

                    st.caption(
                        "The query plan could not express this "
                        f"question: {planned['rules_reason']} Passing "
                        "it to the AI analyst instead, which composes "
                        "its own answer from tool calls."
                    )

                    with st.spinner("Analysing the dataset..."):

                        agent_started = time.perf_counter()
                        agent_response = {"messages": []}
                        agent_failed = False
                        answer_already_shown = False

                        try:
                            recent_messages = (
                                st.session_state
                                .agent_chat_messages[-12:]
                            )

                            messages_for_agent = []

                            for message_index, message in enumerate(
                                recent_messages
                            ):
                                message_content = message["content"]

                                is_latest_user_message = (
                                    message_index
                                    == len(recent_messages) - 1
                                    and message["role"] == "user"
                                )

                                if is_latest_user_message:
                                    # The agent is given the
                                    # session-scoped handle, not the
                                    # bare file name, so its tool calls
                                    # can only reach this session's
                                    # dataset.
                                    message_content = (
                                        f"The active uploaded dataset "
                                        f"is named '{name}' and its "
                                        f"dataset id is "
                                        f"'{dataset_key}'. Always "
                                        f"pass '{dataset_key}' as the "
                                        f"'name' argument to tools. "
                                        f"Refer to it as '{name}' "
                                        f"when talking to the "
                                        f"user.\n\n"
                                        f"User question:\n"
                                        f"{message_content}"
                                    )

                                messages_for_agent.append({
                                    "role": message["role"],
                                    "content": message_content
                                })

                            # Streamed, so the first words appear while
                            # the agent is still working.
                            agent_stream = AgentStream(messages_for_agent)
                            streamed_text = st.write_stream(
                                agent_stream.text()
                            )
                            answer_already_shown = bool(streamed_text)

                            agent_response = (
                                agent_stream.final_state
                                or {"messages": []}
                            )

                            assistant_answer = (
                                agent_stream.answer()
                                or (
                                    streamed_text
                                    if isinstance(streamed_text, str)
                                    else ""
                                )
                            )

                            # Record what the answer was actually
                            # built from, so the reader can check it.
                            answer_trail = extract_calculation_trail(
                                agent_response
                            )

                        except Exception as error:  # noqa: BLE001
                            assistant_answer = describe_agent_error(
                                error
                            )
                            answer_trail = []
                            agent_failed = True
                            # Whatever was streamed before the failure
                            # stays on screen; the explanation follows.
                            answer_already_shown = False

                        agent_record = telemetry.agent_event(
                            request_id,
                            (time.perf_counter() - agent_started) * 1000,
                            agent_response.get("messages", []),
                            len(answer_trail),
                            agent_failed,
                        )
                        telemetry.emit(agent_record)
                        session_usage.add_agent(agent_record)

                    if not answer_already_shown:
                        st.markdown(assistant_answer)

                    render_calculation_trail(answer_trail, "latest")

                st.session_state.agent_chat_messages.append({
                    "role": "assistant",
                    "content": assistant_answer,
                    "planned_by": "agent",
                    "trail": answer_trail
                })

    with usage_container:
        render_session_usage(session_usage)

