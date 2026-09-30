"""
The agent's answer streams as prose only, and the final state is kept for
the calculation trail.
"""

import os
import unittest

os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

from agent_data_analysis import AgentStream  # noqa: E402


class ScriptedAgent:
    """Replays a fixed stream, as the compiled graph would emit it."""

    def __init__(self, items):
        self.items = items
        self.calls = []

    def stream(self, payload, stream_mode=None):
        self.calls.append((payload, stream_mode))

        yield from self.items


class AgentStreamTest(unittest.TestCase):

    def test_a_real_agent_graph_streams_its_answer(self):
        model = GenericFakeChatModel(messages=iter([
            AIMessage(content="North leads with half the revenue.")
        ]))
        agent = create_agent(model=model, tools=[], system_prompt="test")

        stream = AgentStream([{"role": "user", "content": "q"}], agent=agent)
        chunks = list(stream.text())

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), "North leads with half the revenue.")
        self.assertEqual(stream.answer(), "North leads with half the revenue.")

    def test_tool_calls_and_tool_results_are_not_streamed(self):
        final = {"messages": [AIMessage(content="Done.")]}
        agent = ScriptedAgent([
            ("messages", (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[{
                        "name": "get_column_schema",
                        "args": "{}",
                        "id": "1",
                        "index": 0,
                    }],
                ),
                {"langgraph_node": "model"},
            )),
            ("messages", (
                ToolMessage(content='{"columns": 7}', tool_call_id="1"),
                {"langgraph_node": "tools"},
            )),
            ("messages", (
                AIMessageChunk(content="Done."),
                {"langgraph_node": "model"},
            )),
            ("values", final),
        ])

        stream = AgentStream([], agent=agent)

        self.assertEqual(list(stream.text()), ["Done."])
        self.assertIs(stream.final_state, final)

    def test_empty_chunks_are_not_streamed_as_brackets(self):
        agent = ScriptedAgent([
            ("messages", (AIMessageChunk(content=[]), {"langgraph_node": "model"})),
            ("messages", (AIMessageChunk(content="Answer."), {"langgraph_node": "model"})),
        ])

        self.assertEqual(list(AgentStream([], agent=agent).text()), ["Answer."])

    def test_block_content_is_reduced_to_text(self):
        agent = ScriptedAgent([
            ("messages", (
                AIMessageChunk(content=[
                    {"type": "text", "text": "Hello", "extras": {"signature": "x"}}
                ]),
                {"langgraph_node": "model"},
            )),
        ])

        self.assertEqual(list(AgentStream([], agent=agent).text()), ["Hello"])

    def test_both_stream_modes_are_requested(self):
        agent = ScriptedAgent([])

        list(AgentStream([{"role": "user", "content": "q"}], agent=agent).text())

        self.assertEqual(agent.calls[0][1], ["messages", "values"])

    def test_no_final_state_means_no_answer(self):
        self.assertEqual(AgentStream([], agent=ScriptedAgent([])).answer(), "")


if __name__ == "__main__":
    unittest.main()
