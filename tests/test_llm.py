"""
The language-model configuration is read from the environment, applied to
every client, and every completion reports what it cost.

No key is needed and no request is sent. The chat model is a fake.
"""

import os
import unittest
from unittest import mock

import llm

CLEAN_ENVIRONMENT = {
    "GOOGLE_API_KEY": "",
    "GEMINI_API_KEY": "",
    "AI_ANALYST_MODEL": "",
    "AI_ANALYST_LLM_TIMEOUT": "",
    "AI_ANALYST_LLM_MAX_RETRIES": "",
    "AI_ANALYST_THINKING_BUDGET": "",
}


def environment(**overrides):
    values = dict(CLEAN_ENVIRONMENT)
    values.update(overrides)

    return mock.patch.dict(os.environ, values)


class FakeMessage:

    def __init__(self, content, usage=None, metadata=None):
        self.content = content
        self.usage_metadata = usage
        self.response_metadata = metadata or {}


class FakeChatModel:

    def __init__(self, message):
        self.message = message
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        return self.message


class ConfigTest(unittest.TestCase):

    def test_no_key_means_no_configuration(self):
        with environment():
            self.assertIsNone(llm.load_config())

    def test_defaults_apply_when_only_a_key_is_set(self):
        with environment(GOOGLE_API_KEY="test-key"):
            config = llm.load_config()

        self.assertEqual(config.model, llm.DEFAULT_MODEL)
        self.assertEqual(config.timeout, llm.DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(config.max_retries, llm.DEFAULT_MAX_RETRIES)
        self.assertEqual(config.temperature, 0.0)

    def test_the_model_can_be_pinned_from_the_environment(self):
        with environment(
            GOOGLE_API_KEY="test-key",
            AI_ANALYST_MODEL="gemini-2.5-flash",
        ):
            config = llm.load_config()

        self.assertEqual(config.model, "gemini-2.5-flash")
        self.assertFalse(config.is_rolling_alias)

    def test_the_default_model_is_flagged_as_a_rolling_alias(self):
        with environment(GOOGLE_API_KEY="test-key"):
            self.assertTrue(llm.load_config().is_rolling_alias)

    def test_timeout_and_retries_are_read_from_the_environment(self):
        with environment(
            GOOGLE_API_KEY="test-key",
            AI_ANALYST_LLM_TIMEOUT="12.5",
            AI_ANALYST_LLM_MAX_RETRIES="0",
        ):
            config = llm.load_config()

        self.assertEqual(config.timeout, 12.5)
        self.assertEqual(config.max_retries, 0)

    def test_unusable_numbers_fall_back_to_the_defaults(self):
        with environment(
            GOOGLE_API_KEY="test-key",
            AI_ANALYST_LLM_TIMEOUT="-3",
            AI_ANALYST_LLM_MAX_RETRIES="many",
        ):
            config = llm.load_config()

        self.assertEqual(config.timeout, llm.DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(config.max_retries, llm.DEFAULT_MAX_RETRIES)

    def test_the_alternative_key_is_mirrored_for_the_provider(self):
        with environment(GEMINI_API_KEY="alt-key"):
            self.assertIsNotNone(llm.load_config())
            self.assertEqual(os.environ["GOOGLE_API_KEY"], "alt-key")

    def test_the_chat_model_receives_timeout_and_retries(self):
        config = llm.LLMConfig(
            provider="google_genai",
            model="gemini-2.5-flash",
            timeout=7.0,
            max_retries=1,
        )

        with mock.patch(
            "langchain.chat_models.init_chat_model"
        ) as init_chat_model:
            llm.build_chat_model(config)

        init_chat_model.assert_called_once_with(
            "gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.0,
            timeout=7.0,
            max_retries=1,
        )


    def test_the_thinking_budget_is_unset_by_default(self):
        with environment(GOOGLE_API_KEY="test-key"):
            self.assertIsNone(llm.load_config().thinking_budget)

    def test_a_thinking_budget_is_read_and_passed_on(self):
        with environment(
            GOOGLE_API_KEY="test-key",
            AI_ANALYST_THINKING_BUDGET="0",
        ):
            config = llm.load_config()

        self.assertEqual(config.thinking_budget, 0)

        with mock.patch(
            "langchain.chat_models.init_chat_model"
        ) as init_chat_model:
            llm.build_chat_model(config)

        self.assertEqual(
            init_chat_model.call_args.kwargs["thinking_budget"], 0
        )

    def test_an_unusable_thinking_budget_is_ignored(self):
        with environment(
            GOOGLE_API_KEY="test-key",
            AI_ANALYST_THINKING_BUDGET="lots",
        ):
            self.assertIsNone(llm.load_config().thinking_budget)


class EmbedderTest(unittest.TestCase):

    def test_no_embedder_unless_a_model_is_named(self):
        config = llm.LLMConfig("google_genai", "m", 1.0, 0)

        with mock.patch.dict(os.environ, {llm.EMBEDDING_MODEL_ENV: ""}):
            self.assertIsNone(llm.build_embedder(config))

    def test_no_embedder_without_a_key(self):
        with mock.patch.dict(os.environ, {llm.EMBEDDING_MODEL_ENV: "some-model"}):
            self.assertIsNone(llm.build_embedder(None))


class CompletionTest(unittest.TestCase):

    def test_tokens_and_the_answering_model_are_recorded(self):
        model = FakeChatModel(FakeMessage(
            "{}",
            usage={"input_tokens": 120, "output_tokens": 9},
            metadata={"model_name": "gemini-2.5-flash-001"},
        ))

        completion = llm.complete(model, "prompt", "gemini-flash-latest")

        self.assertEqual(completion.text, "{}")
        self.assertEqual(completion.input_tokens, 120)
        self.assertEqual(completion.output_tokens, 9)
        self.assertEqual(completion.model, "gemini-2.5-flash-001")
        self.assertGreaterEqual(completion.latency_ms, 0)

    def test_missing_usage_is_reported_as_unknown(self):
        model = FakeChatModel(FakeMessage("{}"))

        completion = llm.complete(model, "prompt", "requested-model")

        self.assertIsNone(completion.input_tokens)
        self.assertIsNone(completion.output_tokens)
        self.assertEqual(completion.model, "requested-model")

    def test_an_empty_block_list_is_empty_text(self):
        # Found by a live streaming run: Gemini streams empty content
        # lists around tool calls, and they reached the reader as "[]".
        self.assertEqual(llm.extract_text_content([]), "")
        self.assertEqual(llm.extract_text_content([{"type": "tool_call"}]), "")

    def test_block_content_is_reduced_to_text(self):
        model = FakeChatModel(FakeMessage([
            {"type": "text", "text": "hello", "extras": {"signature": "x"}},
        ]))

        self.assertEqual(llm.complete(model, "p", "m").text, "hello")


if __name__ == "__main__":
    unittest.main()
