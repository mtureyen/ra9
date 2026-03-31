"""Tests for LLM adapter layer."""

import pytest

from emotive.config.schema import LLMProviderConfig
from emotive.llm.adapter import LLMAdapter, create_adapter
from emotive.llm.types import LLMMessage, LLMResponse, StreamChunk


class TestLLMTypes:
    def test_llm_message(self):
        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"

    def test_llm_response(self):
        resp = LLMResponse(content="hi there", model="test")
        assert resp.content == "hi there"

    def test_stream_chunk(self):
        chunk = StreamChunk(text="hel", done=False)
        assert not chunk.done


class TestAdapterFactory:
    def test_creates_ollama(self):
        config = LLMProviderConfig(provider="ollama")
        adapter = create_adapter(config)
        from emotive.llm.ollama import OllamaAdapter

        assert isinstance(adapter, OllamaAdapter)

    def test_unknown_provider_raises(self):
        config = LLMProviderConfig(provider="unknown_provider")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_adapter(config)

    def test_anthropic_without_key_raises(self):
        config = LLMProviderConfig(provider="anthropic")
        with pytest.raises(
            (ValueError, ImportError)
        ):
            create_adapter(config)


class TestOllamaAdapter:
    def test_build_payload(self):
        from emotive.llm.ollama import OllamaAdapter

        config = LLMProviderConfig(
            provider="ollama", model="test:7b", temperature=0.5
        )
        adapter = OllamaAdapter(config)
        payload = adapter._build_payload(
            "system prompt",
            [{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert payload["model"] == "test:7b"
        assert payload["stream"] is True
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["options"]["temperature"] == 0.5

    def test_host_configuration(self):
        from emotive.llm.ollama import OllamaAdapter

        # localhost
        config = LLMProviderConfig(host="http://localhost:11434")
        adapter = OllamaAdapter(config)
        assert adapter._host == "http://localhost:11434"

        # remote (MacBook+PC deployment)
        config = LLMProviderConfig(host="http://192.168.1.100:11434/")
        adapter = OllamaAdapter(config)
        assert adapter._host == "http://192.168.1.100:11434"  # trailing slash stripped
