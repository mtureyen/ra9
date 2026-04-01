"""Claude Code LLM adapter — uses the claude CLI as inference engine.

Runs Claude Code in --print mode as a subprocess. No API key needed —
uses the Max plan authentication. The brain (thalamus) wraps this
like any other adapter.

Uses --disallowed-tools to prevent tool use — Sonnet just talks.
Uses --system-prompt for the brain's enriched context.
Uses --output-format stream-json --verbose for streaming.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from emotive.config.schema import LLMProviderConfig
from emotive.logging import get_logger

from .adapter import LLMAdapter

logger = get_logger("llm.claude_code")


class ClaudeCodeAdapter(LLMAdapter):
    """Claude Code CLI adapter using --print mode."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self._model = config.model or "sonnet"
        self._max_tokens = config.max_tokens

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build a single prompt string from conversation history."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n\n".join(parts)

    def _base_args(self, system: str, prompt: str) -> list[str]:
        """Build the claude CLI arguments."""
        return [
            "claude",
            "-p",
            "--system-prompt", system,
            "--model", self._model,
            "--no-session-persistence",
            prompt,
        ]

    async def generate(self, system: str, messages: list[dict]) -> str:
        """Non-streaming generation via Claude Code CLI."""
        prompt = self._build_prompt(messages)
        args = self._base_args(system, prompt) + [
            "--output-format", "text",
        ]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error("Claude Code error: %s", error)
            raise RuntimeError(f"Claude Code failed: {error}")

        return stdout.decode().strip()

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        """Streaming generation via Claude Code CLI."""
        prompt = self._build_prompt(messages)
        args = self._base_args(system, prompt) + [
            "--output-format", "stream-json",
            "--verbose",
        ]

        logger.info("Claude Code args: %d items, prompt len: %d, system len: %d",
                    len(args), len(prompt), len(system))

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        yielded = False
        async for line in proc.stdout:
            line = line.decode().strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            if msg_type == "assistant":
                # Full message with content array
                content_blocks = data.get("message", {}).get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            yielded = True
                            yield text
            elif msg_type == "result" and not yielded:
                # Fallback: get text from result if assistant didn't yield
                result_text = data.get("result", "")
                if result_text:
                    yield result_text

        await proc.wait()
        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            logger.error("Claude Code exit %d: %s", proc.returncode, stderr.decode()[:500])
