"""Tests for the sensory buffer."""

from datetime import datetime, timezone

from emotive.runtime.sensory_buffer import MAX_INPUT_CHARS, SensoryBuffer


def test_process_normal_input():
    sb = SensoryBuffer()
    result = sb.process("Hello world")
    assert result.text == "Hello world"
    assert result.truncated is False
    assert result.char_count == 11


def test_process_strips_whitespace():
    sb = SensoryBuffer()
    result = sb.process("  hello  ")
    assert result.text == "hello"


def test_process_truncates_long_input():
    sb = SensoryBuffer()
    result = sb.process("x" * 3000)
    assert result.truncated is True
    assert result.char_count == MAX_INPUT_CHARS


def test_process_exactly_at_limit():
    sb = SensoryBuffer()
    result = sb.process("x" * MAX_INPUT_CHARS)
    assert result.truncated is False
    assert result.char_count == MAX_INPUT_CHARS


def test_process_sets_timestamp():
    sb = SensoryBuffer()
    before = datetime.now(timezone.utc)
    result = sb.process("test")
    after = datetime.now(timezone.utc)
    assert before <= result.timestamp <= after


def test_custom_max_chars():
    sb = SensoryBuffer(max_chars=10)
    result = sb.process("12345678901234")
    assert result.truncated is True
    assert result.char_count == 10


def test_process_empty_string():
    sb = SensoryBuffer()
    result = sb.process("")
    assert result.char_count == 0
    assert result.truncated is False
