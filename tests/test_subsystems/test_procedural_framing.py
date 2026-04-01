"""Tests for procedural memory instruction framing."""

from emotive.subsystems.prefrontal.context import _format_procedural, build_system_prompt


class TestProceduralInstructionFraming:
    def test_uses_behavioral_instructions_header(self):
        memories = [{"content": "Use lowercase always", "memory_type": "procedural"}]
        result = _format_procedural(memories)
        assert "Behavioral Instructions" in result
        assert "Learned Behaviors" not in result

    def test_no_behavior_tag_prefix(self):
        memories = [{"content": "Be casual and warm", "memory_type": "procedural"}]
        result = _format_procedural(memories)
        # Should NOT have [behavior] prefix — it's an instruction, not a recall
        assert "[behavior]" not in result
        assert "Be casual and warm" in result

    def test_follow_naturally_instruction(self):
        memories = [{"content": "Use lowercase", "memory_type": "procedural"}]
        result = _format_procedural(memories)
        assert "Follow them naturally" in result

    def test_multiple_instructions(self):
        memories = [
            {"content": "Use lowercase always", "memory_type": "procedural"},
            {"content": "Don't end with questions", "memory_type": "procedural"},
            {"content": "Be warm and casual", "memory_type": "procedural"},
        ]
        result = _format_procedural(memories)
        assert "lowercase" in result
        assert "questions" in result
        assert "warm" in result

    def test_in_system_prompt(self):
        memories = [
            {"content": "Use lowercase", "memory_type": "procedural"},
        ]
        prompt = build_system_prompt(procedural_memories=memories)
        assert "Behavioral Instructions" in prompt
        assert "Use lowercase" in prompt
