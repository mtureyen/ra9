"""Tests for procedural encoding of behavioral coaching (Design Decision #12)."""

from emotive.subsystems.hippocampus.encoding import detect_behavioral_coaching


class TestBehavioralCoachingDetection:
    def test_be_more_casual(self):
        assert detect_behavioral_coaching("be more casual and relaxed")

    def test_use_lowercase(self):
        assert detect_behavioral_coaching("use lowercase from now on")

    def test_dont_ask_questions(self):
        assert detect_behavioral_coaching("don't be so formal")

    def test_remember_to(self):
        assert detect_behavioral_coaching("remember to always be friendly")

    def test_try_to_be(self):
        assert detect_behavioral_coaching("try to be more natural")

    def test_talk_like(self):
        assert detect_behavioral_coaching("talk like a normal person")

    def test_from_now_on(self):
        assert detect_behavioral_coaching("from now on speak casually")

    def test_normal_content_not_coaching(self):
        assert not detect_behavioral_coaching("what do you think about love")

    def test_question_not_coaching(self):
        assert not detect_behavioral_coaching("how are you feeling today")

    def test_factual_not_coaching(self):
        assert not detect_behavioral_coaching("the weather is nice")


class TestProceduralInContext:
    def test_learned_behaviors_in_prompt(self):
        from emotive.subsystems.prefrontal.context import build_system_prompt

        procedural = [
            {"content": "Be more casual and use lowercase", "memory_type": "procedural"},
            {"content": "Don't ask questions at the end of every response", "memory_type": "procedural"},
        ]
        prompt = build_system_prompt(procedural_memories=procedural)
        assert "Behavioral Instructions" in prompt
        assert "casual" in prompt
        assert "lowercase" in prompt

    def test_no_procedural_no_section(self):
        from emotive.subsystems.prefrontal.context import build_system_prompt

        prompt = build_system_prompt(procedural_memories=None)
        assert "Behavioral Instructions" not in prompt

    def test_empty_procedural_no_section(self):
        from emotive.subsystems.prefrontal.context import build_system_prompt

        prompt = build_system_prompt(procedural_memories=[])
        assert "Behavioral Instructions" not in prompt
