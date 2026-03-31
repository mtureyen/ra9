"""Tests for the Prefrontal Cortex subsystem (buffer + context)."""

from emotive.layers.appraisal import AppraisalResult, AppraisalVector
from emotive.subsystems.dmn.schema import SelfSchema
from emotive.subsystems.prefrontal.buffer import (
    ConversationBuffer,
    ConversationTurn,
    compress_to_gist,
)
from emotive.subsystems.prefrontal.context import build_messages, build_system_prompt


class TestConversationBuffer:
    def test_add_turn(self):
        buf = ConversationBuffer(buffer_size=6, primacy_pins=2)
        evicted = buf.add_turn("user", "hello")
        assert evicted == []
        assert len(buf.get_active_turns()) == 1

    def test_primacy_pinning(self):
        buf = ConversationBuffer(buffer_size=4, primacy_pins=2)
        buf.add_turn("user", "first")
        buf.add_turn("assistant", "second")
        assert len(buf._pinned) == 2

        # Add more turns — pinned turns stay
        for i in range(10):
            buf.add_turn("user", f"turn {i}")

        active = buf.get_active_turns()
        assert active[0].content == "first"
        assert active[1].content == "second"

    def test_eviction(self):
        buf = ConversationBuffer(buffer_size=4, primacy_pins=2)
        # Fill pinned slots
        buf.add_turn("user", "pinned1")
        buf.add_turn("assistant", "pinned2")

        # Fill remaining active slots (4 - 2 pinned = 2 active)
        buf.add_turn("user", "active1")
        buf.add_turn("assistant", "active2")

        # This should evict "active1"
        evicted = buf.add_turn("user", "active3")
        assert len(evicted) == 1
        assert evicted[0].content == "active1"

    def test_full_session_tracks_everything(self):
        buf = ConversationBuffer(buffer_size=4, primacy_pins=1)
        for i in range(10):
            buf.add_turn("user", f"msg {i}")
        assert len(buf.get_full_session()) == 10

    def test_clear(self):
        buf = ConversationBuffer()
        buf.add_turn("user", "test")
        buf.clear()
        assert len(buf.get_active_turns()) == 0
        assert len(buf.get_full_session()) == 0

    def test_no_eviction_when_under_capacity(self):
        buf = ConversationBuffer(buffer_size=10, primacy_pins=2)
        for i in range(8):
            evicted = buf.add_turn("user", f"msg {i}")
        # Total turns = 8, pinned = 2, active = 6, capacity = 10-2 = 8
        assert len(buf.get_active_turns()) == 8


class TestGistCompression:
    def test_basic_gist(self):
        turns = [
            ConversationTurn(role="user", content="What's the weather?"),
            ConversationTurn(role="assistant", content="It's sunny today."),
        ]
        gist = compress_to_gist(turns)
        assert "Conversation summary:" in gist
        assert "weather" in gist.lower()
        assert "sunny" in gist.lower()

    def test_long_content_truncated(self):
        turns = [
            ConversationTurn(role="user", content="x" * 500),
        ]
        gist = compress_to_gist(turns)
        assert "..." in gist
        assert len(gist) < 600

    def test_empty_turns(self):
        gist = compress_to_gist([])
        assert "Conversation summary:" in gist


class TestContextBuilder:
    def test_build_messages(self):
        turns = [
            ConversationTurn(role="user", content="hi"),
            ConversationTurn(role="assistant", content="hello"),
        ]
        msgs = build_messages(turns)
        assert msgs == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_system_prompt_minimal(self):
        prompt = build_system_prompt()
        assert "Ryo" in prompt
        assert "naturally" in prompt.lower()

    def test_system_prompt_with_schema(self):
        schema = SelfSchema(
            traits={"curious": 0.8, "direct": 0.7},
            core_facts=["My name is Ryo"],
            active_values=["continuity"],
            person_context={"mertcan": {"role": "creator"}},
        )
        prompt = build_system_prompt(self_schema=schema)
        assert "Ryo" in prompt
        assert "curious" in prompt
        assert "continuity" in prompt
        assert "mertcan" in prompt

    def test_system_prompt_with_emotional_state(self):
        state = AppraisalResult(
            vector=AppraisalVector(0.7, 0.5, 0.8, 0.5, 0.6),
            primary_emotion="joy",
            secondary_emotions=["trust"],
            intensity=0.6,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )
        prompt = build_system_prompt(emotional_state=state)
        assert "joy" in prompt
        assert "0.60" in prompt

    def test_system_prompt_with_memories(self):
        memories = [
            {"content": "Mertcan is building ra9", "memory_type": "semantic"},
        ]
        prompt = build_system_prompt(recalled_memories=memories)
        assert "ra9" in prompt
        assert "semantic" in prompt

    def test_system_prompt_with_temperament(self):
        temp = {
            "novelty_seeking": 0.8,  # notable (far from 0.5)
            "social_bonding": 0.5,   # not notable (baseline)
            "analytical_depth": 0.5,
            "playfulness": 0.5,
            "caution": 0.5,
            "expressiveness": 0.5,
            "sensitivity": 0.5,
            "resilience": 0.5,
        }
        prompt = build_system_prompt(temperament=temp)
        assert "novelty" in prompt.lower()

    def test_system_prompt_episodes(self):
        episodes = [
            {"primary_emotion": "awe", "intensity": 0.7, "trigger_event": "read the doc"},
        ]
        prompt = build_system_prompt(active_episodes=episodes)
        assert "awe" in prompt


class TestPrefrontalSubsystem:
    def test_initialization(self, app_context, event_bus):
        from emotive.subsystems.prefrontal import PrefrontalCortex

        pfc = PrefrontalCortex(app_context, event_bus)
        assert pfc.name == "prefrontal_cortex"

    def test_add_turn_and_build_context(self, app_context, event_bus):
        from emotive.subsystems.prefrontal import PrefrontalCortex

        pfc = PrefrontalCortex(app_context, event_bus)
        pfc.add_turn("user", "hello")
        pfc.add_turn("assistant", "hi there")

        system, messages = pfc.build_context()
        assert "Ryo" in system
        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_clear(self, app_context, event_bus):
        from emotive.subsystems.prefrontal import PrefrontalCortex

        pfc = PrefrontalCortex(app_context, event_bus)
        pfc.add_turn("user", "test")
        pfc.clear()
        _, messages = pfc.build_context()
        assert len(messages) == 0
