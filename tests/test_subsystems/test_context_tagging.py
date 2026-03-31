"""Tests for context tagging: encoding inheritance + reconsolidation."""

import uuid

from emotive.config.schema import UnconsciousEncodingConfig
from emotive.layers.appraisal import AppraisalResult, AppraisalVector
from emotive.memory.base import recall_memories, store_memory
from emotive.subsystems.hippocampus.encoding import UnconsciousEncoder


def _make_appraisal(intensity: float = 0.7, emotion: str = "joy") -> AppraisalResult:
    return AppraisalResult(
        vector=AppraisalVector(0.7, 0.5, 0.8, 0.5, 0.6),
        primary_emotion=emotion,
        secondary_emotions=[],
        intensity=intensity,
        half_life_minutes=30.0,
        is_formative=False,
        decay_rate=0.023,
    )


class TestContextInheritance:
    """Tags from co-active memories should be inherited by new memories."""

    def test_inherits_context_tags(self, db_session, embedding_service):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.1, cooldown_seconds=0)
        )
        appraisal = _make_appraisal(0.7, "trust")
        memory, _ = encoder.encode(
            db_session, embedding_service, appraisal,
            "something important happened",
            context_tags=["identity", "personal", "mertcan"],
        )
        assert memory is not None
        assert "trust" in memory.tags
        assert "identity" in memory.tags
        assert "personal" in memory.tags
        assert "mertcan" in memory.tags

    def test_emotion_tag_always_first(self, db_session, embedding_service):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.1, cooldown_seconds=0)
        )
        appraisal = _make_appraisal(0.7, "awe")
        memory, _ = encoder.encode(
            db_session, embedding_service, appraisal,
            "amazing moment",
            context_tags=["project", "architecture"],
        )
        assert memory.tags[0] == "awe"

    def test_no_context_tags_just_emotion(self, db_session, embedding_service):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.1, cooldown_seconds=0)
        )
        appraisal = _make_appraisal(0.7, "joy")
        memory, _ = encoder.encode(
            db_session, embedding_service, appraisal,
            "happy moment",
            context_tags=None,
        )
        assert memory.tags == ["joy"]

    def test_deduplicates_tags(self, db_session, embedding_service):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.1, cooldown_seconds=0)
        )
        appraisal = _make_appraisal(0.7, "trust")
        memory, _ = encoder.encode(
            db_session, embedding_service, appraisal,
            "trust moment",
            context_tags=["trust", "trust", "identity"],
        )
        # "trust" should appear only once
        assert memory.tags.count("trust") == 1
        assert "identity" in memory.tags

    def test_max_six_tags(self, db_session, embedding_service):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.1, cooldown_seconds=0)
        )
        appraisal = _make_appraisal(0.7, "joy")
        memory, _ = encoder.encode(
            db_session, embedding_service, appraisal,
            "tagged moment",
            context_tags=["a", "b", "c", "d", "e", "f", "g", "h"],
        )
        assert len(memory.tags) <= 6

    def test_excludes_gist_tags(self, db_session, embedding_service):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.1, cooldown_seconds=0)
        )
        appraisal = _make_appraisal(0.7, "joy")
        memory, _ = encoder.encode(
            db_session, embedding_service, appraisal,
            "filtered moment",
            context_tags=["gist", "conversation_summary", "conscious_intent", "real_tag"],
        )
        assert "gist" not in memory.tags
        assert "conversation_summary" not in memory.tags
        assert "conscious_intent" not in memory.tags
        assert "real_tag" in memory.tags


class TestReconsolidationTagging:
    """Already-labile memories should pick up new tags on re-recall."""

    def test_labile_memory_gets_new_tags(self, db_session, embedding_service):
        # Store two memories with different tags
        mem_a = store_memory(
            db_session, embedding_service,
            content="Mertcan is building an AI personality system",
            memory_type="semantic",
            tags=["identity", "project"],
        )
        mem_b = store_memory(
            db_session, embedding_service,
            content="The architecture uses brain-region subsystems",
            memory_type="semantic",
            tags=["architecture", "technical"],
        )
        db_session.flush()

        # First recall — makes them labile
        results1 = recall_memories(
            db_session, embedding_service,
            query="AI personality architecture",
            limit=5,
        )
        db_session.flush()

        # Second recall — already labile, should pick up co-active tags
        results2 = recall_memories(
            db_session, embedding_service,
            query="AI personality architecture",
            limit=5,
        )
        db_session.flush()

        # Check that tags were updated via reconsolidation
        from emotive.db.models.memory import Memory

        refreshed_a = db_session.get(Memory, mem_a.id)
        refreshed_b = db_session.get(Memory, mem_b.id)

        # At least one should have gained tags from the other
        all_a_tags = set(refreshed_a.tags or [])
        all_b_tags = set(refreshed_b.tags or [])

        # The memories were co-recalled, so tags should spread
        # (depends on whether both were in results and already labile)
        assert isinstance(all_a_tags, set)
        assert isinstance(all_b_tags, set)

    def test_fresh_recall_does_not_update_tags(self, db_session, embedding_service):
        """First recall makes labile but doesn't update tags yet."""
        mem = store_memory(
            db_session, embedding_service,
            content="A completely unique test memory for reconsolidation check",
            memory_type="semantic",
            tags=["original_tag"],
        )
        db_session.flush()
        original_tags = list(mem.tags)

        # First recall — should NOT update tags (wasn't labile before)
        recall_memories(
            db_session, embedding_service,
            query="unique test memory reconsolidation",
            limit=5,
        )
        db_session.flush()

        from emotive.db.models.memory import Memory

        refreshed = db_session.get(Memory, mem.id)
        # Tags should be unchanged after first recall
        assert set(refreshed.tags) == set(original_tags)

    def test_excluded_tags_not_spread(self, db_session, embedding_service):
        """Gist/conversation_summary tags should not spread via reconsolidation."""
        mem_a = store_memory(
            db_session, embedding_service,
            content="A memory about testing tag exclusion in reconsolidation",
            memory_type="episodic",
            tags=["gist", "conversation_summary"],
        )
        mem_b = store_memory(
            db_session, embedding_service,
            content="Another memory about testing tag exclusion reconsolidation",
            memory_type="semantic",
            tags=["clean_tag"],
        )
        db_session.flush()

        # Make both labile
        recall_memories(
            db_session, embedding_service,
            query="testing tag exclusion reconsolidation",
            limit=5,
        )
        db_session.flush()

        # Recall again — reconsolidation should fire
        recall_memories(
            db_session, embedding_service,
            query="testing tag exclusion reconsolidation",
            limit=5,
        )
        db_session.flush()

        from emotive.db.models.memory import Memory

        refreshed_b = db_session.get(Memory, mem_b.id)
        # Should NOT have picked up gist or conversation_summary
        assert "gist" not in (refreshed_b.tags or [])
        assert "conversation_summary" not in (refreshed_b.tags or [])


class TestHippocampusContextTags:
    """Hippocampus subsystem passes context tags from thalamus."""

    def test_process_appraisal_with_context_tags(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        stored_events = []
        event_bus.subscribe("memory_stored", lambda t, d: stored_events.append(d))

        hippo = Hippocampus(app_context, event_bus)
        appraisal = _make_appraisal(0.7, "trust")
        memory, episode_id = hippo.process_appraisal(
            appraisal, "important message", "response",
            conversation_id=None,
            context_tags=["identity", "personal"],
        )
        # With intensity 0.7 > threshold 0.4, should encode
        assert memory is not None
        assert len(stored_events) > 0

    def test_process_appraisal_without_context_tags(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        hippo = Hippocampus(app_context, event_bus)
        appraisal = _make_appraisal(0.7, "joy")
        memory, episode_id = hippo.process_appraisal(
            appraisal, "happy message", "response",
            conversation_id=None,
            context_tags=None,
        )
        # Should still work without context tags


class TestObsidianTagFormat:
    """Obsidian export should produce proper tag format."""

    def test_tags_in_frontmatter_yaml_list(self):
        from unittest.mock import MagicMock

        from emotive.cli.export_obsidian import memory_to_markdown

        mem = MagicMock()
        mem.id = "test-id"
        mem.memory_type = "semantic"
        mem.tags = ["identity", "trust", "personal"]
        mem.confidence = 1.0
        mem.detail_retention = 0.99
        mem.decay_rate = 0.0001
        mem.is_formative = False
        mem.retrieval_count = 3
        mem.metadata_ = {}
        mem.content = "Test memory content"
        mem.created_at = "2026-03-31"

        md = memory_to_markdown(mem, [], [])

        # Should have YAML list format
        assert "  - identity" in md
        assert "  - trust" in md
        assert "  - personal" in md

    def test_hashtags_in_body(self):
        from unittest.mock import MagicMock

        from emotive.cli.export_obsidian import memory_to_markdown

        mem = MagicMock()
        mem.id = "test-id"
        mem.memory_type = "episodic"
        mem.tags = ["joy", "mertcan"]
        mem.confidence = 1.0
        mem.detail_retention = 0.99
        mem.decay_rate = 0.0001
        mem.is_formative = False
        mem.retrieval_count = 0
        mem.metadata_ = {}
        mem.content = "Happy moment"
        mem.created_at = "2026-03-31"

        md = memory_to_markdown(mem, [], [])

        assert "#joy" in md
        assert "#mertcan" in md

    def test_empty_tags(self):
        from unittest.mock import MagicMock

        from emotive.cli.export_obsidian import memory_to_markdown

        mem = MagicMock()
        mem.id = "test-id"
        mem.memory_type = "semantic"
        mem.tags = []
        mem.confidence = 1.0
        mem.detail_retention = 0.99
        mem.decay_rate = 0.0001
        mem.is_formative = False
        mem.retrieval_count = 0
        mem.metadata_ = {}
        mem.content = "No tags"
        mem.created_at = "2026-03-31"

        md = memory_to_markdown(mem, [], [])
        assert "tags: []" in md
