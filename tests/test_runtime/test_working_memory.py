"""Tests for working memory."""

from emotive.runtime.working_memory import WorkingMemory, WorkingMemoryItem


def test_add_item_below_capacity():
    wm = WorkingMemory(capacity=5)
    evicted = wm.add(WorkingMemoryItem(content="a"))
    assert evicted is None
    assert wm.size == 1


def test_add_item_at_capacity_evicts_oldest():
    wm = WorkingMemory(capacity=2)
    wm.add(WorkingMemoryItem(content="first"))
    wm.add(WorkingMemoryItem(content="second"))
    evicted = wm.add(WorkingMemoryItem(content="third"))
    assert evicted is not None
    assert evicted.content == "first"
    assert wm.size == 2


def test_eviction_publishes_event(event_bus):
    received = []
    event_bus.subscribe("working_memory_evicted", lambda t, d: received.append(d))
    wm = WorkingMemory(capacity=1, event_bus=event_bus)
    wm.add(WorkingMemoryItem(content="a"))
    wm.add(WorkingMemoryItem(content="b"))
    assert len(received) == 1
    assert received[0]["content"] == "a"


def test_get_all_returns_all_items():
    wm = WorkingMemory(capacity=10)
    for i in range(3):
        wm.add(WorkingMemoryItem(content=f"item-{i}"))
    items = wm.get_all()
    assert len(items) == 3
    assert items[0].content == "item-0"


def test_get_above_threshold():
    wm = WorkingMemory(capacity=10)
    wm.add(WorkingMemoryItem(content="low", significance=0.1))
    wm.add(WorkingMemoryItem(content="mid", significance=0.5))
    wm.add(WorkingMemoryItem(content="high", significance=0.9))
    above = wm.get_above_threshold(0.5)
    assert len(above) == 2
    assert all(i.significance >= 0.5 for i in above)


def test_clear_returns_items_and_empties_buffer():
    wm = WorkingMemory(capacity=10)
    wm.add(WorkingMemoryItem(content="a"))
    wm.add(WorkingMemoryItem(content="b"))
    cleared = wm.clear()
    assert len(cleared) == 2
    assert wm.size == 0


def test_size_and_capacity_properties():
    wm = WorkingMemory(capacity=7)
    assert wm.capacity == 7
    assert wm.size == 0
    wm.add(WorkingMemoryItem(content="x"))
    assert wm.size == 1


def test_working_memory_item_defaults():
    item = WorkingMemoryItem(content="test")
    assert item.significance == 0.5
    assert item.tags == []
    assert item.metadata == {}
    assert item.timestamp is not None
