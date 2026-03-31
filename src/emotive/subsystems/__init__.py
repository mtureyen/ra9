"""Brain region subsystems for the cognitive pipeline.

Each subsystem models a brain region: it owns its state, registers event
handlers on the EventBus, and exposes methods the Thalamus can call directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus


class Subsystem:
    """Base class for brain region subsystems.

    Subsystems are initialized with shared services (AppContext) and the
    EventBus (nervous system). They register event handlers in
    _register_handlers() and expose methods for the Thalamus to call.
    """

    name: str = "base"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        self._app = app
        self._bus = event_bus
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Override to subscribe to events on the EventBus."""
