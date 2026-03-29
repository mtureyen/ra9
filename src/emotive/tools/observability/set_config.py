"""set_config: Update configuration with audit trail."""

from __future__ import annotations

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.config.audit import audit_config_change
from emotive.runtime.event_bus import CONFIG_CHANGED


async def set_config_tool(
    ctx: Context,
    key: str,
    value: object,
    reason: str,
) -> dict:
    """Update a configuration setting. Every change is audited.

    Args:
        key: Config key using dot notation (e.g., "consolidation.significance_threshold").
        value: New value.
        reason: Why this change is being made (required for research traceability).
    """
    app: AppContext = ctx.lifespan_context
    config = app.config_manager.get()

    # Navigate to the config field
    parts = key.split(".")
    obj = config
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return {
                "status": "error",
                "error": "invalid_key",
                "message": f"Config key '{key}' not found (failed at '{part}')",
            }

    final_key = parts[-1]
    if not hasattr(obj, final_key):
        return {
            "status": "error",
            "error": "invalid_key",
            "message": f"Config key '{key}' not found (no attribute '{final_key}')",
        }

    old_value = getattr(obj, final_key)

    # Update the config object
    try:
        setattr(obj, final_key, value)
        # Re-validate by reconstructing
        new_config = config.model_copy()
    except Exception as e:
        # Revert
        setattr(obj, final_key, old_value)
        return {
            "status": "error",
            "error": "validation_failed",
            "message": str(e),
        }

    # Save to disk
    app.config_manager.save(new_config)

    # Audit to database
    session = app.session_factory()
    try:
        audit_config_change(session, key, old_value, value, reason=reason)
        session.commit()
    finally:
        session.close()

    app.event_bus.publish(
        CONFIG_CHANGED,
        {"key": key, "old_value": str(old_value), "new_value": str(value), "reason": reason},
    )

    return {
        "status": "ok",
        "data": {
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "reason": reason,
        },
    }
