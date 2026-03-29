"""Audit config changes to Postgres."""

from sqlalchemy.orm import Session

from emotive.db.models.config_changes import ConfigChange


def audit_config_change(
    session: Session,
    key: str,
    old_value: object,
    new_value: object,
    reason: str | None = None,
) -> ConfigChange:
    """Record a config change in the database audit log."""
    entry = ConfigChange(
        config_key=key,
        old_value=_to_jsonb(old_value),
        new_value=_to_jsonb(new_value),
        reason=reason,
    )
    session.add(entry)
    session.flush()
    return entry


def _to_jsonb(value: object) -> object:
    """Wrap scalar values so they're valid JSONB."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    # Wrap scalars in a dict for JSONB compatibility
    return {"value": value}
