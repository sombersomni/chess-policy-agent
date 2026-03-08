"""Factory for creating MetricTracker instances."""
from __future__ import annotations

import logging

from chess_sim.config import AimConfig
from chess_sim.tracking.protocol import MetricTracker

logger = logging.getLogger(__name__)


def make_tracker(cfg: AimConfig) -> MetricTracker:
    """Return AimTracker when aim is installed and cfg.enabled, else NoOpTracker.

    Never raises ImportError. Falls back to NoOpTracker with a warning log
    when aim is not available or when cfg.enabled is False.

    Args:
        cfg: AimConfig with enabled flag and experiment settings.

    Returns:
        MetricTracker implementation (AimTracker or NoOpTracker).

    Example:
        >>> from chess_sim.config import AimConfig
        >>> tracker = make_tracker(AimConfig(enabled=True))

    Edge cases:
        If aim package is not installed, returns NoOpTracker and logs
        a warning instead of raising ImportError.
    """
    from chess_sim.tracking.noop_tracker import NoOpTracker

    if not cfg.enabled:
        return NoOpTracker()
    try:
        import aim  # noqa: F401
    except ImportError:
        logger.warning(
            "aim not installed; falling back to NoOpTracker"
        )
        return NoOpTracker()

    from chess_sim.tracking.aim_tracker import AimTracker
    return AimTracker(cfg)
