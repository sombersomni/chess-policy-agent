"""AimLogHandler: routes Python logging records to a MetricTracker."""
from __future__ import annotations

import logging

from chess_sim.tracking.protocol import MetricTracker


class _ExcludeAimFilter(logging.Filter):
    """Drops log records from aim's own logger to prevent recursion."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for aim-internal records, True otherwise.

        Args:
            record: The log record to evaluate.

        Returns:
            False if the record originates from the aim library.
        """
        return not record.name.startswith("aim")


class AimLogHandler(logging.Handler):
    """Logging handler forwarding records to a MetricTracker.

    Attach to the root logger after creating a tracker. On emit(),
    the record is formatted and sent via tracker.log_text(). Records
    below the handler's level are silently dropped by the base class
    before emit() is called.

    Args:
        tracker: Any MetricTracker implementation.
        level: Minimum logging level; defaults to logging.INFO.
    """

    def __init__(
        self,
        tracker: MetricTracker,
        level: int = logging.INFO,
    ) -> None:
        """Initialise the handler with a tracker and minimum level.

        Args:
            tracker: MetricTracker to receive formatted log text.
            level: Minimum log level (default: INFO).
        """
        super().__init__(level)
        self._tracker = tracker
        self.addFilter(_ExcludeAimFilter())

    def emit(self, record: logging.LogRecord) -> None:
        """Format the record and forward it to the tracker.

        Calls self.format(record) to apply any attached Formatter,
        then delegates to tracker.log_text(). If formatting or
        tracking raises, handleError() is called rather than
        propagating the exception.

        Args:
            record: The log record produced by a Logger.
        """
        try:
            message = self.format(record)
            self._tracker.log_text(message)
        except Exception:  # noqa: BLE001
            self.handleError(record)
