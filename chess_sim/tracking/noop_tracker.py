"""NoOpTracker: silent fallback when aim is absent or disabled."""
from __future__ import annotations


class NoOpTracker:
    """Silent no-op metric tracker; zero external dependencies.

    All methods accept the same arguments as MetricTracker but
    perform no work. Used when aim is not installed or tracking
    is disabled via AimConfig.enabled=False.

    Example:
        >>> tracker = NoOpTracker()
        >>> tracker.track_step(0.5, step=1)  # silent
        >>> tracker.close()  # silent
    """

    def track_step(self, loss: float, step: int) -> None:
        """Accept and discard step-level metrics."""
        pass

    def track_epoch(
        self, metrics: dict[str, float], epoch: int, lr: float
    ) -> None:
        """Accept and discard epoch-level metrics."""
        pass

    def log_text(
        self, message: str, step: int | None = None
    ) -> None:
        """Accept and discard log text."""
        pass

    def track_image(
        self,
        fig: object,
        name: str,
        step: int | None = None,
    ) -> None:
        """Accept and discard image artifacts."""
        pass

    def close(self) -> None:
        """No-op close; nothing to flush."""
        pass
