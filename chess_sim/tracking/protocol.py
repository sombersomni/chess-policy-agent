"""MetricTracker protocol defining the tracker interface."""
from __future__ import annotations

from typing import Protocol


class MetricTracker(Protocol):
    """Interface for experiment metric tracking.

    Implementations must provide track_step (per-batch),
    track_epoch (per-epoch), and close (flush/seal).

    Example:
        >>> tracker: MetricTracker = NoOpTracker()
        >>> tracker.track_step(0.5, step=1)
        >>> tracker.close()
    """

    def track_step(self, loss: float, step: int) -> None:
        """Log a step-level scalar (e.g. train_loss).

        Args:
            loss: Scalar loss value for this step.
            step: Global training step (1-indexed).
        """
        ...

    def track_epoch(
        self, metrics: dict[str, float], epoch: int, lr: float
    ) -> None:
        """Log epoch-level metrics and learning rate.

        Args:
            metrics: Dict of metric name to value (e.g. val_loss).
            epoch: Current epoch number (1-indexed).
            lr: Current learning rate.
        """
        ...

    def close(self) -> None:
        """Flush pending data and seal the tracking run."""
        ...
