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

    def log_text(
        self, message: str, step: int | None = None
    ) -> None:
        """Send a human-readable log message to the tracker.

        Args:
            message: Formatted log string to display in the
                tracker UI.
            step: Optional global step to associate with the
                message.
        """
        ...

    def track_scalars(
        self, metrics: dict[str, float], step: int
    ) -> None:
        """Log multiple named scalars at the same step.

        Args:
            metrics: Dict of metric name to float value.
            step: Global training step to associate with these
                scalars.
        """
        ...

    def track_image(
        self,
        fig: object,
        name: str,
        step: int | None = None,
    ) -> None:
        """Log a matplotlib Figure as an image artifact.

        Args:
            fig: A matplotlib Figure to serialize and log.
            name: Metric name / display label for the image.
            step: Optional global step or epoch to associate.
        """
        ...

    def close(self) -> None:
        """Flush pending data and seal the tracking run."""
        ...
