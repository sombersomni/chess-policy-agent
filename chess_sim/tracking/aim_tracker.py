"""AimTracker: wraps aim.Run to implement MetricTracker."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aim  # noqa: F401

from chess_sim.config import AimConfig


class AimTracker:
    """Wraps aim.Run; logs step-level and epoch-level scalars.

    Constructed from an AimConfig. Opens an aim.Run on init, stores
    hyperparameters, and provides track_step / track_epoch / close.

    Args:
        cfg: AimConfig with experiment settings.
        run: Injected aim.Run instance (for testing); if None,
             constructed internally from cfg.

    Example:
        >>> from chess_sim.config import AimConfig
        >>> tracker = AimTracker(AimConfig(enabled=True))
        >>> tracker.track_step(0.5, step=10)
        >>> tracker.close()
    """

    def __init__(self, cfg: AimConfig, run: Any = None) -> None:
        """Initialize AimTracker with config and optional injected Run.

        Opens an aim.Run (or uses the injected one). Stores hparams
        on the run as run["hparams"]. Saves log_every_n_steps from cfg.

        Args:
            cfg: AimConfig with experiment name, repo, and logging freq.
            run: Optional pre-built aim.Run for dependency injection.

        Raises:
            NotImplementedError: Stub — to be implemented.
        """
        self._log_every_n: int = cfg.log_every_n_steps
        if run is not None:
            self._run = run
        else:
            import aim
            self._run = aim.Run(
                experiment=cfg.experiment_name,
                repo=cfg.repo,
                log_system_params=False,
                capture_terminal_logs=False,
            )
        self._run["hparams"] = {
            "experiment": cfg.experiment_name,
            "repo": cfg.repo,
            "log_every_n_steps": cfg.log_every_n_steps,
        }

    def track_step(self, loss: float, step: int) -> None:
        """Log train_loss at step if step > 0 and step % N == 0.

        Calls run.track(loss, name="train_loss", step=step) only when
        the step is a positive multiple of log_every_n_steps.

        Args:
            loss: Scalar training loss for this step.
            step: Global training step (1-indexed).

        Example:
            >>> tracker.track_step(0.42, step=50)

        Edge cases:
            Step 0 is never logged (pre-training guard).
        """
        if step > 0 and step % self._log_every_n == 0:
            self._run.track(loss, name="train_loss", step=step)

    def track_epoch(
        self, metrics: dict[str, float], epoch: int, lr: float
    ) -> None:
        """Log all metric keys + lr at epoch boundary.

        Iterates the metrics dict and calls run.track(v, name=k, epoch=epoch)
        for each entry. Also logs the learning rate as a separate metric.

        Args:
            metrics: Dict of metric name to float value.
            epoch: Current epoch number (1-indexed).
            lr: Current optimizer learning rate.

        Example:
            >>> tracker.track_epoch({"val_loss": 2.1}, epoch=1, lr=3e-4)
        """
        for k, v in metrics.items():
            self._run.track(v, name=k, epoch=epoch)
        self._run.track(lr, name="lr", epoch=epoch)

    def close(self) -> None:
        """Flush and seal the aim Run.

        Calls run.close() to persist all tracked data. Safe to call
        multiple times (idempotent in aim SDK).

        Example:
            >>> tracker.close()
        """
        self._run.close()
