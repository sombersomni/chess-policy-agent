"""Unit tests for aim experiment tracking integration — TC01 through TC15.

Tests cover: make_tracker factory, AimTracker, NoOpTracker,
Phase1Trainer tracker integration, AimConfig YAML loading,
and _mean_entropy computation.

All tests run on CPU only (per project convention).
"""

from __future__ import annotations

import logging
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from chess_sim.config import AimConfig, load_v2_config
from chess_sim.tracking.aim_tracker import AimTracker
from chess_sim.tracking.noop_tracker import NoOpTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(content: str) -> Path:
    """Write a YAML string to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    )
    tmp.write(textwrap.dedent(content))
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# TC01-TC03: make_tracker factory
# ---------------------------------------------------------------------------

class TestMakeTracker(unittest.TestCase):
    """Tests for the make_tracker factory function."""

    def test_tc01_enabled_returns_aim_tracker(self) -> None:
        """TC01: make_tracker(AimConfig(enabled=True)) returns AimTracker."""
        from chess_sim.tracking.factory import make_tracker

        cfg = AimConfig(enabled=True)
        tracker = make_tracker(cfg)
        self.assertIsInstance(tracker, AimTracker)

    def test_tc02_disabled_returns_noop_no_aim_import(self) -> None:
        """TC02: make_tracker(AimConfig(enabled=False)) returns NoOpTracker.

        No aim import is attempted when enabled=False.
        """
        from chess_sim.tracking.factory import make_tracker

        cfg = AimConfig(enabled=False)
        tracker = make_tracker(cfg)
        self.assertIsInstance(tracker, NoOpTracker)

    def test_tc03_aim_absent_returns_noop_with_warning(self) -> None:
        """TC03: make_tracker when aim not installed returns NoOpTracker.

        A warning is logged; no ImportError propagates.
        """
        from chess_sim.tracking.factory import make_tracker

        cfg = AimConfig(enabled=True)
        with patch.dict("sys.modules", {"aim": None}):
            with self.assertLogs(
                "chess_sim.tracking.factory", level="WARNING"
            ) as cm:
                tracker = make_tracker(cfg)
        self.assertIsInstance(tracker, NoOpTracker)
        self.assertTrue(
            any("aim" in msg.lower() for msg in cm.output),
            f"Expected warning about aim, got: {cm.output}",
        )


# ---------------------------------------------------------------------------
# TC04-TC07: AimTracker
# ---------------------------------------------------------------------------

class TestAimTracker(unittest.TestCase):
    """Tests for AimTracker wrapping aim.Run."""

    def test_tc04_track_step_respects_log_every_n(self) -> None:
        """TC04: track_step calls run.track only on step % N == 0 (not step 0).

        With log_every_n_steps=10, calling track_step for steps 1..25
        should invoke run.track exactly 2 times (steps 10 and 20).
        """
        cfg = AimConfig(enabled=True, log_every_n_steps=10)
        mock_run = MagicMock()
        tracker = AimTracker(cfg, run=mock_run)
        for step in range(1, 26):
            tracker.track_step(0.5, step)
        self.assertEqual(mock_run.track.call_count, 2)

    def test_tc05_track_step_zero_never_calls_run(self) -> None:
        """TC05: track_step(step=0) never calls run.track.

        Step 0 is pre-training; the guard is step > 0 AND step % N == 0.
        """
        cfg = AimConfig(enabled=True, log_every_n_steps=1)
        mock_run = MagicMock()
        tracker = AimTracker(cfg, run=mock_run)
        tracker.track_step(0.5, step=0)
        mock_run.track.assert_not_called()

    def test_tc06_track_epoch_logs_all_keys_plus_lr(self) -> None:
        """TC06: track_epoch calls run.track once per metric key + once for lr.

        With 3 metrics + lr = 4 calls total.
        """
        cfg = AimConfig(enabled=True)
        mock_run = MagicMock()
        tracker = AimTracker(cfg, run=mock_run)
        metrics = {
            "val_loss": 1.0,
            "val_accuracy": 0.5,
            "mean_entropy": 1.2,
        }
        tracker.track_epoch(metrics, epoch=3, lr=1e-4)
        self.assertEqual(mock_run.track.call_count, 4)

    def test_tc07_init_sets_hparams_once(self) -> None:
        """TC07: AimTracker.__init__ sets run['hparams'] exactly once."""
        cfg = AimConfig(enabled=True, experiment_name="test")
        mock_run = MagicMock()
        _ = AimTracker(cfg, run=mock_run)
        mock_run.__setitem__.assert_called_once()
        args = mock_run.__setitem__.call_args
        self.assertEqual(args[0][0], "hparams")


# ---------------------------------------------------------------------------
# TC08: NoOpTracker
# ---------------------------------------------------------------------------

class TestNoOpTracker(unittest.TestCase):
    """Tests for the NoOpTracker fallback."""

    def test_tc08_all_methods_return_none_no_exception(self) -> None:
        """TC08: track_step, track_epoch, and close all return None."""
        tracker = NoOpTracker()
        result_step = tracker.track_step(0.5, step=1)
        self.assertIsNone(result_step)
        result_epoch = tracker.track_epoch(
            {"val_loss": 1.0}, epoch=1, lr=3e-4
        )
        self.assertIsNone(result_epoch)
        result_close = tracker.close()
        self.assertIsNone(result_close)


# ---------------------------------------------------------------------------
# TC12-TC13: AimConfig YAML loading
# ---------------------------------------------------------------------------

class TestAimConfigYAML(unittest.TestCase):
    """Tests for AimConfig integration in load_v2_config."""

    def test_tc12_aim_section_loads_log_every(self) -> None:
        """TC12: AimConfig loads log_every_n_steps=100 from YAML aim section."""
        path = _write_yaml("""
            aim:
              enabled: true
              experiment_name: test_exp
              log_every_n_steps: 100
        """)
        cfg = load_v2_config(path)
        self.assertEqual(cfg.aim.log_every_n_steps, 100)
        self.assertTrue(cfg.aim.enabled)
        self.assertEqual(cfg.aim.experiment_name, "test_exp")

    def test_tc13_missing_aim_section_uses_defaults(self) -> None:
        """TC13: load_v2_config on YAML missing aim: section uses AimConfig defaults."""
        path = _write_yaml("""
            trainer:
              epochs: 5
        """)
        cfg = load_v2_config(path)
        default = AimConfig()
        self.assertEqual(cfg.aim.enabled, default.enabled)
        self.assertEqual(
            cfg.aim.experiment_name, default.experiment_name
        )
        self.assertEqual(cfg.aim.repo, default.repo)
        self.assertEqual(
            cfg.aim.log_every_n_steps, default.log_every_n_steps
        )


# ---------------------------------------------------------------------------
# TC16-TC19: AimLogHandler
# ---------------------------------------------------------------------------

class TestAimLogHandler(unittest.TestCase):
    """Tests for the AimLogHandler logging integration."""

    def test_tc16_emit_calls_log_text(self) -> None:
        """TC16: AimLogHandler.emit() calls tracker.log_text."""
        from chess_sim.tracking.log_handler import AimLogHandler

        mock_tracker = MagicMock()
        handler = AimLogHandler(mock_tracker)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        handler.emit(record)
        mock_tracker.log_text.assert_called_once()
        call_msg = mock_tracker.log_text.call_args[0][0]
        self.assertIn("hello world", call_msg)

    def test_tc17_emit_handles_tracker_exception(self) -> None:
        """TC17: emit() calls handleError when log_text raises."""
        from chess_sim.tracking.log_handler import AimLogHandler

        mock_tracker = MagicMock()
        mock_tracker.log_text.side_effect = RuntimeError(
            "aim down"
        )
        handler = AimLogHandler(mock_tracker)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="msg",
            args=(),
            exc_info=None,
        )
        with patch.object(
            handler, "handleError"
        ) as mock_handle_err:
            handler.emit(record)
            mock_handle_err.assert_called_once_with(record)

    def test_tc18_noop_log_text_returns_none(self) -> None:
        """TC18: NoOpTracker.log_text() silently discards."""
        tracker = NoOpTracker()
        result = tracker.log_text("some message", step=5)
        self.assertIsNone(result)

    def test_tc19_aim_tracker_log_text_tracks_text(self) -> None:
        """TC19: AimTracker.log_text calls run.track with aim.Text."""
        try:
            import aim
        except ImportError:
            self.skipTest("aim not installed")

        cfg = AimConfig(enabled=True)
        mock_run = MagicMock()
        tracker = AimTracker(cfg, run=mock_run)
        tracker.log_text("epoch started", step=1)
        mock_run.track.assert_called_once()
        tracked_obj = mock_run.track.call_args[0][0]
        self.assertIsInstance(tracked_obj, aim.Text)
        self.assertEqual(
            mock_run.track.call_args[1]["name"], "logs"
        )
        self.assertEqual(
            mock_run.track.call_args[1]["step"], 1
        )


# ---------------------------------------------------------------------------
# TC20-TC22: track_scalars and track_epoch step fix
# ---------------------------------------------------------------------------

class TestTrackScalars(unittest.TestCase):
    """Tests for track_scalars and the track_epoch step= fix."""

    def test_tc20_track_epoch_passes_step_equal_epoch(
        self,
    ) -> None:
        """TC20: track_epoch passes step=epoch to every run.track call.

        With 2 metric keys + lr = 3 calls, each must include step=2.
        """
        cfg = AimConfig(enabled=True)
        mock_run = MagicMock()
        tracker = AimTracker(cfg, run=mock_run)
        tracker.track_epoch(
            {"val_loss": 1.0, "pg_loss": 0.5},
            epoch=2,
            lr=1e-4,
        )
        for call in mock_run.track.call_args_list:
            kwargs = call[1]
            self.assertIn(
                "step",
                kwargs,
                f"Expected step= in track call kwargs, "
                f"got: {kwargs}",
            )
            self.assertEqual(kwargs["step"], 2)

    def test_tc21_track_scalars_calls_run_track_per_key(
        self,
    ) -> None:
        """TC21: track_scalars calls run.track once per key
        with correct step."""
        cfg = AimConfig(enabled=True)
        mock_run = MagicMock()
        tracker = AimTracker(cfg, run=mock_run)
        metrics = {
            "pg_loss": 1.5,
            "ce_loss": 0.8,
            "total_loss": 2.3,
        }
        tracker.track_scalars(metrics, step=7)
        self.assertEqual(mock_run.track.call_count, 3)
        tracked_names = {
            call[1]["name"]
            for call in mock_run.track.call_args_list
        }
        self.assertEqual(
            tracked_names,
            {"pg_loss", "ce_loss", "total_loss"},
        )
        for call in mock_run.track.call_args_list:
            self.assertEqual(call[1]["step"], 7)

    def test_tc22_noop_track_scalars_returns_none(
        self,
    ) -> None:
        """TC22: NoOpTracker.track_scalars() returns None
        without raising."""
        tracker = NoOpTracker()
        result = tracker.track_scalars(
            {"pg_loss": 1.0, "ce_loss": 0.5}, step=3
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
