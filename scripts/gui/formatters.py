"""Pure formatting helpers for the GUI stats panel.

No tkinter or matplotlib dependencies -- safe to import in headless tests.
"""
from __future__ import annotations


def _fmt_loss(v: float) -> str:
    """Format a loss value to 4 decimal places.

    Args:
        v: Float loss value.

    Returns:
        String like "3.1416".
    """
    return f"{v:.4f}"


def _fmt_acc(v: int) -> str:
    """Format accuracy integer as a symbol.

    Args:
        v: 1 for correct, 0 for wrong, -1 for ignored/terminal.

    Returns:
        Checkmark for 1, x-mark for 0, en-dash for -1.
    """
    if v == 1:
        return "\u2713"
    if v == 0:
        return "\u2717"
    return "\u2013"


def _fmt_entropy(v: float) -> str:
    """Format an entropy value to 4 decimal places.

    Args:
        v: Float entropy in nats.

    Returns:
        String like "2.0794".
    """
    return f"{v:.4f}"
