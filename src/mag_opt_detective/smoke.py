"""Self-test used to check that a packaged app works (``mag-opt-detective --smoke-test``).

It writes a small synthetic field sweep, loads it through the main window, processes
it, draws every plot type and exits. No user settings are read or written.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _write_sweep(folder: Path) -> tuple[list[str], list[str]]:
    x = np.linspace(400.0, 4000.0, 400)
    base = 1.0 + 0.3 * np.sin(x / 100.0)

    def write(name: str, y: np.ndarray) -> str:
        path = folder / name
        np.savetxt(path, np.column_stack([x, y]), delimiter="\t", fmt="%.8f")
        return str(path)

    zero = [
        write("smoke_a00p000T_a00p000T.txt", base),
        write("smoke_a00p000T_a02p000T.txt", 1.01 * base),
    ]
    field = [write(f"smoke_a0{b:.3f}T.txt".replace(".", "p", 1), (1 + 0.05 * b) * base)
             for b in (0.5, 1.0, 1.5, 2.0)]  # fmt: skip
    return zero, field


def run(window) -> None:
    """Drive *window* through loading, processing, plotting and a unit switch; raise on failure."""
    from mag_opt_detective.gui.controller import SweepFiles

    controller = window.controller
    with tempfile.TemporaryDirectory(prefix="mag-opt-smoke-") as tmp:
        zero, field = _write_sweep(Path(tmp))
        controller.set_processing(sample_files=SweepFiles(tuple(zero), tuple(field)))
        errors: list[str] = []
        window.report_error = lambda title, message, **_kw: errors.append(f"{title}: {message}")
        window.commands["process"].trigger()
        if errors or controller.result is None:
            raise RuntimeError("; ".join(errors) or "processing produced no result")
        image = window.plots.map.image
        for kind in window.toolbar.kind.options():
            window.toolbar.kind.set_value(kind)
            if image.image is None:
                raise RuntimeError(f"no image for {kind}")
        for unit in reversed(window.toolbar.unit.options()):
            window.toolbar.unit.set_value(unit)
            if controller.unit != unit or window.plots.map.image.image is None:
                raise RuntimeError(f"the plots did not switch to {unit}")
        if errors:
            raise RuntimeError("; ".join(errors))
    logger.info("Smoke test passed: %d field spectra processed.", len(field))
