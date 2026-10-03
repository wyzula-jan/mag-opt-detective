"""Self-test used to check that a packaged app works (``mag-opt-detective --smoke-test``).

It writes a small synthetic field sweep, loads it through the main window, processes
it, draws every plot type, switches the unit, saves a small journal figure in every format
(matplotlib's backends, fonts and Pillow's writers) and exits. A bundle must also carry its
third-party notices. No user settings are read or written.
"""

from __future__ import annotations

import logging
import sys
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
        save_figures(controller, Path(tmp))
    if getattr(sys, "frozen", False):
        from mag_opt_detective.gui.licences import notices_path

        if not notices_path().is_file():
            raise RuntimeError(f"the bundle has no {notices_path().name}")
    logger.info("Smoke test passed: %d field spectra processed.", len(field))


# the first bytes of a file in each figure format
MAGIC = {
    ".pdf": b"%PDF",
    ".svg": b"<?xml",
    ".eps": b"%!PS-Adobe",
    ".png": b"\x89PNG",
    ".tif": (b"II*\x00", b"MM\x00*"),
}


def save_figures(controller, folder: Path) -> None:
    """Save the map and the stacked plot as small journal figures in every format."""
    from mag_opt_detective import export
    from mag_opt_detective.gui.export_state import FigureContent, figure_state

    for kind in ("map", "stacked"):
        state = figure_state(controller, FigureContent(kind=kind))
        fig = export.render(state, preset="nature", width_mm=89, height_mm=60, dpi=72)
        for suffix, magic in MAGIC.items():
            path = export.save(fig, folder / f"{kind}{suffix}", dpi=72)
            if not path.read_bytes().startswith(magic):
                raise RuntimeError(f"the {kind} figure saved as {suffix} is not a {suffix} file")
