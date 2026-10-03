"""Drawing journal figures off the GUI thread, one at a time.

matplotlib keeps its settings (rcParams) global, and :func:`export.render` changes them while it
draws, so every figure is drawn and saved by one worker thread, in order; the GUI thread only
builds the :class:`FigureJob` and collects the result (no Qt object crosses threads). A preview
asked for while another is drawn replaces any older one that has not started, so quick changes
cost one extra drawing at most.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

from mag_opt_detective.export import (
    FigureState,
    FigureStyle,
    JournalPreset,
    rasterize,
    render,
    save,
)

logger = logging.getLogger("mag_opt_detective")

POLL_MS = 15
_executor: ThreadPoolExecutor | None = None


def executor() -> ThreadPoolExecutor:
    """The single thread all figures are drawn on (started when first needed)."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="figure")
    return _executor


@dataclass(frozen=True)
class FigureJob:
    """A figure to draw: what it shows, its preset, print size and style, and the dpi to draw
    at."""

    state: FigureState
    preset: JournalPreset
    width_mm: float
    height_mm: float
    font_pt: float
    line_pt: float
    panel_label: str | None
    dpi: float
    style: FigureStyle = field(default_factory=FigureStyle)

    def figure(self):
        return render(
            self.state,
            preset=self.preset,
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            font_size_pt=self.font_pt,
            line_width_pt=self.line_pt,
            panel_label=self.panel_label or None,
            dpi=self.dpi,
            style=self.style,
        )


def draw(job: FigureJob) -> np.ndarray:
    """The figure as an RGBA array at the job's dpi (worker thread)."""
    return rasterize(job.figure(), job.dpi)


def write(job: FigureJob, path: Path) -> Path:
    """Save the figure to *path* in the format of its suffix (worker thread)."""
    return save(job.figure(), path, dpi=job.dpi)


def to_image(rgba: np.ndarray, pixel_ratio: float = 1.0) -> QImage:
    """An RGBA array as a QImage that owns its pixels."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width = rgba.shape[:2]
    image = QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
    image.setDevicePixelRatio(pixel_ratio)
    return image


def error_text(exc: BaseException) -> str:
    """A short message for people (file errors name the file and the reason)."""
    if isinstance(exc, OSError) and exc.strerror:
        return f"{exc.strerror}: {exc.filename}" if exc.filename else exc.strerror
    return str(exc) or type(exc).__name__


class FigureRenderer(QObject):
    """Draws previews and saves files on the worker thread and reports back on the GUI one.

    Saves run in the order asked, before any waiting preview; ``busyChanged`` tells whether
    anything is drawn or waiting.
    """

    previewReady = Signal(QImage)
    saved = Signal(str)
    failed = Signal(str, str)  # "preview" or "save", message
    busyChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._running: tuple[str, Future, float] | None = None  # kind, future, pixel ratio
        self._preview: tuple[FigureJob, float] | None = None
        self._saves: list[tuple[FigureJob, Path]] = []
        self._busy = False

    def preview(self, job: FigureJob, pixel_ratio: float = 1.0) -> None:
        """Draw *job* for the screen (replaces a preview that has not started)."""
        self._preview = (job, pixel_ratio)
        self._next()

    def save(self, job: FigureJob, path: str | Path) -> None:
        self._saves.append((job, Path(path)))
        self._next()

    def busy(self) -> bool:
        return self._running is not None or self._preview is not None or bool(self._saves)

    def saving(self) -> bool:
        running = self._running is not None and self._running[0] == "save"
        return running or bool(self._saves)

    def _next(self) -> None:
        if self._running is None:
            if self._saves:
                job, path = self._saves.pop(0)
                self._running = ("save", executor().submit(write, job, path), 1.0)
            elif self._preview is not None:
                (job, ratio), self._preview = self._preview, None
                self._running = ("preview", executor().submit(draw, job), ratio)
        if self._running is not None:
            self._timer.start()
        else:
            self._timer.stop()
        if self.busy() != self._busy:
            self._busy = self.busy()
            self.busyChanged.emit(self._busy)

    def _poll(self) -> None:
        if self._running is None or not self._running[1].done():
            return
        (kind, future, ratio), self._running = self._running, None
        exc = future.exception()
        if exc is not None:
            if not isinstance(exc, ValueError | OSError):
                logger.error("Drawing the figure failed", exc_info=exc)
            self.failed.emit(kind, error_text(exc))
        elif kind == "preview":
            self.previewReady.emit(to_image(future.result(), ratio))
        else:
            self.saved.emit(str(future.result()))
        self._next()
