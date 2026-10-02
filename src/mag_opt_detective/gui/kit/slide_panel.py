"""A splitter pane that slides open and closed."""

from __future__ import annotations

import json

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QSplitter, QWidget

from mag_opt_detective.gui.kit._common import (
    DURATION_MS,
    json_object,
    make_animation,
    to_bool,
    to_float,
)


class SlidePanel(QWidget):
    """A pane in a QSplitter whose open/close animates the splitter sizes.

    Add it to a splitter like any widget. While it moves, the content keeps its open width
    (height in a vertical splitter) and is clipped, anchored to the edge next to the rest of
    the window, so it slides instead of being squeezed. The user can drag the splitter handle;
    dragging it to zero closes the panel, and reopening restores the last open size. Space is
    taken from (and given back to) the largest sibling that is not a SlidePanel. ``openChanged``
    fires on every change of :meth:`is_open`. Settings protocol: JSON
    ``{"open": bool, "size": int}``.
    """

    openChanged = Signal(bool)
    duration_ms = DURATION_MS  # animation length; tests may lengthen it per instance

    def __init__(self, content: QWidget, default_size: int = 280, parent: QWidget | None = None):
        super().__init__(parent)
        self._content = content
        content.setParent(self)
        self._size = max(1, int(default_size))
        self._open = True
        self._animating = False
        self._pending = False
        self._splitter: QSplitter | None = None
        self._donor: int | None = None
        self._animation = make_animation(self)
        self._animation.valueChanged.connect(self._on_step)
        self._animation.finished.connect(self._on_finished)
        self._attach()

    # --- public API --------------------------------------------------------------------
    def content(self) -> QWidget:
        return self._content

    def is_open(self) -> bool:
        return self._open

    def is_animating(self) -> bool:
        return self._animating

    def open_size(self) -> int:
        """The size the panel has (or gets) when open."""
        return self._size

    def set_open(self, open: bool, animate: bool = True) -> None:
        open = bool(open)
        if open == self._open:
            return
        self._open = open
        self._run(animate)
        self.openChanged.emit(open)

    def toggle(self, animate: bool = True) -> None:
        self.set_open(not self._open, animate)

    # --- splitter bookkeeping ----------------------------------------------------------
    def _attach(self) -> QSplitter | None:
        parent = self.parentWidget()
        splitter = parent if isinstance(parent, QSplitter) else None
        if splitter is not self._splitter:
            if self._splitter is not None:
                self._splitter.splitterMoved.disconnect(self._on_splitter_moved)
            self._splitter = splitter
            if splitter is not None:
                splitter.splitterMoved.connect(self._on_splitter_moved)
        return splitter

    def _horizontal(self) -> bool:
        return self._splitter is None or self._splitter.orientation() == Qt.Orientation.Horizontal

    def _along(self, size: QSize) -> int:
        return size.width() if self._horizontal() else size.height()

    def _anchor_end(self) -> bool:
        """The leading pane slides towards its far edge; the others towards their near edge."""
        return self._splitter is not None and self._splitter.indexOf(self) == 0

    def _sizes(self, splitter: QSplitter) -> list[int]:
        sizes = splitter.sizes()
        if sum(sizes) > 0:
            return sizes
        # not laid out yet: start from the size hints (setSizes treats them as weights)
        hints = []
        for i in range(splitter.count()):
            widget = splitter.widget(i)
            hints.append(0 if widget.isHidden() else max(0, self._along(widget.sizeHint())))
        return hints

    def _pick_donor(self, splitter: QSplitter, sizes: list[int]) -> int | None:
        index = splitter.indexOf(self)
        others = [
            i for i in range(splitter.count()) if i != index and not splitter.widget(i).isHidden()
        ]
        plain = [i for i in others if not isinstance(splitter.widget(i), SlidePanel)]
        candidates = plain or others
        if not candidates:
            return None
        return max(candidates, key=lambda i: (sizes[i], -abs(i - index)))

    def _resize_to(self, size: int, sizes: list[int], donor: int | None) -> None:
        splitter = self._splitter
        index = splitter.indexOf(self)
        new = list(sizes)
        new[index] = size
        if donor is not None:
            new[donor] = max(0, sizes[donor] - (size - sizes[index]))
        splitter.setSizes(new)

    # --- open / close ------------------------------------------------------------------
    def _run(self, animate: bool) -> None:
        self._animation.stop()
        splitter = self._attach()
        if splitter is None:
            self._set_animating(False)
            self._content.setVisible(self._open)
            return
        splitter.setCollapsible(splitter.indexOf(self), True)
        target = self._size if self._open else 0
        sizes = splitter.sizes()
        current = sizes[splitter.indexOf(self)]
        if not animate or not splitter.isVisible() or sum(sizes) == 0 or current == target:
            self._finish()
            return
        self._donor = self._pick_donor(splitter, sizes)
        self._content.setVisible(True)
        self._set_animating(True)
        self._animation.setStartValue(current)
        self._animation.setEndValue(target)
        self._animation.setDuration(self.duration_ms)
        self._animation.start()

    def _set_animating(self, animating: bool) -> None:
        self._animating = animating
        self.updateGeometry()
        self._place_content()

    def _on_step(self, value) -> None:
        # from the live sizes, so panels animating at the same time do not undo each other
        if self._animating and self._splitter is not None:
            self._resize_to(round(value), self._splitter.sizes(), self._donor)

    def _on_finished(self) -> None:
        if self._animating:
            self._finish()

    def _finish(self) -> None:
        """Jump to the final size (the restored minimum size lets the splitter collapse us)."""
        self._set_animating(False)
        splitter = self._splitter
        if splitter is None:
            return
        sizes = self._sizes(splitter)
        self._resize_to(self._size if self._open else 0, sizes, self._pick_donor(splitter, sizes))
        self._content.setVisible(self._open)
        self._place_content()
        self._pending = not splitter.isVisible() or sum(splitter.sizes()) == 0

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        if self._animating or self._splitter is None:
            return
        size = self._splitter.sizes()[self._splitter.indexOf(self)]
        if size > 0:
            self._size = size
        if (size > 0) != self._open:
            self._open = size > 0
            self._content.setVisible(self._open)
            self._place_content()
            self.openChanged.emit(self._open)

    # --- layout ------------------------------------------------------------------------
    def sizeHint(self) -> QSize:
        hint = self._content.sizeHint()
        along = self._size if self._open else 0
        if self._horizontal():
            return QSize(along, max(0, hint.height()))
        return QSize(max(0, hint.width()), along)

    def _content_minimum(self) -> QSize:
        """The content's minimum size: explicit minimumSize() first, else its hint."""
        explicit, hint = self._content.minimumSize(), self._content.minimumSizeHint()
        return QSize(
            explicit.width() or max(0, hint.width()), explicit.height() or max(0, hint.height())
        )

    def minimumSizeHint(self) -> QSize:
        minimum = self._content_minimum()
        width, height = max(1, minimum.width()), max(1, minimum.height())
        if self._animating:  # let the splitter make us smaller than the content
            return QSize(0, height) if self._horizontal() else QSize(width, 0)
        return QSize(width, height)

    def _place_content(self) -> None:
        rect = self.rect()
        if not self._animating:
            self._content.setGeometry(rect)
            return
        minimum = self._content_minimum()
        if self._horizontal():
            width = max(self._size, minimum.width())
            x = rect.width() - width if self._anchor_end() else 0
            self._content.setGeometry(x, 0, width, rect.height())
        else:
            height = max(self._size, minimum.height())
            y = rect.height() - height if self._anchor_end() else 0
            self._content.setGeometry(0, y, rect.width(), height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_content()
        if self._pending and self._splitter is not None and self._splitter.isVisible():
            self._pending = False
            QTimer.singleShot(0, self._settle)

    def _settle(self) -> None:
        """Re-apply the size once the splitter is laid out (setSizes before show is relative)."""
        if not self._animating and self._splitter is not None:
            self._finish()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ParentChange:
            self._attach()
        elif event.type() == QEvent.Type.LayoutRequest:  # the content's size hints changed
            self.updateGeometry()
            self._place_content()
        return super().event(event)

    # --- settings protocol -------------------------------------------------------------
    def settings_value(self) -> str:
        return json.dumps({"open": self._open, "size": int(self._size)})

    def set_settings_value(self, value) -> bool:
        data = json_object(value)
        if data is None:
            return False
        open = to_bool(data.get("open"))
        size = to_float(data.get("size"))
        if open is None or size is None or size < 1:
            return False
        self._size = int(size)
        changed = open != self._open
        self._open = open
        self._run(animate=False)
        if changed:
            self.openChanged.emit(open)
        return True
