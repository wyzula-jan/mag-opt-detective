"""Delete the windows still open when the interpreter exits.

PySide deletes the remaining Qt objects in its own exit hook, in no useful order: a window
alive at that point (a script that raised, or ended without closing it) gets layout events
through our Python event filters into pyqtgraph items whose children are already gone, which
prints a long cascade of "Internal C++ object already deleted" tracebacks. :func:`install`
registers :func:`delete_windows` after PySide is imported, so that it runs before that hook
(exit hooks run last registered first): the windows are deleted in Qt's order while Python
and Qt are still whole.
"""

from __future__ import annotations

import atexit

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

_installed = False


def delete_widget(widget: QWidget) -> None:
    """Delete *widget* and its children now (as ``deleteLater`` does on the next loop)."""
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def delete_windows() -> None:
    """Close and delete the main windows that are still alive, with their dialogs.

    Other top-level widgets (menus and the like) are left alone: many belong to Python
    objects of pyqtgraph, and deleting them under their owners crashes.
    """
    if QApplication.instance() is None:
        return
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QMainWindow) and widget.parent() is None:
            widget.close()  # what a window does on close (e.g. let go of the log)
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def install() -> None:
    """Run :func:`delete_windows` at exit, before PySide's own clean-up (once)."""
    global _installed
    if not _installed:
        atexit.register(delete_windows)
        _installed = True
