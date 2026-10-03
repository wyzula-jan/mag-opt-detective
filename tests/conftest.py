import http.client
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from helpers import sweep_name, write_text

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DATA_DIR = Path(__file__).resolve().parents[1] / "Data_to_test"


def pytest_configure(config):
    config.addinivalue_line("markers", "network: the test may reach the network (none does)")


class NetworkGuard:
    """Refuses every request and records it, also one made on a worker thread (the update
    check), where the refusal would only end as a message in the app."""

    def __init__(self):
        self.attempts: list[str] = []

    def refuse(self, what: str):
        def blocked(*_args, **_kwargs):
            self.attempts.append(what)
            raise AssertionError(f"a test tried to reach the network ({what})")

        return blocked

    def install(self, monkeypatch) -> None:
        from mag_opt_detective import updates

        monkeypatch.setattr(updates, "open_github", self.refuse("the update check"))
        monkeypatch.setattr(urllib.request.OpenerDirector, "open", self.refuse("urllib"))
        monkeypatch.setattr(http.client.HTTPConnection, "connect", self.refuse("http.client"))

    def check(self) -> None:
        """Fail the test if it tried to reach the network."""
        if self.attempts:
            pytest.fail(
                f"the test tried to reach the network: {', '.join(self.attempts)} "
                "(mark it network if it must)",
                pytrace=False,
            )


@pytest.fixture(autouse=True)
def network_guard(request, monkeypatch):
    """No test reaches the network (the update check, any urllib or http.client request)
    unless it is marked ``network``: a test that tries fails, also when the request was made
    on a worker thread. A test that tries on purpose clears ``attempts``."""
    if request.node.get_closest_marker("network") is not None:
        yield None
        return
    guard = NetworkGuard()
    guard.install(monkeypatch)
    yield guard
    guard.check()


@pytest.fixture(autouse=True)
def _delete_closed_widgets():
    """Delete the widgets a test closed, and the menus pyqtgraph leaves without a parent.

    Qt deletes closed widgets only when an event loop next runs, so they would pile up until
    some later test first waits (and spends seconds deleting them), and a theme change would
    restyle them all. pyqtgraph's view and colour-map menus and colour dialogs have no parent
    and outlive their window unless deleted here.
    """
    yield
    if "PySide6.QtWidgets" not in sys.modules:  # a test without Qt
        return
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication, QColorDialog, QMenu

    if QApplication.instance() is None:
        return
    for widget in QApplication.topLevelWidgets():
        orphan = widget.parent() is None and not widget.isVisible()
        if orphan and isinstance(widget, QMenu | QColorDialog):
            widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def data_dir() -> Path:
    """Real measurement data; not part of the repository, so tests skip without it."""
    if not DATA_DIR.is_dir():
        pytest.skip("Data_to_test/ not available")
    return DATA_DIR


@pytest.fixture
def sweep(tmp_path: Path):
    """Synthetic sweep: zero files before/after and field files 0.5..2.0 T.

    Intensity is ``(1 + 0.1 B) * spectrum`` and the zero reference drifts from 1x to 2x.
    """
    x = np.linspace(100.0, 1000.0, 91)
    base = 1.0 + 0.5 * np.sin(x / 50.0)
    fields = np.array([0.5, 1.0, 1.5, 2.0])
    zero = [
        write_text(tmp_path / "Sample_4p2K_Sam1_a00p000T_a00p000T.txt", x, base),
        write_text(tmp_path / "Sample_4p2K_Sam1_a00p000T_a02p000T.txt", x, 2 * base),
    ]
    field = [write_text(tmp_path / sweep_name(b), x, (1 + 0.1 * b) * base) for b in fields]
    return {"x": x, "base": base, "fields": fields, "zero": zero, "field": field}
