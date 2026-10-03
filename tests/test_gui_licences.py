"""Help > About and the Licences… window."""

from PySide6.QtWidgets import QMessageBox

import gui_helpers
from mag_opt_detective.gui import licences
from mag_opt_detective.gui.licences import LicencesDialog

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures


def test_about_opens_the_licences(window):
    box = licences.about_box(window, "<b>Magneto-Optical Detective</b>")
    assert box.licences_button.text() == "Licences…"
    assert box.buttonRole(box.licences_button) == QMessageBox.ButtonRole.ActionRole
    box.licences_button.click()
    (dialog,) = [d for d in window.findChildren(LicencesDialog) if d.isVisible()]
    assert dialog.windowTitle() == "Licences" and dialog.text.isReadOnly()
    assert "Lucide" in dialog.text.toPlainText()


def test_a_bundle_shows_its_third_party_notices(tmp_path, monkeypatch):
    notices = tmp_path / licences.NOTICES
    notices.write_text("all the licence texts", encoding="utf-8")
    monkeypatch.setattr(licences, "notices_path", lambda: notices)
    assert licences.licences_text() == "all the licence texts"


def test_from_source_the_installed_packages_are_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(licences, "notices_path", lambda: tmp_path / "missing.txt")
    text = licences.licences_text()
    for name in ("numpy", "scipy", "matplotlib", "pyqtgraph", "PySide6"):
        assert name.lower() in text.lower()
    assert "GNU Lesser General Public License" in text
    assert "ISC" in text  # the Lucide licence
