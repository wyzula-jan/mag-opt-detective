"""The Models section: model cards, live units, Zeeman couplings, custom expressions, fits to
the picked points, the results table, figure_state overlays and the saved model list."""

import json
import threading

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QApplication, QStyle

import gui_helpers
from gui_helpers import inspector_page, load_sweep, process, save_to, set_unit
from mag_opt_detective.core.expressions import ExpressionError, parse
from mag_opt_detective.core.fitting import Assignment
from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.units import Unit, convert
from mag_opt_detective.core.zeeman import MU_B, Form, branch_energy
from mag_opt_detective.gui.controller import AppController
from mag_opt_detective.gui.inspector import model_state as ms
from mag_opt_detective.gui.inspector import models as models_module
from mag_opt_detective.gui.inspector.model_widgets import FIELD_HEIGHT, ParamRow
from mag_opt_detective.gui.kit.nudge_slider import jog_fraction
from mag_opt_detective.gui.main_window import INSPECTOR_MIN_WIDTH, MainWindow
from mag_opt_detective.gui.settings import PREFIX

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

MEV = 8.0656  # cm-1 per meV
THZ = 33.35641  # cm-1 per THz


def models_of(window):
    return inspector_page(window, "models").models


def card_of(window, entry):
    return models_of(window).cards[entry]


def add(window, kind):
    """Add a model through the section's Add menu."""
    inspector_page(window, "models").add_actions[kind].trigger()
    return models_of(window).entries[-1]


def choose(area, curve: str, text: str) -> None:
    combo = area.combos[curve]
    index = combo.findText(text)
    assert index >= 0, f"{text!r} is not offered for {curve}"
    combo.setCurrentIndex(index)


def load_table(window, tmp_path, field, columns, unit="meV", name="points.csv"):
    """Picked points from a file (energies in *unit*), as the Points panel imports them."""
    path = tmp_path / name
    cm1 = {k: convert(np.asarray(v, float), unit, "cm-1") for k, v in columns.items()}
    PointTable(field, cm1).save_tsv(path, unit=unit)  # the table keeps cm-1
    window.controller.load_points(path)


def result_cells(area) -> list[list[str]]:
    """The texts of the fit result table, row by row (name, value, ±, sigma, unit)."""
    grid = area.result_table.grid
    return [
        [grid.itemAtPosition(row, column).widget().text() for column in range(5)]
        for row in range(grid.rowCount())
        if grid.itemAtPosition(row, 0) is not None
    ]


def zeeman_points(rng, noise=0.02):
    """Two Zeeman branches, E0 20 / 35 meV, g 2.0 / 1.5, m +1 / -1, with noise (meV)."""
    field = np.linspace(0.5, 15.0, 30)
    lower = branch_energy(field, 20.0, 2.0, 1.0) + rng.normal(0, noise, field.size)
    upper = branch_energy(field, 35.0, 1.5, -1.0) + rng.normal(0, noise, field.size)
    return field, {"LL 1": lower, "LL 2": upper}


# ---------------------------------------------------------------------- the list
def test_add_remove_show_and_hide_models(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    models = models_of(window)
    page = inspector_page(window, "models")
    assert [e.kind for e in models.entries] == ["dirac"]
    assert not models.entries[0].visible and page.empty.isHidden()
    assert window.inspector["models"].trailing() is page.add_button

    zeeman = add(window, "zeeman")
    assert zeeman.key == "zeeman" and zeeman.visible and zeeman.name == "Zeeman / magnon"
    # its first branch lies inside the map (25 % up the energy range)
    e0 = ms.params(zeeman)["e0_0"].value
    lo, hi = sweep["x"][0] / MEV, sweep["x"][-1] / MEV
    assert lo < e0 < hi
    drawn = models.curve_data()
    assert drawn["dirac"] == [] and len(drawn["zeeman"]) == 1
    assert window.plots.map.layer("models 2").curve_data()  # second slot
    custom = add(window, "custom")
    assert models.curve_data()["custom"] == []  # no expression yet
    assert len({e.color for e in models.entries}) == 3

    card = card_of(window, zeeman)
    card.visible_switch.setChecked(False)
    assert not zeeman.visible and models.curve_data()["zeeman"] == []
    card.visible_switch.setChecked(True)
    assert len(models.curve_data()["zeeman"]) == 1

    card.swatch.colorChosen.emit(ms.MODEL_COLORS[4])
    assert zeeman.color == ms.MODEL_COLORS[4] and card.swatch.color() == ms.MODEL_COLORS[4]
    card.chevron.click()
    assert not zeeman.expanded and card.body.isHidden()
    card.chevron.click()
    assert zeeman.expanded and not card.body.isHidden()

    card_of(window, models.entries[0]).remove_button.click()
    assert models.entries == [zeeman, custom]
    assert len(window.plots.map.layer("models").curve_data()) == 1  # moved up a slot
    assert window.plots.map.layer("models 3").curve_data() == []
    card_of(window, zeeman).remove_button.click()
    card_of(window, custom).remove_button.click()
    assert models.entries == [] and not page.empty.isHidden()
    again = add(window, "dirac")
    assert again.key == "dirac" and again.visible
    assert not errors


def test_dirac_defaults_are_the_former_overlay(window):
    dirac = models_of(window).entries[0]
    p = ms.params(dirac)
    assert (p["velocity"].value, p["delta"].value, dirac.model.n_lines) == (5.0, 0.0, 5)
    assert not dirac.visible
    editor = card_of(window, dirac).editor
    assert editor.velocity.field.text() == "5"
    assert editor.n_lines.field.text() == "5"


def test_old_overlay_settings_are_taken_over(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    old = QSettings(ini, QSettings.Format.IniFormat)
    for key, value in {"show_dirac": True, "velocity": 5.4, "delta": 12.0, "n_lines": 3}.items():
        old.setValue(f"{PREFIX}/models/{key}", value)
    old.sync()
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    dirac = models_of(w).entries[0]
    p = ms.params(dirac)
    assert dirac.visible
    assert (p["velocity"].value, p["delta"].value, dirac.model.n_lines) == (5.4, 12.0, 3)
    editor = card_of(w, dirac).editor
    assert float(editor.delta.field.text()) == pytest.approx(12.0 * MEV, rel=1e-5)  # cm-1
    assert editor.n_lines.field.text() == "3"
    w.close()
    stored = QSettings(ini, QSettings.Format.IniFormat)
    assert stored.value(f"{PREFIX}/models/velocity") is None  # taken over once
    data = json.loads(stored.value(f"{PREFIX}/models/list"))
    assert data["models"][0]["delta_meV"] == 12.0 and data["models"][0]["visible"]


# ---------------------------------------------------------------------- units
def test_energy_parameters_are_shown_in_the_display_unit(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    result = c.result
    models = models_of(window)
    dirac = models.entries[0]
    card = card_of(window, dirac)
    card.visible_switch.setChecked(True)
    card.editor.delta.field.edit.setText(f"{10 * MEV:g}")  # typed in cm-1
    assert ms.params(dirac)["delta"].value == pytest.approx(10.0)
    assert card.editor.delta.field.unit_label.text() == "cm⁻¹"

    zeeman = add(window, "zeeman")
    zcard = card_of(window, zeeman)
    zcard.editor.add_button.click()
    zcard.editor.coupled.setChecked(True)
    row = zcard.editor.rows[0]
    row.e0.field.edit.setText("161.312")  # 20 meV in cm-1
    assert ms.params(zeeman)["e0_0"].value == pytest.approx(20.0)

    set_unit(window, "meV")
    assert c.result is result  # nothing processed again
    assert card.editor.delta.field.text() == "10"
    assert card.editor.delta.field.unit_label.text() == "meV"
    assert zcard.editor.rows[0].e0.field.text() == "20"
    assert zcard.editor.couplings[(0, 1)].field.text() == "1"  # a new coupling starts at 1 meV
    field, energy = models.curve_data()["dirac"][1]
    np.testing.assert_allclose(energy, dirac_interband(field, 5.0, 10.0, 5)[1])

    set_unit(window, "THz")
    assert float(card.editor.delta.field.text()) == pytest.approx(10 * MEV / THZ, rel=1e-5)
    assert float(zcard.editor.couplings[(0, 1)].field.text()) == pytest.approx(MEV / THZ, rel=1e-5)
    field, energy = models.curve_data()["dirac"][1]
    expected = convert(dirac_interband(field, 5.0, 10.0, 5)[1], "meV", "THz")
    np.testing.assert_allclose(energy, expected)
    zcard.editor.rows[1].e0.field.edit.setText("12")  # THz
    assert ms.params(zeeman)["e0_1"].value == pytest.approx(12 * THZ / MEV)
    assert ms.params(dirac)["delta"].value == pytest.approx(10.0)  # kept in meV
    assert not errors


# ---------------------------------------------------------------------- Zeeman
def test_zeeman_branches_and_couplings(window, errors):
    zeeman = add(window, "zeeman")
    editor = card_of(window, zeeman).editor
    assert [r.label.edit.text() for r in editor.rows] == ["Branch 1"]
    assert not editor.coupled.isEnabled() and not editor.coupling_note.isHidden()
    assert not editor.rows[0].remove.isEnabled()  # one branch stays

    editor.add_button.click()
    assert [r.label.edit.text() for r in editor.rows] == ["Branch 1", "Branch 2"]
    assert editor.coupled.isEnabled() and editor.coupling_box.isHidden()
    editor.coupled.setChecked(True)
    assert zeeman.model.coupled and not editor.coupling_box.isHidden()
    assert ms.params(zeeman)["delta_0_1"].value == 1.0  # meV, not 0
    editor.couplings[(0, 1)].field.edit.setText(f"{2 * MEV:g}")  # 2 meV typed in cm-1
    assert ms.params(zeeman)["delta_0_1"].value == pytest.approx(2.0)
    assert zeeman.branch_names() == ["Mode 1", "Mode 2"]

    editor.coupled.setChecked(False)  # uncoupled holds 0 and keeps the user's value
    delta = ms.params(zeeman)["delta_0_1"]
    assert (delta.value, delta.fixed) == (0.0, True)
    assert zeeman.model.couplings[(0, 1)] == pytest.approx(2.0)
    assert editor.coupling_box.isHidden()
    editor.coupled.setChecked(True)
    assert ms.params(zeeman)["delta_0_1"].value == pytest.approx(2.0)

    editor.add_button.click()
    couplings = zeeman.model.couplings
    assert couplings[(0, 1)] == pytest.approx(2.0)
    assert couplings[(0, 2)] == couplings[(1, 2)] == 1.0
    assert sorted(editor.couplings) == [(0, 1), (0, 2), (1, 2)]

    editor.rows[1].label.edit.setText("")
    assert zeeman.model.branches[1].label == "Branch 2"
    editor.rows[1].label.edit.editingFinished.emit()  # the field shows the name it fell back to
    assert editor.rows[1].label.edit.text() == "Branch 2"
    editor.rows[1].label.edit.setText("magnon")
    editor.rows[1].m.edit.setText("0")
    assert ms.params(zeeman)["g_1"].fixed  # g of an m = 0 line does not matter
    editor.rows[1].form.click()
    assert zeeman.model.branches[1].form is Form.HYPERBOLIC
    assert editor.rows[1].form.text() == "hyp"
    editor.rows[0].remove.click()  # the couplings of the others are kept
    assert [b.label for b in zeeman.model.branches] == ["magnon", "Branch 3"]
    assert zeeman.model.couplings == {(0, 1): 1.0}
    assert [r.label.edit.text() for r in editor.rows] == ["magnon", "Branch 3"]
    assert not errors


# ---------------------------------------------------------------------- custom expressions
def test_custom_expression_validates_as_you_type(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    models = models_of(window)
    entry = add(window, "custom")
    editor = card_of(window, entry).editor
    assert editor.code.edit.placeholderText() == "E0 + g*muB*B"
    assert editor.message.level() == "muted" and editor.table.isHidden()

    editor.code.edit.setPlainText("E0 + g*muB*B")
    assert list(editor.rows) == ["E0", "g"]
    editor.rows["E0"].value.field.edit.setText("40")
    editor.rows["g"].value.field.edit.setText("2")
    editor.rows["g"].fixed.setChecked(True)
    editor.rows["E0"].lo.edit.setText("10")
    assert ms.params(entry)["E0"].lo == 10.0
    field, energy = models.curve_data()["custom"][0]
    np.testing.assert_allclose(energy, convert(40 + 2 * MU_B * field, "meV", "cm-1"))

    bad = "E0 + g*muB*B\nE0 - g*muB*sqrt(B"
    editor.code.edit.setPlainText(bad)
    with pytest.raises(ExpressionError) as exc:
        parse(bad)
    error = exc.value
    assert error.line == 2
    assert editor.message.level() == "err"
    assert editor.message.text() == f"Line 2, column {error.column}: {error.message}"
    assert editor.code.is_invalid()
    selections = editor.code.edit.extraSelections()
    assert selections and selections[0].cursor.blockNumber() == 1  # the line is marked
    assert list(editor.rows) == ["E0", "g"]  # the last valid parameters stay
    assert models.curve_data()["custom"] == []  # an invalid text draws nothing

    editor.code.edit.setPlainText("E0 + g*muB*B\nE1 - g*muB*B  # lower")
    assert editor.message.level() == "muted" and not editor.code.is_invalid()
    assert list(editor.rows) == ["E0", "g", "E1"]
    assert editor.rows["E0"].value.field.text() == "40"
    assert editor.rows["g"].fixed.isChecked()
    assert entry.branch_names() == ["E0 + g*muB*B", "lower"]

    editor.code.edit.setPlainText("E0 + h*B")
    assert list(editor.rows) == ["E0", "h"]
    editor.code.edit.setPlainText("E0 + g*muB*B")  # g comes back as it was
    assert editor.rows["g"].value.field.text() == "2" and editor.rows["g"].fixed.isChecked()
    assert ms.params(entry)["E0"].lo == 10.0
    assert "h" not in entry.memory  # never set by the user: forgotten
    text = "E0 + amplitude*B"
    for end in range(1, len(text) + 1):  # typed letter by letter
        editor.code.edit.setPlainText(text[:end])
    saved = [p["name"] for p in ms.entry_to_dict(entry)["params"]]
    assert sorted(saved) == ["E0", "amplitude", "g"]  # not E, a, am, ...; g was set

    editor.code.edit.setPlainText("foo(B)")
    assert editor.message.text() == "Line 1, column 1: unknown function foo"

    editor.code.edit.setPlainText("E0 + g*muB*B")
    editor.unit.set_value("THz")  # the parameters stay as typed; the curve is converted
    assert ms.params(entry)["E0"].value == 40.0
    field, energy = models.curve_data()["custom"][0]
    mu_b = float(convert(MU_B, "meV", "THz"))  # the constants follow the output unit
    np.testing.assert_allclose(energy, convert(40 + 2 * mu_b * field, "THz", "cm-1"))
    assert not errors


def test_curves_with_gaps_are_drawn(window, sweep, recwarn, errors):
    load_sweep(window, sweep)
    process(window)
    entry = add(window, "custom")
    card_of(window, entry).editor.code.edit.setPlainText("100*sqrt(B - 1)\n1/(B - 1)")
    (field, root), (_field, pole) = models_of(window).curve_data()["custom"]
    assert np.isnan(root[field < 1]).all() and np.isfinite(root[field > 1]).all()
    assert np.isfinite(pole).sum() >= field.size - 1  # inf became a gap
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    assert not errors


# ---------------------------------------------------------------------- fitting
def test_fit_zeeman_branches_to_picked_points(window, sweep, tmp_path, errors):
    load_sweep(window, sweep)
    process(window)
    rng = np.random.default_rng(7)
    field, columns = zeeman_points(rng)
    load_table(window, tmp_path, field, columns)
    models = models_of(window)
    entry = add(window, "zeeman")
    card = card_of(window, entry)
    editor = card.editor
    editor.add_button.click()
    editor.rows[0].e0.field.edit.setText(f"{18 * MEV:g}")
    editor.rows[1].e0.field.edit.setText(f"{37 * MEV:g}")
    editor.rows[1].m.edit.setText("-1")
    editor.rows[1].g.field.edit.setText("1")
    start = {p.name: p.value for p in entry.model.params}

    card.fit_button.click()
    area = card.fit_area
    assert not area.isHidden() and list(area.combos) == ["LL 1", "LL 2"]
    assert [area.combos[n].currentText() for n in area.combos] == ["Branch 1", "Branch 2"]
    area.fit_button.click()
    result = models.result(entry)
    assert result is not None and area.error.isHidden()
    truth = {"e0_0": 20.0, "g_0": 2.0, "e0_1": 35.0, "g_1": 1.5}
    for name, value in truth.items():
        sigma = result.stderr[name]
        assert 0 < sigma < 0.05
        assert abs(result.values[name] - value) < 4 * sigma
    assert result.dof == 60 - 4
    assert not area.results.isHidden()
    value, sigma = ms.format_with_sigma(result.values["e0_0"] * MEV, result.stderr["e0_0"] * MEV)
    assert result_cells(area)[0] == ["E₀ (Branch 1)", value, "±", sigma, "cm⁻¹"]
    report = ms.fit_report(entry, result, window.controller.unit)
    assert report.chi2 == pytest.approx(result.chi2 * MEV**2, rel=1e-4)  # (cm-1)^2
    assert area.stats.text().splitlines() == [
        f"χ² {report.chi2:.3g} (cm⁻¹)²",
        f"reduced χ² {report.reduced_chi2:.3g} (cm⁻¹)²",
        "dof 56 · 60 points",
    ]
    preview = models.preview_data(entry)
    assert len(preview) == 2
    np.testing.assert_allclose(
        preview[0][1], convert(branch_energy(preview[0][0], *_lower(result)), "meV", "cm-1")
    )

    area.discard_button.click()
    assert models.result(entry) is None and models.preview_data(entry) == []
    assert {p.name: p.value for p in entry.model.params} == start  # unchanged

    area.fit_button.click()
    fitted = dict(models.result(entry).values)
    area.apply_button.click()
    assert models.result(entry) is None and area.results.isHidden()
    for p in entry.model.params:
        assert p.value == pytest.approx(fitted[p.name])
    shown_e0 = float(editor.rows[0].e0.field.text())  # five significant digits
    assert shown_e0 == pytest.approx(fitted["e0_0"] * MEV, rel=1e-4)
    assert float(editor.rows[1].g.field.text()) == pytest.approx(fitted["g_1"], rel=1e-5)
    # the sliders follow (energies in the display unit)
    assert editor.rows[0].e0.slider.value() == pytest.approx(fitted["e0_0"] * MEV, rel=1e-4)
    assert editor.rows[1].g.slider.value() == pytest.approx(fitted["g_1"])
    assert not errors


def _lower(result):
    v = result.values
    return v["e0_0"], v["g_0"], 1.0


def test_fit_a_custom_expression(window, sweep, tmp_path, errors):
    load_sweep(window, sweep)
    process(window)
    rng = np.random.default_rng(11)
    field = np.linspace(0.5, 16.0, 32)
    thz = 1.5 + 0.01 * field**2 + rng.normal(0, 0.002, field.size)  # THz
    other = np.full(field.size, np.nan)
    other[::4] = 3.0
    load_table(window, tmp_path, field, {"CR": thz, "noise": other}, unit="THz")
    models = models_of(window)
    entry = add(window, "custom")
    card = card_of(window, entry)
    editor = card.editor
    editor.unit.set_value("THz")
    editor.code.edit.setPlainText("E0 + a*B**2  # parabola")
    editor.rows["E0"].value.field.edit.setText("1")
    editor.rows["a"].value.field.edit.setText("0.02")

    card.fit_button.click()
    area = card.fit_area
    assert [area.combos[n].currentText() for n in area.combos] == ["parabola", "Skip"]
    area.fit_button.click()
    result = models.result(entry)
    assert result is not None and result.dof == 30
    for name, value in {"E0": 1.5, "a": 0.01}.items():
        assert abs(result.values[name] - value) < 4 * result.stderr[name]
        assert 0 < result.stderr[name] < 0.01
    report = ms.fit_report(entry, result, window.controller.unit)
    assert [row[0] for row in report.rows] == ["E0", "a"]
    assert report.rows[0][3] == ""  # custom parameters keep the expression's unit
    # chi2 is in the display unit squared: THz residuals -> cm-1
    assert report.chi2 == pytest.approx(result.chi2 * THZ**2)

    area.mode.set_value("nearest")
    assert [area.combos[n].currentText() for n in area.combos] == ["Use", "Skip"]
    choose(area, "noise", "Use")
    area.fit_button.click()  # the noise points pull the curve away
    assert models.result(entry).chi2 > result.chi2
    area.apply_button.click()
    assert float(editor.rows["E0"].value.field.text()) == pytest.approx(
        ms.params(entry)["E0"].value, rel=1e-5
    )
    assert not errors


def test_nearest_keeps_the_branches_of_the_other_modes(window, sweep, tmp_path, errors):
    load_sweep(window, sweep)
    process(window)
    field, columns = zeeman_points(np.random.default_rng(2))
    load_table(window, tmp_path, field, columns)
    entry = add(window, "zeeman")
    card = card_of(window, entry)
    card.editor.add_button.click()
    entry.fit.assignment = Assignment.NEAREST  # e.g. restored: nothing chosen yet
    card.fit_button.click()
    area = card.fit_area

    def shown():
        return [area.combos[n].currentText() for n in area.combos]

    assert shown() == ["Use", "Use"]
    choose(area, "LL 1", "Skip")
    area.mode.set_value("branch")
    assert shown() == ["Skip", "Branch 2"]  # not Branch 1, the "Use" it showed
    area.mode.set_value("nearest")
    choose(area, "LL 2", "Skip")
    choose(area, "LL 2", "Use")
    choose(area, "LL 1", "Use")
    area.mode.set_value("branch")
    assert shown() == ["Branch 1", "Branch 2"]
    assert not errors


def test_edits_that_change_a_model_drop_its_fit(window, sweep, tmp_path, errors):
    load_sweep(window, sweep)
    process(window)
    field = np.linspace(0.5, 15.0, 30)
    load_table(window, tmp_path, field, {"line": 30.0 + 2.0 * MU_B * field})  # meV
    models = models_of(window)
    entry = add(window, "custom")
    card = card_of(window, entry)
    card.editor.code.edit.setPlainText("E0 + g*muB*B")
    card.fit_button.click()
    area = card.fit_area
    area.fit_button.click()
    assert models.result(entry).values["E0"] == pytest.approx(30.0)
    card.editor.unit.set_value("THz")  # the fit was in meV: its values mean nothing in THz
    assert models.result(entry) is None and models.preview_data(entry) == []
    assert area.results.isHidden()
    area.apply_button.click()
    assert ms.params(entry)["E0"].value == 1.0  # nothing stale was written
    area.fit_button.click()
    assert models.result(entry) is not None
    card.editor.code.edit.setPlainText("E0 - g*muB*B")  # the same names, another model
    assert models.result(entry) is None and area.results.isHidden()

    zeeman = add(window, "zeeman")
    zcard = card_of(window, zeeman)
    zcard.fit_button.click()
    zarea = zcard.fit_area
    zarea.fit_button.click()
    assert models.result(zeeman) is not None
    zcard.editor.rows[0].form.click()  # fitted as linear, now hyperbolic
    assert models.result(zeeman) is None and zarea.results.isHidden()
    zarea.fit_button.click()
    zcard.editor.rows[0].m.edit.setText("1.")  # still m = 1: the fit stays
    assert models.result(zeeman) is not None
    zcard.editor.rows[0].m.edit.setText("1.5")
    assert models.result(zeeman) is None and zeeman.model.branches[0].m == 1.5
    assert zcard.editor.rows[0].m.text() == "1.5"  # the refresh keeps what is typed
    assert not errors


def test_a_long_fit_runs_in_the_background_and_can_be_cancelled(
    window, sweep, tmp_path, qtbot, monkeypatch, errors
):
    load_sweep(window, sweep)
    process(window)
    field, columns = zeeman_points(np.random.default_rng(5))
    load_table(window, tmp_path, field, {"LL 1": columns["LL 1"]})
    release = threading.Event()
    original = AppController.run_fit

    def slow(model, observations, assignment):  # waits until the test lets it go on
        release.wait(5)
        return original(model, observations, assignment)

    monkeypatch.setattr(AppController, "run_fit", staticmethod(slow))
    monkeypatch.setattr(models_module, "FIT_WAIT", 0.02)
    models = models_of(window)
    entry = add(window, "zeeman")
    card = card_of(window, entry)
    card.fit_button.click()
    area = card.fit_area
    area.fit_button.click()
    assert models.is_fitting(entry) and models.result(entry) is None  # busy, not frozen
    assert area.fit_button.text() == "Fitting…" and not area.fit_button.isEnabled()
    assert not area.cancel_button.isHidden() and not area.mode.isEnabled()
    release.set()
    qtbot.waitUntil(lambda: models.result(entry) is not None, timeout=5000)
    assert not models.is_fitting(entry) and area.cancel_button.isHidden()
    assert area.fit_button.text() == "Fit" and not area.results.isHidden()

    release.clear()
    area.fit_button.click()
    job = models.jobs[entry]
    area.cancel_button.click()
    assert not models.is_fitting(entry) and area.fit_button.isEnabled()
    release.set()  # the worker stops at its next evaluation of the model
    qtbot.waitUntil(lambda: job.wait(0), timeout=5000)
    qtbot.wait(20)  # the worker's signal arrives and is ignored
    assert isinstance(job.error, models_module.FitCancelled)
    assert models.result(entry) is None and area.error.isHidden()

    release.clear()
    area.fit_button.click()
    job = models.jobs[entry]
    card.editor.rows[0].form.click()  # a fit of the linear branch is void: it stops
    assert not models.is_fitting(entry) and area.cancel_button.isHidden()
    release.set()
    qtbot.waitUntil(lambda: job.wait(0), timeout=5000)
    qtbot.wait(20)
    assert models.result(entry) is None
    assert not errors


def test_fit_problems_are_shown_inline(window, sweep, tmp_path, errors):
    load_sweep(window, sweep)
    process(window)
    entry = add(window, "zeeman")
    card = card_of(window, entry)
    card.fit_button.click()
    area = card.fit_area
    assert not area.empty.isHidden() and not area.fit_button.isEnabled()  # no points yet
    assert not area.points_link.isHidden()
    area.points_link.click()
    assert window.current_panel() == "points" and window.side_panel.is_open()

    load_table(window, tmp_path, np.array([1.0, 2.0, 3.0]), {"LL 1": [20.0, 21.0, np.nan]})
    assert list(area.combos) == ["LL 1"] and area.fit_button.isEnabled()
    area.fit_button.click()  # two points, two free parameters
    assert not area.error.isHidden()
    assert area.error.text() == (
        "2 points cannot fit 2 free parameters; pick more points or fix parameters"
    )
    choose(area, "LL 1", "Skip")
    area.fit_button.click()
    assert area.error.text() == "Assign at least one picked curve to a branch"
    assert models_of(window).result(entry) is None

    custom = add(window, "custom")
    ccard = card_of(window, custom)
    ccard.fit_button.click()
    ccard.editor.code.edit.setPlainText("E0 +")
    assert not ccard.fit_area.fit_button.isEnabled()
    assert not errors  # nothing went to the error bar


def test_results_can_be_copied_and_exported(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    process(window)
    field, columns = zeeman_points(np.random.default_rng(3))
    load_table(window, tmp_path, field, {"LL 1": columns["LL 1"]})
    models = models_of(window)
    entry = add(window, "zeeman")
    card = card_of(window, entry)
    card.editor.rows[0].label.edit.setText("upper")
    card.fit_button.click()
    card.fit_area.fit_button.click()
    result = models.result(entry)
    set_unit(window, "meV")
    text = models.results_text(entry)
    lines = text.splitlines()
    assert lines[0] == "# Model: Zeeman / magnon"
    assert lines[1] == "# Assignment: branch; LL 1 -> upper"
    assert lines[2] == f"# Points: 30; dof: {result.dof}"
    assert "# upper: m = 1, linear" in lines
    header = lines.index("parameter\tvalue\tsigma\tunit")
    rows = [line.split("\t") for line in lines[header + 1 :]]
    assert [r[0] for r in rows] == ["E0 (upper)", "g (upper)"]
    assert rows[0][3] == "meV" and rows[1][3] == ""
    assert float(rows[0][1]) == pytest.approx(result.values["e0_0"])
    assert float(rows[0][2]) == pytest.approx(result.stderr["e0_0"])
    set_unit(window, "cm-1")
    rows_cm1 = [line.split("\t") for line in models.results_text(entry).splitlines()]
    energy = next(r for r in rows_cm1 if r[0] == "E0 (upper)")
    assert float(energy[1]) == pytest.approx(result.values["e0_0"] * MEV)
    assert energy[3] == "cm-1"

    card.fit_area.copy_button.click()
    assert QApplication.clipboard().text() == models.results_text(entry)
    save_to(monkeypatch, tmp_path / "fit")
    card.fit_area.export_button.click()
    assert (tmp_path / "fit.tsv").read_text(encoding="utf-8") == models.results_text(entry)
    assert not errors


# ---------------------------------------------------------------------- figure state
def test_visible_models_are_overlays_of_the_figure(window, sweep, qtbot, errors):
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    models = models_of(window)
    dirac = models.entries[0]
    card_of(window, dirac).visible_switch.setChecked(True)
    zeeman = add(window, "zeeman")
    add(window, "custom")  # no expression: nothing to draw, not an overlay
    state = c.figure_state()
    assert set(state.overlays) == {"dirac", "zeeman"}
    assert len(state.overlays["dirac"]) == 5
    set_unit(window, "THz")
    field, energy = c.figure_state().overlays["zeeman"][0]
    e0 = ms.params(zeeman)["e0_0"].value
    np.testing.assert_allclose(energy, convert(branch_energy(field, e0, 2.0, 1.0), "meV", "THz"))
    with qtbot.waitSignal(c.overlaysChanged, timeout=1000):
        card_of(window, zeeman).editor.rows[0].g.field.edit.setText("3")
    card_of(window, zeeman).visible_switch.setChecked(False)
    assert set(c.figure_state().overlays) == {"dirac"}
    card_of(window, dirac).remove_button.click()
    assert c.figure_state().overlays == {}
    assert not errors


def test_the_section_never_widens_the_inspector(window, sweep, tmp_path, errors):
    """With every card open and a fit shown, Models fits the inspector at its narrowest (its
    content cannot scroll sideways, so wider content would be clipped)."""
    load_sweep(window, sweep)
    process(window)
    field, columns = zeeman_points(np.random.default_rng(1))
    load_table(window, tmp_path, field, columns)
    models = models_of(window)
    add(window, "zeeman")
    custom = add(window, "custom")
    card_of(window, custom).editor.code.edit.setPlainText("E0 + a*B\nE1 + b*B**2  # second")
    zeeman = models.entries[1]
    card_of(window, zeeman).editor.add_button.click()
    card_of(window, zeeman).editor.coupled.setChecked(True)
    for entry in models.entries:
        card_of(window, entry).fit_button.click()
    card_of(window, zeeman).fit_area.fit_button.click()
    assert models.result(zeeman) is not None
    scrollbar = window.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
    narrowest = INSPECTOR_MIN_WIDTH - scrollbar
    assert window.inspector["models"].minimumSizeHint().width() <= narrowest
    assert not errors


def test_zeeman_energies_fit_their_fields_in_a_narrow_window(window, sweep, qtbot, errors):
    """At 1100 px with the side panel open the inspector keeps its narrow width (280 px) and
    each E0 shows all its digits (once shown as "ꞁ22.624", read as 22.6)."""
    window.resize(1100, 800)
    window.show()
    qtbot.waitExposed(window)
    load_sweep(window, sweep)
    process(window)
    zeeman = add(window, "zeeman")
    editor = card_of(window, zeeman).editor
    editor.add_button.click()
    p = ms.params(zeeman)
    p["e0_0"].value, p["e0_1"].value = 40.0, 60.0  # meV
    models_of(window).edited(zeeman, structure=True)
    qtbot.wait(50)
    assert window.inspector_panel.width() >= 280 and window.side_panel.is_open()
    texts = []
    for row in editor.rows:
        edit = row.e0.field.edit
        texts.append(edit.text())
        assert edit.width() >= edit.fontMetrics().horizontalAdvance(edit.text())
        assert edit.cursorPosition() == 0  # a longer number would show its start
    assert texts == ["322.62", "483.94"]  # five significant digits, in cm⁻¹
    assert not errors


# ---------------------------------------------------------------------- settings
def test_models_are_remembered(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.toolbar.unit.set_value("meV")
    models = models_of(w)
    dirac = models.entries[0]
    dcard = card_of(w, dirac)
    dcard.visible_switch.setChecked(True)
    dcard.editor.delta.field.edit.setText("7.5")  # meV
    dcard.editor.n_lines.field.edit.setText("4")
    zeeman = add(w, "zeeman")
    zcard = card_of(w, zeeman)
    zcard.editor.add_button.click()
    zcard.editor.rows[0].e0.field.edit.setText("12")
    zcard.editor.rows[1].label.edit.setText("magnon")
    zcard.editor.rows[1].form.click()
    zcard.editor.coupled.setChecked(True)
    zcard.editor.couplings[(0, 1)].field.edit.setText("2.5")
    zcard.editor.coupled.setChecked(False)  # the 2.5 meV is kept aside
    zcard.swatch.colorChosen.emit(ms.MODEL_COLORS[3])
    custom = add(w, "custom")
    ccard = card_of(w, custom)
    ccard.editor.code.edit.setPlainText("E0 + a*B")
    ccard.editor.unit.set_value("THz")
    ccard.editor.rows["a"].value.field.edit.setText("0.25")
    ccard.editor.rows["a"].fixed.setChecked(True)
    ccard.editor.rows["E0"].hi.edit.setText("5")
    ccard.chevron.click()  # collapsed
    card_of(w, custom).fit_area.mode.set_value("sorted")
    w.close()

    data = json.loads(QSettings(ini, QSettings.Format.IniFormat).value(f"{PREFIX}/models/list"))
    assert data["models"][1]["branches"][0]["e0_meV"] == 12.0  # energies in meV
    assert data["models"][1]["couplings_meV"] == [[0, 1, 2.5]]

    w2 = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w2)
    restored = models_of(w2).entries
    assert [(e.kind, e.key, e.name) for e in restored] == [
        ("dirac", "dirac", "Massive Dirac"),
        ("zeeman", "zeeman", "Zeeman / magnon"),
        ("custom", "custom", "Custom expression"),
    ]
    d, z, x = restored
    assert d.visible and ms.params(d)["delta"].value == 7.5 and d.model.n_lines == 4
    assert [b.label for b in z.model.branches] == ["Branch 1", "magnon"]
    assert z.model.branches[1].form is Form.HYPERBOLIC
    assert ms.params(z)["e0_0"].value == 12.0
    assert not z.model.coupled and z.model.couplings == {(0, 1): 2.5}
    assert z.color == ms.MODEL_COLORS[3]
    assert x.text == "E0 + a*B" and x.unit is Unit.THZ and not x.expanded
    assert ms.params(x)["a"].value == 0.25 and ms.params(x)["a"].fixed
    assert ms.params(x)["E0"].hi == 5.0 and x.edited == {"E0", "a"}
    assert x.fit.assignment == "sorted"
    zcard2 = card_of(w2, z)
    assert zcard2.editor.rows[0].e0.field.text() == "12"  # shown in meV, the restored unit
    zcard2.editor.coupled.setChecked(True)
    assert ms.params(z)["delta_0_1"].value == 2.5
    assert card_of(w2, x).body.isHidden()


def test_stored_transitions_stay_within_the_limit():
    data = ms.entry_to_dict(ms.new_entry("dirac", []))
    for stored, expected in ((100000, 40), (0, 1), (-3, 1), (12, 12)):
        assert ms.entry_from_dict(data | {"n_lines": stored}).model.n_lines == expected


def test_corrupt_stored_parameters_are_rejected():
    entry = ms.new_entry("custom", [])
    ms.set_expression(entry, "E0 + a*B")
    data = ms.entry_to_dict(entry)
    assert ms.entry_from_dict(data).text == "E0 + a*B"
    for params in (["E0"], [None], "E0"):
        with pytest.raises(ValueError, match="invalid model"):
            ms.entry_from_dict(data | {"params": params})


def test_invalid_stored_models_keep_the_defaults(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    s = QSettings(ini, QSettings.Format.IniFormat)
    s.setValue(f"{PREFIX}/models/list", json.dumps({"models": [{"kind": "nonsense"}]}))
    s.sync()
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    assert [e.kind for e in models_of(w).entries] == ["dirac"]


# ---------------------------------------------------------------------- parameter rows
def shown_window(window, qtbot, size=(1100, 800)):
    window.resize(*size)
    window.show()
    qtbot.waitExposed(window)


def settle(qtbot) -> None:
    QApplication.processEvents()
    qtbot.wait(30)


def every_model(window) -> None:
    """Dirac (shown), a coupled Zeeman model with three branches and a custom expression with
    five parameters, every card open."""
    models = models_of(window)
    models.set_visible(models.entries[0], True)
    zeeman = add(window, "zeeman")
    editor = card_of(window, zeeman).editor
    editor.add_button.click()
    editor.add_button.click()
    editor.coupled.setChecked(True)
    custom = add(window, "custom")
    card_of(window, custom).editor.code.edit.setPlainText(
        "E0 + amplitude*sqrt(B) + b*B  # LL\nE1 + g*muB*B"
    )


def shown_in(widget, ancestor) -> bool:
    """Whether *widget* shows with *ancestor* (nothing between them was hidden)."""
    explicit = Qt.WidgetAttribute.WA_WState_ExplicitShowHide
    while widget is not ancestor:
        if widget.isHidden() and widget.testAttribute(explicit):
            return False
        widget = widget.parentWidget()
    return True


def param_rows(window) -> list[ParamRow]:
    """The parameter rows of every card that are not hidden (e.g. uncoupled couplings)."""
    return [
        row
        for card in models_of(window).cards.values()
        for row in card.findChildren(ParamRow)
        if shown_in(row, card)
    ]


@pytest.mark.parametrize(
    ("size", "inspector"), [((1100, 800), None), ((1400, 900), "narrowest"), ((1400, 900), 480)]
)
def test_value_fields_share_one_column(window, sweep, qtbot, size, inspector, errors):
    """In every card the number fields share x and width (the same column in every card), at
    1100 px, with the inspector dragged to its narrowest and wide; captions sit level with
    their fields, and the slider moves under them when the inspector is narrow."""
    shown_window(window, qtbot, size)
    load_sweep(window, sweep)
    process(window)
    every_model(window)
    if inspector is not None:
        splitter = window.body_splitter
        sizes = splitter.sizes()
        width = INSPECTOR_MIN_WIDTH if inspector == "narrowest" else inspector
        splitter.setSizes([sizes[0], sum(sizes) - sizes[0] - width, width])
    settle(qtbot)
    width = INSPECTOR_MIN_WIDTH if inspector in (None, "narrowest") else inspector
    assert window.inspector_panel.width() == width
    section = window.inspector["models"]
    rows = param_rows(window)
    assert len(rows) == 3 + 6 + 3 + 5  # Dirac v, Δ, N; E₀ and g of 3 branches; Δij; custom
    qtbot.waitUntil(  # the new rows are laid out (one line or two)
        lambda: all(r.height() == r.heightForWidth(r.width()) for r in rows), timeout=5000
    )
    boxes = {(r.field.mapTo(section, QPoint(0, 0)).x(), r.field.width()) for r in rows}
    assert len(boxes) == 1, boxes
    wide = inspector == 480
    for row in rows:
        field, caption, slider = row.field.geometry(), row.caption.geometry(), row.slider.geometry()
        assert field.height() == FIELD_HEIGHT and caption.center().y() == field.center().y()
        assert row.is_wide() == wide
        if wide:  # caption | slider | field on one line
            assert caption.right() < slider.left() and slider.right() < field.left()
            assert abs(slider.center().y() - field.center().y()) <= 1
        else:  # caption and field, the slider under them across the row
            assert slider.top() > field.bottom() and slider.width() == row.width()
        edit = row.field.edit
        assert edit.width() >= edit.fontMetrics().horizontalAdvance(edit.text())
    # captions share their column too (wide: the slider column starts in one place)
    assert len({r.slider.mapTo(section, QPoint(0, 0)).x() for r in rows}) == 1
    scrollbar = window.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
    assert section.minimumSizeHint().width() <= INSPECTOR_MIN_WIDTH - scrollbar
    assert not errors


def drag_by(qtbot, slider, offsets, release=True) -> None:
    """Press a slider in its centre and move by *offsets* (of the half track)."""
    x0, x1 = slider.track()
    y = slider.height() // 2
    centre = round(slider.centre_x())
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(centre, y))
    for offset in offsets:
        qtbot.mouseMove(slider, QPoint(round(centre + offset * (x1 - x0) / 2), y))
    if release:
        end = round(centre + offsets[-1] * (x1 - x0) / 2)
        qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(end, y))


def test_g_has_a_slider_that_drags_the_zeeman_curve(window, sweep, qtbot, errors):
    shown_window(window, qtbot)
    load_sweep(window, sweep)
    process(window)
    models = models_of(window)
    zeeman = add(window, "zeeman")
    row = card_of(window, zeeman).editor.rows[0]
    p = ms.params(zeeman)
    e0 = p["e0_0"].value
    assert p["g_0"].value == 2.0 and row.g.slider.mode() == "relative"  # ±10 % by default
    assert row.g.slider.accessibleName() == "g factor of Branch 1"
    before = models.curve_data()["zeeman"][0][1].copy()

    drag_by(qtbot, row.g.slider, [0.5, 1.2], release=False)  # past the end: the whole span
    assert p["g_0"].value == pytest.approx(2.2) and row.g.field.text() == "2.2"  # at once
    qtbot.waitUntil(lambda: not np.allclose(models.curve_data()["zeeman"][0][1], before))
    x1 = row.g.slider.track()[1]
    qtbot.mouseRelease(row.g.slider, Qt.MouseButton.LeftButton, pos=QPoint(round(x1), 10))
    field, energy = models.curve_data()["zeeman"][0]  # drawn on release, without waiting
    np.testing.assert_allclose(energy, convert(branch_energy(field, e0, 2.2, 1.0), "meV", "cm-1"))
    qtbot.waitUntil(lambda: not row.g.slider.is_springing(), timeout=2000)
    assert row.g.slider.handle_offset() == 0.0 and p["g_0"].value == pytest.approx(2.2)

    # E₀ is dragged in the display unit (cm⁻¹) and kept in meV
    drag_by(qtbot, row.e0.slider, [-0.5])
    expected = e0 * (1 - 0.1 * jog_fraction(0.5))
    assert p["e0_0"].value == pytest.approx(expected, rel=2e-3)
    assert float(row.e0.field.text()) == pytest.approx(p["e0_0"].value * MEV, rel=1e-4)
    assert row.e0.slider.value() == pytest.approx(p["e0_0"].value * MEV, rel=1e-6)
    field, energy = models.curve_data()["zeeman"][0]
    np.testing.assert_allclose(
        energy, convert(branch_energy(field, p["e0_0"].value, 2.2, 1.0), "meV", "cm-1")
    )

    # range mode: the handle spans 0 … 10 for g
    inspector_page(window, "models").models.slider_mode.set("range", 10.0)
    slider = row.g.slider
    assert slider.mode() == "range" and slider.range() == (0.0, 10.0)
    assert slider.handle_x() == pytest.approx(slider.x_for(2.2))
    qtbot.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(round(slider.x_for(5)), 10))
    assert p["g_0"].value == pytest.approx(5.0, abs=0.05)
    assert float(row.g.field.text()) == pytest.approx(p["g_0"].value)
    field, energy = models.curve_data()["zeeman"][0]
    np.testing.assert_allclose(
        energy,
        convert(branch_energy(field, p["e0_0"].value, p["g_0"].value, 1.0), "meV", "cm-1"),
    )
    assert not errors


def test_every_continuous_parameter_has_a_slider(window, sweep, qtbot, errors):
    shown_window(window, qtbot)
    load_sweep(window, sweep)
    process(window)
    every_model(window)
    models = models_of(window)
    dirac, zeeman, custom = models.entries
    names = {row.slider.accessibleName() for row in param_rows(window)}
    assert {"Fermi velocity v", "Half-gap Δ", "Transitions shown"} <= names
    assert {"E₀ of Branch 3", "g factor of Branch 2", "Coupling Δ 1–3"} <= names
    assert {"E0", "amplitude", "b", "E1", "g"} <= names
    transitions = card_of(window, dirac).editor.n_lines.slider
    assert transitions.mode() == "range" and transitions.modes() == ("range",)  # whole numbers

    coupling = card_of(window, zeeman).editor.couplings[(0, 2)]
    drag_by(qtbot, coupling.slider, [1.0])  # 1 meV + 10 %
    assert ms.params(zeeman)["delta_0_2"].value == pytest.approx(1.1, rel=1e-3)
    row = card_of(window, custom).editor.rows["amplitude"].value
    drag_by(qtbot, row.slider, [1.0])  # a new parameter is 1: +10 %
    assert ms.params(custom)["amplitude"].value == pytest.approx(1.1)
    assert "amplitude" in custom.edited  # remembered as a value the user set
    row.set_value(0.0)
    ms.params(custom)["amplitude"].value = 0.0
    drag_by(qtbot, row.slider, [-1.0])  # zero is not a dead end: 10 % of 1
    assert ms.params(custom)["amplitude"].value == pytest.approx(-0.1)
    assert not errors


def test_one_slider_mode_for_every_card_is_remembered(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    page = inspector_page(w, "models")
    models = page.models
    zeeman = add(w, "zeeman")
    assert page.mode_button.text() == "Relative ±10 %" and not page.mode_row.isHidden()
    g = card_of(w, zeeman).editor.rows[0].g.slider
    g.mode_menu().actions()[3].trigger()  # a slider's context menu: Relative ±50 %
    sliders = [row.slider for row in param_rows(w) if row.slider.modes() != ("range",)]
    assert len(sliders) >= 4
    assert {(s.mode(), s.span()) for s in sliders} == {("relative", 50.0)}
    assert page.mode_button.text() == "Relative ±50 %"
    menu = models.slider_mode.menu(page.mode_button)  # the section's control
    assert [a.text() for a in menu.actions() if a.isChecked()] == ["Relative ±50 %"]
    menu.actions()[0].trigger()
    assert {s.mode() for s in sliders} == {"range"} and page.mode_button.text() == "Range"
    custom = add(w, "custom")  # rows made later take the mode
    card_of(w, custom).editor.code.edit.setPlainText("E0 + a*B")
    assert card_of(w, custom).editor.rows["a"].value.slider.mode() == "range"
    menu.actions()[2].trigger()  # Relative ±10 %
    menu = models.slider_mode.menu(page.mode_button)
    menu.actions()[1].trigger()  # Relative ±1 %
    w.close()
    assert QSettings(ini, QSettings.Format.IniFormat).value(f"{PREFIX}/models/slider_mode") == (
        "relative-1"
    )

    w2 = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w2)
    assert inspector_page(w2, "models").mode_button.text() == "Relative ±1 %"
    assert {(r.slider.mode(), r.slider.span()) for r in param_rows(w2)} >= {("relative", 1.0)}
    w2.close()
    s = QSettings(ini, QSettings.Format.IniFormat)
    s.setValue(f"{PREFIX}/models/slider_mode", "sideways")
    s.sync()
    w3 = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w3)
    assert inspector_page(w3, "models").mode_button.text() == "Relative ±10 %"  # the default


def test_a_unit_switch_keeps_values_suffixes_and_sliders(window, sweep, qtbot, errors):
    shown_window(window, qtbot)
    load_sweep(window, sweep)
    process(window)
    every_model(window)
    models = models_of(window)
    dirac, zeeman, _custom = models.entries
    zeditor = card_of(window, zeeman).editor
    ms.params(zeeman)["e0_0"].value = 20.0
    ms.params(dirac)["delta"].value = 10.0
    models.refresh()
    models.slider_mode.set("range", 10.0)
    rows = {
        "E₀": (zeditor.rows[0].e0, lambda: ms.params(zeeman)["e0_0"].value),
        "Δ": (card_of(window, dirac).editor.delta, lambda: ms.params(dirac)["delta"].value),
        "Δ 1–2": (zeditor.couplings[(0, 1)], lambda: zeeman.model.couplings[(0, 1)]),
    }
    for unit, text in (("meV", "meV"), ("THz", "THz"), ("cm-1", "cm⁻¹")):
        set_unit(window, unit)
        settle(qtbot)
        for name, (row, stored) in rows.items():
            shown = float(convert(stored(), "meV", unit))
            assert row.field.unit_label.text() == text, name
            assert float(row.field.text()) == pytest.approx(shown, rel=1e-4), name
            assert row.slider.value() == pytest.approx(shown), name
            lo, hi = row.slider.range()
            assert lo <= shown <= hi, name  # the range follows the unit
            edit = row.field.edit
            assert edit.width() >= edit.fontMetrics().horizontalAdvance(edit.text()), name
    assert ms.params(zeeman)["e0_0"].value == pytest.approx(20.0)  # kept in meV
    hi = zeditor.rows[0].e0.slider.range()[1]
    assert hi >= sweep["x"][-1]  # E₀ spans the processed map (cm⁻¹ now)
    assert not errors


def test_the_fit_area_takes_the_place_of_its_button(window, errors):
    zeeman = add(window, "zeeman")
    card = card_of(window, zeeman)
    assert not card.fit_button.isHidden() and card.fit_area.isHidden()
    card.fit_button.click()
    assert card.fit_button.isHidden() and not card.fit_area.isHidden()
    card.fit_area.close_button.click()
    assert not card.fit_button.isHidden() and card.fit_area.isHidden()
    assert not errors
