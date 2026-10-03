"""Models section: model transition energies E(B) over the map, fitted to the picked points.

The models are cards (``model_cards``): massive Dirac, Zeeman / magnon branches (optionally
coupled) and custom expressions in B, each with its own colour, a show switch and a fit area
(``model_fit``). Every visible model is drawn dashed with a dark shadow in the overlay layer of
its place in the list ("models" for the first, then "models 2", ...); a fit is previewed in
"models fit", "models fit 2", ... Curves are drawn in the display unit, live while parameters
are edited and on unit switches, and each visible model is registered with
``controller.set_overlay(key, fn(unit))`` so ``figure_state()`` (and the export) include it.

The list is saved under ``models/list`` (JSON, energies in meV). The keys of the former
Overlays section (``models/show_dirac``, ``velocity``, ``delta``, ``n_lines``) are taken over
by the Dirac card once.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QVBoxLayout, QWidget

from mag_opt_detective.core.fitting import FitResult, apply
from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import EXPECTED_ERRORS, AppController, user_action
from mag_opt_detective.gui.inspector import model_state as ms
from mag_opt_detective.gui.inspector.model_cards import ModelCard
from mag_opt_detective.gui.inspector.model_fit import FitArea
from mag_opt_detective.gui.inspector.view import update_sections
from mag_opt_detective.gui.panels.common import LinkButton, hint
from mag_opt_detective.gui.points_view import curve_color
from mag_opt_detective.gui.settings import PREFIX
from mag_opt_detective.gui.widgets import parse_float, save_file

logger = logging.getLogger("mag_opt_detective")

LAYER, PREVIEW_LAYER = "models", "models fit"
SETTINGS_KEY = "models/list"
LEGACY_KEYS = ("models/show_dirac", "models/velocity", "models/delta", "models/n_lines")
SHADOW_PEN = pg.mkPen((0, 0, 0, 110), width=3)  # as the processing guides
NO_MODELS = (
    "No models. Add one to draw transition energies over the map and fit them to the picked points."
)
KIND_ICONS = {ms.DIRAC: "activity", ms.ZEEMAN: "git-merge", ms.CUSTOM: "sigma"}
KIND_TIPS = {
    ms.DIRAC: "Interband transitions L₋ₙ → Lₙ₊₁ of a massive Dirac band",
    ms.ZEEMAN: "Branches E₀ ± m g μB B, linear or hyperbolic, optionally coupled",
    ms.CUSTOM: "Your own expressions in B, one branch per line",
}


def layer_name(slot: int, preview: bool = False) -> str:
    """The overlay layer of the model at *slot* in the list (or of its fit preview)."""
    base = PREVIEW_LAYER if preview else LAYER
    return base if slot == 0 else f"{base} {slot + 1}"


def curve_pen(color: str) -> pg.QtGui.QPen:
    qcolor = QColor(color)
    qcolor.setAlpha(230)
    return pg.mkPen(qcolor, width=1.5, style=Qt.PenStyle.DashLine)


def preview_pen(color: str) -> pg.QtGui.QPen:
    pen = pg.mkPen(QColor(color), width=2.4, style=Qt.PenStyle.CustomDashLine)
    pen.setDashPattern([1.5, 1.5])
    return pen


class ModelsPage(QWidget):
    """The model cards; "Add model…" sits in the section header."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.add_button = LinkButton("Add model…", "Add a model")
        self.add_menu = QMenu(self.add_button)
        self.add_actions = {}
        for kind, name in ms.KIND_NAMES.items():
            action = self.add_menu.addAction(name)
            action.setToolTip(KIND_TIPS[kind])
            icons.set_icon(action, KIND_ICONS[kind], "muted")
            self.add_actions[kind] = action
        self.add_menu.setToolTipsVisible(True)
        self.add_button.clicked.connect(
            lambda: self.add_menu.exec(
                self.add_button.mapToGlobal(self.add_button.rect().bottomLeft())
            )
        )
        self.empty = hint(NO_MODELS)
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.empty)
        layout.addLayout(self.cards_layout)
        self.models: Models | None = None  # set by install()

    def cards(self) -> list[ModelCard]:
        items = (self.cards_layout.itemAt(i).widget() for i in range(self.cards_layout.count()))
        return [card for card in items if isinstance(card, ModelCard)]


class Models(QObject):
    """The model list, its cards, the curves on the map and the fits."""

    def __init__(self, window, page: ModelsPage):
        super().__init__(page)
        self.window = window
        self.page = page
        self.c: AppController = window.controller
        self.entries: list[ms.ModelEntry] = ms.default_entries()
        self.cards: dict[ms.ModelEntry, ModelCard] = {}
        self.results: dict[ms.ModelEntry, tuple[FitResult, dict[str, int | None]]] = {}
        self._registered: set[str] = set()
        self._slots = 0  # model layers in use
        self._previews: set[str] = set()  # preview layers created
        page.models = self
        for kind, action in page.add_actions.items():
            action.triggered.connect(lambda _checked=False, k=kind: self.add(k))
        self.rebuild()

    @property
    def unit(self) -> Unit:
        return self.c.unit

    # --- the list -------------------------------------------------------------------------
    def rebuild(self) -> None:
        """Cards for every entry (after a restore), then draw."""
        for card in self.cards.values():
            self.page.cards_layout.removeWidget(card)
            card.hide()
            card.deleteLater()
        self.cards = {}
        self.results = {}
        for entry in self.entries:
            self._add_card(entry)
        self.page.empty.setVisible(not self.entries)
        self.draw()

    def _add_card(self, entry: ms.ModelEntry) -> ModelCard:
        card = ModelCard(entry, self, FitArea(entry, self))
        card.fit_area.close_button.clicked.connect(lambda: card.fit_button.setChecked(False))
        self.page.cards_layout.addWidget(card)
        self.cards[entry] = card
        card.refresh()
        return card

    def set_entries(self, entries: list[ms.ModelEntry]) -> None:
        self.entries = list(entries)
        self.rebuild()

    def add(self, kind: str) -> ms.ModelEntry:
        """Add a model of *kind* ("dirac", "zeeman", "custom"), shown and expanded."""
        entry = ms.new_entry(kind, self.entries, self.energy_span())
        self.entries.append(entry)
        card = self._add_card(entry)
        self.page.empty.hide()
        self.draw()
        logger.info("Model added: %s", entry.name)
        if kind == ms.CUSTOM:
            card.editor.code.edit.setFocus()
        return entry

    def remove(self, entry: ms.ModelEntry) -> None:
        if entry not in self.entries:
            return
        self.entries.remove(entry)
        self.results.pop(entry, None)
        card = self.cards.pop(entry)
        self.page.cards_layout.removeWidget(card)
        card.hide()
        card.deleteLater()
        self.page.empty.setVisible(not self.entries)
        self.draw()
        logger.info("Model removed: %s", entry.name)

    def entry(self, key: str) -> ms.ModelEntry:
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def energy_span(self) -> tuple[float, float] | None:
        """The energy range of the processed map in meV (to place new branches in it)."""
        if self.c.result is None:
            return None
        energy = from_cm1(self.c.result.ratio.energy, Unit.MEV)
        energy = energy[np.isfinite(energy)]
        return (float(energy.min()), float(energy.max())) if energy.size else None

    # --- edits from the cards -------------------------------------------------------------
    def edited(self, entry: ms.ModelEntry, structure: bool = False) -> None:
        """*entry*'s parameters changed; *structure*: its branches or parameters did."""
        card = self.cards.get(entry)
        if structure:
            self.results.pop(entry, None)
            if card is not None:
                card.refresh()
        if card is not None:
            card.fit_area.refresh()
        self.draw()

    def set_visible(self, entry: ms.ModelEntry, visible: bool) -> None:
        entry.visible = bool(visible)
        self.cards[entry].refresh()
        self.draw()

    def set_color(self, entry: ms.ModelEntry, color: str) -> None:
        entry.color = color
        self.cards[entry].swatch.set_color(color)
        self.draw()

    def set_expanded(self, entry: ms.ModelEntry, expanded: bool) -> None:
        entry.expanded = bool(expanded)
        self.cards[entry].refresh()

    def refresh(self) -> None:
        """Show every card again (e.g. in another unit)."""
        if self.c.is_restoring():
            return
        for card in self.cards.values():
            card.refresh()
            card.fit_area.refresh()
        self.draw()

    # --- drawing --------------------------------------------------------------------------
    def curves(self, entry: ms.ModelEntry, unit: Unit | str) -> list[ms.Curve]:
        """The curves of *entry* over the processed map's fields, in *unit* ([] if hidden)."""
        if self.c.result is None or not entry.visible:
            return []
        try:
            return entry.curves(self.c.result.ratio.field, unit)
        except ValueError as exc:  # e.g. a parameter the expression cannot use
            logger.debug("%s cannot be drawn: %s", entry.name, exc)
            return []

    def preview(self, entry: ms.ModelEntry, unit: Unit | str) -> list[ms.Curve]:
        """The curves of *entry*'s fit result (not applied yet)."""
        item = self.results.get(entry)
        if item is None or self.c.result is None or entry.model is None:
            return []
        grid = ms.field_grid(self.c.result.ratio.field, entry.kind)
        return ms.evaluate_curves(entry.model, grid, unit, item[0].values)

    def draw(self) -> None:
        """Draw every model in its layer, the fit previews, and register the overlays."""
        plot = self.window.plots.map
        unit = self.c.unit
        for slot, entry in enumerate(self.entries):
            layer = plot.layer(layer_name(slot))
            curves = self.curves(entry, unit)
            if curves:
                layer.set_curves(curves, curve_pen(entry.color), SHADOW_PEN)
            else:
                layer.clear()
            preview = self.preview(entry, unit)
            name = layer_name(slot, preview=True)
            if preview:
                self._previews.add(name)
                plot.layer(name).set_curves(preview, preview_pen(entry.color), SHADOW_PEN)
            elif name in self._previews:
                plot.layer(name).clear()
        for slot in range(len(self.entries), self._slots):
            plot.layer(layer_name(slot)).clear()
            if layer_name(slot, preview=True) in self._previews:
                plot.layer(layer_name(slot, preview=True)).clear()
        self._slots = len(self.entries)
        self._register()
        self.c.notify_overlays()

    def _register(self) -> None:
        keys = set()
        for entry in self.entries:
            if entry.visible and entry.is_drawable():
                self.c.set_overlay(entry.key, lambda unit, e=entry: self.curves(e, unit))
                keys.add(entry.key)
        for key in self._registered - keys:
            self.c.set_overlay(key, None)
        self._registered = keys

    def curve_data(self) -> dict[str, list[ms.Curve]]:
        """(field, energy) of the curves drawn per model key (for tests and checks)."""
        plot = self.window.plots.map
        return {
            entry.key: plot.layer(layer_name(slot)).curve_data()
            for slot, entry in enumerate(self.entries)
        }

    def preview_data(self, entry: ms.ModelEntry) -> list[ms.Curve]:
        name = layer_name(self.entries.index(entry), preview=True)
        return self.window.plots.map.layer(name).curve_data() if name in self._previews else []

    # --- fitting --------------------------------------------------------------------------
    def picked(self) -> list[tuple[str, int, QColor]]:
        """The picked curves with points: name, count and their colour on the plots."""
        names = self.c.curve_names()
        return [
            (name, count, curve_color(names.index(name)))
            for name, count in self.c.picked_curves().items()
        ]

    def mapping(self, entry: ms.ModelEntry) -> dict[str, int | None]:
        return ms.resolve_mapping(entry, list(self.c.picked_curves()))

    def fit(self, entry: ms.ModelEntry) -> None:
        """Fit *entry* to the picked curves; ValueError explains what is missing."""
        if entry.model is None or not entry.is_drawable():
            raise ValueError("enter a valid expression first")
        mapping = self.mapping(entry)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.c.fit_model(entry.model, mapping, entry.fit.assignment)
        except EXPECTED_ERRORS:
            raise
        except Exception as exc:  # a bug, not the data: the dialog and the log say so
            logger.exception("Fit of %s failed", entry.name)
            self.window.report_error("Fit", str(exc) or type(exc).__name__, expected=False)
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.results[entry] = (result, mapping)
        self.draw()

    def result(self, entry: ms.ModelEntry) -> FitResult | None:
        item = self.results.get(entry)
        return None if item is None else item[0]

    def apply(self, entry: ms.ModelEntry) -> None:
        """Write the fitted values into the model (the preview becomes the model)."""
        item = self.results.pop(entry, None)
        if item is None:
            return
        apply(entry.model, item[0])
        card = self.cards[entry]
        card.refresh()
        card.fit_area.refresh()
        self.draw()
        logger.info("Fit applied to %s", entry.name)

    def discard(self, entry: ms.ModelEntry) -> None:
        if self.results.pop(entry, None) is not None:
            self.cards[entry].fit_area.refresh()
            self.draw()

    def results_text(self, entry: ms.ModelEntry) -> str:
        """The fit result of *entry* as a tab-separated table (energies in the display unit)."""
        result, mapping = self.results[entry]
        return ms.report_tsv(entry, result, self.c.unit, ms.mapping_text(mapping, entry))

    def copy_results(self, entry: ms.ModelEntry) -> None:
        if entry in self.results:
            QApplication.clipboard().setText(self.results_text(entry))

    def export_results(self, entry: ms.ModelEntry) -> None:
        export_results(self.window, self, entry)

    def open_points(self) -> None:
        self.window.show_panel("points")

    def points_changed(self) -> None:
        """The picked curves changed: the open fit areas list them again."""
        for card in self.cards.values():
            card.fit_area.refresh()


@user_action("Export fit results")
def export_results(window, models: Models, entry: ms.ModelEntry) -> None:
    if entry not in models.results:
        return
    path = save_file(window, "Export fit results")
    if not path:
        return
    out = Path(path) if Path(path).suffix else Path(path).with_suffix(".tsv")
    out.write_text(models.results_text(entry), encoding="utf-8")
    logger.info("Fit results of %s exported to %s", entry.name, out)


# ---------------------------------------------------------------------- settings
class ModelsSetting:
    """Settings protocol for the model list: JSON with the energies in meV, applied once
    settings are restored. Without a stored list the old Overlays keys are taken over."""

    def __init__(self, models: Models):
        self.models = models
        self.pending: list[ms.ModelEntry] | None = None

    def settings_value(self) -> str:
        return ms.entries_to_json(self.models.entries)

    def set_settings_value(self, value) -> bool:
        try:
            entries = ms.entries_from_json(value)
        except (ValueError, TypeError):
            return False
        self.pending = entries
        return True

    def apply(self) -> None:
        if self.pending is not None:
            entries, self.pending = self.pending, None
            self.models.set_entries(entries)
        else:
            self.migrate()
        self.models.refresh()

    def migrate(self) -> None:
        """Give the Dirac card the values of the old Overlays section, then forget them."""
        p = self.models.window.persistence
        if p is None:
            return
        stored = {key: p.value(key) for key in LEGACY_KEYS}
        if all(value is None for value in stored.values()):
            return
        dirac = next((e for e in self.models.entries if e.kind == ms.DIRAC), None)
        if dirac is not None:
            params = ms.params(dirac)
            show = _stored_bool(stored["models/show_dirac"])
            velocity = _stored_number(stored["models/velocity"])
            delta = _stored_number(stored["models/delta"])
            n_lines = _stored_number(stored["models/n_lines"])
            if show is not None:
                dirac.visible = show
            if velocity is not None and velocity > 0:
                params["velocity"].value = velocity
            if delta is not None and delta >= 0:
                params["delta"].value = delta  # the old field was in meV
            if n_lines is not None and n_lines.is_integer() and 1 <= n_lines <= 40:
                dirac.model.n_lines = int(n_lines)
            self.models.rebuild()
            logger.info("Took over the Dirac overlay settings of the previous version.")
        for key in LEGACY_KEYS:
            p.settings.remove(f"{PREFIX}/{key}")


def _stored_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def _stored_number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    number = float(value) if isinstance(value, int | float) else parse_float(str(value))
    return number if number is not None and math.isfinite(number) else None


# ---------------------------------------------------------------------- install
def install(window) -> None:
    c = window.controller
    page = ModelsPage()
    section = window.add_inspector_section("models", "Models", page)
    section.set_trailing(page.add_button)
    models = Models(window, page)
    window.inspector_views.pop("overlays", None)
    window.inspector_views["models"] = ("map",)
    update_sections(window)

    setting = ModelsSetting(models)
    c.restored.connect(setting.apply)
    c.resultChanged.connect(models.draw)
    c.unitChanged.connect(lambda _old, _new: models.refresh())
    c.pointsChanged.connect(models.points_changed)
    if window.persistence is not None:
        window.persistence.bind(SETTINGS_KEY, setting)
