"""The models of the Models section, kept apart from their widgets (no Qt).

A :class:`ModelEntry` is one card: a core fit model (massive Dirac, Zeeman / magnon branches or
a custom expression in B) with its name, colour, visibility and fit setup. Built-in energy
parameters (E0, Δ, Δij) are kept in meV and shown in the display unit; a custom expression
keeps its parameters in its own output unit and only its curves are converted. Entries are
saved as JSON with the energies in meV.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from mag_opt_detective.core.expressions import ExpressionError, ExpressionModel
from mag_opt_detective.core.fitting import Assignment, FitResult, Model, Param
from mag_opt_detective.core.models import DiracModel
from mag_opt_detective.core.units import Unit, convert
from mag_opt_detective.core.zeeman import Branch, Form, ZeemanModel

DIRAC, ZEEMAN, CUSTOM = "dirac", "zeeman", "custom"
KIND_NAMES = {DIRAC: "Massive Dirac", ZEEMAN: "Zeeman / magnon", CUSTOM: "Custom expression"}
# light data colours: dashed lines with a dark shadow stay readable on every colour map
MODEL_COLORS = ("#ffffff", "#8be9fd", "#ffb86c", "#9cf27a", "#ff8fd8", "#f4f78a")
COLOR_NAMES = ("White", "Cyan", "Orange", "Green", "Pink", "Yellow")
COUPLING_START = 1.0  # meV: a new coupling (a zero coupling has no gradient)
DIRAC_DEFAULTS = {"velocity": 5.0, "delta": 0.0, "n_lines": 5}
ZEEMAN_E0 = 10.0  # meV, first branch of a new model when no map is shown
FIELD_SAMPLES = 300
EXAMPLE = "E0 + g*muB*B"
ASSIGNMENTS = (
    (Assignment.BRANCH, "By branch", "Each curve is fitted to the branch chosen for it"),
    (Assignment.SORTED, "Sorted", "The k-th choice is the k-th lowest model energy at each field"),
    (Assignment.NEAREST, "Nearest", "Each point is fitted to the branch nearest to it"),
)
SAVE_VERSION = 1

Curve = tuple[np.ndarray, np.ndarray]


@dataclass
class FitSetup:
    """How picked curves are matched with the branches: *mapping* is curve -> branch index
    (the energy rank when sorted; any int means "use" for nearest), None skips the curve."""

    assignment: Assignment = Assignment.BRANCH
    mapping: dict[str, int | None] = field(default_factory=dict)


@dataclass(eq=False)
class ModelEntry:
    """One model card. *model* is None for a custom expression that was never valid."""

    kind: str
    key: str  # unique id; also the name of the overlay in ``figure_state()``
    name: str
    color: str
    model: Model | None
    visible: bool = True
    expanded: bool = True
    text: str = ""  # custom: the text in the editor (may be invalid)
    unit: Unit = Unit.MEV  # custom: the output unit of the expression
    error: ExpressionError | None = None  # custom: why the text is invalid
    memory: dict[str, Param] = field(default_factory=dict)  # custom: every parameter seen
    fit: FitSetup = field(default_factory=FitSetup)

    @property
    def model_unit(self) -> Unit:
        return self.model.unit if self.model is not None else self.unit

    def is_drawable(self) -> bool:
        """Whether the model has curves to draw (a custom text must be valid)."""
        if self.model is None or self.error is not None:
            return False
        return self.kind != CUSTOM or bool(self.text.strip())

    def branch_names(self) -> list[str]:
        """Names of the branches as shown in the mapping (1-based, readable)."""
        if not self.is_drawable():
            return []
        if self.kind == DIRAC:
            minus = chr(0x2212)
            return [f"L{minus if n else ''}{n} → L{n + 1}" for n in range(self.model.n_lines)]
        if self.kind == ZEEMAN and self.model.coupled and len(self.model.branches) > 1:
            return [f"Mode {k + 1}" for k in range(len(self.model.branches))]
        return self.model.branch_names()

    def curves(self, field_values: np.ndarray, unit: Unit | str) -> list[Curve]:
        """The branches over the field range of *field_values*, energies in *unit*."""
        if not self.is_drawable() or field_values.size == 0:
            return []
        grid = field_grid(field_values, self.kind)
        return evaluate_curves(self.model, grid, unit)


# ---------------------------------------------------------------------- new entries
def unique_key(kind: str, entries: Sequence[ModelEntry]) -> tuple[str, str]:
    """A free key and name for a new model of *kind* ("dirac", "dirac-2", ...)."""
    keys = {e.key for e in entries}
    names = {e.name for e in entries}
    n = 1
    while True:
        key = kind if n == 1 else f"{kind}-{n}"
        name = KIND_NAMES[kind] if n == 1 else f"{KIND_NAMES[kind]} {n}"
        if key not in keys and name not in names:
            return key, name
        n += 1


def free_color(entries: Sequence[ModelEntry]) -> str:
    """The first model colour not in use (cycling when all are)."""
    used = [e.color for e in entries]
    for color in MODEL_COLORS:
        if color not in used:
            return color
    return MODEL_COLORS[len(entries) % len(MODEL_COLORS)]


def new_entry(
    kind: str, entries: Sequence[ModelEntry], energy_span: tuple[float, float] | None = None
) -> ModelEntry:
    """A new model of *kind*; *energy_span* (meV) of the map shown places Zeeman's first
    branch inside it."""
    key, name = unique_key(kind, entries)
    color = free_color(entries)
    if kind == DIRAC:
        model = DiracModel(**DIRAC_DEFAULTS)
        return ModelEntry(DIRAC, key, name, color, model)
    if kind == ZEEMAN:
        e0 = ZEEMAN_E0
        if energy_span is not None:
            lo, hi = energy_span
            e0 = round_nice(lo + 0.25 * (hi - lo))
        model = ZeemanModel([Branch(e0, 2.0, 1.0, Form.LINEAR, "Branch 1")])
        return ModelEntry(ZEEMAN, key, name, color, model)
    if kind == CUSTOM:
        return ModelEntry(CUSTOM, key, name, color, None)
    raise ValueError(f"unknown model kind {kind!r}")


def round_nice(value: float) -> float:
    """*value* rounded to two significant digits (start values typed by nobody)."""
    if not math.isfinite(value) or value == 0:
        return 0.0
    digits = 1 - math.floor(math.log10(abs(value)))
    return round(value, digits)


def default_entries() -> list[ModelEntry]:
    """What a new window shows: the massive Dirac model, hidden (today's overlay)."""
    entry = new_entry(DIRAC, [])
    entry.visible = False
    return [entry]


# ---------------------------------------------------------------------- curves
def field_grid(field_values: np.ndarray, kind: str) -> np.ndarray:
    """Fields to draw a model at: the data's field range (from 0 for Dirac)."""
    finite = np.asarray(field_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = float(finite.min()), float(finite.max())
    if kind == DIRAC:
        lo = max(0.0, lo)
    return np.linspace(lo, hi, FIELD_SAMPLES)


def evaluate_curves(
    model: Model, field_values: np.ndarray, unit: Unit | str, values: Mapping | None = None
) -> list[Curve]:
    """``(field, energy in unit)`` per branch of *model* (NaN where it is not finite)."""
    energies = np.asarray(model.evaluate(field_values, values), dtype=float)
    energies = convert(energies.reshape(-1, field_values.size), model.unit, unit)
    energies[~np.isfinite(energies)] = np.nan
    return [(field_values, row) for row in energies]


# ---------------------------------------------------------------------- editing
def make_zeeman(
    branches: Sequence[Branch], couplings: Mapping[tuple[int, int], float], coupled: bool
) -> ZeemanModel:
    """A Zeeman model; missing couplings start at :data:`COUPLING_START` and ``g`` of a
    branch with ``m = 0`` is held fixed (it does not change the energy)."""
    n = len(branches)
    pairs = {
        (i, j): couplings.get((i, j), COUPLING_START) for i in range(n) for j in range(i + 1, n)
    }
    model = ZeemanModel(branches, pairs, coupled)
    hold_unused_g(model)
    return model


def zeeman_rebuild(
    model: ZeemanModel,
    branches: Sequence[Branch] | None = None,
    couplings: Mapping[tuple[int, int], float] | None = None,
    coupled: bool | None = None,
) -> ZeemanModel:
    """A Zeeman model like *model* with other branches, couplings or coupling switch."""
    return make_zeeman(
        list(model.branches if branches is None else branches),
        dict(model.couplings if couplings is None else couplings),
        model.coupled if coupled is None else coupled,
    )


def hold_unused_g(model: ZeemanModel) -> None:
    """Fix ``g`` of the branches with ``m = 0`` (and free it again otherwise)."""
    params = {p.name: p for p in model.params}
    for i, branch in enumerate(model.branches):
        params[f"g_{i}"].fixed = branch.m == 0


def next_branch(model: ZeemanModel) -> Branch:
    """A new branch below the others: like the last one, 10 % higher, labelled "Branch n"."""
    branches = model.branches
    labels = {b.label for b in branches}
    n = len(branches) + 1
    while f"Branch {n}" in labels:
        n += 1
    if not branches:
        return Branch(ZEEMAN_E0, 2.0, 1.0, Form.LINEAR, f"Branch {n}")
    last = branches[-1]
    e0 = round_nice(last.e0 * 1.1) if last.e0 else ZEEMAN_E0
    return Branch(e0, last.g, last.m, last.form, f"Branch {n}")


def add_branch(entry: ModelEntry) -> None:
    model = entry.model
    entry.model = zeeman_rebuild(model, [*model.branches, next_branch(model)])


def remove_branch(entry: ModelEntry, index: int) -> None:
    """Remove branch *index*; the couplings of the others are kept (renumbered)."""
    model = entry.model
    branches = model.branches
    if len(branches) <= 1:
        raise ValueError("a Zeeman model needs at least one branch")
    keep = [i for i in range(len(branches)) if i != index]
    old = model.couplings
    couplings = {
        (a, b): old[(keep[a], keep[b])] for a in range(len(keep)) for b in range(a + 1, len(keep))
    }
    entry.model = zeeman_rebuild(model, [branches[i] for i in keep], couplings)


def set_branch(entry: ModelEntry, index: int, **changes) -> None:
    """Change ``label``, ``m`` or ``form`` of a branch (``e0``/``g`` are plain parameters)."""
    branches = entry.model.branches
    b = branches[index]
    branches[index] = Branch(
        changes.get("e0", b.e0),
        changes.get("g", b.g),
        float(changes.get("m", b.m)),
        changes.get("form", b.form),
        changes.get("label", b.label),
    )
    entry.model = zeeman_rebuild(entry.model, branches)


def set_expression(entry: ModelEntry, text: str) -> ExpressionError | None:
    """Use *text* as the expression of a custom entry; returns the error if it is invalid.

    Parameters keep their value, bounds and fixed flag while their name stays, and get them
    back when a name returns. An invalid text leaves the model as it was (not drawn).
    """
    entry.text = text
    if not text.strip():
        entry.error = None
        return None
    try:
        if entry.model is None:
            entry.model = ExpressionModel(text, entry.unit)
        else:
            entry.model.set_text(text)
    except ExpressionError as exc:
        entry.error = exc
        return exc
    entry.error = None
    for p in entry.model.params:
        old = entry.memory.get(p.name)
        if old is not None and old is not p:
            p.value, p.lo, p.hi, p.fixed = old.value, old.lo, old.hi, old.fixed
        entry.memory[p.name] = p
    return None


def set_output_unit(entry: ModelEntry, unit: Unit | str) -> None:
    entry.unit = Unit(unit)
    if entry.model is not None:
        entry.model.unit = entry.unit


def params(entry: ModelEntry) -> dict[str, Param]:
    return {p.name: p for p in entry.model.params} if entry.model is not None else {}


# ---------------------------------------------------------------------- fitting
def resolve_mapping(entry: ModelEntry, curves: Sequence[str]) -> dict[str, int | None]:
    """The branch of each picked curve: the one chosen, else the i-th curve on branch i.

    Choices beyond the model's branches fall back to None (skipped); for nearest any choice
    means "use" (0), and every curve is used by default.
    """
    n = len(entry.branch_names())
    nearest = entry.fit.assignment is Assignment.NEAREST
    mapping: dict[str, int | None] = {}
    for i, name in enumerate(curves):
        if name in entry.fit.mapping:
            branch = entry.fit.mapping[name]
            if branch is not None and nearest:
                branch = 0
            if branch is not None and not 0 <= branch < n:
                branch = None
        else:
            branch = 0 if nearest else (i if i < n else None)
        mapping[name] = branch if n else None
    return mapping


def mapping_text(mapping: Mapping[str, int | None], entry: ModelEntry) -> str:
    """The assignment as text, e.g. ``LL 1 -> Branch 1, LL 2 skipped``."""
    names = entry.branch_names()
    nearest = entry.fit.assignment is Assignment.NEAREST
    parts = []
    for curve, branch in mapping.items():
        if branch is None:
            parts.append(f"{curve} skipped")
        elif nearest:
            parts.append(f"{curve} -> nearest")
        elif entry.fit.assignment is Assignment.SORTED:
            parts.append(f"{curve} -> rank {branch + 1}")
        else:
            parts.append(f"{curve} -> {names[branch]}")
    return ", ".join(parts)


# ---------------------------------------------------------------------- display
def param_label(entry: ModelEntry, name: str, ascii_only: bool = False) -> str:
    """A readable name: v, Δ, E₀ (Branch 1), g (Branch 1), Δ 1–2, or the expression's name."""
    delta = "Delta" if ascii_only else "Δ"
    if entry.kind == DIRAC:
        return {"velocity": "v", "delta": delta}.get(name, name)
    if entry.kind == ZEEMAN:
        kind, *index = name.split("_")
        if kind == "delta":
            i, j = (int(k) + 1 for k in index)
            return f"{delta} {i}-{j}" if ascii_only else f"Δ {i}–{j}"
        label = entry.model.branches[int(index[0])].label
        symbol = {"e0": "E0" if ascii_only else "E₀", "g": "g"}[kind]
        return f"{symbol} ({label})"
    return name


def is_energy(entry: ModelEntry, p: Param) -> bool:
    """Built-in energy parameters are shown in the display unit; custom ones never are."""
    return entry.kind != CUSTOM and p.kind == "energy"


def unit_text(entry: ModelEntry, p: Param, unit: Unit | str) -> str:
    if is_energy(entry, p):
        return str(Unit(unit))
    if p.kind == "velocity":
        return "1e5 m/s"
    return ""


def shown(entry: ModelEntry, p: Param, value: float, unit: Unit | str) -> float:
    """*value* of parameter *p* as shown (energies in *unit*)."""
    if is_energy(entry, p):
        return float(convert(value, entry.model_unit, unit))
    return float(value)


def stored(entry: ModelEntry, p: Param, value: float, unit: Unit | str) -> float:
    """The inverse of :func:`shown`: a value typed in *unit* as kept by the model."""
    if is_energy(entry, p):
        return float(convert(value, unit, entry.model_unit))
    return float(value)


def format_number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else "–"


def format_with_sigma(value: float, sigma: float) -> tuple[str, str]:
    """*value* and *sigma* rounded to two significant digits of *sigma*."""
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(value):
        return format_number(value), "–" if not math.isfinite(sigma) else format_number(sigma)
    decimals = 1 - math.floor(math.log10(sigma))
    if decimals > 9 or abs(value) >= 1e7:
        return f"{value:.6g}", f"{sigma:.2g}"
    if decimals <= 0:
        return f"{round(value, decimals):.0f}", f"{round(sigma, decimals):.0f}"
    return f"{value:.{decimals}f}", f"{sigma:.{decimals}f}"


@dataclass
class FitReport:
    """A fit result as shown: rows of (label, value, sigma, unit) in the display unit."""

    rows: list[tuple[str, float, float, str]]
    fixed: list[tuple[str, float, str]]
    chi2: float
    reduced_chi2: float
    dof: int
    n_points: int
    unit: Unit


def fit_report(
    entry: ModelEntry, result: FitResult, unit: Unit | str, ascii_only: bool = False
) -> FitReport:
    """The free parameters ± sigma and the statistics of *result* in *unit*."""
    unit = Unit(unit)
    by_name = params(entry)
    rows, fixed = [], []
    for name, value in result.values.items():
        p = by_name.get(name)
        if p is None:
            continue
        label = param_label(entry, name, ascii_only)
        text = unit_text(entry, p, unit)
        if name in result.free:
            sigma = result.stderr[name]
            scaled = shown(entry, p, sigma, unit) if math.isfinite(sigma) else sigma
            rows.append((label, shown(entry, p, value, unit), abs(scaled), text))
        else:
            fixed.append((label, shown(entry, p, value, unit), text))
    factor = float(convert(1.0, entry.model_unit, unit)) ** 2  # chi2 is in energy^2
    n_points = sum(r.size for r in result.residuals)
    return FitReport(
        rows, fixed, result.chi2 * factor, result.reduced_chi2 * factor, result.dof, n_points, unit
    )


def report_tsv(entry: ModelEntry, result: FitResult, unit: Unit | str, mapping_text: str) -> str:
    """Tab-separated results: ``#`` comment lines, then parameter, value, sigma, unit.

    Fixed parameters are listed with an empty sigma.
    """
    report = fit_report(entry, result, unit, ascii_only=True)
    squared = f"{report.unit}^2"
    lines = [
        f"# Model: {entry.name}",
        f"# Assignment: {entry.fit.assignment.value}; {mapping_text}",
        f"# Points: {report.n_points}; dof: {report.dof}",
        f"# chi2: {report.chi2:.6g} {squared}; reduced chi2: {report.reduced_chi2:.6g} {squared}",
    ]
    lines += [f"# {line}" for line in structure_lines(entry)]
    lines.append("parameter\tvalue\tsigma\tunit")
    for label, value, sigma, text in report.rows:
        lines.append(f"{label}\t{value:.10g}\t{sigma:.10g}\t{text}")
    for label, value, text in report.fixed:
        lines.append(f"{label}\t{value:.10g}\t\t{text}")
    return "\n".join(lines) + "\n"


def structure_lines(entry: ModelEntry) -> list[str]:
    """What else defines the model (not fitted): transitions, branch forms, the expression."""
    model = entry.model
    if entry.kind == DIRAC:
        return [f"Transitions: {model.n_lines}"]
    if entry.kind == ZEEMAN:
        lines = [f"{b.label}: m = {b.m:g}, {Form(b.form).value}" for b in model.branches]
        return [*lines, f"Coupled: {'yes' if model.coupled else 'no'}"]
    lines = [f"Expression ({entry.unit}): {line}" for line in entry.text.splitlines() if line]
    return lines


# ---------------------------------------------------------------------- saving
def entry_to_dict(entry: ModelEntry) -> dict:
    data: dict = {
        "kind": entry.kind,
        "key": entry.key,
        "name": entry.name,
        "color": entry.color,
        "visible": entry.visible,
        "expanded": entry.expanded,
        "assignment": entry.fit.assignment.value,
        "mapping": dict(entry.fit.mapping),
    }
    model = entry.model
    if entry.kind == DIRAC:
        p = params(entry)
        data |= {
            "velocity": p["velocity"].value,
            "delta_meV": p["delta"].value,
            "n_lines": model.n_lines,
        }
    elif entry.kind == ZEEMAN:
        data["branches"] = [
            {"label": b.label, "e0_meV": b.e0, "g": b.g, "m": b.m, "form": Form(b.form).value}
            for b in model.branches
        ]
        data["coupled"] = model.coupled
        data["couplings_meV"] = [[i, j, v] for (i, j), v in sorted(model.couplings.items())]
    else:
        data["text"] = entry.text
        data["unit"] = entry.unit.value
        data["params"] = [
            {
                "name": p.name,
                "value": p.value,
                "fixed": p.fixed,
                "lo": _bound(p.lo),
                "hi": _bound(p.hi),
            }
            for p in entry.memory.values()
        ]
    return data


def entry_from_dict(data: Mapping) -> ModelEntry:
    """Rebuild an entry from :func:`entry_to_dict`; raises ValueError if *data* is invalid."""
    try:
        kind = data["kind"]
        if kind not in KIND_NAMES:
            raise ValueError(f"unknown model kind {kind!r}")
        entry = ModelEntry(
            kind,
            str(data["key"]),
            str(data["name"]),
            str(data["color"]),
            None,
            visible=_bool(data.get("visible", True)),
            expanded=_bool(data.get("expanded", True)),
        )
        entry.fit = FitSetup(
            Assignment(data.get("assignment", Assignment.BRANCH)),
            {
                str(k): None if v is None else _int(v)
                for k, v in dict(data.get("mapping", {})).items()
            },
        )
        if kind == DIRAC:
            entry.model = DiracModel(
                _number(data["velocity"]), _number(data["delta_meV"]), _int(data["n_lines"])
            )
        elif kind == ZEEMAN:
            branches = [
                Branch(
                    _number(b["e0_meV"]),
                    _number(b["g"]),
                    _number(b["m"]),
                    Form(b["form"]),
                    str(b["label"]),
                )
                for b in data["branches"]
            ]
            if not branches:
                raise ValueError("no branches")
            couplings = {(_int(i), _int(j)): _number(v) for i, j, v in data["couplings_meV"]}
            entry.model = make_zeeman(branches, couplings, _bool(data["coupled"]))
        else:
            entry.unit = Unit(data.get("unit", Unit.MEV))
            for p in data.get("params", []):
                lo = -math.inf if p.get("lo") is None else _number(p["lo"])
                hi = math.inf if p.get("hi") is None else _number(p["hi"])
                name = str(p["name"])
                entry.memory[name] = Param(
                    name, _number(p["value"]), lo, hi, _bool(p.get("fixed", False))
                )
            set_expression(entry, str(data.get("text", "")))
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"invalid model: {exc}") from exc
    return entry


def entries_to_json(entries: Sequence[ModelEntry]) -> str:
    return json.dumps({"version": SAVE_VERSION, "models": [entry_to_dict(e) for e in entries]})


def entries_from_json(value) -> list[ModelEntry]:
    """Entries saved by :func:`entries_to_json` (a JSON text or the decoded dict)."""
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("not a saved model list")
    entries = [entry_from_dict(item) for item in data["models"]]
    keys = [e.key for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("model keys must be unique")
    return entries


def _bound(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _number(value) -> float:
    if value is None:
        raise TypeError("missing number")
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"not a number: {value!r}")
    number = float(value)
    if math.isnan(number):
        raise ValueError("not a number")
    return number


def _int(value) -> int:
    number = _number(value)
    if not number.is_integer():
        raise ValueError(f"not an integer: {value!r}")
    return int(number)


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise TypeError(f"not a bool: {value!r}")
