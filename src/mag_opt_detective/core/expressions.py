"""User-defined transition energies E(B): one safe arithmetic expression per branch.

Each non-empty line is one branch and ``#`` starts a comment. ``B`` is the field in T, the
constants below are known, and every other name is a fit parameter shared between lines.
Expressions are checked against a strict whitelist of the Python syntax tree and evaluated
by walking it with numpy, never with ``eval``. All literals are floats.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy import constants

from mag_opt_detective.core.fitting import Param, param_values
from mag_opt_detective.core.units import Unit
from mag_opt_detective.core.zeeman import MU_B

MAX_LENGTH = 2000
MAX_NODES = 500
VARIABLE = "B"

CONSTANTS: dict[str, float] = {
    "muB": MU_B,  # meV/T
    "hbar": constants.physical_constants["reduced Planck constant in eV s"][0] * 1e3,  # meV s
    "kB": constants.physical_constants["Boltzmann constant in eV/K"][0] * 1e3,  # meV/K
    "e": constants.e,  # C
    "c": constants.c,  # m/s
    "pi": math.pi,
}

# name -> (numpy function, number of arguments)
FUNCTIONS: dict[str, tuple[Callable[..., np.ndarray], int]] = {
    "sqrt": (np.sqrt, 1),
    "exp": (np.exp, 1),
    "log": (np.log, 1),
    "log10": (np.log10, 1),
    "sin": (np.sin, 1),
    "cos": (np.cos, 1),
    "tan": (np.tan, 1),
    "arcsin": (np.arcsin, 1),
    "arccos": (np.arccos, 1),
    "arctan": (np.arctan, 1),
    "arctan2": (np.arctan2, 2),
    "sinh": (np.sinh, 1),
    "cosh": (np.cosh, 1),
    "tanh": (np.tanh, 1),
    "abs": (np.abs, 1),
    "sign": (np.sign, 1),
    "minimum": (np.minimum, 2),
    "maximum": (np.maximum, 2),
    "hypot": (np.hypot, 2),
}

_BINARY: dict[type[ast.operator], Callable[..., np.ndarray]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}
_UNARY: dict[type[ast.unaryop], Callable[..., np.ndarray]] = {
    ast.UAdd: np.positive,
    ast.USub: np.negative,
}
_OPERATORS = {
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.MatMult: "@",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.Invert: "~",
    ast.Not: "not",
}
_REJECTED: dict[type[ast.AST], str] = {
    ast.Attribute: "attribute access is not allowed",
    ast.Subscript: "subscripts are not allowed",
    ast.Lambda: "lambda functions are not allowed",
    ast.ListComp: "comprehensions are not allowed",
    ast.SetComp: "comprehensions are not allowed",
    ast.DictComp: "comprehensions are not allowed",
    ast.GeneratorExp: "comprehensions are not allowed",
    ast.Compare: "comparisons are not allowed",
    ast.BoolOp: "'and' and 'or' are not allowed",
    ast.IfExp: "conditional expressions are not allowed",
    ast.NamedExpr: "assignments are not allowed",
    ast.Starred: "starred arguments are not allowed",
    ast.JoinedStr: "text is not allowed",
    ast.Tuple: "commas are only allowed between function arguments",
    ast.List: "lists are not allowed",
    ast.Set: "sets are not allowed",
    ast.Dict: "dicts are not allowed",
}
_IMPORT_RE = re.compile(r"(import|from)\b")

_Node = Callable[[Mapping[str, np.ndarray]], np.ndarray]


class ExpressionError(ValueError):
    """A rejected expression; *line* and *column* are 1-based positions in the text."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"line {line}, column {column}: {message}")
        self.message = message
        self.line = line
        self.column = column


@dataclass(frozen=True)
class ExpressionLine:
    """One branch: its expression, its label (the comment, or else the expression), its line."""

    text: str
    label: str
    line: int


class CompiledExpression:
    """Parsed branches and their parameters, in order of first appearance."""

    def __init__(self, branches: list[ExpressionLine], parameters: list[str], nodes: list[_Node]):
        self.branches = branches
        self.parameters = parameters
        self._nodes = nodes

    def evaluate(self, field: np.ndarray, values: Mapping[str, float]) -> np.ndarray:
        """Energies of every branch, shape ``(n_branch, *field.shape)``."""
        field = np.asarray(field, dtype=float)
        env: dict[str, np.ndarray] = {VARIABLE: field}
        for name in self.parameters:
            if name not in values:
                raise ValueError(f"no value for parameter {name}")
            env[name] = np.float64(values[name])
        with np.errstate(all="ignore"):
            rows = [
                np.broadcast_to(np.asarray(node(env), dtype=float), field.shape)
                for node in self._nodes
            ]
        return np.stack(rows)


def parse(text: str) -> CompiledExpression:
    """Check and compile *text*; raises :class:`ExpressionError` with the position."""
    if len(text) > MAX_LENGTH:
        line, column = _position(text, MAX_LENGTH)
        raise ExpressionError(f"the text is longer than {MAX_LENGTH} characters", line, column)
    compiler = _Compiler()
    branches, nodes = [], []
    for number, raw in enumerate(text.splitlines(), start=1):
        code, _, comment = raw.partition("#")
        source = code.strip()
        if not source:
            continue
        indent = len(code) - len(code.lstrip())
        nodes.append(compiler.compile_line(source, number, indent))
        branches.append(ExpressionLine(source, comment.strip() or source, number))
    if not branches:
        raise ExpressionError("enter at least one expression", 1, 1)
    return CompiledExpression(branches, compiler.parameters(), nodes)


class ExpressionModel:
    """A fit model from user expressions; parameters are in the expression's own *unit*."""

    def __init__(
        self,
        text: str,
        unit: Unit | str = Unit.MEV,
        initial: Mapping[str, float] | None = None,
    ):
        self.unit = Unit(unit)
        self.params: list[Param] = []
        self.set_text(text, initial)

    def set_text(self, text: str, initial: Mapping[str, float] | None = None) -> None:
        """Parse *text*; parameters that keep their name keep their value and settings.

        Unknown names in *initial* are ignored, new parameters start at 1.0. On an
        :class:`ExpressionError` the model is left unchanged.
        """
        expression = parse(text)
        old = {p.name: p for p in self.params}
        params = []
        for name in expression.parameters:
            p = old.get(name) or Param(name, 1.0, kind="other")
            if initial is not None and name in initial:
                p.value = float(initial[name])
            params.append(p)
        self.text = text
        self.expression = expression
        self.params[:] = params

    def branch_names(self) -> list[str]:
        return [branch.label for branch in self.expression.branches]

    def evaluate(self, field: np.ndarray, values: Mapping[str, float] | None = None) -> np.ndarray:
        """Energies in :attr:`unit`, shape ``(n_branch, n_field)``."""
        return self.expression.evaluate(field, param_values(self.params, values))


class _Compiler:
    """Turns whitelisted syntax trees into closures over an environment of arrays."""

    def __init__(self) -> None:
        self._first: dict[str, tuple[int, int]] = {}
        self._nodes = 0

    def parameters(self) -> list[str]:
        return sorted(self._first, key=self._first.__getitem__)

    def compile_line(self, source: str, line: int, indent: int) -> _Node:
        self._bytes = source.encode()
        self._line = line
        self._indent = indent
        if _IMPORT_RE.match(source):
            raise ExpressionError("imports are not allowed", line, indent + 1)
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            column = exc.offset if exc.lineno == 1 and exc.offset else len(source) + 1
            raise ExpressionError(exc.msg, line, indent + column) from None
        except (RecursionError, MemoryError):
            raise ExpressionError("the expression is nested too deeply", line, indent + 1) from None
        self._nodes += sum(isinstance(node, ast.expr) for node in ast.walk(tree))
        if self._nodes > MAX_NODES:
            raise ExpressionError(
                f"the expressions have more than {MAX_NODES} elements", line, indent + 1
            )
        return self._compile(tree.body)

    def _error(self, message: str, node: ast.AST, offset: int | None = None) -> ExpressionError:
        offset = node.col_offset if offset is None else offset
        column = len(self._bytes[:offset].decode("utf-8", errors="replace"))
        return ExpressionError(message, self._line, self._indent + column + 1)

    def _compile(self, node: ast.AST) -> _Node:
        match node:
            case ast.Constant(value=value):
                return self._constant(node, value)
            case ast.Name(id=name):
                return self._name(node, name)
            case ast.BinOp(left=left, op=op, right=right):
                func = _BINARY.get(type(op))
                if func is None:
                    raise self._operator_error(op, node, left, right)
                a, b = self._compile(left), self._compile(right)
                return lambda env: func(a(env), b(env))
            case ast.UnaryOp(op=op, operand=operand):
                func = _UNARY.get(type(op))
                if func is None:
                    raise self._error(f"operator {_OPERATORS[type(op)]} is not allowed", node)
                a = self._compile(operand)
                return lambda env: func(a(env))
            case ast.Call():
                return self._call(node)
        message = _REJECTED.get(type(node), f"{type(node).__name__} is not allowed")
        raise self._error(message, node)

    def _constant(self, node: ast.AST, value: object) -> _Node:
        if isinstance(value, bool) or not isinstance(value, int | float):
            if isinstance(value, str | bytes):
                message = "text is not allowed, only numbers"
            elif isinstance(value, complex):
                message = "complex numbers are not allowed"
            else:
                message = f"{value!r} is not allowed, only numbers"
            raise self._error(message, node)
        try:
            number = np.float64(float(value))
        except OverflowError:
            raise self._error("the number is too large", node) from None
        return lambda env: number

    def _name(self, node: ast.AST, name: str) -> _Node:
        if name.startswith("_"):
            raise self._error("names starting with an underscore are not allowed", node)
        if name in FUNCTIONS:
            raise self._error(f"{name} is a function, write {name}(...)", node)
        if name in CONSTANTS:
            number = np.float64(CONSTANTS[name])
            return lambda env: number
        if name != VARIABLE:
            self._first.setdefault(name, (self._line, node.col_offset))
        return lambda env: env[name]

    def _call(self, node: ast.Call) -> _Node:
        if not isinstance(node.func, ast.Name):
            self._compile(node.func)  # names a forbidden construct such as np.sqrt
            raise self._error("only functions can be called", node)
        name = node.func.id
        if name.startswith("_"):
            raise self._error("names starting with an underscore are not allowed", node.func)
        if name not in FUNCTIONS:
            known = name == VARIABLE or name in CONSTANTS
            raise self._error(
                f"{name} is not a function" if known else f"unknown function {name}", node.func
            )
        if node.keywords:
            raise self._error("keyword arguments are not allowed", node.keywords[0])
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise self._error("starred arguments are not allowed", arg)
        func, arity = FUNCTIONS[name]
        if len(node.args) != arity:
            plural = "s" if arity > 1 else ""
            raise self._error(f"{name} takes {arity} argument{plural}, got {len(node.args)}", node)
        args = [self._compile(arg) for arg in node.args]
        return lambda env: func(*(a(env) for a in args))

    def _operator_error(
        self, op: ast.operator, node: ast.AST, left: ast.expr, right: ast.expr
    ) -> ExpressionError:
        """Point at the operator itself when it can be found between the operands."""
        text = _OPERATORS.get(type(op), type(op).__name__)
        gap = self._bytes[left.end_col_offset : right.col_offset]
        at = gap.find(text.encode())
        offset = left.end_col_offset + at if at >= 0 else None
        return self._error(f"operator {text} is not allowed", node, offset)


def _position(text: str, index: int) -> tuple[int, int]:
    """1-based (line, column) of character *index* in *text*, with lines as in :func:`parse`."""
    lines = (text[:index] + "x").splitlines()
    return len(lines), len(lines[-1])
