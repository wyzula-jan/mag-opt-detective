# Contributing

## Setup

```bash
uv sync                                                            # .venv with app + dev tools
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The hooks run ruff (lint + format) on every commit and reject commit messages that do
not follow the convention below.

## Checks

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

GitHub Actions runs the same checks on every push to `main` and on pull requests
(Linux with Python 3.12–3.15, macOS and Windows with 3.14). Results:
<https://github.com/wyzula-jan/mag-opt-detective/actions>.

- `src/mag_opt_detective/core` holds the numerical code and must not import Qt.
  Every change there needs a test in `tests/`.
- `src/mag_opt_detective/gui` is the PySide6 interface, written in code (no `.ui` files).
- Never commit measurement data. Tests needing real data use the `data_dir` fixture,
  which skips when `Data_to_test/` is missing.

## Branches and the task board

`main` always works. Do the work on a short-lived branch (`feat/merge-by-field`,
`fix/opus-csf`, …) and merge it back when the tests pass.

Open work is tracked on the task board at
<https://claude.ai/artifact/GYM8suwEvzhze7WGGhvNRF> (private, ask Jan for access).
Each task there has an ID (`P2-04`), acceptance criteria and a suggested branch name
(`feat/p2-04-merge-by-field`). Claim a task on the board before starting it and set it
to *In review* when its branch is ready. The board's *Agent handoff* section explains
how coding agents update it.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <summary>

<optional body>

<optional footer>
```

**Summary line**

- imperative mood, lowercase, no trailing period: `add`, not `added` / `adds`
- at most ~50 characters; it should finish the sentence "this commit will …"
- one logical change per commit

**Types**

| type | use for |
| --- | --- |
| `feat` | a new feature for the user |
| `fix` | a bug fix |
| `refactor` | code change that neither fixes a bug nor adds a feature |
| `perf` | performance improvement |
| `test` | adding or fixing tests only |
| `docs` | documentation only |
| `build` | packaging, dependencies, `pyproject.toml`, `uv.lock` |
| `ci` | CI configuration |
| `chore` | maintenance that does not touch the app (cleanup, tooling) |
| `style` | formatting only, no code change |
| `revert` | revert of an earlier commit |

**Scopes** (optional): `core`, `gui`, `io`, `opus`, `points`, `deps`, …

**Body** (optional): explain *why*, wrapped at 72 characters, separated from the
summary by a blank line.

**Breaking changes**: add `!` after the type/scope and a `BREAKING CHANGE:` footer,
e.g. when the export file format changes.

**Examples**

```
feat(gui): add merge by field
fix(core): apply opus scaling factor
refactor(core): split readers into opus and text
docs: describe zero-field drift correction
build(deps): bump pyside6 to 6.12
feat(io)!: write units into points export

BREAKING CHANGE: point tables now start with an "Energy (unit)" header.
```
