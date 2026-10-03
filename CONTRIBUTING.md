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

Every behaviour change needs a test in `tests/`. Never commit measurement data: tests
needing real data use the `data_dir` fixture, which skips when `Data_to_test/` is missing.

## Architecture

Everything lives in `src/mag_opt_detective/`:

- `core/`: the numerical code (readers, processing, units, points, picking, fitting,
  Zeeman and expression models, colour-map tables). No Qt.
- `export/`: journal figures with matplotlib's object API only: no Qt and no pyplot
  (`test_export_does_not_import_qt_or_pyplot` checks it in a subprocess).
- `gui/`: the PySide6 interface, written in code (no `.ui` files).
  - `controller.py`: the state (processing, view, unit, results, points) and the actions,
    with signals for every change; `figure_state()` is the snapshot that exports draw.
  - `main_window.py`: only the frame (toolbar, rail, plot area, inspector, log drawer,
    status bar). The area modules fill it.
  - `kit/`: our own widgets (range slider and control, segmented control, switch, slide
    panel, collapsible section, info bar); `python -m mag_opt_detective.gui.kit.gallery`
    shows them all.
  - `plots/`: the map and stacked views, colour scales and overlay layers.
  - `panels/`: the rail's panels (Sample, Reference, Processing, Library, Points).
  - `inspector/`: the inspector's sections (View, Colour, Traces, Models).
  - `tools/`: plot tools beyond pan, zoom and pick (auto-pick).
  - `theme.py` (light and dark tokens, palette, style sheet) and `icons/` (tinted icons).
  - `plot_panel.py` (plot area, plot toolbar, tool registry), `export_menu.py` and
    `export_dialog.py` (Export menu, journal figure window), `points_*.py` (point table,
    markers, undo), `console.py` (log drawer), `licences.py` (About, Licences…).
- `smoke.py`: the self-test behind `--smoke-test`.

**Area modules.** Each panel, inspector section and tool exposes `install(window)`, which
builds its widgets, puts them into the window (`add_panel`, `add_inspector_section`,
`tools.register`, …), wires them to the controller and **binds its own settings keys**
with `window.persistence.bind("area/key", widget)`. `main_window.py` only lists the
modules. Handlers that can fail use the `user_action` decorator, which reports expected
errors in the info bar above the plot and logs unexpected ones.

**Settings.** Values are stored under the `v2/` prefix (`gui/settings.py`). Bump `PREFIX`
when a stored key changes meaning: everything older is then ignored once, so no migration
code is needed. Widgets store plain values: Qt's own widgets are handled, and our widgets
implement the settings protocol, `settings_value()` returning a str, int, float or bool
and `set_settings_value(value)` returning False for a value it cannot take (the default
stays). Energies are stored in cm⁻¹ whatever unit is shown.

**Icons.** The icons are [Lucide](https://lucide.dev) SVGs from `lucide-static` 1.50.0
(24 × 24, stroke 2, `stroke="currentColor"`). Copy the SVG unchanged into `gui/icons/`
under its Lucide name, use it with `icons.set_icon(widget, "name")`, and keep
`LICENSE-lucide.txt` (ISC, with the MIT notice for icons derived from Feather) next to
them. Lucide has outlines only: for a filled icon pass `fill=`, which draws the glyph on a
rounded tile (`icons.set_icon(button, "layers", "accent-fg", fill="accent")`, as the rail's
data states do); never edit the SVGs.

## GUI tests

GUI tests use pytest-qt and run offscreen (`QT_QPA_PLATFORM=offscreen`, set in
`tests/conftest.py`). Take the shared fixtures from `tests/gui_helpers.py` with
`window, errors = gui_helpers.window, gui_helpers.errors`: `window` is a main window
without stored settings, `errors` collects the messages it reports (and no dialog
blocks). The helpers load the synthetic `sweep` fixture, process, choose plots and units
and click on the plots. Test through the controller and public widget methods, not
private attributes. Open and close slide panels and sections with `animate=False`, or
wait with a generous timeout. After each test `conftest.py` deletes the closed windows
and the menus pyqtgraph leaves without a parent; without that they pile up and slow down
later tests.

## Bundle

The app bundles are built with PyInstaller from `packaging/mag-opt-detective.spec`:

```bash
uv sync --locked --no-default-groups --group bundle   # without the dev tools
uv run --no-sync pyinstaller packaging/mag-opt-detective.spec --noconfirm
dist/mag-opt-detective/mag-opt-detective --smoke-test  # macOS: inside the .app
uv sync                                                # back to the dev environment
```

`--smoke-test` loads a synthetic sweep, processes it, draws every plot, switches the unit,
saves journal figures in every format and exits with 0. The spec also writes
`THIRD_PARTY_NOTICES.txt` (`packaging/third_party_notices.py`); the *App bundles*
workflow builds and smoke-tests the bundles on Linux, macOS and Windows for pull requests
that touch the packaging and for version tags, which also publish a release. The README
screenshots are rendered by `docs/make_screenshots.py` (synthetic data only).

## Docs site

The documentation and download site lives in `docs/site/` (see its README): page fragments,
one layout and a standard-library build script, `python docs/site/build.py`.
`tests/test_site.py` builds it and checks its links and images. Its screenshots come from
`python docs/make_screenshots.py --site`. The *Docs site* workflow deploys it to GitHub Pages
and runs only when started by hand.

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

## Releases

Versions follow [Semantic Versioning](https://semver.org) and come from the commit messages:
`tools/release.py` (standard library only) reads the commits since the last `vX.Y.Z` tag,
merges skipped, and picks the next version.

| Commits since the last release | Before 1.0.0 | From 1.0.0 |
| --- | --- | --- |
| a breaking change (`!` or a `BREAKING CHANGE:` footer), any type | minor | major |
| `feat` | minor | minor |
| `fix`, `perf` or `revert` | patch | patch |
| only `docs`, `test`, `build`, `ci`, `chore`, `style` or `refactor` | no release | no release |

After every merge to `main`, on `main` with a clean tree:

```bash
uv run python tools/release.py --dry-run   # the next version and its changelog section
uv run python tools/release.py             # release it
```

A release writes the version into `pyproject.toml`, `uv.lock`,
`src/mag_opt_detective/__init__.py` and `CITATION.cff` (with `date-released`), adds a section
at the top of `CHANGELOG.md` (`feat` under Added, breaking changes and reverts under Changed,
`fix` under Fixed, `perf` under Performance, each with its scope), commits these files as
`chore(release): vX.Y.Z` (the hooks run) and creates the annotated tag `vX.Y.Z` with the
section as its message. It pushes nothing. It refuses to run with uncommitted changes or on a
commit that is already tagged, and says so when the new commits need no release. The *Release
notes* page of the docs site is built from `CHANGELOG.md`.

- `--version X.Y.Z` releases a version of your choice instead of the computed one. **1.0.0**,
  the first public release, is made this way (`--version 1.0.0`); after it a breaking change
  bumps the major version. At 1.0.0 also update the docs site's lines that say no release is
  published yet: the home page's "In development" eyebrow and the notes on the home and
  download pages.
- `--notes FILE` replaces the generated section with your own Markdown (`###` headings at
  most). The first release had no tag to count from: it was made with
  `--first 0.1.0 --notes tools/release-notes-0.1.0.md`, a summary of the app instead of every
  commit since the start.

**Publishing.** Push with the tags: `git push --follow-tags origin main`. The *Release*
workflow runs on every push to `main` (and by hand). When the version in `pyproject.toml` has a
section in `CHANGELOG.md` and no GitHub release yet, it builds the app bundles with the *App
bundles* workflow and creates the GitHub release `vX.Y.Z` on the release commit, from the
pushed tag (or creating the tag when it was not pushed), with the section as its notes and the
bundles attached; versions before 1.0.0 are marked as pre-releases. Any other push publishes
nothing, and no version is published twice. If several versions were released between two
pushes, only the newest gets a GitHub release; the others keep their tags and their
`CHANGELOG.md` sections.
