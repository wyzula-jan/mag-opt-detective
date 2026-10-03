"""Third-party licence notices for the app bundles (THIRD_PARTY_NOTICES.txt).

The PyInstaller spec writes the file at build time and puts it into the bundle, where
Help > About > Licences… shows it. It holds, from the environment the bundle is built in:

- the licence files of every package the app needs at run time (the requirements of
  mag-opt-detective, followed recursively; extras and other platforms' packages left out)
- the Python licence and the Lucide icon licence
- the GNU LGPL v3 and GNU GPL v3 texts for Qt and PySide6, whose wheels ship no licence file
  (``packaging/licenses/``, from the Qt sources)

Run it alone to check the result: ``python packaging/third_party_notices.py OUT.txt``.
"""

from __future__ import annotations

import re
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

APP = "mag-opt-detective"
HERE = Path(__file__).resolve().parent
QT_LICENCES = (HERE / "licenses" / "LGPL-3.0-only.txt", HERE / "licenses" / "GPL-3.0-only.txt")
QT_PACKAGES = {"pyside6", "pyside6-essentials", "pyside6-addons", "shiboken6"}
LICENCE_NAME = re.compile(r"(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", re.IGNORECASE)
RULE = "=" * 78

QT_NOTE = """\
Qt and PySide6 are used under the GNU Lesser General Public License v3 (LGPL-3.0-only),
whose text follows, together with the GNU General Public License v3 that it refers to.
The bundle contains the unmodified Qt and PySide6 libraries of the PySide6 wheels on PyPI,
as separate shared libraries (in the PySide6 folder) that can be replaced by other builds.
Source code: https://code.qt.io/cgit/qt/ (Qt) and
https://code.qt.io/cgit/pyside/pyside-setup.git/ (PySide6 and Shiboken6)."""


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_distributions(root: str = APP) -> list[metadata.Distribution]:
    """The installed distributions *root* needs at run time on this platform, by name."""
    env = default_environment()
    found: dict[str, metadata.Distribution] = {}
    todo = [root]
    while todo:
        name = normalise(todo.pop())
        if name in found:
            continue
        dist = metadata.distribution(name)
        found[name] = dist
        for text in dist.requires or []:
            requirement = Requirement(text)
            marker = requirement.marker
            if marker is None or marker.evaluate({**env, "extra": ""}):
                todo.append(requirement.name)
    del found[normalise(root)]
    return sorted(found.values(), key=lambda d: normalise(d.metadata["Name"]))


def licence_files(dist: metadata.Distribution) -> list[tuple[str, str]]:
    """(name, text) of the licence files in *dist*'s metadata folder."""
    files = []
    for path in dist.files or []:
        parts = path.parts
        if not parts[0].endswith(".dist-info"):
            continue
        in_licences = len(parts) > 2 and parts[1] in ("licenses", "license_files")
        if in_licences or LICENCE_NAME.search(path.name):
            text = path.locate().read_text(encoding="utf-8", errors="replace")
            name = "/".join(parts[2:] if in_licences else parts[1:])
            files.append((name, text))
    return files


def licence_summary(dist: metadata.Distribution) -> str:
    """The licence as the package declares it (SPDX expression, short field or classifier)."""
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    text = (meta.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 80:
        return text
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or []]
    licences = [c for c in classifiers if "License" in c or "Licence" in c]
    return ", ".join(licences) or "see the licence text below"


def home_page(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        if label.strip().lower() in ("homepage", "home", "source", "repository"):
            return url.strip()
    return meta.get("Home-page") or ""


def python_licence() -> str:
    """The licence of the Python interpreter the bundle is built with."""
    base = Path(sys.base_prefix)
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    for path in (stdlib / "LICENSE.txt", base / "LICENSE.txt", base / "LICENSE"):
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return "Python Software Foundation License: https://docs.python.org/3/license.html\n"


def lucide_licence() -> str:
    import mag_opt_detective  # only its version: no Qt

    path = Path(mag_opt_detective.__file__).parent / "gui" / "icons" / "LICENSE-lucide.txt"
    return path.read_text(encoding="utf-8")


def section(title: str, body: str) -> str:
    return f"{RULE}\n{title}\n{RULE}\n\n{body.rstrip()}\n"


def build_notices() -> str:
    """The text of THIRD_PARTY_NOTICES.txt."""
    from mag_opt_detective import __version__

    dists = runtime_distributions()
    rows = [f"  Python {sys.version.split()[0]}: Python Software Foundation License"]
    rows += [
        f"  {d.metadata['Name']} {d.version}: {licence_summary(d)}"
        for d in dists
        if normalise(d.metadata["Name"]) not in QT_PACKAGES
    ]
    rows += [
        f"  {d.metadata['Name']} {d.version} (with Qt): LGPL-3.0-only"
        for d in dists
        if normalise(d.metadata["Name"]) in QT_PACKAGES
    ]
    rows.append("  Lucide icons: ISC (parts MIT)")
    parts = [
        f"Magneto-Optical Detective {__version__}: third-party notices\n\n"
        "This app contains the following third-party software. Each is distributed under\n"
        "its own licence, reproduced in full below.\n\n" + "\n".join(rows) + "\n",
        section(f"Python {sys.version.split()[0]}", python_licence()),
    ]
    for dist in dists:
        name = dist.metadata["Name"]
        if normalise(name) in QT_PACKAGES:
            continue
        head = f"Licence: {licence_summary(dist)}"
        if home_page(dist):
            head += f"\nHome page: {home_page(dist)}"
        texts = [f"--- {file} ---\n\n{text.rstrip()}\n" for file, text in licence_files(dist)]
        body = "\n\n".join([head, *texts]) if texts else f"{head}\n(no licence file installed)"
        parts.append(section(f"{name} {dist.version}", body))
    qt = [d for d in dists if normalise(d.metadata["Name"]) in QT_PACKAGES]
    qt_names = ", ".join(f"{d.metadata['Name']} {d.version}" for d in qt)
    qt_texts = [path.read_text(encoding="utf-8") for path in QT_LICENCES]
    parts.append(section(f"Qt and PySide6 ({qt_names})", "\n\n".join([QT_NOTE, *qt_texts])))
    parts.append(section("Lucide icons (gui/icons)", lucide_licence()))
    return "\n\n".join(parts)


def write_notices(path: str | Path) -> Path:
    """Write :func:`build_notices` to *path* (UTF-8); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_notices(), encoding="utf-8")
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} OUT.txt")
    print(write_notices(sys.argv[1]))
