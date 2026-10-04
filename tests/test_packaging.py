"""Project metadata and the bundle's build-time helpers in packaging/."""

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

import mag_opt_detective

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    """A module of packaging/ (not a package) by file name."""
    spec = importlib.util.spec_from_file_location(f"packaging_{name}", ROOT / "packaging" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_version_is_the_same_in_pyproject_and_the_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == mag_opt_detective.__version__


def test_the_texts_give_the_minimum_macos_the_bundle_declares():
    spec = (ROOT / "packaging" / "mag-opt-detective.spec").read_text(encoding="utf-8")
    declared = re.search(r'"LSMinimumSystemVersion": "(\d+)\.0"', spec)[1]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"| macOS {declared} or newer on Apple silicon" in readme
    assert f"On a Mac this needs macOS {declared} or newer" in readme
    pages = ROOT / "docs" / "site" / "pages"
    download = (pages / "download.html").read_text(encoding="utf-8")
    assert f'<p class="req">macOS {declared} ' in download
    assert f"on macOS {declared} or newer, also on Intel Macs" in download
    index = (pages / "index.html").read_text(encoding="utf-8")
    assert f'<p class="req">macOS {declared} or newer on Apple silicon</p>' in index
    for text in (readme, download, index):  # no other minimum
        assert set(re.findall(r"macOS (\d+)(?: [A-Z][a-z]+)? or newer", text)) == {declared}


def test_the_licence_and_the_citation_agree_with_the_project():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "GPL-3.0-only"
    assert project["license-files"] == ["LICENSE"]
    licence = " ".join((ROOT / "LICENSE").read_text(encoding="utf-8").split())
    assert licence.startswith("GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {mag_opt_detective.__version__}\n" in citation
    assert "license: GPL-3.0-only\n" in citation
    assert "please cite it" in citation


# ---------------------------------------------------------------------- launcher
@pytest.mark.parametrize(
    ("platform", "environ", "expected"),
    [
        ("darwin", {}, "home/Library/Caches/mag-opt-detective/matplotlib"),
        ("win32", {"LOCALAPPDATA": "/appdata"}, "/appdata/mag-opt-detective/matplotlib"),
        ("win32", {}, "home/AppData/Local/mag-opt-detective/matplotlib"),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, "/xdg/mag-opt-detective/matplotlib"),
        ("linux", {}, "home/.cache/mag-opt-detective/matplotlib"),
    ],
)
def test_matplotlib_cache_dir_is_per_user(platform, environ, expected):
    launcher = load("launcher.py")
    folder = launcher.matplotlib_cache_dir(platform, environ, home=Path("home"))
    assert folder == Path(expected)


def test_the_bundle_keeps_the_matplotlib_cache(tmp_path):
    """The launcher replaces PyInstaller's temporary MPLCONFIGDIR (importing it starts no app)."""
    launcher = load("launcher.py")
    environ = {"MPLCONFIGDIR": "/tmp/_MEI-run"}
    folder = tmp_path / "cache" / "mag-opt-detective" / "matplotlib"
    launcher.use_persistent_matplotlib_cache(environ, folder)
    assert folder.is_dir() and environ["MPLCONFIGDIR"] == str(folder)

    blocked = tmp_path / "file"
    blocked.write_text("")  # a file where the folder should be: it cannot be made
    environ = {"MPLCONFIGDIR": "/tmp/_MEI-run"}
    launcher.use_persistent_matplotlib_cache(environ, blocked / "matplotlib")
    assert environ["MPLCONFIGDIR"] == "/tmp/_MEI-run"  # the temporary folder stays


# ---------------------------------------------------------------------- notices
@pytest.fixture(scope="module")
def notices():
    return load("third_party_notices.py")


def test_notices_cover_the_runtime_packages_only(notices):
    names = {notices.normalise(d.metadata["Name"]) for d in notices.runtime_distributions()}
    assert {"numpy", "scipy", "matplotlib", "pillow", "pyqtgraph", "shiboken6"} <= names
    assert {"pyside6-essentials", "fonttools", "contourpy", "kiwisolver"} <= names
    assert not names & {"pytest", "pytest-qt", "ruff", "pre-commit", "pyyaml", "pyinstaller"}
    assert "mag-opt-detective" not in names


def test_notices_hold_the_licence_texts(notices, tmp_path):
    path = notices.write_notices(tmp_path / "THIRD_PARTY_NOTICES.txt")
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"Magneto-Optical Detective {mag_opt_detective.__version__}")
    for dist in notices.runtime_distributions():
        assert f"{dist.metadata['Name']} {dist.version}" in text
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text  # Qt and PySide6
    assert "GNU GENERAL PUBLIC LICENSE" in text  # which the LGPL refers to
    assert "PYTHON SOFTWARE FOUNDATION LICENSE" in text.upper()
    assert "Lucide" in text
    assert "NumPy Developers" in text  # numpy's own licence file
