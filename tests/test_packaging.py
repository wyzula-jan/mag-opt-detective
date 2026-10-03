"""The bundle's helpers in packaging/."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    """A module of packaging/ (not a package) by file name."""
    spec = importlib.util.spec_from_file_location(f"packaging_{name}", ROOT / "packaging" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
