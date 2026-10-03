"""Entry script for the PyInstaller bundle."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = "mag-opt-detective"


def matplotlib_cache_dir(
    platform: str = sys.platform, environ=os.environ, home: Path | None = None
) -> Path:
    """The per-user folder that keeps matplotlib's font cache between launches."""
    home = Path.home() if home is None else home
    if platform == "darwin":
        base = home / "Library" / "Caches"
    elif platform == "win32":
        base = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    else:
        base = Path(environ.get("XDG_CACHE_HOME") or home / ".cache")
    return base / APP_DIR / "matplotlib"


def use_persistent_matplotlib_cache(environ=os.environ, folder: Path | None = None) -> None:
    """Point MPLCONFIGDIR at :func:`matplotlib_cache_dir` instead of a new temporary folder.

    PyInstaller's matplotlib hook gives every launch a new temporary MPLCONFIGDIR, so the
    first figure export of each session rebuilt the font cache (about 20 s). matplotlib reads
    the variable when it is first imported, which happens later (the export window). If the
    folder cannot be made, the temporary one stays. A cached font file that has moved (the
    app was moved) makes matplotlib rebuild the cache itself.
    """
    folder = matplotlib_cache_dir(environ=environ) if folder is None else folder
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    environ["MPLCONFIGDIR"] = str(folder)


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        use_persistent_matplotlib_cache()

    from mag_opt_detective.app import main

    raise SystemExit(main())
