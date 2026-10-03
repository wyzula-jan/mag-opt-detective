# PyInstaller spec: one-folder build of the desktop app.
#   uv sync --group bundle
#   uv run pyinstaller packaging/mag-opt-detective.spec --noconfirm
# Result: dist/mag-opt-detective/ (and dist/Magneto-Optical Detective.app on macOS), each with
# THIRD_PARTY_NOTICES.txt (see third_party_notices.py).
import os
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

from mag_opt_detective import __version__

sys.path.insert(0, SPECPATH)
from third_party_notices import write_notices  # noqa: E402

NAME = "mag-opt-detective"
NOTICES = write_notices(Path(workpath) / "THIRD_PARTY_NOTICES.txt")
ICONS = os.path.join(SPECPATH, "icons")
# Windows embeds the .ico in the .exe; macOS needs .icns (Linux has no file icon)
ICON = os.path.join(ICONS, NAME + (".icns" if sys.platform == "darwin" else ".ico"))

# matplotlib only draws figure exports (Agg, PDF, SVG, PS); leave out its GUI backends
MPL_GUI_BACKENDS = [
    f"matplotlib.backends.{name}"
    for name in (
        "_backend_tk",
        "_tkagg",
        "backend_tkagg",
        "backend_tkcairo",
        "backend_qt",
        "backend_qtagg",
        "backend_qtcairo",
        "backend_qt5",
        "backend_qt5agg",
        "backend_qt5cairo",
        "qt_compat",
        "qt_editor",
        "backend_webagg",
        "backend_webagg_core",
        "backend_nbagg",
        "backend_wx",
        "backend_wxagg",
        "backend_wxcairo",
        "_backend_gtk",
        "backend_gtk3",
        "backend_gtk3agg",
        "backend_gtk3cairo",
        "backend_gtk4",
        "backend_gtk4agg",
        "backend_gtk4cairo",
        "_macosx",
        "backend_macosx",
    )
]

a = Analysis(
    ["launcher.py"],
    # package data: Lucide icons, the window icon in mag_opt_detective/resources
    datas=[*collect_data_files("mag_opt_detective"), (str(NOTICES), "mag_opt_detective")],
    hiddenimports=["mag_opt_detective.smoke"],
    excludes=[
        "pyqtgraph.opengl",
        "tkinter",
        "pandas",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "IPython",
        "pytest",
        # The export never uses pyplot. Bundled, it would be imported by pyqtgraph's colour
        # map menu, so every start-up would load matplotlib's font cache; without it pyqtgraph
        # skips its matplotlib maps. The export window loads the cache, which launcher.py
        # keeps in a per-user folder so that it is built only once.
        "matplotlib.pyplot",
        *MPL_GUI_BACKENDS,
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=False,
    # macOS: keep argv so "--smoke-test" reaches the app
    argv_emulation=False,
    icon=ICON,
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
# the notices also next to the program, where people unpacking the archive see them
shutil.copy(NOTICES, Path(DISTPATH) / NAME / NOTICES.name)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Magneto-Optical Detective.app",
        icon=ICON,
        bundle_identifier="io.github.wyzula-jan.mag-opt-detective",
        version=__version__,
        info_plist={"NSHighResolutionCapable": True},
    )
