# PyInstaller spec: one-folder build of the desktop app.
#   uv sync --group bundle
#   uv run pyinstaller packaging/mag-opt-detective.spec --noconfirm
# Result: dist/mag-opt-detective/ (and dist/Magneto-Optical Detective.app on macOS).
import os
import sys

from PyInstaller.utils.hooks import collect_data_files

from mag_opt_detective import __version__

NAME = "mag-opt-detective"
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
    datas=collect_data_files("mag_opt_detective"),
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

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Magneto-Optical Detective.app",
        icon=ICON,
        bundle_identifier="io.github.wyzula-jan.mag-opt-detective",
        version=__version__,
        info_plist={"NSHighResolutionCapable": True},
    )
