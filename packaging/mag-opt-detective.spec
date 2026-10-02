# PyInstaller spec: one-folder build of the desktop app.
#   uv sync --group bundle
#   uv run pyinstaller packaging/mag-opt-detective.spec --noconfirm
# Result: dist/mag-opt-detective/ (and dist/Magneto-Optical Detective.app on macOS).
import sys

from mag_opt_detective import __version__

NAME = "mag-opt-detective"

a = Analysis(
    ["launcher.py"],
    hiddenimports=["mag_opt_detective.smoke"],
    excludes=[
        "pyqtgraph.opengl",
        "tkinter",
        "matplotlib",
        "pandas",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "IPython",
        "pytest",
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
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Magneto-Optical Detective.app",
        bundle_identifier="ch.psi.mag-opt-detective",
        version=__version__,
        info_plist={"NSHighResolutionCapable": True},
    )
