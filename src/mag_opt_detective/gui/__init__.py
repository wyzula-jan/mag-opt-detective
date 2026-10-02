"""Qt user interface (PySide6 + pyqtgraph)."""

import os

# make sure pyqtgraph binds to PySide6 even if other Qt bindings are installed
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
