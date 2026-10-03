"""The Help menu's web links: the documentation site and the GitHub issue forms.

*Report a bug…* opens the bug form (``.github/ISSUE_TEMPLATE/bug_report.yml``) filled in with
the environment only: the app's version, how it runs, the system, the Python, Qt and library
versions and the plot shown. GitHub fills a form field from the query parameter named after
the field's ``id``. Never file paths, user or host names, data or settings: nothing is sent
until the user submits the form in the browser.
"""

from __future__ import annotations

import logging
import platform
import sys
from collections.abc import Mapping
from urllib.parse import quote, urlencode

import numpy as np
import pyqtgraph as pg
import scipy
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QByteArray, QUrl, qVersion
from PySide6.QtGui import QDesktopServices

from mag_opt_detective import __version__

logger = logging.getLogger("mag_opt_detective")

REPOSITORY = "https://github.com/wyzula-jan/mag-opt-detective"
# The documentation site built from docs/site (the url in CITATION.cff). It goes live when
# Jan publishes it with the Docs site workflow; until then the address does not answer.
DOCS_URL = "https://wyzula-jan.github.io/mag-opt-detective/"
ISSUES = f"{REPOSITORY}/issues"
NEW_ISSUE = f"{ISSUES}/new"
BUG_FORM, FEATURE_FORM = "bug_report.yml", "feature_request.yml"  # in .github/ISSUE_TEMPLATE
# the bug form's fields the app fills in (their ids), in the form's order
ENVIRONMENT_FIELDS = ("version", "install", "os", "python", "qt", "libraries", "plot")
MAX_VALUE = 100  # characters of one field: the whole address stays well under 8000
STANDALONE, FROM_SOURCE = "Standalone app", "From source"


def new_issue_url(form: str, fields: Mapping[str, str] | None = None) -> str:
    """The address of a new issue with the issue form *form*, its *fields* filled in."""
    query = {"template": form}
    query.update({key: value[:MAX_VALUE] for key, value in (fields or {}).items()})
    return f"{NEW_ISSUE}?{urlencode(query, quote_via=quote)}"


def feature_request_url() -> str:
    return new_issue_url(FEATURE_FORM)


def bug_report_url(environment: Mapping[str, str]) -> str:
    """The bug form filled in with *environment* (see :func:`environment`)."""
    unknown = set(environment) - set(ENVIRONMENT_FIELDS)
    if unknown:
        raise KeyError(f"not a field of the bug form: {', '.join(sorted(unknown))}")
    return new_issue_url(BUG_FORM, environment)


def system() -> str:
    """The operating system with its version and the processor, e.g. ``macOS 15.6 (arm64)``."""
    if sys.platform == "darwin":
        name = f"macOS {platform.mac_ver()[0] or platform.release()}"
    elif sys.platform == "win32":
        name = f"Windows {platform.release()} {platform.version()}"
    else:
        try:
            name = platform.freedesktop_os_release()["PRETTY_NAME"]
        except (OSError, KeyError):
            name = platform.system() or "Unknown system"
        name += f", kernel {platform.release()}"
    return f"{name} ({platform.machine()})"


def environment(plot: str = "") -> dict[str, str]:
    """What the bug form is filled in with; *plot* names the plot shown (may be empty)."""
    fields = {
        "version": __version__,
        "install": STANDALONE if getattr(sys, "frozen", False) else FROM_SOURCE,
        "os": system(),
        "python": platform.python_version(),
        "qt": f"Qt {qVersion()}, PySide6 {pyside_version}",
        "libraries": f"numpy {np.__version__}, scipy {scipy.__version__}, "
        f"pyqtgraph {pg.__version__}",
    }
    if plot:
        fields["plot"] = plot
    return fields


def open_url(url: str) -> None:
    """Open *url* in the web browser; OSError when none opens it (the log keeps the address)."""
    if not QDesktopServices.openUrl(QUrl.fromEncoded(QByteArray(url.encode()))):
        logger.info("Address: %s", url)
        raise OSError("No web browser opened the page: its address is in the log")
