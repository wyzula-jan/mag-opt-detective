"""New versions on GitHub: version numbers, the release feed and which release to offer (no Qt).

The app only says that a new version exists; it never installs one (the app bundles are not
code-signed, so people download the new one and replace the app). :func:`fetch_releases` is one
anonymous GET of GitHub's releases API: no account, no cookies, no data, a short timeout, and
Python's own proxy and certificate handling. Releases are tagged ``vX.Y.Z`` by
``tools/release.py``; the *Release* workflow marks versions before 1.0.0 as GitHub pre-releases,
so while the app is 0.x pre-releases count (:func:`default_channel`).
"""

from __future__ import annotations

import http.client
import json
import re
import ssl
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import total_ordering
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "wyzula-jan/mag-opt-detective"
RELEASES_API = f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=10"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases"
DOWNLOADS = f"{RELEASES_PAGE}/download/"  # where the files attached to a release are
TIMEOUT = 5.0  # seconds
MAX_BYTES = 4_000_000  # ten releases with their notes are far less
STABLE, PRERELEASE = "stable", "pre-release"
CHANNELS = (STABLE, PRERELEASE)
# the archives the App bundles workflow attaches to every release, by sys.platform
ASSETS = {
    "darwin": "mag-opt-detective-macos.zip",
    "win32": "mag-opt-detective-windows.zip",
    "linux": "mag-opt-detective-linux.tar.gz",
}

_NUMBER = r"(0|[1-9]\d*)"
_VERSION = re.compile(
    rf"v?{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
    r"(?:-(?P<pre>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)"  # SemVer: 1.2.0-rc.1
    r"|(?P<pep>a|b|rc)(?P<pep_n>\d+)"  # PEP 440: 1.2.0rc1
    r"|\.dev(?P<dev>\d*))?"  # PEP 440: 1.2.0.dev0, a version in development
    r"(?:\+[0-9A-Za-z.-]+)?"  # build metadata (a local build): ignored
)


@total_ordering
@dataclass(frozen=True)
class Version:
    """A version number ordered as Semantic Versioning orders it: 0.10.0 > 0.9.1, and a
    pre-release before its release (1.2.0-rc.1 < 1.2.0).

    :meth:`parse` also reads the PEP 440 forms a version in development has: ``1.2.0.dev0``
    and ``1.2.0rc1`` are pre-releases of 1.2.0 (newer than 1.1.x, older than 1.2.0).
    """

    major: int
    minor: int
    patch: int
    pre: tuple[int | str, ...] = ()  # pre-release identifiers; () for a release

    @classmethod
    def parse(cls, text: str) -> Version:
        """``vX.Y.Z`` or ``X.Y.Z``, with an optional pre-release; ValueError otherwise."""
        match = _VERSION.fullmatch(text.strip()) if isinstance(text, str) else None
        if match is None:
            raise ValueError(f"not a version number: {text!r}")
        major, minor, patch = (int(n) for n in match.group(1, 2, 3))
        if match["pre"] is not None:
            pre = tuple(int(p) if p.isdigit() else p for p in match["pre"].split("."))
        elif match["pep"] is not None:
            pre = (match["pep"], int(match["pep_n"]))
        elif match["dev"] is not None:
            pre = ("dev", int(match["dev"] or 0))
        else:
            pre = ()
        return cls(major, minor, patch, pre)

    @property
    def is_prerelease(self) -> bool:
        return bool(self.pre)

    def _key(self) -> tuple:
        # numeric identifiers sort before alphanumeric ones, and a release after its
        # pre-releases (a longer list of identifiers sorts after its prefix, as tuples do)
        pre = tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in self.pre)
        return self.major, self.minor, self.patch, not self.pre, pre

    def __lt__(self, other: Version) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() < other._key()

    def __str__(self) -> str:
        text = f"{self.major}.{self.minor}.{self.patch}"
        return f"{text}-{'.'.join(str(p) for p in self.pre)}" if self.pre else text


@dataclass(frozen=True)
class Release:
    """A published release: its version, tag, page on GitHub and attached files."""

    version: Version
    tag: str
    page: str  # the release page on GitHub (its notes and files)
    prerelease: bool  # marked as one on GitHub, or a pre-release version
    assets: dict[str, str] = field(default_factory=dict, compare=False)  # file name -> URL

    def download_url(self, platform: str = sys.platform) -> str:
        """The archive of the app for *platform*, or the release page if it has none."""
        return self.assets.get(ASSETS.get(platform, ""), self.page)


class UpdateCheckError(Exception):
    """The releases could not be read; the message says why, for people."""


def default_channel(current: Version) -> str:
    """Pre-releases count while the app is 0.x (every 0.x release is one), not from 1.0.0."""
    return PRERELEASE if current.major == 0 else STABLE


def _ours(url: object, prefix: str) -> bool:
    """A link into this repository's releases (the only pages the app opens)."""
    return isinstance(url, str) and url.startswith(prefix) and url.isprintable()


def parse_releases(payload: object) -> list[Release]:
    """The releases in an answer of GitHub's list-releases API.

    Drafts, tags that are no version number and files outside this repository's releases are
    left out; a release page outside them is replaced by the tag's page.
    """
    if not isinstance(payload, list):
        raise UpdateCheckError("GitHub sent an answer the app cannot read")
    releases = []
    for item in payload:
        if not isinstance(item, dict) or item.get("draft") is not False:
            continue
        tag = item.get("tag_name")
        try:
            version = Version.parse(tag)
        except ValueError:
            continue
        page = item.get("html_url")
        if not _ours(page, f"{RELEASES_PAGE}/"):
            page = f"{RELEASES_PAGE}/tag/{tag}"
        assets = {}
        for asset in item.get("assets") or ():
            if not isinstance(asset, dict) or asset.get("state", "uploaded") != "uploaded":
                continue
            name, url = asset.get("name"), asset.get("browser_download_url")
            if isinstance(name, str) and _ours(url, DOWNLOADS):
                assets[name] = url
        prerelease = item.get("prerelease") is not False or version.is_prerelease
        releases.append(Release(version, tag, page, prerelease, assets))
    return releases


def newest(releases: Iterable[Release], channel: str) -> Release | None:
    """The newest release of *channel*: every release for pre-release, else the others."""
    candidates = [r for r in releases if channel == PRERELEASE or not r.prerelease]
    return max(candidates, key=lambda r: r.version, default=None)


def update_for(releases: Iterable[Release], current: Version, channel: str) -> Release | None:
    """The newest release of *channel* if it is newer than *current*, else None."""
    release = newest(releases, channel)
    return release if release is not None and release.version > current else None


def releases_request(version: str) -> Request:
    """The GET of the newest releases: who asks (the app and its version) and nothing else."""
    return Request(
        RELEASES_API,
        headers={
            "User-Agent": f"mag-opt-detective/{version}",
            "Accept": "application/vnd.github+json",
        },
    )


def _error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code in (403, 429):  # the limit for requests without an account
            return "GitHub limits how often it answers; try again in an hour"
        return f"GitHub answered with an error (HTTP {exc.code})"
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, TimeoutError):
        return f"GitHub did not answer within {TIMEOUT:g} s"
    if isinstance(reason, ssl.SSLError):
        return "the secure connection to GitHub failed"
    if isinstance(reason, OSError):
        return "no connection to GitHub"
    return "GitHub sent an answer the app cannot read"


def fetch_releases(
    version: str, opener: Callable | None = None, timeout: float = TIMEOUT
) -> list[Release]:
    """Ask GitHub for the newest releases (blocks: call it off the GUI thread).

    *version* is the app's, sent in the User-Agent header as GitHub asks. *opener* takes the
    request and the timeout as :func:`urllib.request.urlopen` (the default) does; tests pass a
    fake. Every failure (offline, timeout, rate limit, an answer that is no release list)
    raises :class:`UpdateCheckError`.
    """
    opener = opener or urlopen
    try:
        with opener(releases_request(version), timeout=timeout) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("the answer is too long")
        payload = json.loads(body)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise UpdateCheckError(_error_text(exc)) from exc
    return parse_releases(payload)
