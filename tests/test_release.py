"""tools/release.py: the next version from Conventional Commits, the version files, the
release commit and tag (on temporary repositories), and the release workflows."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-03"


def load_release():
    spec = importlib.util.spec_from_file_location("release_tool", ROOT / "tools" / "release.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release = load_release()

PYPROJECT = """\
[project]
name = "mag-opt-detective"
version = "5.0.0.dev0"
requires-python = ">=3.12"

[tool.other]
version = "9.9.9"
"""
LOCK = """\
version = 1
requires-python = ">=3.12"

[[package]]
name = "matplotlib"
version = "3.11.2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "mag-opt-detective"
version = "5.0.0.dev0"
source = { editable = "." }

[[package]]
name = "mag-opt-detective-plugin"
version = "5.0.0.dev0"
"""
INIT = '"""The package."""\n\n__version__ = "5.0.0.dev0"\n'
CITATION = "cff-version: 1.2.0\ntitle: Test\nversion: 5.0.0.dev0\nlicense: GPL-3.0-only\n"
NOTES = "The first release: it plots sweeps.\n\n### Added\n\n- **Maps:** colour maps\n"


def git(repo: Path, *args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def commit(repo: Path, message: str) -> None:
    """A commit with *message* that changes a scratch file."""
    path = repo / "work.txt"
    write(path, (path.read_text(encoding="utf-8") if path.exists() else "") + message + "\n")
    git(repo, "add", "work.txt")
    git(repo, "commit", "--quiet", "--file=-", stdin=message)


def run(repo: Path, *args: str) -> int:
    return release.main(["-C", str(repo), "--date", DATE, *args])


def head(repo: Path) -> str:
    return git(repo, "log", "-1", "--format=%H %s").strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with the version files at 5.0.0.dev0 and no tag; git hooks and the
    user's git configuration are off for it."""
    config = tmp_path / "gitconfig"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "--quiet", "--initial-branch=main")
    settings = {
        "user.name": "Test",
        "user.email": "test@example.org",
        "core.hooksPath": str(tmp_path / "no-hooks"),
        "core.autocrlf": "false",
        "commit.gpgsign": "false",
        "tag.gpgsign": "false",
    }
    for key, value in settings.items():
        git(path, "config", key, value)
    files = {
        "pyproject.toml": PYPROJECT,
        "uv.lock": LOCK,
        "src/mag_opt_detective/__init__.py": INIT,
        "CITATION.cff": CITATION,
        "CHANGELOG.md": release.CHANGELOG_HEADER,
        "notes.md": NOTES,
    }
    for name, text in files.items():
        write(path / name, text)
    git(path, "add", "--all")
    git(path, "commit", "--quiet", "-m", "feat: start the app")
    return path


@pytest.fixture
def released(repo, capsys):
    """*repo* after its first release, 0.1.0."""
    assert run(repo, "--first", "0.1.0", "--notes", str(repo / "notes.md")) == 0
    capsys.readouterr()
    return repo


# ---------------------------------------------------------------------- versions
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("feat(gui): add merge by field", ("feat", "gui", "add merge by field", False, "")),
        ("fix: apply the scaling factor", ("fix", None, "apply the scaling factor", False, "")),
        (
            "feat(io)!: write units\n\nBody.\n\nBREAKING CHANGE: tables name the unit.\nOld"
            " ones load.\n\nRefs: 12",
            ("feat", "io", "write units", True, "tables name the unit. Old ones load."),
        ),
        ("refactor!: split readers", ("refactor", None, "split readers", True, "")),
        (
            "build: drop python 3.11\n\nBREAKING-CHANGE: needs 3.12",
            ("build", None, "drop python 3.11", True, "needs 3.12"),
        ),
        ("Merge branch 'p4-09'", (None, None, "Merge branch 'p4-09'", False, "")),
        ("fix:no space", (None, None, "fix:no space", False, "")),
    ],
)
def test_commit_messages_are_parsed(message, expected):
    assert tuple(release.parse_commit(message))[1:] == expected


@pytest.mark.parametrize(
    ("last", "messages", "expected"),
    [
        ("0.1.0", ["fix: a"], "0.1.1"),
        ("0.1.0", ["perf(core): a", "docs: b"], "0.1.1"),
        ("0.1.0", ["revert: feat: a"], "0.1.1"),
        ("0.1.4", ["fix: a", "feat(gui): b"], "0.2.0"),
        # while the major version is 0 a breaking change bumps the minor version
        ("0.1.4", ["feat!: a"], "0.2.0"),
        ("0.1.4", ["fix: a\n\nBREAKING CHANGE: b"], "0.2.0"),
        ("0.1.4", ["refactor(core)!: a"], "0.2.0"),
        ("1.2.3", ["fix: a"], "1.2.4"),
        ("1.2.3", ["feat: a", "fix: b"], "1.3.0"),
        ("1.2.3", ["feat!: a"], "2.0.0"),
        ("1.2.3", ["docs: a\n\nBREAKING-CHANGE: b"], "2.0.0"),
    ],
)
def test_the_next_version_follows_the_commits(last, messages, expected):
    commits = [release.parse_commit(message) for message in messages]
    level = release.release_level(commits)
    version = release.next_version(release.parse_version(last), level)
    assert release.format_version(version) == expected


def test_docs_test_and_maintenance_commits_alone_make_no_release():
    messages = [
        "docs: a",
        "test: b",
        "chore: c",
        "ci: d",
        "build(deps): e",
        "style: f",
        "refactor(gui): g",
        "chore(release): v0.1.0",
        "Merge branch 'x'",
    ]
    assert release.release_level([release.parse_commit(m) for m in messages]) is None


def test_versions_are_plain_x_y_z():
    assert release.parse_version("v1.20.3") == (1, 20, 3)
    for text in ("1.2", "1.2.3.dev0", "01.2.3", "1.2.3-rc1", "v1.2-beta"):
        with pytest.raises(ValueError):
            release.parse_version(text)


# ---------------------------------------------------------------------- changelog
def test_the_changelog_section_groups_the_commits():
    messages = [
        "fix(gui): keep the zoom",
        "feat(gui): add a ruler",
        "docs: explain the ruler",
        "perf: draw faster",
        "feat(core): read `.dpt` files",
        "fix(gui): keep the zoom",  # the same subject twice: listed once
        "feat(io)!: write units\n\nBREAKING CHANGE: tables name the unit.",
        "refactor!: rename the settings",
        "revert: fix(core): round fields",
        "test: cover the ruler",
    ]
    section = release.changelog_section([release.parse_commit(m) for m in messages])
    assert section == (
        "### Added\n\n"
        "- **core:** read `.dpt` files\n"
        "- **gui:** add a ruler\n\n"
        "### Changed\n\n"
        "- rename the settings. **Breaking change.**\n"
        "- revert fix(core): round fields\n"
        "- **io:** write units. **Breaking:** tables name the unit.\n\n"
        "### Fixed\n\n"
        "- **gui:** keep the zoom\n\n"
        "### Performance\n\n"
        "- draw faster"
    )
    assert release.changelog_section([release.parse_commit("docs: a")]) == ""


def test_sections_go_on_top_of_the_changelog():
    text = release.add_section("", "0.1.0", "2026-10-01", "First.")
    text = release.add_section(text, "0.2.0", "2026-10-03", "### Fixed\n\n- a")
    assert text.startswith(release.CHANGELOG_HEADER)
    assert release.changelog_sections(text) == [
        ("0.2.0", "2026-10-03", "### Fixed\n\n- a"),
        ("0.1.0", "2026-10-01", "First."),
    ]
    with pytest.raises(release.ReleaseError, match="not a release heading"):
        release.changelog_sections(text + "\n## Unreleased\n")


# ---------------------------------------------------------------------- version files
def test_the_version_files_are_updated_in_place():
    pyproject = release.set_pyproject_version(PYPROJECT, "0.1.0")
    assert pyproject == PYPROJECT.replace('version = "5.0.0.dev0"', 'version = "0.1.0"')
    assert 'version = "9.9.9"' in pyproject  # another table's version stays
    lock = release.set_lock_version(LOCK, "0.1.0")
    assert lock.count('version = "5.0.0.dev0"') == 1  # the plugin's
    assert 'name = "mag-opt-detective"\nversion = "0.1.0"\n' in lock
    assert 'version = "3.11.2"' in lock
    assert release.set_init_version(INIT, "0.1.0").endswith('__version__ = "0.1.0"\n')

    citation = release.set_citation_version(CITATION, "0.1.0", "2026-10-03")
    assert 'version: 0.1.0\ndate-released: "2026-10-03"\nlicense' in citation
    citation = release.set_citation_version(citation, "0.2.0", "2026-11-01")
    assert citation.count("date-released") == 1
    assert 'version: 0.2.0\ndate-released: "2026-11-01"\n' in citation
    assert "cff-version: 1.2.0\n" in citation

    with pytest.raises(release.ReleaseError, match="no version"):
        release.set_lock_version("[[package]]\nname = 'other'\n", "0.1.0")


# ---------------------------------------------------------------------- releases
def test_the_first_release_needs_its_version_and_notes(repo, capsys):
    assert run(repo) == 1
    assert "--first X.Y.Z" in capsys.readouterr().err
    assert run(repo, "--first", "0.1.0") == 1
    assert "--notes FILE" in capsys.readouterr().err
    assert run(repo, "--version", "0.1.0", "--notes", str(repo / "notes.md")) == 1
    assert "--first" in capsys.readouterr().err
    assert git(repo, "tag") == ""


def test_the_first_release_commits_and_tags(released):
    repo = released
    assert head(repo).endswith(" chore(release): v0.1.0")
    assert git(repo, "status", "--porcelain") == ""
    assert set(release.read_versions(repo).values()) == {"0.1.0"}
    assert release.citation_date(repo) == DATE
    changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release.changelog_sections(changelog) == [("0.1.0", DATE, NOTES.strip())]
    # an annotated tag on the release commit, with the section ("###" lines kept)
    assert git(repo, "cat-file", "-t", "v0.1.0").strip() == "tag"
    assert git(repo, "rev-parse", "v0.1.0^{commit}") == git(repo, "rev-parse", "HEAD")
    message = git(repo, "tag", "-l", "--format=%(contents)", "v0.1.0")
    assert message.startswith("v0.1.0\n\nThe first release")
    assert "### Added\n\n- **Maps:** colour maps" in message
    # only the version files and the changelog changed
    changed = git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(changed) == sorted([*release.VERSION_FILES, release.CHANGELOG])


def test_a_fix_after_a_release_makes_a_patch_release(released, capsys):
    repo = released
    commit(repo, "fix(gui): keep the zoom")
    commit(repo, "docs: describe the zoom")
    assert run(repo) == 0
    out = capsys.readouterr().out
    assert "Next version: 0.1.1 (patch: fix)" in out
    assert "Tagged v0.1.1" in out
    assert head(repo).endswith(" chore(release): v0.1.1")
    assert set(release.read_versions(repo).values()) == {"0.1.1"}
    sections = release.changelog_sections((repo / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert [version for version, _date, _body in sections] == ["0.1.1", "0.1.0"]
    assert sections[0][2] == "### Fixed\n\n- **gui:** keep the zoom"
    assert release.last_release(repo) == "v0.1.1"


def test_a_breaking_change_before_1_0_bumps_the_minor_version(released, capsys):
    commit(released, "feat(io)!: write units\n\nBREAKING CHANGE: tables name the unit.")
    assert run(released) == 0
    assert "Next version: 0.2.0 (minor: a breaking change while the major version is 0)" in (
        capsys.readouterr().out
    )
    assert git(released, "describe", "--tags").strip() == "v0.2.0"


def test_version_1_0_0_is_set_by_hand_then_breaking_changes_bump_the_major(released, capsys):
    repo = released
    commit(repo, "fix: a")
    assert run(repo, "--version", "0.1.0") == 1
    assert "not after v0.1.0" in capsys.readouterr().err
    assert run(repo, "--version", "1.0.0") == 0
    assert "Next version: 1.0.0 (set with --version)" in capsys.readouterr().out
    commit(repo, "feat!: a")
    assert run(repo) == 0
    assert "Next version: 2.0.0 (major: a breaking change)" in capsys.readouterr().out
    assert release.release_tags(repo) == ["v0.1.0", "v1.0.0", "v2.0.0"]


def test_no_release_without_a_feat_fix_or_perf(released, capsys):
    before = head(released)
    commit(released, "docs: a")
    commit(released, "test(gui): b")
    commit(released, "chore: c")
    assert run(released) == 0
    out = capsys.readouterr().out
    assert out.startswith("No release: the 3 commits since v0.1.0 are ")
    assert "1 docs" in out and "1 test" in out
    assert head(released) != before  # the new commits, but no release commit
    assert head(released).endswith(" chore: c")
    assert release.release_tags(released) == ["v0.1.0"]


def test_a_dry_run_writes_nothing(released, capsys):
    repo = released
    commit(repo, "perf(gui): draw faster")
    before = head(repo)
    assert run(repo, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "Next version: 0.1.1 (patch: perf)" in out
    assert "Dry run, nothing written" in out
    assert f"## 0.1.1 - {DATE}\n\n### Performance\n\n- **gui:** draw faster" in out
    assert head(repo) == before
    assert git(repo, "status", "--porcelain") == ""
    assert release.release_tags(repo) == ["v0.1.0"]
    assert set(release.read_versions(repo).values()) == {"0.1.0"}


def test_a_dry_run_of_the_first_release_writes_nothing(repo, capsys):
    assert run(repo, "--dry-run", "--first", "0.1.0", "--notes", str(repo / "notes.md")) == 0
    out = capsys.readouterr().out
    assert "Commits since the start: 1 (1 feat)" in out
    assert "Next version: 0.1.0 (the first release)" in out
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "tag") == ""
    assert set(release.read_versions(repo).values()) == {"5.0.0.dev0"}


def test_uncommitted_changes_are_refused(released, capsys):
    commit(released, "fix: a")
    write(released / "pyproject.toml", PYPROJECT + "\n")
    assert run(released) == 1
    assert "uncommitted changes" in capsys.readouterr().err
    assert release.release_tags(released) == ["v0.1.0"]
    # a dry run only warns
    assert run(released, "--dry-run") == 0
    assert "Warning: uncommitted changes" in capsys.readouterr().out


def test_a_tagged_head_is_refused(released, capsys):
    assert run(released) == 1
    assert "already released as v0.1.0" in capsys.readouterr().err
    assert run(released, "--version", "0.2.0") == 1
    capsys.readouterr()
    commit(released, "fix: a")
    assert run(released, "--first", "0.2.0", "--notes", str(released / "notes.md")) == 1
    assert "--first is for the first release only" in capsys.readouterr().err


def test_notes_need_level_three_headings(released, capsys):
    commit(released, "fix: a")
    write(released / "bad.md", "## Added\n\n- a\n")
    assert run(released, "--notes", str(released / "bad.md")) == 1
    assert "### headings" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32", reason="the test hook is a shell script")
def test_a_rejected_commit_leaves_the_files_unchanged(released, tmp_path, capsys):
    commit(released, "fix: a")
    before = head(released)
    hooks = tmp_path / "hooks"
    write(hooks / "commit-msg", "#!/bin/sh\necho 'message rejected' >&2\nexit 1\n")
    (hooks / "commit-msg").chmod(0o755)
    git(released, "config", "core.hooksPath", str(hooks))
    assert run(released) == 1
    assert "message rejected" in capsys.readouterr().err
    assert head(released) == before
    assert git(released, "status", "--porcelain") == ""
    assert release.release_tags(released) == ["v0.1.0"]


def test_the_workflow_reads_the_version_and_the_notes(released, capsys):
    assert release.main(["-C", str(released), "--print-version"]) == 0
    assert capsys.readouterr().out == "0.1.0\n"
    assert release.main(["-C", str(released), "--print-notes", "v0.1.0"]) == 0
    assert capsys.readouterr().out == NOTES.strip() + "\n"
    assert release.main(["-C", str(released), "--print-notes", "0.2.0"]) == 1
    assert "no section for 0.2.0" in capsys.readouterr().err


def test_the_script_runs_from_the_command_line(released):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "release.py"), "-C", str(released), "--dry-run"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 1
    assert "already released as v0.1.0" in result.stderr


# ---------------------------------------------------------------------- this repository
def test_the_repository_version_files_agree():
    versions = release.read_versions(ROOT)
    assert len(set(versions.values())) == 1, versions
    version = versions["pyproject.toml"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = release.changelog_sections(changelog)
    if sections:  # released: the newest section is this version, of the citation's date
        newest, date, _body = sections[0]
        assert (newest, date) == (version, release.citation_date(ROOT))
        release.parse_version(version)
    else:
        assert release.citation_date(ROOT) is None
