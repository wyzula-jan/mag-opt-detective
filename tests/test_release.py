"""The release setup: python-semantic-release's configuration and the release workflows."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import mag_opt_detective

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def workflow(name: str) -> tuple[dict, dict]:
    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return loaded, loaded.get("on", loaded.get(True))  # YAML 1.1 reads "on" as True


def test_every_file_names_the_same_version():
    version = pyproject()["project"]["version"]
    assert mag_opt_detective.__version__ == version

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^version: *(\S+)", citation, re.MULTILINE)[1] == version

    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert f'name = "mag-opt-detective"\nversion = "{version}"' in lock

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## v?(\d+\.\d+\.\d+)", changelog, re.MULTILINE)[1] == version


def test_semantic_release_writes_every_version_file():
    config = pyproject()["tool"]["semantic_release"]
    assert config["version_toml"] == ["pyproject.toml:project.version"]
    assert set(config["version_variables"]) == {
        "src/mag_opt_detective/__init__.py:__version__",
        "CITATION.cff:version",
    }
    assert set(config["assets"]) == {"uv.lock", "CITATION.cff"}  # changed by the build command
    assert "uv lock" in config["build_command"]
    assert "date-released" in config["build_command"]


def test_a_release_commit_is_one_conventional_subject_by_the_author():
    config = pyproject()["tool"]["semantic_release"]
    assert config["commit_message"] == "chore(release): v{version}"
    assert config["commit_author"]["default"] == "wyzula-jan <wyzula.jan@gmail.com>"
    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert 'GIT_COMMIT_AUTHOR: "wyzula-jan <wyzula.jan@gmail.com>"' in release


def test_only_features_fixes_and_breaking_changes_make_a_release():
    options = pyproject()["tool"]["semantic_release"]["commit_parser_options"]
    assert options["minor_tags"] == ["feat"]
    assert options["patch_tags"] == ["fix", "perf"]
    assert options["default_bump_level"] == 0


def test_a_release_keeps_the_written_changelog_sections():
    changelog_options = pyproject()["tool"]["semantic_release"]["changelog"]
    assert changelog_options["mode"] == "update"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog_options["insertion_flag"] in changelog.partition("\n## ")[0]


def test_only_the_release_workflow_publishes_releases():
    publishing = sorted(
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if "gh release create" in path.read_text(encoding="utf-8")
    )
    assert publishing == ["release.yml"]

    loaded, triggers = workflow("release.yml")
    assert triggers == {"push": {"branches": ["main"]}}
    assert loaded["permissions"] == {"contents": "read"}
    jobs = loaded["jobs"]
    assert jobs["bundles"]["uses"] == "./.github/workflows/bundles.yml"
    assert jobs["publish"]["permissions"] == {"contents": "write"}
    assert set(jobs["publish"]["needs"]) == {"release", "bundles", "pypi"}  # PyPI first
    assert set(jobs["pypi"]["needs"]) == {"release", "bundles"}  # only when all build
    assert jobs["pypi"]["permissions"] == {"id-token": "write"}  # trusted publishing
    assert jobs["pypi"]["environment"]["name"] == "pypi"

    steps = jobs["release"]["steps"]
    checkout = next(step for step in steps if "actions/checkout" in step.get("uses", ""))
    assert checkout["with"]["ssh-key"] == "${{ secrets.CI_DEPLOY_SSH_KEY }}"
    assert checkout["with"]["fetch-depth"] == 0
    # the GitHub release is created last, so a release without its files is never the latest
    run = "\n".join(step.get("run", "") for step in steps)
    assert "semantic-release -vv version --no-vcs-release" in run


def test_the_bundles_are_built_for_pull_requests_that_change_the_app():
    bundles, triggers = workflow("bundles.yml")
    assert "src/**" in triggers["pull_request"]["paths"]
    assert "packaging/**" in triggers["pull_request"]["paths"]
    assert "ref" in triggers["workflow_call"]["inputs"]
    assert "tags" not in triggers["push"]  # a pushed tag builds nothing on its own
    assert list(bundles["jobs"]) == ["bundle"]


def test_the_release_texts_name_no_earlier_app():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    texts = {
        "CHANGELOG.md header": changelog.partition("\n## ")[0],
        "CITATION.cff": (ROOT / "CITATION.cff").read_text(encoding="utf-8"),
        "release-notes.html": (ROOT / "docs/site/pages/release-notes.html").read_text("utf-8"),
        "release.yml": (WORKFLOWS / "release.yml").read_text(encoding="utf-8"),
    }
    for name, text in texts.items():
        found = re.findall(r"4\.9|4p9|legacy|rewritten|PyQt5", text, re.IGNORECASE)
        assert not found, f"{name}: {found}"
