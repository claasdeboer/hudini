"""Repository hygiene checks that guard against publishing sensitive content."""

import subprocess
from pathlib import Path

import pytest

FORBIDDEN_STRINGS_FILE = Path(__file__).resolve().parent.parent / "data" / "forbidden_strings.txt"


@pytest.mark.integration
def test_tracked_tree_contains_no_forbidden_strings() -> None:
    """Grep the tracked tree for the strings in the gitignored local list;
    skip when the list is absent."""
    if not FORBIDDEN_STRINGS_FILE.exists():
        pytest.skip("no local forbidden-strings list")
    forbidden = [
        line.strip() for line in FORBIDDEN_STRINGS_FILE.read_text().splitlines() if line.strip()
    ]
    if not forbidden:
        pytest.skip("forbidden-strings list is empty")

    repo_root = FORBIDDEN_STRINGS_FILE.parent.parent
    hits: list[str] = []
    for needle in forbidden:
        result = subprocess.run(
            ["git", "grep", "-i", "-l", "--fixed-strings", needle, "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            files = result.stdout.strip().splitlines()
            hits.extend(f"{needle!r} in {name}" for name in files)

    if hits:
        raise AssertionError("forbidden strings found in tracked tree: " + "; ".join(hits))
