"""Shared lattice-estimator checkout pin and discovery for Akita scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# malb/lattice-estimator#217 head; strict descendant of malb#213 @ 27a581b.
# Euclidean (BDGL16, norm=2) output is unchanged from 27a581b; infinity needs #217.
PINNED_LATTICE_ESTIMATOR_SHA = "c667a48546f140c3a5454c7503c3ca44a264cce2"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def locate_estimator(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("LATTICE_ESTIMATOR_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = repo_root()
    candidates.extend(
        [
            root / "third_party" / "lattice-estimator",
            root / "lattice-estimator",
            root.parent / "lattice-estimator",
        ]
    )
    for candidate in candidates:
        if (candidate / "estimator" / "__init__.py").exists():
            return candidate.resolve()
    raise SystemExit(
        "Could not locate lattice-estimator. "
        "Initialize `third_party/lattice-estimator`, set LATTICE_ESTIMATOR_PATH, "
        "or pass --estimator-path."
    )


def estimator_git_sha(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def assert_pinned_estimator(path: Path) -> None:
    actual = estimator_git_sha(path)
    if actual != PINNED_LATTICE_ESTIMATOR_SHA:
        raise SystemExit(
            "lattice-estimator SHA mismatch: "
            f"expected {PINNED_LATTICE_ESTIMATOR_SHA}, got {actual} at {path}"
        )


def normalize_git_remote_url(url: str) -> str:
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    if url.startswith("ssh://git@github.com/"):
        return "https://github.com/" + url.removeprefix("ssh://git@github.com/")
    return url


def estimator_remote_url(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        return normalize_git_remote_url(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
