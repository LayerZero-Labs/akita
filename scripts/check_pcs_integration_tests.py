#!/usr/bin/env python3
"""Check that akita-pcs has one complete integration-test target."""

from __future__ import annotations

import re
import sys
import tomllib
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CRATE_ROOT = REPO_ROOT / "crates" / "akita-pcs"
TESTS_ROOT = CRATE_ROOT / "tests"
MODULES_ROOT = TESTS_ROOT / "integration_tests"
SUITE_PATH = TESTS_ROOT / "integration_tests.rs"
SUITE_RELATIVE_PATH = "tests/integration_tests.rs"
PATH_ATTRIBUTE = re.compile(r'#\[path\s*=\s*"([^"]+)"\]')


def fail(messages: list[str]) -> None:
    for message in messages:
        print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def expected_modules() -> set[str]:
    files = {
        f"integration_tests/{path.name}"
        for path in MODULES_ROOT.glob("*.rs")
    }
    directories = {
        f"integration_tests/{path.name}/mod.rs"
        for path in MODULES_ROOT.iterdir()
        if path.is_dir() and (path / "mod.rs").is_file()
    }
    return files | directories


def main() -> None:
    errors: list[str] = []
    manifest = tomllib.loads((CRATE_ROOT / "Cargo.toml").read_text())

    if manifest["package"].get("autotests") is not False:
        errors.append("crates/akita-pcs/Cargo.toml must set autotests = false")

    test_targets = manifest.get("test", [])
    expected_target = {
        "name": "integration_tests",
        "path": SUITE_RELATIVE_PATH,
    }
    if test_targets != [expected_target]:
        errors.append(
            "crates/akita-pcs/Cargo.toml must declare only the "
            f"integration_tests target at {SUITE_RELATIVE_PATH}"
        )

    top_level_sources = sorted(path.name for path in TESTS_ROOT.glob("*.rs"))
    if top_level_sources != [SUITE_PATH.name]:
        errors.append(
            "crates/akita-pcs/tests must contain only integration_tests.rs "
            f"at the top level; found {top_level_sources}"
        )

    declared = PATH_ATTRIBUTE.findall(SUITE_PATH.read_text())
    counts = Counter(declared)
    duplicates = sorted(path for path, count in counts.items() if count != 1)
    if duplicates:
        errors.append(f"suite module paths must be unique; duplicates: {duplicates}")

    declared_set = set(declared)
    expected_set = expected_modules()
    missing = sorted(expected_set - declared_set)
    stale = sorted(declared_set - expected_set)
    if missing:
        errors.append(f"suite does not declare these modules: {missing}")
    if stale:
        errors.append(f"suite declares missing or nested modules: {stale}")

    if errors:
        fail(errors)

    print(
        "ok: akita-pcs has one integration-test target with "
        f"{len(expected_set)} source modules"
    )


if __name__ == "__main__":
    main()
