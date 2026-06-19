#!/usr/bin/env python3
"""Return 0 when `crate` defines `feature` in its Cargo.toml, else 1."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def feature_exists(repo_root: Path, crate: str, feature: str) -> bool:
    manifest = repo_root / "crates" / crate / "Cargo.toml"
    if not manifest.is_file():
        return False
    text = manifest.read_text(encoding="utf-8")
    pattern = rf"^\s*{re.escape(feature)}\s*="
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: cargo_feature_exists.py <crate> <feature>", file=sys.stderr)
        return 2
    crate, feature = sys.argv[1], sys.argv[2]
    repo_root = Path(__file__).resolve().parent.parent
    return 0 if feature_exists(repo_root, crate, feature) else 1


if __name__ == "__main__":
    raise SystemExit(main())
