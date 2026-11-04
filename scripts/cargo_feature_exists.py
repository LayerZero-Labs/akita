#!/usr/bin/env python3
"""Return 0 when `crate` defines `feature` in its Cargo.toml, else 1.

With `--default`, return 0 only when `feature` is in the crate's `default`
feature list. With `--root`, read the crate under that checkout instead of the
one that contains this script.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


def feature_exists(repo_root: Path, crate: str, feature: str) -> bool:
    manifest = repo_root / "crates" / crate / "Cargo.toml"
    if not manifest.is_file():
        return False
    text = manifest.read_text(encoding="utf-8")
    pattern = rf"^\s*{re.escape(feature)}\s*="
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def feature_in_default(repo_root: Path, crate: str, feature: str) -> bool:
    manifest = repo_root / "crates" / crate / "Cargo.toml"
    if not manifest.is_file():
        return False
    features = tomllib.loads(manifest.read_text(encoding="utf-8")).get("features", {})
    return feature in features.get("default", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", action="store_true", dest="in_default")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("crate")
    parser.add_argument("feature")
    args = parser.parse_args()
    check = feature_in_default if args.in_default else feature_exists
    return 0 if check(args.root, args.crate, args.feature) else 1


if __name__ == "__main__":
    raise SystemExit(main())
