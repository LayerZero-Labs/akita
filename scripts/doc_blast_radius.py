#!/usr/bin/env python3
"""Suggest documentation updates for a PR based on changed paths.

Reads docs/doc-blast-radius.json and prints a Markdown report (or upsert body
with marker for GitHub Actions). Advisory only — see docs/documentation.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

import json

MARKER = "<!-- akita-doc-blast-radius -->"
REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = REPO_ROOT / "docs" / "doc-blast-radius.json"


def git_changed_files(base: str, head: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def path_matches_glob(changed: str, pattern: str) -> bool:
    # Normalize: yaml uses crates/foo/** ; git paths are forward-slash.
    pat = pattern.rstrip("/")
    if pat.endswith("/**"):
        prefix = pat[:-3]
        return changed == prefix or changed.startswith(prefix + "/")
    return fnmatch.fnmatch(changed, pat)


def region_hits(changed_files: list[str], region: dict) -> bool:
    for cf in changed_files:
        for pat in region.get("code", []):
            if path_matches_glob(cf, pat):
                return True
    return False


def load_map() -> list[dict]:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    return data.get("regions", [])


def render_report(changed_files: list[str], matched: list[tuple[dict, list[str]]]) -> str:
    lines: list[str] = [
        MARKER,
        "## Documentation blast radius (advisory)",
        "",
        "These regions *may* need doc/spec/book updates based on changed paths.",
        "This is not a merge gate. See [`docs/documentation.md`](docs/documentation.md).",
        "",
    ]
    if not changed_files:
        lines.append("_No changed files detected._")
        return "\n".join(lines) + "\n"

    lines.append(f"**Changed files in this PR:** {len(changed_files)}")
    lines.append("")

    if not matched:
        lines.append(
            "_No blast-radius regions matched. If this PR changes user-visible "
            "behavior, consider whether `docs/doc-blast-radius.json` needs a new region._"
        )
        return "\n".join(lines) + "\n"

    for region, touched in matched:
        rid = region.get("id", "unknown")
        desc = region.get("description", "")
        lines.append(f"### `{rid}`")
        if desc:
            lines.append(desc)
        lines.append("")
        lines.append("**Code paths touched:**")
        for t in touched:
            lines.append(f"- `{t}`")
        lines.append("")
        lines.append("**Consider updating:**")
        for doc in region.get("consider_updating", []):
            lines.append(f"- `{doc}`")
        lines.append("")

    lines.append("---")
    lines.append(
        "**Per-PR checklist:** spec `Status` / acceptance criteria; book owning "
        "page; `AGENTS.md` if contracts changed; archive spec after fold."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="git merge base ref")
    parser.add_argument("--head", default="HEAD", help="git head ref")
    parser.add_argument(
        "--marker-only",
        action="store_true",
        help="print marker line only (for workflow debugging)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write report to this file instead of stdout",
    )
    args = parser.parse_args()

    if not MAP_PATH.is_file():
        print(f"error: missing {MAP_PATH}", file=sys.stderr)
        return 2

    try:
        changed = git_changed_files(args.base, args.head)
    except subprocess.CalledProcessError as e:
        print(f"error: git diff failed: {e}", file=sys.stderr)
        return 2

    regions = load_map()
    matched: list[tuple[dict, list[str]]] = []
    for region in regions:
        touched = [
            cf
            for cf in changed
            if any(path_matches_glob(cf, pat) for pat in region.get("code", []))
        ]
        if touched:
            matched.append((region, touched))

    body = render_report(changed, matched)
    if args.marker_only:
        print(MARKER)
    elif args.output is not None:
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
