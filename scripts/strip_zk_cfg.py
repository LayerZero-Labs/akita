#!/usr/bin/env python3
"""Strip paired `#[cfg(feature = "zk")]` / `#[cfg(not(feature = "zk"))]` arms (transparent only)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ZK = re.compile(r'^\s*#\[cfg\(feature = "zk"\)\]\s*$')
NOTZK = re.compile(r'^\s*#\[cfg\(not\(feature = "zk"\)\)\]\s*$')
INLINE_NOT = re.compile(r'#\[cfg\(not\(feature = "zk"\)\)\]\s*')
ZK_DOC = re.compile(r'^\s*///.*(masked|ZK plain-opening|ZkHiding|zk_hiding|B-blinding)', re.I)


def paren_depth(s: str) -> int:
    d = 0
    for c in s:
        if c in "({[":
            d += 1
        elif c in ")}]":
            d -= 1
    return d


def skip_statement(lines: list[str], start: int) -> int:
    d = 0
    i = start
    while i < len(lines):
        d += paren_depth(lines[i])
        stripped = lines[i].strip()
        if d <= 0 and (
            ";" in lines[i]
            or stripped.endswith(",")
            or stripped.endswith("),")
            or stripped == "}"
        ):
            return i + 1
        i += 1
    return i


def skip_brace(lines: list[str], start: int) -> int:
    d, i = 0, start
    while i < len(lines):
        for c in lines[i]:
            if c == "{":
                d += 1
            elif c == "}":
                d -= 1
                if d == 0:
                    return i + 1
        i += 1
    return i


def strip_cfg_if(text: str) -> str:
    out, i = [], 0
    while i < len(text):
        idx = text.find("cfg_if!", i)
        if idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:idx])
        j = text.find("{", idx)
        depth = 0
        k = j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        inner = text[j + 1 : k]
        m = re.search(r'if\s+#\[cfg\(feature\s*=\s*"zk"\)\]\s*\{', inner)
        if m:
            zk_open = inner.find("{", m.start())
            d = 0
            p = zk_open
            while p < len(inner):
                if inner[p] == "{":
                    d += 1
                elif inner[p] == "}":
                    d -= 1
                    if d == 0:
                        break
                p += 1
            rest = inner[p + 1 :].lstrip()
            if rest.startswith("else"):
                eo = rest.find("{")
                d = 0
                p = eo
                while p < len(rest):
                    if rest[p] == "{":
                        d += 1
                    elif rest[p] == "}":
                        d -= 1
                        if d == 0:
                            break
                    p += 1
                out.append(rest[eo + 1 : p].strip() + "\n")
            else:
                out.append(text[idx : k + 1])
        else:
            out.append(text[idx : k + 1])
        i = k + 1
    return "".join(out)


def process_lines(lines: list[str]) -> list[str]:
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        if ZK.match(line):
            i += 1
            if i < n and lines[i].strip().startswith("{"):
                i = skip_brace(lines, i)
            elif i < n:
                i = skip_statement(lines, i)
            continue
        if NOTZK.match(line):
            i += 1
            if i < n and lines[i].strip().startswith("{"):
                end = skip_brace(lines, i)
                body = "".join(lines[i:end]).strip()[1:-1]
                for bl in body.splitlines(keepends=True):
                    out.append(bl)
                i = end
            continue
        if "#[cfg(feature = \"zk\")]" in line:
            i += 1
            continue
        line = INLINE_NOT.sub("", line)
        out.append(line)
        i += 1
    cleaned = [ln for ln in out if not ZK_DOC.match(ln)]
    return cleaned


def strip_file(path: Path) -> None:
    text = strip_cfg_if(path.read_text())
    path.write_text("".join(process_lines(text.splitlines(keepends=True))))
    print("stripped", path)


def main() -> None:
    for arg in sys.argv[1:]:
        strip_file(Path(arg))


if __name__ == "__main__":
    main()
