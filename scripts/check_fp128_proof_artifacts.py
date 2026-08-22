#!/usr/bin/env python3
"""Check that the proved and production Fp128 add/sub words agree."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterable
import re
import shutil
import subprocess
from pathlib import Path


ADD_BODY = [
    0xAB020005,  # adds x5, x0, x2
    0xBA030026,  # adcs x6, x1, x3
    0x1A9F37E7,  # cset w7, hs
    0xAB0400A8,  # adds x8, x5, x4
    0xBA1F00C9,  # adcs x9, x6, xzr
    0x7A4038E0,  # ccmp w7, #0, #0, lo
    0x9A851100,  # csel x0, x8, x5, ne
    0x9A861121,  # csel x1, x9, x6, ne
]
SUB_BODY = [
    0xEB020005,  # subs x5, x0, x2
    0xFA030026,  # sbcs x6, x1, x3
    0x9A8423E7,  # csel x7, xzr, x4, hs
    0xEB0700A0,  # subs x0, x5, x7
    0xDA1F00C1,  # sbc x1, x6, xzr
]
RET = 0xD65F03C0
LOAD_A7F7_INTO_W4 = 0x128B0104  # mov w4, #-0x5809

ADD_OBJECT_WORDS = [*ADD_BODY, RET]
SUB_OBJECT_WORDS = [*SUB_BODY, RET]
ADD_PRODUCTION_WITNESS_WORDS = [LOAD_A7F7_INTO_W4, *ADD_BODY, RET]
SUB_PRODUCTION_WITNESS_WORDS = [LOAD_A7F7_INTO_W4, *SUB_BODY, RET]
VERIFIER_SEQUENCES = {
    "addition": [LOAD_A7F7_INTO_W4, *ADD_BODY],
    "subtraction": [LOAD_A7F7_INTO_W4, *SUB_BODY],
}

SYMBOL_RE = re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
INSTRUCTION_RE = re.compile(r"^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})(?:\s|$)")


def parse_symbol_words(disassembly: str) -> dict[str, list[int]]:
    """Return instruction words keyed by symbol, without a Mach-O underscore."""
    symbols: dict[str, list[int]] = {}
    current: list[int] | None = None
    for line in disassembly.splitlines():
        symbol_match = SYMBOL_RE.match(line)
        if symbol_match:
            name = symbol_match.group(1).removeprefix("_")
            current = symbols.setdefault(name, [])
            continue
        instruction_match = INSTRUCTION_RE.match(line)
        if instruction_match and current is not None:
            current.append(int(instruction_match.group(1), 16))
    return symbols


def find_llvm_objdump(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    llvm_objdump = shutil.which("llvm-objdump")
    if llvm_objdump:
        return [llvm_objdump]
    xcrun = shutil.which("xcrun")
    if xcrun:
        result = subprocess.run(
            [xcrun, "--find", "llvm-objdump"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [result.stdout.strip()]
    raise SystemExit("llvm-objdump was not found")


def read_symbol_words(tool: list[str], binary: Path, symbol: str) -> list[int]:
    for candidate in (symbol, f"_{symbol}"):
        result = subprocess.run(
            [*tool, f"--disassemble-symbols={candidate}", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        words = parse_symbol_words(result.stdout).get(symbol)
        if words is not None:
            return words
    raise SystemExit(f"symbol {symbol!r} was not found in {binary}")


def count_sequences_in_symbols(
    lines: Iterable[str],
    symbol_fragment: str,
    sequences: dict[str, list[int]],
) -> dict[str, int]:
    """Count instruction sequences within symbols whose names match a fragment."""
    counts = dict.fromkeys(sequences, 0)
    windows = {name: deque(maxlen=len(words)) for name, words in sequences.items()}
    in_matching_symbol = False

    for line in lines:
        symbol_match = SYMBOL_RE.match(line)
        if symbol_match:
            in_matching_symbol = symbol_fragment in symbol_match.group(1)
            for window in windows.values():
                window.clear()
            continue
        instruction_match = INSTRUCTION_RE.match(line)
        if not in_matching_symbol or instruction_match is None:
            continue

        word = int(instruction_match.group(1), 16)
        for name, expected in sequences.items():
            window = windows[name]
            window.append(word)
            if list(window) == expected:
                counts[name] += 1

    return counts


def read_verifier_sequence_counts(
    tool: list[str], binary: Path, symbol_fragment: str
) -> dict[str, int]:
    process = subprocess.Popen(
        [*tool, "--disassemble", str(binary)],
        stdout=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None:
        raise SystemExit("failed to read llvm-objdump output")
    counts = count_sequences_in_symbols(
        process.stdout,
        symbol_fragment,
        VERIFIER_SEQUENCES,
    )
    if process.wait() != 0:
        raise SystemExit(f"llvm-objdump failed for {binary}")
    return counts


def format_words(words: list[int]) -> str:
    return " ".join(f"{word:08x}" for word in words)


def require_words(label: str, actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        raise SystemExit(
            f"{label} instruction mismatch\n"
            f"expected: {format_words(expected)}\n"
            f"actual:   {format_words(actual)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-object", required=True, type=Path)
    parser.add_argument("--sub-object", required=True, type=Path)
    parser.add_argument("--production-witness", required=True, type=Path)
    parser.add_argument("--profile-binary", required=True, type=Path)
    parser.add_argument("--llvm-objdump")
    args = parser.parse_args()

    tool = find_llvm_objdump(args.llvm_objdump)
    add_object_words = read_symbol_words(tool, args.add_object, "akita_fp128_add_asm")
    sub_object_words = read_symbol_words(tool, args.sub_object, "akita_fp128_sub_asm")
    add_witness_words = read_symbol_words(
        tool,
        args.production_witness,
        "akita_fp128_add_production_witness",
    )
    sub_witness_words = read_symbol_words(
        tool,
        args.production_witness,
        "akita_fp128_sub_production_witness",
    )
    require_words("standalone addition proof object", add_object_words, ADD_OBJECT_WORDS)
    require_words(
        "production addition witness",
        add_witness_words,
        ADD_PRODUCTION_WITNESS_WORDS,
    )
    require_words(
        "standalone subtraction proof object",
        sub_object_words,
        SUB_OBJECT_WORDS,
    )
    require_words(
        "production subtraction witness",
        sub_witness_words,
        SUB_PRODUCTION_WITNESS_WORDS,
    )
    verifier_counts = read_verifier_sequence_counts(
        tool,
        args.profile_binary,
        "akita_verifier",
    )
    missing = [name for name, count in verifier_counts.items() if count == 0]
    if missing:
        raise SystemExit(
            "proved A7F7 sequences were not found in verifier symbols: "
            + ", ".join(missing)
        )
    count_text = ", ".join(
        f"{name}={count}" for name, count in verifier_counts.items()
    )
    print("Fp128 add/sub proof objects and production witness bytes match.")
    print(f"Final verifier profile contains proved A7F7 sequences: {count_text}.")


if __name__ == "__main__":
    main()
