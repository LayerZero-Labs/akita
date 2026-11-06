#!/usr/bin/env python3
"""Validate checked-quartet metadata and colex contribution tables."""
from math import comb
from pathlib import Path
import re

def contribution(offset, count, mask):
    rank = 0
    for bit in range(4):
        if mask >> bit & 1:
            count += 1
            rank += comb(offset + bit, count)
    return rank

def main():
    text = (Path(__file__).resolve().parents[1] / "crates/akita-challenges/src/checked_hint.rs").read_text()
    tables = {name: list(map(int, values.split(","))) for name, values in
              re.findall(r"const (META|RANK_ONE|RANK_TWO): \[u64; \d+\] = \[(.*?)\];", text, re.S)}
    expected = []
    for code in range(625):
        value = code
        n1 = n2 = m1 = m2 = s1 = s2 = free = 0
        for bit in range(4):
            coefficient = value % 5 - 2
            value //= 5
            if abs(coefficient) == 1:
                m1 |= 1 << bit
                s1 |= (coefficient < 0) << n1
                n1 += 1
            else:
                if abs(coefficient) == 2:
                    m2 |= 1 << free
                    s2 |= (coefficient < 0) << n2
                    n2 += 1
                free += 1
        expected.append(n1 | n2 << 3 | m1 << 6 | m2 << 10 | s1 << 14 | s2 << 18)
    assert tables["META"] == expected
    assert tables["RANK_ONE"] == [contribution(4*q,c,m) for q in range(16) for c in range(61) for m in range(16)]
    assert tables["RANK_TWO"] == [contribution(f,c,m) for f in range(61) for c in range(61) for m in range(16)]
    print("All quartet metadata and colex contribution tables match math.comb")

if __name__ == "__main__":
    main()
