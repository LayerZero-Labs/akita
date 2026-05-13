#!/usr/bin/env python3
"""Sanity-check the Python SIS table parser against the Rust unit tests in
`crates/akita-planner/src/sis_security.rs`. Any mismatch is a parser bug we
must fix before trusting the analysis."""

from extract_params import load_sis_table, min_rank_for_secure_width, \
    ceil_supported_collision

table = load_sis_table()

CASES = [
    # From rank_lookup test
    ((32, 7, 500), 1),
    ((32, 7, 959), 1),
    ((32, 7, 960), 2),
    # From d128_rank_lookup
    ((128, 2, 4_862_955_514), 1),
    ((128, 2, 4_862_955_515), 2),
    ((128, 63, 4_900_937), 1),
    ((128, 63, 4_900_938), 2),
    ((128, 31, 20_241_230), 1),
    ((128, 31, 20_241_231), 2),
    # Suspicious case from our mismatch list
    ((128, 255, 524288), None),
]

CEIL_CASES = [
    ((32, 248), 255),
    ((64, 62), 63),
    ((128, 62), 63),
    ((128, 248), 255),
    ((128, 7_812), 8191),
    # The case my analysis flagged: a_extraction = 156
    ((128, 156), 255),
    ((128, 104), 255),
]

print("--- rank lookups ---")
for (d, coll, w), expected in CASES:
    got = min_rank_for_secure_width(table, d, coll, w)
    print(f"  D={d} bucket={coll} width={w}: rank={got} expected={expected}",
          "OK" if expected is None or got == expected else "MISMATCH")

print()
print("--- ceil_supported_collision ---")
for (d, val), expected in CEIL_CASES:
    got = ceil_supported_collision(d, val)
    print(f"  D={d} val={val}: bucket={got} expected={expected}",
          "OK" if got == expected else "MISMATCH")

print()
print("--- Spot-check the alleged 'mismatch': D=128, root onehot tensor, n_a=1 ---")
print("  fold: m_vars=19, r_vars=14, delta_commit=1, log_basis=2, stored n_a=1")
print("  inner_width = 2^19 * 1 = 524288")
print("  extraction_linf = 4 * omega = 4 * 13 = 52")
print("  at root level with log_commit_bound=1: a_raw = 2")
print("  a_extraction = 2 * 52 = 104")
bucket = ceil_supported_collision(128, 104)
print(f"  a_bucket = ceil_supported(128, 104) = {bucket}")
widths = table.get((128, 255))
print(f"  table (128, 255) widths: {widths}")
rank = min_rank_for_secure_width(table, 128, 255, 524288)
print(f"  rank required for width 524288 at (128, 255) = {rank}")
print()
print("If rank > 1 here, then the planner stored n_a=1 is below the SIS floor.")
print("Need to triage: either (a) my model is missing something, (b) the planner")
print("converged to a different layout shape than the table entry shows, or")
print("(c) there is a real security bug.")
