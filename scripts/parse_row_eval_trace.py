#!/usr/bin/env python3
"""Aggregate per-level / per-component verifier timings from a tracing-chrome
trace produced by `examples/profile` (with AKITA_VERIFY_REPEAT).

Segments the main thread (tid 0) by `verify_iter` spans, treats each
`stage2_expected_output_claim` inside a verify as one ring-switch level, and
sums child span durations per component, averaged across verify iterations.

Usage:
    parse_row_eval_trace.py <trace.json> [label]
"""
import json
import sys
from collections import defaultdict

LEVEL_MARK = "stage2_expected_output_claim"
VERIFY_MARK = "verify_iter"
# Components reported per level. `others` catches whatever is inside the level
# span but not attributed to a named child.
COMPONENTS = [
    "stage2_witness_eval",
    "e_structured",
    "t_structured",
    "z_structured",
    "setup_contribution",
    "r_structured",
    "r_dense",
    "structured_chunks",
    "stage2_ring_switch_row_eval",
]


def load_events(path):
    with open(path) as fh:
        text = fh.read().strip()
    if not text.endswith("]"):
        text = text.rstrip().rstrip(",") + "]"
    return json.loads(text)


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    events = load_events(path)

    # (verify_idx, level_idx, name) -> total_us ; and count of verifies seen.
    agg = defaultdict(float)
    level_dur = defaultdict(float)  # (verify, level) -> expected_output_claim us
    stack = []
    cur_verify = -1
    cur_level = -1
    verifies = set()

    for ev in events:
        if ev.get("tid") != 0:
            continue
        ph = ev.get("ph")
        name = ev.get("name")
        if ph not in ("B", "E"):
            continue
        ts = ev["ts"]
        if ph == "B":
            if name == VERIFY_MARK:
                cur_verify += 1
                cur_level = -1
                verifies.add(cur_verify)
            if name == LEVEL_MARK:
                cur_level += 1
            stack.append((name, ts, cur_verify, cur_level))
        else:  # E
            # pop matching name (stack is well nested per thread)
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    _, start_ts, v, lvl = stack.pop(i)
                    break
            else:
                continue
            if v < 0:
                continue
            dur = ts - start_ts
            if name == LEVEL_MARK:
                level_dur[(v, lvl)] += dur
            if lvl >= 0:
                agg[(v, lvl, name)] += dur

    nverify = len(verifies)
    if nverify == 0:
        print(f"[{label}] no verify_iter spans found")
        return

    # levels present
    levels = sorted({lvl for (_, lvl) in level_dur})
    print(f"\n==== {label}  (avg over {nverify} verify iterations) ====")
    header = ["lvl", "total(us)"] + [c.replace("stage2_", "").replace("_structured", "")
                                     .replace("_contribution", "").replace("ring_switch_row_eval", "row_eval")
                                     for c in COMPONENTS]
    widths = [4, 11] + [max(9, len(h) + 1) for h in header[2:]]
    print("".join(h.rjust(w) for h, w in zip(header, widths)))

    grand = defaultdict(float)
    for lvl in levels:
        tot = sum(level_dur[(v, lvl)] for v in verifies) / nverify
        row = [str(lvl), f"{tot:.1f}"]
        for c in COMPONENTS:
            val = sum(agg[(v, lvl, c)] for v in verifies) / nverify
            grand[c] += val
            row.append(f"{val:.1f}" if val > 0 else "-")
        grand["total"] += tot
        print("".join(cell.rjust(w) for cell, w in zip(row, widths)))

    # totals row
    trow = ["ALL", f"{grand['total']:.1f}"] + [f"{grand[c]:.1f}" if grand[c] > 0 else "-"
                                               for c in COMPONENTS]
    print("".join(cell.rjust(w) for cell, w in zip(trow, widths)))
    return grand, nverify, levels


if __name__ == "__main__":
    main()
