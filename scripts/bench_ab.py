#!/usr/bin/env python3
"""Read criterion median point estimates for a saved baseline (ns/lane) and
optionally diff two baselines. Local A/B helper for the field-microbench work.

Usage:
  bench_ab.py <baseline>                 # print medians
  bench_ab.py <baseline_pre> <baseline_post>  # print pre -> post % change
"""
import glob
import json
import os
import sys

ROOT = "target/criterion"


def load(baseline):
    out = {}
    for est in glob.glob(f"{ROOT}/**/{baseline}/estimates.json", recursive=True):
        rel = os.path.relpath(os.path.dirname(os.path.dirname(est)), ROOT)
        out[rel] = json.load(open(est))["median"]["point_estimate"]
    return out


def tag(rel):
    return (
        rel.replace("field_arith_", "")
        .replace("_2048x16_ns_lane", "")
        .replace("_8x16x256_ns_lane", "")
        .replace("_512x16_ns_lane", "")
        .replace("_8x16x128_ns_lane", "")
        .replace("_2048_ns_per_op", "")
    )


def main():
    if len(sys.argv) == 2:
        data = load(sys.argv[1])
        for rel in sorted(data):
            print(f"  {tag(rel):66} {data[rel]:8.4f} ns")
    elif len(sys.argv) == 3:
        pre, post = load(sys.argv[1]), load(sys.argv[2])
        for rel in sorted(set(pre) & set(post)):
            a, b = pre[rel], post[rel]
            print(f"  {tag(rel):66} {a:8.4f} -> {b:8.4f} ns  ({(b/a-1)*100:+6.1f}%)")
    else:
        print(__doc__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
