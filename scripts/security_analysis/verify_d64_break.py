#!/usr/bin/env python3
"""Cross-validate the D=64 finding with the full SIS.estimate() routine.

The bits we got from SIS.lattice() use the lattice-attack family
specifically (BDGL16 + lgsa). SIS.estimate() runs every attack the
estimator knows about and returns the minimum.

If the full estimate also lands ~90 bits, the D=64 production recursive
schedule is genuinely broken.
"""
import os
import sys
from pathlib import Path

ESTIMATOR_PATH = Path(os.environ.get("LATTICE_ESTIMATOR_PATH",
                                     str(Path.home() / "GitHub" / "lattice-estimator")))
sys.path.insert(0, str(ESTIMATOR_PATH))

from estimator import SIS
from sage.all import log

Q = (1 << 128) - 275


def show(label, D, rank, width, coll):
    n = rank * D
    m = width * D
    length_bound = (m ** 0.5) * coll
    print(f"\n=== {label} ===")
    print(f"  D={D} rank={rank} width={width} coll={coll}")
    print(f"  n={n} m={m} length_bound=sqrt(m)*coll = {length_bound:.0f}")
    print(f"  q = 2^128-275")
    print()
    # full estimate over all SIS attacks
    params = SIS.Parameters(n=n, q=Q, m=m, length_bound=length_bound, norm=2, tag=label)
    print("  SIS.estimate.rough:")
    r = SIS.estimate.rough(params)
    if isinstance(r, dict):
        for tag, info in r.items():
            print(f"    {tag}: rop={float(log(info['rop'], 2)):.1f} bits")
    print()
    print("  SIS.estimate (all attacks):")
    r = SIS.estimate(params)
    if isinstance(r, dict):
        for tag, info in r.items():
            print(f"    {tag}: rop={float(log(info['rop'], 2)):.1f} bits")


# 1. The D=64 production case at the as-shipped rank.
show("D64_TENSOR_RECURSIVE_PRODUCTION", D=64, rank=1, width=234, coll=1080)

# 2. The same with rank bumped to satisfy the SIS table.
show("D64_TENSOR_RECURSIVE_FIXED", D=64, rank=2, width=234, coll=1080)

# 3. Best-case D=64 flat (no tensor) for comparison.
show("D64_FLAT_RECURSIVE", D=64, rank=1, width=234, coll=15)
