# SIS golden reference cells

Offline regression cells for `scripts/gen_sis_table.py` against the pinned
`third_party/lattice-estimator` checkout.

## Setup

```bash
git submodule update --init third_party/lattice-estimator
```

## Refresh golden

Regenerates `golden.csv` and updates `metadata.json` from the grid in `grid.py`:

```bash
sage -python scripts/sis_golden/refresh_golden.py
```

The grid covers q32/q64/q128, `d ∈ {32,64,128,256}`, ranks `{1,5,20}`, and
includes the degenerate knee at `(q32, d=32, collision_l2_sq=16384)`.

## Check

Replay golden cells, rank monotonicity, and secure/insecure brackets:

```bash
sage -python scripts/sis_golden/check.py
```

## Full table regen (smoke / production)

Per family, shard over `(d, collision)` work items:

```bash
sage -python scripts/gen_sis_table.py --family q32 --jobs 6
sage -python scripts/gen_sis_table.py --family q64 --jobs 6
sage -python scripts/gen_sis_table.py --family q128 --jobs 6
```

Stitch rust output into `crates/akita-types/src/sis/generated_sis_table/` in the
table cutover PR (separate from golden refresh).

Manual workflow only. Rust CI does not require Sage or an initialized submodule.
