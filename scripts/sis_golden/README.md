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

## Full table regen

Regenerate and stitch every SIS table row with the pinned
`third_party/lattice-estimator` checkout:

```bash
sage -python scripts/stitch_generated_sis_table.py --jobs 6
```

The stitcher uses `--max-rank 20`, passes `--estimator-path
third_party/lattice-estimator` to every shard, and rejects any estimator checkout
whose `HEAD` does not match `metadata.json`.

Manual workflow only. Rust CI does not require Sage or an initialized submodule.
