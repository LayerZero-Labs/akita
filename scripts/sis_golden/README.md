# SIS golden reference cells

Offline regression cells for `scripts/gen_sis_table.py` against the pinned
`third_party/lattice-estimator` checkout.

## Refresh

```bash
git submodule update --init third_party/lattice-estimator
sage -python scripts/gen_sis_table.py --family q32 --d 32 --collision 16384 --format csv > scripts/sis_golden/golden.csv
```

Update `metadata.json` with the submodule SHA from:

```bash
git -C third_party/lattice-estimator rev-parse HEAD
```

## Check

```bash
sage -python scripts/sis_golden/check.py
```

Manual workflow only. Normal Rust CI does not require Sage.
