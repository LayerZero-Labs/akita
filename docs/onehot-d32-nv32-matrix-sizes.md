# SIS matrix sizes — `onehot_d32`, `nv = 32`

Per-level dimensions of the three SIS commitment matrices `A`, `B`, `D`
for the `onehot_d32` preset at `nv = 32` (8 levels: 1 root + 7
recursive). Each entry is a ring element of degree `D = 32`; total
field-element count is `rows × cols × 32`.

Captured from `AKITA_MODE=onehot_d32 AKITA_NUM_VARS=32 cargo run --release
--example profile`, parsing the `planned fold level` tracing events.

## Formulas

Pulled from `LevelParams::with_decomp` in
`crates/akita-types/src/layout/params.rs`:

```text
A : rows = n_a, cols = block_len · δcommit
B : rows = n_b, cols = n_a · δopen · num_blocks
D : rows = n_d, cols = δopen · num_blocks
```

`δcommit = num_digits_commit`, `δopen = num_digits_open`. The witness
ranks `n_a`, `n_b`, `n_d` are the row counts of the inner/outer/D Ajtai
keys.

## Per-level inputs

| Level | n_a | n_b | n_d | num_blocks | block_len | δcommit | δopen | δfold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 2 | 2 | 2048 | 65 536 | 1 | 64 | 10 |
| 1 | 2 | 2 | 2 | 256  | 4 611  | 1 | 64 | 9  |
| 2 | 2 | 2 | 1 | 64   | 1 425  | 1 | 64 | 8  |
| 3 | 2 | 2 | 2 | 32   | 755    | 1 | 32 | 4  |
| 4 | 2 | 2 | 2 | 16   | 397    | 1 | 32 | 4  |
| 5 | 2 | 2 | 2 | 8    | 423    | 1 | 26 | 3  |
| 6 | 2 | 2 | 2 | 8    | 263    | 1 | 26 | 3  |

## Per-level matrix sizes (rows × cols, ring elements)

| Level | A (n_a × block_len·δcommit) | B (n_b × n_a·δopen·num_blocks) | D (n_d × δopen·num_blocks) |
|---:|---:|---:|---:|
| 0 | 3 × 65 536  | 2 × 393 216 | 2 × 131 072 |
| 1 | 2 × 4 611   | 2 × 32 768  | 2 × 16 384  |
| 2 | 2 × 1 425   | 2 × 8 192   | 1 × 4 096   |
| 3 | 2 × 755     | 2 × 2 048   | 2 × 1 024   |
| 4 | 2 × 397     | 2 × 1 024   | 2 × 512     |
| 5 | 2 × 423     | 2 × 416     | 2 × 208     |
| 6 | 2 × 263     | 2 × 416     | 2 × 208     |

## Per-level matrix sizes (total ring elements)

| Level | A | B | D | total |
|---:|---:|---:|---:|---:|
| 0 | 196 608 | 786 432 | 262 144 | 1 245 184 |
| 1 | 9 222   | 65 536  | 32 768  | 107 526   |
| 2 | 2 850   | 16 384  | 4 096   | 23 330    |
| 3 | 1 510   | 4 096   | 2 048   | 7 654     |
| 4 | 794     | 2 048   | 1 024   | 3 866     |
| 5 | 846     | 832     | 416     | 2 094     |
| 6 | 526     | 832     | 416     | 1 774     |

Total across all 7 levels: **A = 212 356**, **B = 876 160**,
**D = 302 912** ring elements (≈ 1.39 M ring elements, ≈ 44.5 M field
elements at `D = 32`). Level 0 alone accounts for ~89.5 % of the total
— column widths shrink rapidly with each fold.

## Notes

- All three matrices alias the same backing storage in the verifier's
  `setup.shared_matrix`; per-level column counts above describe the
  *active sub-rectangle* used by W (`D`-half), T (`B`-half), and Z
  (`A`-half) during `compute_setup_contribution`.
- For `D = 32` ring elements, multiply `rows × cols` by 32 to get the
  field-element count. The setup envelope chooses
  `max(rows) × max(stride)` across all levels; here that is
  `max_rows = max(n_a, n_b, n_d) = 3`, `max_stride =
  max(block_len·δcommit, n_a·δopen·num_blocks, δopen·num_blocks)
  = 393 216` (level 0's `B` width).
