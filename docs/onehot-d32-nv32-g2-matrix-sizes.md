# SIS matrix sizes — `onehot_d32`, `nv = 32`, groups `[2, 2]`

Per-level dimensions of the three SIS commitment matrices `A`, `B`, `D`
for the `onehot_d32` preset at `nv = 32`, opening 4 one-hot polynomials
at one point as two commitment groups of size 2.

Each entry is a ring element of degree `D = 32`; total field-element count
is `rows × cols × 32`.

Captured with:

```text
AKITA_MODE=onehot_d32 AKITA_NUM_VARS=32 AKITA_NUM_POLYS=4 \
AKITA_GROUP_SIZE=2 AKITA_SCHEDULE_ONLY=1 cargo run --release --example profile
```

## Formulas

For recursive levels, the usual layout formulas apply:

```text
A : rows = n_a, cols = block_len · δcommit
B : rows = n_b, cols = n_a · δopen · num_blocks
D : rows = n_d, cols = δopen · num_blocks
```

At the grouped batched root, `B` and `D` are claim/group aware:

```text
A_root : rows = n_a, cols = block_len · δcommit
B_root : rows = n_b, cols = max_group_size · n_a · δopen · num_blocks
D_root : rows = n_d, cols = num_claims · δopen · num_blocks
```

For `[2, 2]`, `num_claims = 4`, `num_groups = 2`, and
`max_group_size = 2`.

Note: logically the M relation has `n_b · num_groups` B rows, but the
optimized verifier evaluates the shared SIS rows once and combines the two
group-specific T patterns. The B table below reports this optimized shared
SIS rectangle. The logical B cells across both groups at the root are twice
the reported B rectangle.

## Per-level inputs

| Level | n_a | n_b | n_d | num_blocks | block_len | δcommit | δopen | δfold | groups | max group |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 2 | 2 | 1024 | 131 072 | 1 | 64 | 11 | 2 | 2 |
| 1 | 2 | 2 | 2 | 256  | 9 731   | 1 | 64 | 9  | 1 | 1 |
| 2 | 2 | 2 | 2 | 64   | 2 145   | 1 | 43 | 6  | 1 | 1 |
| 3 | 2 | 2 | 2 | 32   | 671     | 1 | 43 | 5  | 1 | 1 |
| 4 | 2 | 2 | 2 | 16   | 490     | 1 | 26 | 4  | 1 | 1 |
| 5 | 2 | 2 | 2 | 8    | 427     | 1 | 26 | 3  | 1 | 1 |
| 6 | 2 | 2 | 2 | 8    | 265     | 1 | 26 | 3  | 1 | 1 |

## Per-level matrix sizes (rows × cols, ring elements)

| Level | A (n_a × block_len·δcommit) | B optimized shared SIS rectangle | D |
|---:|---:|---:|---:|
| 0 | 3 × 131 072 | 2 × 393 216 | 2 × 262 144 |
| 1 | 2 × 9 731   | 2 × 32 768  | 2 × 16 384  |
| 2 | 2 × 2 145   | 2 × 5 504   | 2 × 2 752   |
| 3 | 2 × 671     | 2 × 2 752   | 2 × 1 376   |
| 4 | 2 × 490     | 2 × 832     | 2 × 416     |
| 5 | 2 × 427     | 2 × 416     | 2 × 208     |
| 6 | 2 × 265     | 2 × 416     | 2 × 208     |

At the root, the B width is:

```text
max_group_size · n_a · δopen · num_blocks
= 2 · 3 · 64 · 1024
= 393 216
```

For comparison, one commitment group of 4 claims with the same root split
would have B width `4 · 3 · 64 · 1024 = 786 432`.

## Per-level matrix sizes (total ring elements)

| Level | A | B optimized | D | total |
|---:|---:|---:|---:|---:|
| 0 | 393 216 | 786 432 | 524 288 | 1 703 936 |
| 1 | 19 462  | 65 536  | 32 768  | 117 766   |
| 2 | 4 290   | 11 008  | 5 504   | 20 802    |
| 3 | 1 342   | 5 504   | 2 752   | 9 598     |
| 4 | 980     | 1 664   | 832     | 3 476     |
| 5 | 854     | 832     | 416     | 2 102     |
| 6 | 530     | 832     | 416     | 1 778     |

Total across all 7 levels: **A = 420 674**, **B = 871 808**,
**D = 566 976** ring elements (≈ 1.86 M ring elements, ≈ 59.5 M field
elements at `D = 32`). Level 0 alone accounts for ~91.6 % of the active
shared SIS rectangles.

## Notes

- The root `B` width is cut in half compared with a single group of 4
  because `max_group_size` drops from 4 to 2.
- The root `D` width does not change with grouping; it still depends on
  the total number of claims, which is 4.
- Recursive levels are singleton recursive witness folds, so they have
  `groups = 1` and `max_group = 1`.
- If you count logical B row blocks in M rather than the optimized shared
  SIS rectangle, the root B total is `num_groups · n_b · B_width =
  2 · 2 · 393 216 = 1 572 864` ring elements. The optimized verifier does
  not scan two disjoint SIS row sets for those rows; it reuses the same
  `n_b` shared SIS rows and combines group-specific T patterns.
