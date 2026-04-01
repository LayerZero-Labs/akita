# Block Order

Hachi currently uses two different block-order conventions, and the split is intentional.

## Rule

- Level 0 uses `BlockOrder::RowMajor`.
- Recursive `w` levels use `BlockOrder::ColumnMajor`.

## Meaning

`RowMajor` splits the outer opening coordinates as:

- first `m_vars` coordinates -> in-block weights `a`
- remaining `r_vars` coordinates -> block weights `b`

This matches the root polynomial layout where block `i` occupies ring elements
`[i * block_len, (i + 1) * block_len)`.

`ColumnMajor` splits the outer opening coordinates as:

- first `r_vars` coordinates -> block weights `b`
- remaining `m_vars` coordinates -> in-block weights `a`

This matches the recursive witness layout where the logical sequential index is
`index = position * 2^r + block`.

## Why The Split Exists

- The level-0 polynomial backends (`DensePoly`, `OneHotPoly`) still operate on
  contiguous root blocks.
- The recursive witness path stores `w` in a strided column-major form, and the
  recursive commitment/opening flow is built around that layout.

## Practical Guidance

- Do not thread raw booleans for this behavior. Use `BlockOrder`.
- Preserve `RowMajor` for level 0 unless you are intentionally doing a full
  semantic cutover of the root polynomial layout.
- Preserve `ColumnMajor` for recursive `w` folding, commitment, and verifier
  replay.

## Main Code Paths

- Opening split: `src/protocol/opening_point.rs`
- Root prove/verify wiring: `src/protocol/commitment_scheme.rs`
- Root block semantics: `src/protocol/hachi_poly_ops/dense.rs`,
  `src/protocol/hachi_poly_ops/onehot.rs`
- Recursive witness semantics: `src/protocol/hachi_poly_ops/recursive_witness.rs`
