# Structured E Term

This note defines the verifier's structured `E` contribution to the relation
matrix evaluation. It uses the canonical dyadic chunk partition from
[`dyadic-chunk-partition.md`](dyadic-chunk-partition.md).

## Definitions

For group `g`:

- `K_g` is the number of claims.
- `B_g` is the exact number of live blocks.
- `C` is the witness chunk count.
- `d_A(g)` is the source ring dimension.
- `d_D(g)` is the opening role ring dimension.
- `d_0` is the common coefficient block size used by the verifier point split.
- `q_D = d_A(g) / d_D(g)` is the number of opening role subcolumns in one
  source ring.
- `L_D = d_D(g) / d_0` is the number of common coefficient lanes in one
  opening role ring.
- `delta_open,g` is the number of opening decomposition digits.
- `lambda_cons(g) = eq(tau_1, i_cons(g))` is the row weight for the consistency
  row of group `g`.
- `c_{g,k,b}(alpha)` is the challenge ring for claim `k` and global block `b`,
  evaluated at the source ring powers of `alpha`.
- `G_j^open` is the opening digit gadget weight.
- `x_addr` is the high address part of the stage 2 evaluation point.

The verifier applies the low coefficient factor outside this term:

```math
K_\alpha(x_{\mathrm{coeff}})
=
\sum_{r=0}^{d_0-1}
\alpha^r eq(x_{\mathrm{coeff}}, r).
```

The equations below describe the remaining high address factor.

## Dyadic Chunk Geometry

The layout does not pad live blocks. Chunk `c`, where `0 <= c < C`, owns the
exact half open range

```math
[S_{g,c},S_{g,c+1}),
\qquad
S_{g,c}=\left\lfloor\frac{cB_g}{C}\right\rfloor.
```

Its block count is

```math
N_{g,c}=S_{g,c+1}-S_{g,c}.
```

The local block index satisfies

```math
0\le\beta<N_{g,c},
```

and its global block is

```math
b=S_{g,c}+\beta.
```

Chunk lengths differ by at most one. If `C > B_g`, some ranges are empty.
Those chunks keep their replicated `Z` segment, while their `E` and `T`
segments have length zero.

## E Segment Address

For every group `h` and chunk `c`, define the unit lengths

```math
Z_h
=
P_h\delta_{\mathrm{wit},h}\delta_{\mathrm{fold},h}d_A(h),
```

```math
E_{h,c}
=
K_hN_{h,c}\delta_{\mathrm{open},h}d_A(h),
```

```math
T_{h,c}
=
K_hN_{h,c}n_{A,h}\delta_{\mathrm{commit},h}d_A(h).
```

The physical width of chunk `c` across all groups is

```math
W_c=\sum_h\left(Z_h+E_{h,c}+T_{h,c}\right).
```

The chunks appear in increasing order. Within a chunk, groups appear in the
authenticated relation order, and each group unit has the order
`Z || E || T`. Therefore the start of group `g`'s `E` segment in chunk `c` is

```math
\mathcal E_{g,c}
=
\sum_{r=0}^{c-1}W_r
+
\sum_{h\prec g}\left(Z_h+E_{h,c}+T_{h,c}\right)
+Z_g.
```

Here `h \prec g` means that group `h` appears before group `g` in the relation
order. Every term is a multiple of `d_0`, so define the relation lane start

```math
\overline{\mathcal E}_{g,c}=\frac{\mathcal E_{g,c}}{d_0}.
```

The physical coefficient address of the opening role ring for
`(g,c,k,beta,s,j)` is

```math
addr_E^{phys,g}(c,k,\beta,s,j)
=
\mathcal E_{g,c}
+
\Big(
\big((kN_{g,c}+\beta)q_D+s\big)\delta_{\mathrm{open},g}
+j
\Big)d_D(g).
```

The corresponding relation lane address, including lane `ell`, is

```math
addr_E^{lane,g}(c,k,\beta,s,j,\ell)
=
\overline{\mathcal E}_{g,c}
+
\Big(
\big((kN_{g,c}+\beta)q_D+s\big)\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell,
```

where

```math
0\le s<q_D,
\qquad
0\le j<\delta_{\mathrm{open},g},
\qquad
0\le\ell<L_D.
```

### Single Chunk

For `C = 1`, the only chunk has

```math
S_{g,0}=0,
\qquad
N_{g,0}=B_g.
```

The general address above then reduces to the original single chunk address.

### Equal Width Special Case

If every `B_h` is divisible by `C`, then `N_{h,c}=B_h/C` is independent of
`c`. Every `W_c` is equal, and each group has a fixed offset inside every
chunk. Only in this special case can the segment start be written as

```math
\mathcal E_{g,c}=cW_{\mathrm{chunk}}+O_g.
```

The verifier may use this affine shape as an optimization. It is not a layout
requirement.

## Structured E Equation

For one group `g`, the high address contribution is

```math
S_E^g
=
\sum_{c=0}^{C-1}
\sum_{k=0}^{K_g-1}
\sum_{\beta=0}^{N_{g,c}-1}
\lambda_{\mathrm{cons}(g)}
c_{g,k,S_{g,c}+\beta}(\alpha)
\sum_{s=0}^{q_D-1}
\sum_{j=0}^{\delta_{\mathrm{open},g}-1}
\sum_{\ell=0}^{L_D-1}
G^{open}_j
\alpha^{\ell d_0}
eq\!\left(
x_{\mathrm{addr}},
addr_E^{lane,g}(c,k,\beta,s,j,\ell)
\right).
```

The complete contribution is

```math
S_E=\sum_g S_E^g.
```

Empty chunks contribute no `E` terms because `N_{g,c}=0` makes the local block
sum empty.

## Interpretation

For each live block, the verifier multiplies the source challenge evaluation by
the consistency row weight. It then scans the opening role representation of
`E` over three coordinates:

1. Opening role subcolumns `s` inside the source ring.
2. Opening digits `j`.
3. Common coefficient lanes `ell` inside the opening role ring.

The factor `alpha^(ell d_0)` selects the common coefficient lane. The shared
factor `K_alpha(x_coeff)` supplies the remaining low coefficient powers.

## Compact Verifier Evaluation

The verifier does not build a rectangular `(chunk, local block)` table. Such a
table would need padding because `N_{g,c}` can depend on `c`. Instead, it builds
one exact tensor family for each nonempty group and chunk unit.

For a fixed `(g,c,k)`, ignore the low lane `ell` for a moment. The setup column
address is

```math
A_{g,c,k}(\beta,s,j)
=
\Big((kB_g+S_{g,c}+\beta)q_D+s\Big)
\delta_{\mathrm{open},g}+j.
```

The high relation address is

```math
R_{g,c,k}(\beta,s,j)
=
\overline{\mathcal E}_{g,c}
+
\Big((kN_{g,c}+\beta)q_D+s\Big)
\delta_{\mathrm{open},g}L_D
+jL_D.
```

Both addresses are affine in `beta`, `s`, and `j`. The verifier records them as
one paired equality tensor with these axes:

| Axis | Length | Setup stride | Relation stride |
|---|---:|---:|---:|
| `beta` | `N_{g,c}` | `q_D delta_open,g` | `q_D delta_open,g L_D` |
| `s` | `q_D` | `delta_open,g` | `delta_open,g L_D` |
| `j` | `delta_open,g` | `1` | `L_D` |

The row weights add one more tensor axis. The role projection handles `ell` and
its `alpha^(ell d_0)` weight without expanding the relation equality table.

The paired carry recurrence evaluates each family directly against the setup
point and the relation point. It does not allocate a table whose length is the
setup size or the relation witness size.

### Chunk Compaction

The implementation may combine several chunk families into one tensor only
when all of these facts match:

- Each unit has the same axes, including the same live block count.
- Consecutive setup offsets have one constant stride.
- Consecutive relation offsets have one constant stride.
- All scalar weights match.

If any check fails, the verifier keeps the original unit families. Unequal
dyadic chunks therefore remain exact. They are never extended with zero blocks
to make the fast path apply.

The evaluation trace contraction follows the same rule. It splits the unit
list into affine runs, divides each run into power of two segments, and combines
only segments with equal unit geometry. An irregular unit remains a separate
segment.

This gives the equal width case a smaller constant factor while preserving the
canonical exact prefix for every block count. The chunk count is capped at 64,
so the unmatched unit fallback is bounded by the public layout limit.

## Correctness Conditions

The compact evaluation must preserve these conditions:

- The dyadic ranges cover `[0,B_g)` exactly once.
- The setup challenge uses the global block `S_{g,c}+beta`.
- The relation address uses the local block `beta` inside the physical unit.
- Empty units contribute no `E` terms.
- Chunk compaction runs only after exact addresses and axis shapes match.
- The dense materialization and compact contraction evaluate the same
  polynomial.

The setup contribution tests compare compact tensors with the dense oracle for
uneven dyadic ownership. The evaluation trace tests also cover irregular cases,
including 253 blocks over 64 chunks and 61 blocks over 64 chunks.

## Implementation Owners

- [`akita_types::dyadic_block_ranges`](../crates/akita-types/src/witness/chunk_partition.rs)
  owns chunk boundaries.
- [`WitnessLayout`](../crates/akita-types/src/witness.rs) owns physical unit
  ranges and `Z || E || T` placement.
- [`setup_index_weight.rs`](../crates/akita-types/src/setup_contribution/plan/setup_index_weight.rs)
  builds and conditionally combines setup contribution tensors.
- [`evaluation_trace.rs`](../crates/akita-verifier/src/protocol/evaluation_trace.rs)
  contracts evaluation trace units and handles irregular dyadic chunks.
