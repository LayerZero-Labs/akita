# Spec: binary source dense root lanes

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-11-04 |
| Status | implemented |
| PR | |
| Supersedes | None |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

> **Implemented: interleaved lane runs** (`ce98d817c`). Section 1 measures the
> layout before the change, with code citations at commit `02bd154b1`.
> Sections 2-6 compare the candidate layouts and choose one. Section 7 is the
> implementation plan, and Section 8 measures the result.

## Summary

The binary root writes its outgoing digit table with every lane padded to a
power-of-two stride: 648 natural coefficients in 1,024 cells for ring lanes and
162 in 256 for scalar lanes. Each padded lane wastes 376 of 1,024 or 94 of 256 cells, 36.7%. Together
with the tail, the domain is 1.6 to 2.9 times the natural cell count. Stage 2
and the dense weight table cost 9 to 10 ns per domain cell
(Section 1.3), about a fifth of proving time.

This note compares stride 648, stride 768 and three ways to pack several lanes
into power-of-two blocks. The chosen layout, **interleaved lane runs**, stores
`W` lanes side by side, coefficient by coefficient, where `W` is the largest
power of two dividing the segment's lane count. The lane index then occupies
the low `log2 W` address bits, and the equality weight of every address factors
into a lane factor and a coefficient factor (Section 2). This keeps the
verifier's closed forms and the rank-1 Stage 3 tensor. The table holds no lane
padding. Its only padding is a 130-cell alignment gap before the prime planes or a
130-cell rounding of the end to a multiple of 256, and the tail.

The domain halves in five of the eight measured shapes, and in every shape it
reaches the smallest power of two that holds the natural cells (Section 3.5).
No layout can do better. The verifier's family evaluation gains six adjoints and
one 162-coefficient kernel per run of prime lanes, a few milliseconds
(Section 4). The planner re-prices the suffix from the smaller
next-level witness. At 2^22 `word-prime` it trades 6.5 KB of proof for a
suffix whose level-2 witness is half as large (Section 5).

## Notation

| Symbol | Meaning |
| --- | --- |
| `n` | Number of Stage 2 variables. The table domain is `{0,1}^n`. |
| `r` | The Stage 2 point in `F^n`. Coordinate `i` binds address bit `i`, low bit first. |
| `eq(u, a)` | `prod_i (u_i a_i + (1-u_i)(1-a_i))`. |
| `r[i..j]` | Coordinates `i` to `j-1` of `r`. |
| Segment | One [`RootRepackingSegment`](../crates/akita-types/src/root_repacking.rs): `E` elements, `d` digits, natural lane width `m` (648 for ring lanes, 162 for scalar lanes). |
| `L` | Lanes of a segment, `E * d`. Lane `l = e*d + j` is digit `j` of element `e`. |
| `W` | The segment's run width, `2^w`, the largest power of two dividing `L`. It is 1 for a segment without lane padding. |
| `s` | The segment start, a multiple of `W`. |
| Natural cells | `sum L*m` over segments. Every layout must give each of them a distinct address. |
| Physical cells | The committed live length: the last segment's end rounded up to a multiple of 256. |

## 1. Measured split before the change

### 1.1 The current layout

The binary root writes one outgoing digit table. Each digit of each element is
a *lane*. Each lane has `m` natural coefficients and is stored at a fixed
physical stride `S`. In
`crates/akita-types/src/root_repacking.rs:152`, coefficient `c` of lane `l` in
a segment that starts at `start` has address `start + l * S + c`. The ring
segments use `m = 648` and `S = 1024`
(`crates/akita-types/src/binary_source/root_plan.rs:59-61`). The scalar
segments use `m = 162` and `S = 256`
(`binary_scalar_geometry`,
`crates/akita-types/src/binary_source/evaluation_weights.rs:332`). The table
domain is the next power of two above the last segment's end
(`root_repacking.rs:74`).

The segments, in order, are:

| Segment | Elements | Digits | m in S | Lanes at N=2^22 | Lanes at N=2^23 |
| --- | --- | ---: | --- | ---: | ---: |
| Z, response | owners · ring rows | 4 | 648 in 1,024 | 16,384 | 32,768 |
| T, image | owners · columns · rank | 8 | 648 in 1,024 | 6,144 | 6,144 |
| U, partials (word shapes) | owners · columns | 1 | 162 in 256 | 256 | 256 |
| K, parity carry (word shapes) | owners | 9 | 162 in 256 | 9 | 9 |
| K_val, value carry (word shapes) | owners | 6 | 162 in 256 | 6 | 6 |
| EP, prime planes | planes · columns | 43 | 162 in 256 | 11,008 per plane | 11,008 per plane |
| ordinary segments (`word-prime-ordinary`) | from the ordinary profile | | no padding | 4,968,448 cells | 9,129,984 cells |

The full-size fixtures use one owner, 256 columns and A rank 3. `word` has Z, T,
U, K and K_val. `word-prime` adds one plane. `word-prime-ordinary` adds one
plane and the ordinary group's segments. `prime5` has Z, T and five planes,
with no U, K or K_val.

Two kinds of padding follow. Each ring lane wastes 376 of its 1,024 cells and
each scalar lane wastes 94 of its 256. This is *lane padding*. The rest of the
power-of-two domain after the last segment is the *tail*.

### 1.2 Domains and padding fractions

The natural cells are the sum of `elements * digits * m` over the segments.
The physical cells are the last segment's end. The table below takes the
physical cells and domain from each baseline run's `native_transition
physical_cells=` and `selected_witness scheduled_domain=` lines. The padding
fraction is `(domain - natural) / domain`.

| N | Shape | Natural cells | Physical cells | Lane padding | Tail | Domain | Padding |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2^22 | word | 14,642,046 | 23,138,048 | 8,496,002 | 10,416,384 | 2^25 | 56.36% |
| 2^22 | word-prime | 16,425,342 | 25,956,096 | 9,530,754 | 7,598,336 | 2^25 | 51.05% |
| 2^22 | word-prime-ordinary | 21,393,790 | 30,924,544 | 9,530,754 | 2,629,888 | 2^25 | 36.24% |
| 2^22 | prime5 | 23,514,624 | 37,158,912 | 13,644,288 | 29,949,952 | 2^26 | 64.96% |
| 2^23 | word | 25,258,878 | 39,915,264 | 14,656,386 | 27,193,600 | 2^26 | 62.36% |
| 2^23 | word-prime | 27,042,174 | 42,733,312 | 15,691,138 | 24,375,552 | 2^26 | 59.70% |
| 2^23 | word-prime-ordinary | 36,172,158 | 51,863,296 | 15,691,138 | 15,245,568 | 2^26 | 46.10% |
| 2^23 | prime5 | 34,131,456 | 53,936,128 | 19,804,672 | 13,172,736 | 2^26 | 49.14% |

As a check, the 2^22 `word-prime` row has `2^25 - 16,425,342 = 17,129,090`
padding cells. This is the padding count in
[binary-source-zero-padding.md](binary-source-zero-padding.md).

### 1.3 What scales with the domain

The prover figures are medians of three runs of `06b8435eb`, the same code as
`02bd154b1`. They come from the interleaved runs behind the opening spec's
prover span profile. The logs are
`$CARGO_TARGET_DIR/r3/m-fin-<log>-<shape>-<run>.txt`, and one-minute loads
stayed between 4.24 and 10.74. Times are in milliseconds. *Table* is the
weight-materialization span minus the five `root_weights_visit_*` family spans.
It is the part of materialization that allocates and walks the dense table.
*Families* is the sum of those five spans. *After root* is complete proving
minus the binary root level span. It holds the ordinary suffix folds and the
work outside the root level.

| N | Shape | Domain | Stage 2 | Stage 2 ns/cell | Table | Families | Stage 1 | After root | Complete |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 2^25 | 327.7 | 9.77 | 30.7 | 117.6 | 18.9 | 282.4 | 1,628.7 |
| 2^22 | word-prime | 2^25 | 323.7 | 9.65 | 29.7 | 197.9 | 15.8 | 258.5 | 1,737.8 |
| 2^22 | word-prime-ordinary | 2^25 | 326.4 | 9.73 | 29.3 | 187.3 | 17.0 | 255.4 | 1,837.1 |
| 2^22 | prime5 | 2^26 | 591.5 | 8.81 | 59.3 | 525.6 | 20.5 | 258.3 | 2,534.3 |
| 2^23 | word | 2^26 | 627.9 | 9.36 | 62.1 | 178.7 | 26.1 | 428.1 | 2,901.9 |
| 2^23 | word-prime | 2^26 | 612.3 | 9.12 | 59.5 | 313.0 | 22.4 | 378.7 | 2,927.9 |
| 2^23 | word-prime-ordinary | 2^26 | 623.5 | 9.29 | 60.0 | 318.4 | 28.1 | 392.3 | 3,369.4 |
| 2^23 | prime5 | 2^26 | 634.3 | 9.45 | 59.5 | 867.5 | 27.0 | 386.6 | 3,726.5 |

The *Families* column sums per-span medians, so it can differ from the median
of per-run sums by a few milliseconds.

- **Stage 2 is linear in the domain.** It costs 8.8 to 9.8 ns per domain cell
  in every configuration, independent of the shape. Doubling the domain from
  2^25 to 2^26 takes 2^22 `prime5` from about 325 ms to 591 ms.
- **Materialization's table part is linear in the domain.** It is about 30 ms
  at 2^25 and about 60 ms at 2^26.
- **The family visits do not depend on the domain.** They scale with the
  natural terms. At the same 2^26 domain, 2^22 `prime5` spends 525.6 ms and
  2^23 `word` spends 178.7 ms. Five planes dominate the former.
- **Stage 1 is small.** It costs 16 to 28 ms.
- **The work after the root level does not follow the domain.** 2^22 `prime5`
  (2^26 domain, 37.2M physical cells) and 2^22 `word-prime` (2^25, 26.0M)
  spend the same 258 ms. The suffix schedule commits the root's physical cells
  as its level-1 witness (`root.output_witness_len`,
  `crates/akita-planner/src/native_root.rs:86`), but at these sizes its time
  is set by the fixed fold structure, not the cell count.

So the domain-proportional prover work is Stage 2 plus the table part of
materialization. At a 2^25 domain this is about 353 ms of 1,738 ms at 2^22
`word-prime`, or 20%. At a 2^26 domain it is about 672 ms of 2,928 ms at 2^23
`word-prime`, or 23%. Halving the domain saves about half of it.

### 1.4 The verifier's family evaluation

The verifier never builds the table. `RootEq` keeps `2^10` low and
`2^(n-10)` high equality weights
(`crates/akita-types/src/native_root_relation/structured.rs:21-23`). This is
32,768 high entries at a 2^25 domain. Each family's closed form reads them per
lane or per run of lanes. The figures below are single baseline verifications
of the `02bd154b1` executable under one-minute loads between 22 and 84. The
logs are `$CARGO_TARGET_DIR/r5/v-base-<log>-<shape>-1-1.txt`. They are
milliseconds, and they bound the order of magnitude only.

| N | Shape | Image | Parity rows | Prime | Authentication | A Stage 3 | Complete verification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 3.51 | 0.58 | absent | 258.79 | 5.21 | 376.61 |
| 2^22 | word-prime | 1.87 | 0.57 | 0.48 | 394.72 | 1.74 | 436.86 |
| 2^22 | word-prime-ordinary | 1.85 | 0.55 | 0.49 | 415.98 | 4.56 | 783.71 |
| 2^22 | prime5 | 1.86 | absent | 2.40 | 961.43 | 1.77 | 1,088.62 |
| 2^23 | word | 1.88 | 1.65 | absent | 259.54 | 1.80 | 305.78 |
| 2^23 | word-prime | 5.93 | 2.11 | 0.69 | 442.88 | 5.26 | 497.15 |
| 2^23 | word-prime-ordinary | 1.86 | 1.07 | 0.68 | 451.39 | 4.43 | 1,133.11 |
| 2^23 | prime5 | 1.89 | absent | 3.35 | 971.45 | 3.73 | 1,070.96 |

The image, parity and prime closed forms cost a few milliseconds and depend on
the lane counts, not on the domain. Authentication evaluates one equality
weight per opened digit. It scales with the opened U and EP digits, not with
the domain, and it is outside this note's scope. The domain enters the
verifier only through the `RootEq` high table and the number of Stage 1 and
Stage 2 rounds, one round per doubling.

## 2. The factorization lemma

Every structured weight family relies on one property of the layout. Within a
segment, the equality weight of a lane's coefficient must split into a factor
that depends only on the lane and a factor that is shared by many lanes. The
stride layout has it because the stride is a power of two. The chosen layout
keeps it with the lane index in the low bits instead.

**Layout.** In a segment with start `s`, `L` lanes, natural width `m` and run
width `W = 2^w`, where `W` divides `L` and `s`, coefficient `c` of lane `l`
lives at

```text
a(l, c) = s + floor(l / W) * W * m + c * W + (l mod W).
```

**Lemma 1 (bijection).** `a` maps `[0, L) x [0, m)` one-to-one onto
`[s, s + L*m)`.

*Proof.* Write `q = floor(l / W)` and `j = l mod W`. Then
`a - s = W * (q*m + c) + j` with `0 <= j < W`, `0 <= c < m` and
`0 <= q < L/W`. This is a mixed-radix numeral with digits `j`, `c` and `q`
and radices `W`, `m` and `L/W`, so it is one-to-one onto `[0, L*m)`. ∎

Consecutive segments therefore occupy disjoint ranges, because each starts at
or after the previous end. The table holds no lane padding.

**Lemma 2 (factorization).** For every `r` in `F^n`,

```text
eq(r, a(l, c)) = eq(r[0..w], l mod W) * eq(r[w..n], s/W + floor(l/W) * m + c).
```

*Proof.* Since `W` divides `s`, `a = W * (s/W + q*m + c) + j` with
`j < W = 2^w`. The low `w` bits of `a` are the bits of `j`, and the remaining
bits are those of `s/W + q*m + c`. The equality polynomial is a product over
bit positions. ∎

The first factor is the *lane factor*. It has `W` values, one per position in
the run. The second is the *coefficient vector* `z_q` of run `q`, of length
`m`, shared by every lane of that run. A family that applies one linear map per
lane, such as the image adjoint or a shifted-remainder kernel, applies it once
per run to `z_q` and contracts with the lane factors afterwards.

The stride layout before the change is the special case in which the lane
sits in the high bits. With stride `S = 2^t`, `eq(r, s + l*S + c)` equals
`eq(r[0..t], c) * eq(r[t..n], s/S + l)`. One coefficient vector serves the whole
segment, and the price is `S - m` padding cells per lane.

**Run width.** `W` is the largest power of two dividing `L`. A larger `W` would
leave a partial last run, which needs padding. A smaller one gives more runs
and more per-run work. A segment without lane padding, such as an ordinary
group's segment with `S = m`, keeps `W = 1`, which is the previous contiguous
layout. The segment start is `s = align_up(previous end, W)`.

## 3. Candidate layouts

### 3.1 What a layout does not change

A layout is the public map from (segment, element, digit, coefficient) to an
address. The prover and the verifier both compute it from public parameters
through `RootCoefficientRepacking::coefficient_address` and its structured
counterpart. Changing it moves where each weight lands. It does not change any
equation. The four parts the brief asks about are affected as follows.

- **The relation.** The root's rows are the same linear identities over the
  same natural coefficients. The deletion rule of
  [binary-source-zero-padding.md](binary-source-zero-padding.md) needs only
  that no row reads a padding address. Every family reads addresses through the
  layout map, so the rule holds for any layout. The regression
  `compiled_root_weights_vanish_on_every_padding_address`
  (`crates/akita-types/src/binary_source/reference_tests.rs`) checks it on
  nineteen shapes.
- **The range check.** Stage 1 and the Stage 2 range-image term read every
  address of `{0,1}^n` and require only a digit in `[-4, 3]` (the consumers
  table of the zero-padding note). Their cost and round count follow `n`. A
  layout changes them only through the domain.
- **The norm argument.** The successor bounds the physical norm of the table it
  commits. The planner's worst-case source energy is 16 per physical cell
  (`crates/akita-planner/src/native_root.rs:56-61`). A denser layout lowers the
  energy in proportion to the physical cells. The bound remains worst-case and
  does not assume a digit distribution.
- **The next level's witness.** The suffix's level-1 witness is the physical
  cell count (`native_root.rs:53` and `:86`). It shrinks with the layout, and
  the planner re-prices every suffix fold from it (Section 5).

The options therefore differ only in the domain, in the verifier's closed
forms, and in whether the typed Stage 3 product keeps its rank-1 form.

### 3.2 Stride 648

Lanes are contiguous at their natural width: `a = s + l*m + c`. This reaches
the natural cell count. However, `m` is not a power of two, so the bits of
`l*m + c` mix lane and coefficient, and Lemma 2 has no analogue. Every lane
needs its own coefficient vector of `m` equality values.

- The image's Hankel step, `<adj(rho, c), z> = <c, adj(rho, z)>`, needs one
  adjoint per lane instead of one per rank row: 6,144 adjoints of 154,872
  multiplications each, or about 9.5 * 10^8.
- The parity rows and the prime response rows need the equality weight of
  every natural Z address that they read, up to 10,616,832 at 2^22.
- The typed Stage 3 product loses its rank-1 form. The prover's rectangular
  kernel takes a row factor and a coefficient factor
  (`BinaryAProduct::factors`, `crates/akita-types/src/binary_source/a_product.rs`).
  The verifier's terminal weight costs `O(rank * 1024 + log prefix_len)`
  (`BinaryAProduct::evaluate`). With `eq(r, l*648 + c)`, the weight on the
  setup prefix no longer separates into a function of the source row and a
  function of the coefficient. Both sides would need a new kernel.

Verification would become linear in the response. This option is rejected.

### 3.3 Stride 768

Ring lanes use `S = 768 = 3 * 256`. Scalar lanes keep `S = 256`. The variant
with `S = 192` changes no domain in the table below. Since
`s + 768 l + c = 256 * (s/256 + 3l + floor(c/256)) + (c mod 256)`, the low
eight bits hold `c mod 256`. The high part `3l + floor(c/256)` mixes the lane
into the coefficient's block. By linearity of the adjoint, the image needs
three adjoints per rank row, one per block of 256. However, `eq(r_high, 3l + k)`
is not a product over the bits of `l`, because multiplying by three carries.
The Stage 3 row factor therefore loses its tensor form, as in Section 3.2.
This option halves the domain in three shapes only.

### 3.4 Several lanes per power-of-two block

`k` lanes share one block of `2^b` cells, each lane contiguous within it:
`a = s + floor(l/k) * 2^b + (l mod k) * m + c`. The low `b` bits hold
`(l mod k) * m + c`, so the block's `k` slots need `k` distinct coefficient
vectors. These are shared by all blocks. The high factor is `eq` at the block
index.

- **Three ring lanes per 2,048 and six scalar lanes per 1,024.** The image
  needs `3 * rank = 9` adjoints and the prime planes need six kernels. The
  digit and source-row bits of the response lane `l = 4 * row + digit` are cut
  by `floor(l/3)`, so the Stage 3 product loses its rank-1 form. The domain
  halves in four shapes and not at 2^22 `word-prime`, where 17,306,624 cells
  exceed 2^24.
- **Twenty-five ring lanes and 101 scalar lanes per 16,384.** The domains
  equal those of Section 3.5. The verifier needs 75 adjoints for the image
  (about 1.2 * 10^7 multiplications) and 101 prime kernels, and Stage 3
  loses its rank-1 form.
- **A power-of-two `k`.** This keeps Stage 3 separable, but it packs no better
  than the stride layout. Two ring lanes fill 1,296 of 2,048 cells, the same
  63.3% as 648 of 1,024.

### 3.5 Interleaved lane runs (chosen)

The layout of Section 2. At the measured shapes, the runs are:

| Segment | Lanes at 2^22 | `W` | Runs | Lanes at 2^23 | `W` | Runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Z, response | 16,384 | 16,384 | 1 | 32,768 | 32,768 | 1 |
| T, image | 6,144 | 2,048 | 3 | 6,144 | 2,048 | 3 |
| U, partials | 256 | 256 | 1 | 256 | 256 | 1 |
| K, parity carry | 9 | 1 | 9 | 9 | 1 | 9 |
| K_val, value carry | 6 | 2 | 3 | 6 | 2 | 3 |
| EP, one plane | 11,008 | 256 | 43 | 11,008 | 256 | 43 |
| EP, five planes | 55,040 | 256 | 215 | 55,040 | 256 | 215 |
| Ordinary segments | | 1 | | | 1 | |

Only EP needs an alignment gap. In the word shapes, K_val ends at 14,642,046
(2^22) or 25,258,878 (2^23), and EP starts at the next multiple of 256,
130 cells later. In `prime5`, T ends at a multiple of 256. The last segment's
end is rounded up to a multiple of 256 (`LIVE_ALIGNMENT`). The successor's
relation reads the live witness in coefficient blocks of its smallest role
dimension, which must divide the live length
(`common_relation_coeff_count`, `crates/akita-params/src/layout/relation_address.rs:82-97`).
The stride layout always ended on a multiple of the 256-cell scalar stride.
In `word`, the dense layout ends at 14,642,046 (2^22) or 25,258,878 (2^23),
which are not multiples of 64. A first version rounded to the destination ring
dimension. That made the live length depend on the successor the planner
selects, which the full-size catalog rejects, so the rounding is the fixed
256 that the stride layout guaranteed. It adds 130 cells in `word`. In the
other shapes, the table already ends on a multiple of 256. Interleaved runs therefore have exactly the physical cells of stride 648.

Each family's closed form keeps its structure (Section 4). Stage 3 keeps its
rank-1 form: the response's lowest point coordinates address the digit, then
the source row, then the owner within its run. The coefficient factor is the
sum over runs of the owner-combined adjoint applied to `z_q`.

### 3.6 Domains

Physical cells and domains per option. *Before* is the stride layout. Every
column rounds the end up to a multiple of 256, as Section 3.5 does.

| N | Shape | Natural | Before | Stride 648 | Stride 768 | 3 per 2,048 | 25 per 16,384 | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 14,642,046 | 23,138,048 (2^25) | 14,642,176 (2^24) | 17,370,880 (2^25) | 15,427,584 (2^24) | 14,860,288 (2^24) | 14,642,176 (2^24) |
| 2^22 | word-prime | 16,425,342 | 25,956,096 (2^25) | 16,425,472 (2^24) | 20,188,928 (2^25) | 17,306,624 (2^25) | 16,646,144 (2^24) | 16,425,472 (2^24) |
| 2^22 | word-prime-ordinary | 21,393,790 | 30,924,544 (2^25) | 21,393,920 (2^25) | 25,157,376 (2^25) | 22,275,072 (2^25) | 21,614,592 (2^25) | 21,393,920 (2^25) |
| 2^22 | prime5 | 23,514,624 | 37,158,912 (2^26) | 23,514,624 (2^25) | 31,391,744 (2^25) | 24,774,656 (2^25) | 23,707,648 (2^25) | 23,514,624 (2^25) |
| 2^23 | word | 25,258,878 | 39,915,264 (2^26) | 25,259,008 (2^25) | 29,953,792 (2^25) | 26,611,712 (2^25) | 25,591,808 (2^25) | 25,259,008 (2^25) |
| 2^23 | word-prime | 27,042,174 | 42,733,312 (2^26) | 27,042,304 (2^25) | 32,771,840 (2^25) | 28,490,752 (2^25) | 27,377,664 (2^25) | 27,042,304 (2^25) |
| 2^23 | word-prime-ordinary | 36,172,158 | 51,863,296 (2^26) | 36,172,288 (2^26) | 41,901,824 (2^26) | 37,620,736 (2^26) | 36,507,648 (2^26) | 36,172,288 (2^26) |
| 2^23 | prime5 | 34,131,456 | 53,936,128 (2^26) | 34,131,456 (2^26) | 43,974,656 (2^26) | 35,958,784 (2^26) | 34,439,168 (2^26) | 34,131,456 (2^26) |

*3 per 2,048* also places six scalar lanes per 1,024. *25 per 16,384* also
places 101 scalar lanes per 16,384. Packed blocks are aligned to the block
size. The *Before* column reproduces Section 1.2.

Interleaved runs halve the domain in five shapes: 2^22 `word`, `word-prime` and
`prime5`, and 2^23 `word` and `word-prime`. In every shape the domain is the
next power of two above the natural cell count. Every layout must hold the
natural cells at distinct addresses, so no layout has a smaller domain. The
other three shapes already have more natural cells than half their domain.

## 4. The verifier's closed forms

The verifier evaluates each family's multilinear extension at the Stage 2 point
`r` without building the table
(`crates/akita-types/src/native_root_relation/structured.rs`). Under the
chosen layout, every family reads a segment through `SegmentEq`, which holds
Lemma 2's two factors. It has `W` lane factors and one coefficient vector per
run. Building it costs about `W + 1024 + runs * m` multiplications: the lane
table, one low equality table of `2^10` entries, and the vectors. `RootEq`
keeps the split table `eq(r, a) = low[a mod 2^10] * high[a >> 10]` for
families that read single addresses. Its high table has `2^(n-10)` entries,
so it halves with the domain.

Below, `rho` is one row of the A matrix and `adj(rho, z)` is the trinomial
adjoint, 154,872 multiplications for degree 648 (opening spec, "Typed A setup
product"). `SRE(z)` is `shifted_remainder_evaluations` on a 162-coefficient
window. It costs about 500 multiplications: a 162-step Horner evaluation and a
two-multiplication recurrence step per output. Counts are for one owner,
256 columns and rank three, at 2^22. The 2^23 counts are the same except for
the Z lane table, which doubles.

| Family | Closed form after the change | Before | After |
| --- | --- | --- | --- |
| Image (T) | `-sum_rho sum_col sum_q (sum_j 8^j lane_j) <c_col, adj(rho, z_q)>`, one adjoint per (rank row, run) touched | 3 adjoints, 464,616 | 9 adjoints, 1,393,848 |
| Parity rows (Z, digit 0) | Four Russians subset-sum tables over `SRE` of the four signed residue classes of `z_q`, built once per Z run, then one lane factor per row quad | tables once | tables once (one Z run) |
| Prime, response part | signed sums `sum_c (-alpha)^c z_q[4c+k]` per Z run | once | once (one Z run) |
| Prime, plane part | `SRE(z_q[..162])` per EP run, read through each column's sparse support | 1 kernel | 43 kernels (one plane), 215 (five planes) |
| Prime, reading part | `<omega, z_q>` per plane and EP run | 162 per plane | 6,966 (one plane), 174,150 (five planes) |
| Authentication (U, EP digits) | one `eq(r, a)` per opened digit address, then the residue kernel per column | one product per address | unchanged |
| Typed A, Stage 3 | owner rows combined per Z run, one adjoint per (rank row, run) | 3 adjoints | 3 adjoints (one Z run) |
| Segment tables | `SegmentEq` for Z, T and EP | none | about 67,000 (one plane) |
| Split table | `RootEq` high part | 2^(n-10) | halved where `n` drops |

The added work is about 1.0 * 10^6 multiplications at 2^22 `word-prime` and
1.3 * 10^6 at 2^22 `prime5`. The baseline image evaluation spends 1.87 ms on
three adjoints, about 4 ns per multiplication (Section 1.4), so the change adds
roughly 4 to 5 ms to verification. A shape whose domain halves loses one
Stage 1 and one Stage 2 round.

The image's six extra adjoints come from the T element order. Element
`(col * rank + row)` spreads each rank row across all three runs. A row-major
T order would put each rank row in one run and restore three adjoints. That is
a change to the response construction, and it is left open (Section 7).

## 5. Planner, SIS and schedule re-pricing

### 5.1 What the planner reads

The planner reads the layout in two places.

- `price_binary_root_planes` (`crates/akita-planner/src/prime_planes.rs`)
  reports the outgoing physical cells and the minimal domain. The full-size
  catalog checks them against the canonical plan
  (`crates/akita-pcs/examples/binary_source_full_size/catalog.rs`).
- `find_native_root_schedule` (`crates/akita-planner/src/native_root.rs`) makes
  the physical cells the level-1 witness, sets the root's
  `output_witness_len` to it, and charges a worst-case energy of 16 per cell.
  It then searches the ordinary suffix with the unchanged objective of
  `crates/akita-planner/src/schedule_params/objective.rs`. That objective
  orders candidates by setup envelope, first direct setup, exact work score and
  then proof bytes.

The D-table pricing does not change. `d_live_digits`, `d_columns` and
`d_rank` count opened digits, not addresses. The planner reports the same
values as before, for example 28,512 columns at rank 2 for one plane. Every
suffix fold is selected again, with its SIS route certified from the existing
generated tables. No SIS table and no ordinary catalog row changes.

### 5.2 Re-priced schedules

The table compares the current-protocol schedules selected before the change
(the Section 1 runs) with those selected after it. *Level-1 witness* is the
root's physical cells. *Level-2 witness* is the first suffix fold's output.

| N | Shape | Level-1 witness | Domain | Level-2 witness | Suffix parameters |
| --- | --- | --- | --- | --- | --- |
| 2^22 | word | 23,138,048 → 14,642,176 | 2^25 → 2^24 | 21,206,528 → 10,526,208 | basis 2^3 → 2^6 from level 1 |
| 2^22 | word-prime | 25,956,096 → 16,425,472 | 2^25 → 2^24 | 21,294,592 → 10,571,264 | basis 2^3 → 2^6 from level 1 |
| 2^22 | word-prime-ordinary | 30,924,544 → 21,393,920 | 2^25 | 21,514,752 → 19,901,952 | level-3 inner rank 6 → 5 |
| 2^22 | prime5 | 37,158,912 → 23,514,624 | 2^26 → 2^25 | 21,778,944 → 21,206,528 | unchanged |
| 2^23 | word | 39,915,264 → 25,259,008 | 2^26 → 2^25 | 17,777,152 → 17,250,816 | unchanged |
| 2^23 | word-prime | 42,733,312 → 27,042,304 | 2^26 → 2^25 | 17,869,312 → 17,306,112 | unchanged |
| 2^23 | word-prime-ordinary | 51,863,296 → 36,172,288 | 2^26 | 18,182,656 → 17,629,696 | unchanged |
| 2^23 | prime5 | 53,936,128 → 34,131,456 | 2^26 | 18,256,384 → 17,574,400 | unchanged |

Where the fold parameters are unchanged, the level-2 witness shrinks by 2.6%
to 3.7%, much less than the level-1 witness. At 2^22 `word` and
`word-prime`, the smaller level-1 witness lets the planner move the suffix
from digit basis 2^3 to 2^6. Every fold from level 1 on then uses basis 2^6
with 22 outer digits at rank 2. Before the change, levels 1 to 3 used 43
outer digits at rank 1 and level 4 used basis 2^5. The inner ranks of levels
2 to 5 change from 6, 6, 4, 4 to 8, 5, 5, 5, and the level-2 witness halves.
The objective ranks proof bytes after the setup envelope, the first direct
setup and the exact work score, so the planner can accept more bytes for a
gain on an earlier coordinate. This note does not identify which coordinate
decided the switch, and it does not change the objective. The dev protocol
makes the same switch in the same two shapes.

### 5.3 Proof size

Proof bytes from one run per row of
`scripts/binary-source-full-size-digests.tsv`, before and after the change.

| Protocol | N | Shape | Before | After | Change |
| --- | --- | --- | ---: | ---: | ---: |
| current | 2^10 | word-prime-ordinary | 87,337 | 87,363 | +26 |
| current | 2^22 | word | 87,804 | 94,319 | +6,515 |
| current | 2^22 | word-prime | 87,805 | 94,334 | +6,529 |
| current | 2^22 | word-prime-ordinary | 87,786 | 87,799 | +13 |
| current | 2^22 | prime5 | 84,879 | 84,807 | -72 |
| current | 2^23 | word | 89,641 | 89,520 | -121 |
| current | 2^23 | word-prime | 89,603 | 89,524 | -79 |
| current | 2^23 | word-prime-ordinary | 89,638 | 89,591 | -47 |
| current | 2^23 | prime5 | 86,595 | 86,600 | +5 |
| dev | 2^10 | word-prime-ordinary | 82,308 | 82,351 | +43 |
| dev | 2^22 | word | 82,739 | 89,249 | +6,510 |
| dev | 2^22 | word-prime | 82,788 | 89,269 | +6,481 |
| dev | 2^22 | word-prime-ordinary | 82,747 | 82,754 | +7 |
| dev | 2^22 | prime5 | 79,860 | 79,776 | -84 |
| dev | 2^23 | word | 84,544 | 84,432 | -112 |
| dev | 2^23 | word-prime | 84,554 | 84,457 | -97 |
| dev | 2^23 | word-prime-ordinary | 84,598 | 84,536 | -62 |
| dev | 2^23 | prime5 | 81,534 | 81,590 | +56 |

- **One round fewer.** Where the domain halves, level 0 loses one Stage 1 and
  one Stage 2 round, which is 114 bytes in the current protocol (9,093 → 8,979
  at 2^22 `word-prime`, 6,214 → 6,100 at 2^22 `prime5`, 9,282 → 9,168 at
  2^23).
- **The suffix switch.** At 2^22 `word` and `word-prime`, levels 1 to 3 grow
  from 9,291 to about 16,045 bytes, so the proof grows by 7.4% (current) and
  7.9% (dev).
- **Everything else.** The remaining rows move by less than 130 bytes. Their
  schedules are unchanged except for one inner rank at 2^22
  `word-prime-ordinary`. The differences lie in the deeper levels, whose
  witnesses and transcripts differ.

## 6. Decision

**Interleaved lane runs.** They reach the smallest possible domain in every
measured shape, as stride 648 does. Unlike stride 648, they keep every closed
form factored and the typed Stage 3 product rank-1, at a verifier cost of a few
milliseconds. Stride 768 and three-per-2,048 packing halve fewer domains and
also break Stage 3. Twenty-five-per-16,384 packing matches the domains, but it
costs a dozen times more verifier adjoints and also breaks Stage 3.

The gain is the domain-proportional prover work of Section 1.3 in the five
shapes whose domain halves. Halving it saves about 175 ms where 2^25 becomes
2^24 and about 330 ms where 2^26 becomes 2^25. The other three shapes keep
their domains and gain only the smaller next-level witness.

The cost is proof size at 2^22 `word` and `word-prime`, which grows by about
6.5 KB because the planner selects a larger-basis suffix (Section 5.2). Every
other row moves by less than 130 bytes (Section 5.3). Section 1.3 shows that
the time after the root level does not follow the level-1 witness under fixed
fold parameters. It says nothing about the switched suffix, whose time is
measured with the totals in the opening spec.

## 7. Implementation plan

The change stays behind `labinius-trinomial`. `root_repacking`,
`native_root_relation`, `binary_source`, `native_root` and `prime_planes` are
compiled only with that feature (`crates/akita-types/src/lib.rs`,
`crates/akita-planner/src/lib.rs`). None of the changed files exists on `main`,
so no extension seam is added. Ordinary bytes cannot change, and
`ordinary_grouped_root_bytes` pins them with the feature off and on.

1. **Layout.** `RootCoefficientRepacking::new` computes each segment's run
   width and aligned start and rounds the live length to `LIVE_ALIGNMENT`.
   `coefficient_address` uses Section 2's formula. `is_padding_address` tests
   the segment ranges, because the gaps are no longer at the end of a lane.
   `repack` transposes each run from the natural digit order. A new
   `live_len` feeds the planner and the catalog.
2. **Closed forms.** `SegmentLanes` addresses a lane and its coefficients.
   `SegmentEq` holds Lemma 2's lane factors and per-run coefficient vectors.
   Image, parity, prime and direct-A weights evaluate through it, and stream
   their prover terms run by run, coefficient by coefficient. `BinaryAProduct`
   reads the digit, row and owner coordinates from the low end of the point and
   combines owners per run.
3. **Authentication.** It needs only an index change: each address comes from
   `SegmentLanes::address`, and each weight from `RootEq::at`. A separate
   behaviour-preserving commit introduces both on the stride layout first.
4. **No-panic contract.** Every verifier-reachable index uses checked
   arithmetic and `get`, and rejects with `AkitaError`. `SegmentLanes::new`
   rejects a misaligned or multi-plane segment. `SegmentEq::new` bounds every
   high index by the point's length before evaluating it.
5. **Tests.** Small-fixture expectations change where they encode addresses or
   physical cell counts. A new test checks run alignment and that the gaps are
   padding. The flat-reference differential test and the padding regression
   cover multi-run segments. `a_product_tests` adds three owners, which gives
   three response runs of one owner each.
6. **Re-pins.** All eighteen rows of `scripts/binary-source-full-size-digests.tsv`
   and the six `assert_base_bytes` pins are regenerated from runs.

## 8. Measured result

The opening spec's prover span profile records the measurement: three
isolated proving runs and three isolated verifications per configuration of
`ce98d817c`, interleaved with three of `06b8435eb`, with a catalog planned
once per executable. Every run reproduced its pin and verified. The host was
heavily shared: one-minute loads were 11.7 to 28.6, against 4.2 to 10.7 in
Section 1.3's runs. Times are medians in milliseconds; *after root* is complete
proving minus the binary root level, and *proving CPU* is the process CPU time
of complete proving.

| N | Shape | Stage 2 | Weight materialization | After root | Complete proving | Proving CPU | Verification | Prover peak RSS GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 362.5 → 208.2 | 182.3 → 185.7 | 357.5 → 483.7 | 1,917.3 → 1,859.3 (-3%) | 10,918 → 11,919 (+9%) | 261.2 → 267.8 | 4.20 → 2.89 |
| 2^22 | word-prime | 369.5 → 203.4 | 247.6 → 291.3 | 346.9 → 486.0 | 2,033.5 → 2,039.9 (+0%) | 12,231 → 13,204 (+8%) | 356.6 → 377.2 | 4.20 → 2.91 |
| 2^22 | word-prime-ordinary | 469.7 → 407.4 | 435.5 → 390.4 | 696.4 → 469.1 | 3,230.5 → 2,637.6 (-18%) | 12,523 → 12,290 (-2%) | 620.5 → 650.4 | 4.76 → 4.80 |
| 2^22 | prime5 | 652.6 → 438.8 | 639.9 → 759.6 | 326.8 → 500.1 | 3,018.3 → 2,972.1 (-2%) | 17,847 → 15,787 (-12%) | 776.1 → 827.6 | 6.97 → 4.21 |
| 2^23 | word | 722.0 → 382.5 | 265.6 → 287.3 | 517.7 → 469.5 | 3,259.4 → 2,869.2 (-12%) | 19,851 → 18,295 (-8%) | 260.1 → 269.4 | 8.05 → 5.33 |
| 2^23 | word-prime | 771.5 → 440.7 | 499.6 → 546.7 | 725.7 → 665.1 | 4,537.8 → 4,142.3 (-9%) | 21,421 → 19,448 (-9%) | 374.0 → 383.2 | 8.05 → 5.32 |
| 2^23 | word-prime-ordinary | 717.2 → 682.8 | 441.4 → 607.7 | 690.1 → 590.1 | 4,411.2 → 4,827.9 (+9%) | 21,671 → 21,862 (+1%) | 835.6 → 849.0 | 8.63 → 8.64 |
| 2^23 | prime5 | 751.1 → 705.3 | 1,043.6 → 1,256.1 | 510.3 → 548.6 | 4,522.6 → 5,502.3 (+22%) | 26,450 → 26,238 (-1%) | 784.3 → 818.1 | 8.06 → 8.05 |

- **The predicted gain arrived.** Stage 2 plus the table part of
  materialization fell by 170 to 182 ms where 2^25 became 2^24, against
  Section 6's 175 ms. Where 2^26 became 2^25 it fell by 244 ms at N=2^22
  `prime5` and by 364 to 373 ms at N=2^23, against 330 ms.
- **Other prover work grew.** Family visits and repacking cost more with
  interleaved runs, which returns part of the gain (opening spec, prover span
  profile). At N=2^22 `word` and `word-prime`, the basis-2^6 suffix of Section
  5.2 adds about 130 ms after the root level and 8% to 9% proving CPU, which
  cancels the gain in those two shapes.
- **Memory fell.** Prover peak RSS fell by 31% to 40% in the five shapes whose
  domain halved.
- **Verification grew by 1.6% to 6.6%.** Image evaluation grew by about 3.3
  ms, as Section 4 predicts for three to nine adjoints. The rest is in
  authentication evaluation, whose arithmetic is unchanged (Section 7, item
  3); its cause was not isolated.
- **Wall-time totals are inconclusive.** Under these loads, spans the layout
  does not touch varied as much as the totals. Proving CPU fell by 8% to 12%
  at N=2^22 `prime5` and N=2^23 `word` and `word-prime`, and moved by 2% or
  less in the three shapes whose domain did not shrink.

Two follow-ups remain open. Ordering T row-major would restore three image
adjoints (Section 4). Writing the family visits run-contiguously, or a
structured Stage 2 oracle, would recover the visit cost. Whether the planner
should prefer the basis-2^3 suffix at N=2^22 is a question for its objective,
which this change does not touch.

## Verification record

The following were checked against the code and the run logs.

- **Layout counts.** A script outside the repository applies each option's
  segment rule to the segment sizes of Section 1.1. Its *Before* column
  reproduces the physical cells of Section 1.2. Its *Runs* column equals the
  `witness ... physical_cells=` line that the full-size driver prints for
  every shape and size. The segment starts and run widths in Section 3.5 are
  the driver's `witness_segment` lines.
- **Lemmas 1 and 2.** `RootCoefficientRepacking` tests check addresses, ranges
  and padding on a mixed ring-plus-ordinary layout and on aligned scalar runs
  with a gap (`interleaved_runs_align_and_their_gaps_are_padding`).
  `structured_root_weights_match_the_flat_reference` compares every family's
  closed form with the flat terms built from `coefficient_address` at random
  points on nineteen shapes. These include multi-run T segments and EP
  segments of up to 215 runs.
- **Stage 3.** `stage3_a_product_matches_independent_signed_remainder_and_response_digits`
  compares the factored product with the direct response weights placed by
  `coefficient_address`, for one, two and three owners. Two owners share one
  run, and three owners give three runs of one owner each.
- **Multiplication counts.** The adjoint's 154,872 is the opening spec's count.
  The shifted-remainder count is read from
  `crates/akita-algebra/src/ring/trinomial/remainder.rs`. The run counts are
  those of Section 3.5.
- **Schedules and bytes.** Section 5's numbers come from one prove and one
  verify per row with the driver at this change, and from the logs behind
  Section 1. Every verification passed.

**Corrected numbers.** A first implementation rounded the live length to the
destination ring dimension. The full-size catalog rejected it at 2^22 `word`,
because the length then depended on the selected successor's dimension. A
version before that left the length unrounded at 14,642,046 cells, and
schedule admission rejected it, because 64 does not divide that length. The
256-cell rounding of Section 3.5 replaces both. No number in Section 1 or in
the opening spec was found to be wrong.
