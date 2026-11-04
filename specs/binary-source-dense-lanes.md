# Spec: binary source dense root lanes

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-11-04 |
| Status | draft |
| PR | |
| Supersedes | None |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

> **Draft: measurements only.** This note will compare denser layouts for the
> binary root's outgoing digit table. This revision records what the current
> layout costs. Code citations are at commit `02bd154b1`.

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
