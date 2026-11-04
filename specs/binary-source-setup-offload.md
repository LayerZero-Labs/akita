# Spec: binary source B/D setup offload

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-11-03 |
| Status | implemented |
| PR | |
| Supersedes | None |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

> **Accepted and implemented.** The opening spec,
> [binary-source-opening.md](binary-source-opening.md), describes the current
> protocol and records the measurements. This note keeps the argument.
> Sections 1 and 2 describe the protocol before the change.

## Summary

The binary root authenticates its outgoing digits through two typed digit
commitments: B for the image digits and D for the opening digits. Before this
change, the verifier evaluated both authentication rows at the Stage 2 point
`r` by reading every B/D setup column. One column is 2 rows of 64
coefficients, each squeezed from SHAKE256, so the read is linear in the setup:
62,208 image columns plus 28,512 opening columns at 2^22 `word-prime`, and
62,208 plus 139,320 at 2^22 `prime5`. After the byte-preserving speedups
earlier on this branch, the read still cost 28.0 ms of a 43.2 ms verification
at 2^22 `word-prime`, and 58.3 ms of 72.3 ms at 2^22 `prime5`.

The typed A product already avoids this cost. Its setup lives in a typed setup
prefix that setup preparation commits. Stage 3 reduces its contribution to one
prefix evaluation, and the ordinary successor folds open that evaluation. This
note moves the B and D rows into the same typed prefix and the same Stage 3
sumcheck. The prefix stores one copy of the B/D setup columns per *period* of
each digit segment. The copies are cut into power-of-two *pieces*, and the
pieces fill the zero padding of the A tensor before any cell is appended
after it. The verifier's B/D terminal weight then factors over the segment
geometry and the piece placement. It costs a few thousand field
multiplications per segment, independent of the number of setup columns.
Online verification reads no setup entry.

The change adds no sumcheck, no batching challenge and no proof field. It
enlarges the typed prefix only where the A padding cannot hold the pieces:
from 2^24 to 2^25 cells at 2^22 `word-prime` and `word-prime-ordinary`, and
to 2^26 for `prime5` at both sizes. At 2^22 `word` and at 2^23 `word`,
`word-prime` and `word-prime-ordinary`, the prefix domain is unchanged. Each
added Stage 3 round adds `2/P = 2^-127` of interactive error, 32 bytes of
round message and the round's scheduled grinding nonce. The prover now builds
a dense Stage 3 weight over the whole prefix. Before the change, the A weight
entered Stage 3 as two small factors.

**Conclusion:** the offload is sound and beats the alternatives in Section 9.
It is implemented with the packed layout of Section 4. Measured, the
binary-only shapes verify in 14.9 to 17.5 ms at 2^22 and 2^23, against 33.7 to
84.1 ms with the byte-preserving speedups alone and 249 to 785 ms before them
(Section 8). The proof grows only where the prefix domain grows. The
prover's peak memory grows in every shape, by 0.14 to 2.39 GB, and Section 8
separates what is known about its sources. An earlier appended
layout was implemented and measured first; Section 9.4 records why it was
replaced. The mixed shape's ordinary-root rows are not offloaded. Their setup
derivation is under 1 ms, and their verifier cost is flat-term construction
rather than setup reads. Section 9.3 records the follow-up.

## Notation

| Symbol | Meaning |
| --- | --- |
| `P` | The proof prime `2^128 - 275`. Challenges are base-field elements (`E = F`). |
| `x` | The outgoing digit table on `{0,1}^n`, fixed by the order-5 outgoing commitment. |
| `r` | The Stage 2 output point in `F^n`. |
| `alpha` | The ring-switch scalar (order 7). |
| `beta_B`, `beta_D` | The two row batches of the B and D authentication rows, part of `Tau1Point` ([`BinaryRootPublic::authentication_batches`](../crates/akita-types/src/binary_source/root_public.rs)). |
| `K(v)` | The residue kernel at `alpha` applied to a length-64 vector, [`ResidueKernelPoint::field_kernel_into`](../crates/akita-algebra/src/ring.rs). It is linear in `v`. |
| `M_{row,c}` | Setup column `c` of role B or D: 64 coefficients of row `row` ([`setup_entry`](../crates/akita-types/src/binary_source/authentication.rs)). |
| `e_c` | The length-64 vector of `eq(r, a)` over the root addresses `a` of the 64 compact digits in column `c` (zero where no digit sits). |
| `S` | The typed prefix table on `2^k'` cells: A, the B/D pieces, zeros elsewhere. |
| `W` | The public Stage 3 weight on the same domain. |
| `k`, `k'` | `log2` of the selected prefix domain before and after the change. |

## 1. The authentication rows

For role B the compact digits are those of segment 1 (the image, degree 648,
stride 1024, 8 digits). For role D they are those of
`plan.opening_segments()`: the U segment (degree 162, stride 256) when the
word is opened, followed by every EP prime segment (degree 162). The compact
order is segment, element, digit, coefficient. Each run of 64 digits forms one
setup column, as in the commitment ([`binary_digit_authentication_weights`](../crates/akita-types/src/binary_source/authentication.rs)).
After `alpha` and the row batches, the batched B/D row contributes this to the
Stage 2 weight at `r`:

```text
W_BD(r) = sum_(role in {B,D}) sum_(row < 2) beta_role[row]
          * sum_c < M_{row,c}, K(e_c) >.
```

The row's claim (the right-hand side, computed from the public B payload and
the D payload) does not read the setup. The weight does. Each of its terms is
a product of a setup coefficient and a public value that depends only on
`(r, alpha, beta)`. That structure is exactly the one the typed A product
already exploits.

## 2. The A product as it stands

The typed A prefix holds the A tensor (`BinaryAPrefixLen`). A cell is
`row * row_stride + rank * 1024 + c`, with the rank axis padded to a power of
two and `c < 648` live. The prover sends one claim `C = <S_A, W_A>` and runs
one degree-two Stage 3 sumcheck over the prefix. It then sends `S(z)` at the
sumcheck point `z`. The verifier checks the final round against
`S(z) * W_A(z)`, where `W_A(z)` is
[`BinaryAProduct::evaluate`](../crates/akita-types/src/binary_source/a_product.rs).
The claim `C` joins the Stage 2 terminal check in place of the A part of the
weight. The pair `(z, S(z))` joins the next ordinary grouped fold, which opens
it against the typed prefix commitment. Setup preparation authenticates that
commitment once, in
[`prepare_binary_a_verifier_setup`](../crates/akita-cpu-backend/src/binary_root/setup_prefix.rs).
The opening spec's order 10 is this Stage 3.

Both `S_A` and `W_A` vanish on the padding of the A tensor. The setup writes
only the 648 live coefficients of each live rank
([`binary_a_setup_coefficients`](../crates/akita-types/src/binary_source/a_product.rs)),
and the weight's coefficient factor writes only the 648 adjoint outputs of
each live rank row. With 3 live ranks padded to 4, the padding is a quarter of
the rank slots plus coefficients `[648, 1024)` of the live ranks. The parts
that form aligned subcubes of at least 64 cells hold 8,126,464 cells at 2^22
and 16,252,928 at 2^23 (Section 4).

## 3. Which claims move

| Claim | Before | After |
| --- | --- | --- |
| A rows (projection rows of the response) | Typed prefix, Stage 3 | Unchanged |
| B row weight `sum_row beta_B[row] <M_B, K(e)>` | Verifier reads all 62,208 image columns | Typed prefix, the same Stage 3 |
| D row weight (U and every EP segment) | Verifier reads every opening column | Typed prefix, the same Stage 3 |
| B and D row claims | Verifier | Verifier, unchanged |
| Parity, value, prime, projection-image, ordinary and `{-1,0}` rows | Verifier | Verifier, unchanged |

Only the setup-dependent weights move. In code, `BinaryRootPublic::relation`
called without a matrix keeps the B/D claims and replaces their weights with
empty term lists ([`root_public.rs`](../crates/akita-types/src/binary_source/root_public.rs)).
The flat reference does the same (`root_relation` in
[`reference_weights.rs`](../crates/akita-types/src/binary_source/reference_weights.rs)).
The prover still runs Stage 2 on the full weights. It calls `relation` with the
matrix, which keeps the B/D weights.

## 4. Layout of the typed prefix

The opening spec's
[Typed prefix layout](binary-source-opening.md#typed-prefix-layout) defines
the layout normatively: the digit segments, the period copies, the pieces,
the regions, the placement order and the identity. This section records the
properties of that layout that the argument uses, and its sizes at the
measured shapes.
[`BinaryTypedPrefix`](../crates/akita-types/src/binary_source/typed_prefix.rs)
computes it from the same inputs that build `BinaryRootWitnessPlan`. The test
`typed_prefix_offloads_the_digit_authentication_rows` checks
`BinaryTypedPrefix::new == from_plan` on every reference shape.

**Periods and copies.** The pattern of columns relative to lanes repeats every
`P = 64 / gcd(n, 64)` lanes. In the measured shapes, `P = 8` for the image
(`n = 648`) and `P = 32` for U and EP (`n = 162`), so one period covers 81
columns in every case. A period's copy carries only the weight of its own
period's digits (`DigitSegment::address` returns `None` outside the period),
so the sum over copies counts every digit exactly once. In all eight measured
shapes the offsets are multiples of 64 and the lane counts are multiples of
`P`. So `J = 81`, and the copies are the role columns exactly. The reference
shapes exercise offsets 4, 8 and 16 modulo 64, `J = 82` and partial last
periods.

**Pieces.** The image segment has 768 = 512 + 256 periods and U has 8. The
single EP segment holds every prime plane and has `344 * planes` periods
(344 = 256 + 64 + 16 + 8 for one plane); `J = 81 = 64 + 16 + 1`. A segment
therefore contributes `2 * popcount(periods) * popcount(J)` pieces: 12 for the
image, 6 for U, and for EP 24 with one plane and 36 with five
(1,720 = 1024 + 512 + 128 + 32 + 16 + 8 periods).

**Regions.** The holes lie in the zero padding of the A tensor: the padded
rank slots, and coefficients `[704, 768)` and `[768, 1024)` of the live
ranks. `[648, 704)` stays unused because only runs of at least 64 cells
qualify. At rank 3 of 4 the holes have `2^22, 2^21, 2^20, 2^19, 2^18` cells at
2^22 (twice that at 2^23). Holes require the A natural length to be a power
of two, which `BinaryAProduct` already requires of its row count.

**Placement.** Pieces are placed largest first, so every piece has a
power-of-two size and a base aligned to it. Its cells are therefore exactly
the points of a subcube in its region's local coordinates, and its cell
equality factors into the region's fixed bits, the slot, column and period
bits (Section 7). The layout requires every digit segment's start to be a
multiple of its stride, because the terminal weight splits lane addresses as
`(start/stride + lane) * stride + coefficient`. This alignment holds by
construction in the root plan
([binary-source-zero-padding.md](binary-source-zero-padding.md), §4.1).

**Sizes.** The digit cells are `2 * 64 * columns`: 7,962,624 for the image
(62,208 columns), 82,944 for U (648) and 3,566,592 per EP plane (27,864). The
image has 6,144 lanes of 648 digits at both 2^22 and 2^23, because only the
response segment grows with N. The natural lengths below are the planner's
`typed_prefix.natural_len`, which the test
`prime_plane_maximum_admission_inputs_at_large_geometries` pins
([`prime_planes.rs`](../crates/akita-planner/src/prime_planes.rs)). The split
into holes and tail follows from the allocation rule above. The A natural
length is 2^24 at 2^22 and 2^25 at 2^23, as before the change.

| N | Shape | Digit cells | In A padding | In tail | Natural cells | `2^k` before | `2^k'` after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 8,045,568 | 8,045,568 | 0 | 16,777,216 | 2^24 | 2^24 |
| 2^22 | word-prime | 11,612,160 | 8,126,464 | 3,485,696 | 20,262,912 | 2^24 | 2^25 |
| 2^22 | word-prime-ordinary | 11,612,160 | 8,126,464 | 3,485,696 | 20,262,912 | 2^24 | 2^25 |
| 2^22 | prime5 | 25,795,584 | 8,126,464 | 17,669,120 | 34,446,336 | 2^24 | 2^26 |
| 2^23 | word | 8,045,568 | 8,045,568 | 0 | 33,554,432 | 2^25 | 2^25 |
| 2^23 | word-prime | 11,612,160 | 11,612,160 | 0 | 33,554,432 | 2^25 | 2^25 |
| 2^23 | word-prime-ordinary | 11,612,160 | 11,612,160 | 0 | 33,554,432 | 2^25 | 2^25 |
| 2^23 | prime5 | 25,795,584 | 16,252,928 | 9,542,656 | 43,097,088 | 2^25 | 2^26 |

At 2^22 `word-prime` the digit cells exceed the A padding by 3,485,696 cells,
so no placement keeps the 2^24 domain. At 2^22 `prime5` they exceed it by
17,669,120 cells, and the A tensor plus all digit cells (42,572,800) exceed
2^25, so no placement keeps 2^25.

The planner sizes the successor prefix from `typed_prefix.natural_len`
through `NativeRootTransition`, so the selected `2^k'` follows the layout.
`validate_prefix` rejects a selected prefix smaller than `domain_len`.

## 5. Draw order and what is fixed before each draw

The root's transcript order is unchanged
([`verify_root`](../crates/akita-verifier/src/fold/binary_root.rs)):

1. **Before any proof atom:** the verifier rebuilds the typed layout from
   `(manifest, word, planes, d_a)`. It rejects unless the setup context for
   the scheduled slot equals `identity(manifest, slot)`. The identity string is
   `akita/binary-source/typed-prefix/a-then-digit-pieces/v3`, followed by the
   manifest bytes, the A geometry, every segment, region and piece field,
   `natural_len` and the slot. Every layout field is a little-endian `u64`, so
   the identity does not depend on the target's pointer width (opening spec,
   [Typed prefix layout](binary-source-opening.md#typed-prefix-layout)). `S` and its commitment are fixed at setup,
   before the proof begins.
2. Statement, field switch, the D payload, the response grinding, the root
   fold challenges and the successor payload. The successor commits `x`.
3. Projection rows, projected values, `alpha`, `tau0` and the row batches
   (`Tau1Point`), which include `beta_B` and `beta_D`.
4. Stage 1, `gamma`, Stage 2, which yields `r` and `x~(r)`.
5. **Stage 3:** absorb the identity; receive the claim `C`; run `k'` rounds,
   each sending two coefficients, applying its scheduled grinding step and
   drawing one challenge; receive `S(z)`.

The weight `W` is a function of `(r, alpha, beta_B, beta_D, rho)` and the
layout, so it is fixed before the prover sends `C`. The table `S` is fixed
before step 1. No new challenge is drawn, and no draw moves.

## 6. Soundness

**Disjointness.** Pieces in one region occupy disjoint aligned ranges of its
local index, and the regions are pairwise disjoint subcubes of the A padding
or lie in the tail after `A natural_len`. Both `S_A` and `W_A` vanish on every
digit cell (Section 2). So `S = S_A + S_BD` and `W = W_A + W_BD` as functions
on cells, and

```text
<S, W> = <S_A, W_A> + <S_BD, W_BD>.
```

**Identity.** For the prover's dense weights `W_BD` on the digit cells,

```text
<S_BD, W_BD> = sum_(role, row) beta_role[row] sum_c <M_{row,c}, K(e_c)> = W_BD(r).
```

Each copy of column `c` stores `M_{row,c}` and carries `beta_role[row] * K(e_c^q)`,
where `e_c^q` keeps only period `q`'s digits. Because `K` is linear, the copies
sum to `K(e_c)`. The test `typed_prefix_offloads_the_digit_authentication_rows`
checks this identity on all 19 reference shapes. It computes
`<S, W> - <S_A, W_A>` from `setup_coefficients`, `prover_weights`,
`binary_a_setup_coefficients` and the A-only weights, and compares the result
with `binary_digit_authentication_weights(...).evaluate(r)`. It also checks
that `BinaryAProduct::evaluate(z) + digit_weight(z)` equals the multilinear
extension of the prover's dense weights at a random `z`. That second check
fails if a piece overlaps a live A cell or if a region factor is wrong
(Verification record).

**Terminal check.** Before the change the verifier checked

```text
out2 = (R_full(r) + C_A) * w + (N(r) + gamma * eq(zeta, r)) * w * (w + 1),
```

where `R_full` includes the B/D weights. After the change it uses
`R(r) + C` with `R = R_full - W_BD` and `C = C_A + C_BD`. If `C = <S, W>`, the
two checks are identical. If `C != <S, W>`, then Stage 3 is a sumcheck over a
`k'`-variate product of two multilinear polynomials, one committed at setup
and one public. Its final check uses `S(z)`, which the successor proves
against the authenticated prefix commitment. Its soundness error is at most
`2k'/P`. The existing order-10 argument applies unchanged. The only
difference is that `W` has more nonzero cells.

**Added error.** Stage 3 already carried `2k/P`. The change replaces it with
`2k'/P`, which adds `2(k' - k)/P`:

| Shapes | `k` → `k'` | Before | After |
| --- | --- | --- | --- |
| 2^22 word | 24 → 24 | `48/P = 2^-122.415` | unchanged |
| 2^22 word-prime, word-prime-ordinary | 24 → 25 | `48/P = 2^-122.415` | `50/P = 2^-122.356` |
| 2^22 prime5 | 24 → 26 | `48/P = 2^-122.415` | `52/P = 2^-122.300` |
| 2^23 word, word-prime, word-prime-ordinary | 25 → 25 | `50/P = 2^-122.356` | unchanged |
| 2^23 prime5 | 25 → 26 | `50/P = 2^-122.356` | `52/P = 2^-122.300` |

There is no batching term, because B and D enter through the `beta` already
priced at `Tau1Point`. There is no new identity-testing term, because
`alpha` and `r` are reused. The sum `C_A + C_BD` travels as one field element,
as `C_A` did before.

**Ledger and grinding.** `binary_root_grinding_plan` copies the level-0
Stage 3 rounds from the ordinary plan of the selected schedule
([`grinding.rs`](../crates/akita-types/src/binary_source/grinding.rs)). That
plan is built for the schedule's `2^k'` prefix, so each added round brings its
own scheduled `SumcheckRound { protocol: Stage3, .. }` run. No root-specific
grinding site changes. The Stage 3 line of the root ledger reads
`2k'/P`. Commitment binding of the prefix and the hash terms are unchanged.

**Verifier no-panic contract.** `digit_weight` validates the batch lengths
and uses checked arithmetic and `get` throughout. The point lengths come from
the schedule; a point shorter than a region's bits is rejected with
`InvalidProof` in `Region::view`. The layout constructor rejects malformed
geometry with `InvalidSetup`.

## 7. Verifier work

`BinaryTypedPrefix::digit_weight` computes `W_BD(z)` as follows:

1. `lambda = K(eq(z[..6], .))`, a single 64-entry kernel. Every region keeps
   the 6 slot bits free and first, so the slot coordinates are `z[..6]` in
   every piece.
2. Per region, `Region::view` returns the product of `z_i` or `1 - z_i` over
   its fixed bits and `1 - z_i` over the bits at or above `top`, and the
   coordinates `z_F` at its free bits.
3. Per segment, form `eq` tables over the low `log2 stride` coordinates of
   `r` and over the next `log2 P` coordinates. One pass over the
   `min(P, L) * n` digits of a period (5,184 in every measured segment) adds
   `eq_coeff * lambda[slot]` to the sum of the digit's lane and column.
4. Per piece, contract each lane's column sums with the column equality over
   `z_F[6 .. 6+e]`. Weight by the lane factor and group by the lane-high value
   of period `q0` and the number of live periods in the piece.
5. Per group, one `paired_offset_eq_sum`. It computes
   `sum_(q < count) eq(z_F[6+e ..], base/2^(6+e) + q) * eq(r_hi, v + q)` with
   an eight-state carry recursion over the bits, which takes `O(bits)` work
   for any `count`. The result is multiplied by the row batch and the region
   factor.

Nothing in these steps loops over periods or setup columns. The period count
enters only through `count` and through the number of pieces. Per segment, the
work is one digit pass of `P * n` terms, `rows * popcount(periods) * P * J`
multiplications for the contractions, and at most four carry recursions per
piece. The number of pieces is logarithmic in the segment's period count. At
the measured shapes it is the same at 2^22 and 2^23, because the image does
not grow with N. Concretely: the image costs 2 × 2 × 8 × 81 = 2,592
contraction products and one EP plane 2 × 4 × 32 × 81 = 20,736. The
[`paired_offset_equality_sums_match_direct_summation`](../crates/akita-types/src/binary_source/typed_prefix_tests.rs)
test checks the recursion against brute force on 300 random cases.

Online verification no longer calls `setup_entry`.
`binary_digit_authentication_weights` still builds its lazy structured
weight on the verifier path, but only to take the claim. It never evaluates
the weight. Measured times are in the opening spec.

## 8. Proof bytes and prover work

**Bytes.** Each added Stage 3 round adds 32 bytes of coefficients and its
grinding nonce, which is one byte at the measured schedules. The claim and the
prefix evaluation are unchanged at 32 bytes. Predicted level-0 change: none at
2^22 `word` and at 2^23 `word`, `word-prime` and `word-prime-ordinary`; +33
bytes at 2^22 `word-prime` and `word-prime-ordinary`; +66 at 2^22 `prime5`;
+33 at 2^23 `prime5`. Where `k' > k`, the successor folds also open a larger
prefix, which changes their schedule. Everywhere, the prefix values change,
and the successors' Golomb–Rice-coded tail responses
([`tail_segments.rs`](../crates/akita-types/src/proof/tail_segments.rs))
change length with them by a few bytes. Measured totals are in the opening
spec and the digest table.

**Prover.** The prover already ran Stage 2 on the full B/D weights. It now
also does the following:

- It builds one dense weight table of `2^k'` field elements
  (`prover_weights`), `16 * 2^k'` bytes. The A part is expanded from
  `BinaryAProduct::factors`. Each 64-cell chunk of a piece is
  `beta_row * K(e^q_c)`, computed in parallel per chunk. Before the change,
  Stage 3 received those factors directly: `2^k / 4096` row entries and 4,096
  coefficient entries. So this table is new state in every shape, also where
  `k' = k`.
- It derives the B/D setup copies into the prefix (`setup_coefficients`)
  during cold preparation, with the same `setup_entry` reads the verifier made
  online before the change.
- It runs Stage 3 over `2^k'` cells, which is unchanged where `k' = k`.

**Measured.** The opening spec's tables record one full-size campaign in each
protocol. The level-0 Stage 3 bytes match the prediction: 824 bytes at
`k' = 24`, 857 at 25 and 890 at 26. Current-protocol totals:

| N | Shape | `k'` | Verification before → after (ms) | Proof bytes before → after | Prover peak RSS before → after (GB) |
| --- | --- | ---: | --- | --- | --- |
| 2^22 | word | 24 | 33.7 → 14.9 | 87,804 → 87,801 | 5.04 → 5.18 |
| 2^22 | word-prime | 25 | 43.2 → 16.1 | 87,805 → 89,464 | 5.07 → 6.06 |
| 2^22 | word-prime-ordinary | 25 | 150.5 → 124.8 | 87,786 → 89,463 | 5.18 → 6.23 |
| 2^22 | prime5 | 26 | 72.3 → 15.2 | 84,879 → 94,242 | 7.61 → 10.00 |
| 2^23 | word | 25 | 34.2 → 16.2 | 89,641 → 89,626 | 9.92 → 10.18 |
| 2^23 | word-prime | 25 | 51.4 → 17.5 | 89,603 → 89,658 | 9.70 → 10.19 |
| 2^23 | word-prime-ordinary | 25 | 241.5 → 215.2 | 89,638 → 89,661 | 9.52 → 10.21 |
| 2^23 | prime5 | 26 | 84.1 → 16.5 | 86,595 → 94,261 | 9.20 → 10.59 |

"Before" verification is the build with the byte-preserving speedups and
without the offload; "before" bytes and prover RSS are the opening spec's
tables before the offload, which those speedups do not change. Where `k'`
equals `k`, the proof changes by at most 55 bytes. Where it grows, the
successor schedule changes with the larger prefix, which adds 1,659 to 1,677
bytes at 2^25 and 7,666 to 9,363 bytes at 2^26. The verifier's B/D work is
gone in every shape. These rows predate protocol v8. After the v8 merge the
offload changes the v8 proofs by the same stages; the opening spec's proof
digest pins section compares both changes row by row.

`begin_factored_stage3` receives the dense table as its index factor with a
one-element coefficient factor. The rectangular product then binds the same
variables in the same order as with the table as its coefficient factor, and
the proof bytes are identical. The choice matters for memory: the kernel's
first pass keeps one unreduced accumulator per coefficient-factor entry in
each parallel task, which at 2^22 `word` raised the prover's peak RSS from
5.18 GB to 7.33 GB.

**Where the prover memory goes.** Three sources add memory, and they do not
peak together.

- *Prefix growth.* Where `k' > k`, the setup prefix, its commitment and the
  successor folds that open it all double or quadruple. This is the only
  source that also changes the proof.
- *Dense weights.* The table above is `16 * 2^k'` bytes in every shape:
  256 MiB at `k' = 24`, 512 MiB at 25 and 1 GiB at 26.
- *Backend state.* With a one-element coefficient factor there is no
  coefficient round, so the rectangular kernel used to build its index table
  at construction. That table is a full-length copy of the setup source, so
  Stage 3 held the source, the dense weight and the copy (three
  `16 * 2^k'`-byte tables), and the first fold added half a table. The
  prover's `Vec`-to-`Arc` conversion of the source also overlapped the dense
  weight. The kernel now reads the index table from the source until the first
  challenge folds it to half length, then releases the source. The prover
  converts the source before it builds the weight. This changes no proof byte,
  and it lowers Stage 3's peak from three and a half tables to two and a half.

The process peak did not move with that change. These are isolated proving
processes, both with matching digest pins, measured back to back on the shared
host before and after it:

| N | Shape | `k'` | Prover peak RSS before → after (bytes) |
| --- | --- | ---: | --- |
| 2^22 | word | 24 | 5,206,458,368 → 5,189,353,472 |
| 2^22 | prime5 | 26 | 10,007,166,976 → 10,012,590,080 |

Removing a 256 MiB (`word`) or 1 GiB (`prime5`) Stage 3 table changed the
process peak by less than 20 MB. Whole-process peak RSS cannot attribute the
peak to a phase, in part because the allocator may retain freed memory, so
which phase sets the peak is unmeasured. Where `k' > k` the growth is
consistent with prefix growth. At 2^22 `word` (+0.14 GB) and the 2^23 shapes
with `k' = k` (+0.26 to +0.69 GB), the prefix domain is unchanged, and the
source of the growth has not been located.

## 9. Alternatives

### 9.1 Cache the B/D columns in the verifier setup

This removes the SHAKE derivation but not the linear pass. The verifier would
still compute `<M_{row,c}, K(e_c)>` for every column. It would also hold
`columns * 2 * 64 * 16` bytes: 185,794,560 bytes at 2^22 `word-prime`
(90,720 columns) and 412,729,344 bytes at 2^22 `prime5` (201,528 columns).
Rejected.

### 9.2 Structured B/D setup

A setup whose columns are generated from a short tensor or from evaluations
of a low-degree map would let the verifier evaluate `W_BD(r)` in closed form
without Stage 3. It changes the distribution of the digit commitments' SIS
matrix, and it needs a new hardness argument. Rejected for this branch.

### 9.3 Offload the mixed shape's ordinary-root rows

At 2^22 `word-prime-ordinary`, the ordinary-root setup derivation reads
131,072 elements and costs 0.69 ms. The remaining ordinary span is the
construction of 9,936,896 flat terms from public geometry, not setup reads.
Offloading those rows through Stage 3 would remove setup reads that cost
almost nothing. It would not shorten term construction, which needs a
structured evaluator for the ordinary rows instead. Deferred.

### 9.4 Append the copies after the A tensor

The first implementation placed every copy after the A tensor in blocks of
`2^e` columns. It is simpler: no regions, one aligned base per block. It
always enlarges the prefix, because the A tensor alone already fills a power
of two: 2^24 → 2^25 at 2^22 `word`, `word-prime` and `word-prime-ordinary`,
2^24 → 2^26 at 2^22 `prime5`, and 2^25 → 2^26 at every 2^23 shape. Measured at
full size, it removed the B/D derivation as intended (verification 15.5 ms to
17.7 ms for the three binary-only shapes at both sizes). But the larger prefix
doubled the prover's peak RSS at 2^22 `word-prime` (5.06 GB to 10.35 GB),
added 1,673 bytes there (87,805 to 89,478) and 9,344 bytes at 2^22 `prime5`,
and raised verifier RSS by 42 to 80 MB at 2^22, because the successor folds
open the larger prefix. Packing into the A padding keeps the verifier gain.
It restores the 2^24 and 2^25 domains wherever the digit cells fit.
Rejected in favour of Section 4.

## Verification record

- `typed_prefix_offloads_the_digit_authentication_rows` (19 shapes): stored
  setup against dense weights equals the direct B/D weight at `r`; the
  terminal weight `BinaryAProduct::evaluate + digit_weight` equals the dense
  weights' extension; `new` agrees with `from_plan`. Each of these mutations
  fails it:
  - removing the `p >= period` cut in `DigitSegment::address`, so a copy also
    weights the next period's digits, breaks the stored-setup identity;
  - replacing the live-period count `ceil((L - p) / P)` in `digit_weight`
    with `L / P` breaks the terminal-extension check on the shapes with a
    partial last period;
  - starting the coefficient-tail holes at 512 instead of `A_DEGREE` in
    `a_padding_regions`, so pieces overwrite live A cells, breaks the
    terminal-extension check;
  - dropping the region factor from `digit_weight` breaks the
    terminal-extension check.
- `paired_offset_equality_sums_match_direct_summation`: 300 random cases
  against brute force.
- `structured_root_weights_match_the_flat_reference`: the verifier relation
  and the flat reference agree with the B/D weights removed.
- `stage3_prefix_context_rejects_manifest_swap_and_missing_context_before_proof`
  and `reject_foreign_prefixes`: identity admission and prefix authentication
  under the new identity.
- `prime_plane_maximum_admission_inputs_at_large_geometries`: the natural
  lengths in Section 4.
