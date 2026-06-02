# Spec: Packed-SIMD sum-check and EOR prover

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao (spec) → hand-off for implementation |
| Created     | 2026-06-02                     |
| Status      | ready for implementation       |
| Pilot       | extension-opening reduction (EOR), then stage1 / stage2 |
| Related     | [`specs/eor-streamed-prover.md`](eor-streamed-prover.md), PR [#142](https://github.com/LayerZero-Labs/akita/pull/142) (`specs/cross-repo-field-microbench.md`) |

## Summary

Akita's sum-check and extension-opening-reduction (EOR) prover hot loops run on
**scalar extension-field arithmetic** today: `Vec<E>` with `E = RingSubfieldFp4<Fp32>`,
folded and accumulated one element at a time (the only "vectorization" is the wide
**scalar** `ProductAccum` of `HasUnreducedOps` plus Rayon). The arch-specific packed
SIMD representation that already exists in `akita-field`
(`PackedField` / `HasPacking`, and the production-quartic `PackedRingSubfieldFp4`) is
**not connected** to any loop above `akita-field` (confirmed: zero `Packing` /
`PackedField` / `pack_slice` hits in `akita-prover`, `akita-sumcheck`, or non-NTT
`akita-algebra`).

This spec threads the packed representation through the **data-parallel prover loops**.
EOR is the pilot (its fold-dominated rounds and its newly-streamed witness path are the
cleanest target); the same recipe then rolls out to the stage1 (eq-factored) and stage2
(standard) sum-check provers. The committed direction includes a **packed unreduced
accumulator** (`PackedHasUnreducedOps`, D1) so the *accumulate*, not just the fold, is
lane-parallel without giving up akita's deferred-reduction win. The design follows two
production references that solved exactly this: **Plonky3 `p3-sumcheck`** and
**leanMultisig `mt-field`** (see References).

The pack representation is a **byte-identical, field-exact** representation change, not a
protocol or algorithm change. It is **orthogonal to and composable with**
`eor-streamed-prover.md` (stream `g`, prefix-suffix `A_η`, relax the cap): packing
changes *how* a loop computes, the streamed prover changes *what* it materializes.

### Honest framing up front (read before scoping)

Packing is **not** a uniform 8× win, and the spec is explicit about this:

- **Base-field ops pack near-linearly** (PR #142, throughput): base `mul` **8.0×** on
  AVX2 (×8 lanes), **3.7×** on NEON (×4). This is the embarrassingly-parallel ceiling.
- **A single fp4 (degree-4 extension) multiply packs only ≈ 1.1-1.2× on throughput**
  (PR #142: fp4 rs `mul` AVX2 9.106 → 1.052 ns/lane = 1.08×; NEON 7.522 → 1.530 =
  1.23×), because one fp4 multiply is itself a multiply-heavy schedule, not lane-parallel
  elementwise work. **Do not promise dense-base-like speedups on extension multiplies.**
- **The real lever is the fold/accumulate structure, not the multiply.** Sum-check rounds
  are dominated by the *linear-interpolation fold* `a + r·(b−a)` and the
  add/sub-heavy `add_constant_product` / `add_quadratic_product` across many hypercube
  points — that work *is* lane-parallel. Akita's own EOR breakdown is **fold-dominated**
  (`eor-streamed-prover.md`, dense nv26: univariate 52 ms vs **fold 293 ms**; one-hot
  nv32: univariate 1252 ms, **fold 719 ms**). Packing the fold realistically lands
  **≈ 2-4×** on fold-heavy rounds; less on mul-heavy univariate rounds. That is still a
  large absolute win on the measured hotspots.
- Akita's fp4 (degree-4) packed `mul` already beats Plonky3's security-equivalent
  KoalaBear degree-5 by **1.7×** (AVX2) / **2.75×** (NEON) — so the field layer is
  competitive; the gap to close is purely that nothing above `akita-field` uses it.

## Background

### The packing layout (already in `akita-field`)

`PackedRingSubfieldFp4<F, F::Packing>` is a **transpose / coefficient** packing: it stores
`[PF; 4]` where `PF = F::Packing` is the packed base field, and each `PF` holds the
corresponding coefficient of an fp4 element across `WIDTH` base lanes
(`packed_ext.rs:1-6` doc, `:558-637`, `:702-710`). So **one packed word represents
`WIDTH` distinct fp4 elements** (one per lane), with arithmetic over them performed by the
existing SIMD base-field ops. This is identical to Plonky3's `ExtensionPacking`
(`[[F::Packing; D]; W]` transpose, `packed_traits.rs:356-363`).

Lane counts (`packed.rs` arch selection): NEON ×4, AVX2 ×8, AVX-512 ×16; the no-SIMD
fallback is `NoPacking<T>` with `WIDTH = 1` (`packed.rs:273`). `RingSubfieldFp4: HasPacking`
is wired (`packed_ext.rs:702`).

Consequence for sum-check: a multilinear table `Vec<E>` of `2^n` evals **is** a
`Vec<PackedRingSubfieldFp4>` of `2^n / WIDTH` words once you pack `WIDTH` consecutive evals
per word. The fold and the per-round univariate then process `WIDTH` hypercube points per
SIMD instruction.

### Reference recipe (Plonky3 `p3-sumcheck` + leanMultisig)

Both converge on the same five moves; copy them.

1. **Store the table packed.** Plonky3 keeps `Poly<EF::ExtensionPacking>`
   (`multilinear-util/src/poly.rs`); leanMultisig keeps `MleGroup::{BasePacked,
   ExtensionPacked}` and reinterprets base `&[F]` via `PFPacking::pack_slice`
   (`mle_group_ref.rs:81-92`). One packed cell = `WIDTH` hypercube nodes; the logical
   variable count adds back `log2(WIDTH)`.
2. **Accumulate in packed words, horizontal-reduce once per round.** There is **no**
   `horizontal_sum` trait; the idiom is to accumulate `(c0, c_inf)` (or the round
   coefficients) as packed extension words across the half-hypercube, then reduce with
   `to_ext_iter([acc]).sum()` exactly once when emitting the scalar round message
   (Plonky3 `product_polynomial.rs:354-358`; leanMultisig `sc_computation.rs:117-120`,
   `packing_unpack_sum`).
3. **Fold with a broadcast challenge.** `fix_var_mut` / lane-wise linear interpolation
   with `Packed::broadcast(r)` (Plonky3 `poly.rs:513-537`; leanMultisig
   `sc_computation.rs:353` `EFPacking::from(challenge)`).
4. **Transition to scalar for the last `log2(WIDTH)` rounds.** When one packed word
   remains, unpack lanes (`to_ext_iter`) and finish scalar (Plonky3
   `product_polynomial.rs:283-300`; leanMultisig auto-unpack when
   `n_vars ≤ 1 + packing_log_width`, `utils.rs:66-68`).
5. **Keep early rounds lane-aligned via storage order.** leanMultisig folds
   *right-to-left* with a **bit-reversed chunk layout** so the bound variable stays *above*
   the SIMD-lane boundary for as many early rounds as possible, only dropping into the
   lane (and unpacking) in the final phase (`air_sumcheck.rs:9-30`,
   `quotient_gkr/mod.rs:19-26`). This is the trick that keeps SIMD live across the big
   rounds.
6. **Tail (length not a multiple of `WIDTH`).** `pack_slice_with_suffix` → packed main
   loop + scalar epilogue, or pad the hypercube to a `WIDTH` multiple
   (Plonky3 `helpers.rs:34-38`, `field.rs:1030-1040`).

### Current akita scalar state (what changes)

| Loop | File:line | Shape |
|------|-----------|-------|
| EOR dense fold+accumulate | `extension_opening_reduction/dense.rs:173-190` (`fused_fold_and_accumulate`), `:60-69` (`accumulate_dense_round_with`) | scalar `E::fold_one`, `acc.add_constant_product` |
| EOR factor fold (materialized) | `sparse.rs:883-903` (`fold_dense_reduction_tables_in_place` → `fold_evals_in_place`) | scalar `Vec<E>` |
| EOR sparse accumulate / factor query | `sparse.rs:322-346` (`accumulate_entries_with_factor_using`), `:749-783` (`factor_pair`) | scalar over support + scalar `ProductAccum` dot |
| EOR partials | `akita-types/src/extension_opening_reduction.rs:260-279` (`tensor_column_partials_split_fold`) | scalar base×ext via `SplitEqEvals` |
| eq table | `akita-algebra/src/eq_poly.rs:232-252` (`SplitEqEvals`), `EqPolynomial::evals` | scalar `Vec<E>` |
| stage1 round + fold | `akita_stage1/mod.rs:392-448`, `round_flow.rs:200-201` | scalar |
| stage2 round + fold | `akita_stage2/dense_terms.rs:189-216`, `round_flow.rs:234` | scalar |
| generic fold | `akita-algebra/src/poly.rs:225-228` (`fold_evals_in_place`) | scalar `E::fold_one` |
| driver (unchanged) | `drivers/standard.rs:112-148` | trait-driven; packing lives in the instance |

## Key design decisions (the hard parts — resolve these first)

### D1 — Packed unreduced accumulator (DECIDED: build it; minimize in-loop reductions)

Akita's EOR/sum-check speed rests on the **wide scalar** `HasUnreducedOps::ProductAccum`
(defer modular reduction across a sum, e.g. `MulBaseUnreduced::mul_base_to_product_accum`,
PR #136): accumulate many products in a wide integer, reduce once. Akita already wires this
into the EOR accumulate; the packed path must keep it.

**The field form (Montgomery vs pseudo-Mersenne) is *not* the obstacle.** Delayed reduction
is clean in Montgomery too: a sum of Montgomery products `Σ â_i·b̂_i = R²·Σ a_i b_i`
reduces to the Montgomery form of the dot with a single reduction. Plonky3 and leanMultisig
both do exactly this *scalarly, in production* — `MontyField31::dot_product` accumulates
products in a `u128` and applies **one** `monty_reduce_u128` (prior art below). So "deferred
reduction needs pseudo-Mersenne" is **false**; akita's prime choice is incidental here, not
enabling.

**What the measured cost actually is — reduction *frequency*, not lane width.** A directional
microbench (Mersenne31, `p = 2^31−1`; 4 `u64`/`u128` lanes laid out lane-major to mimic the
packed transpose; autovectorized `-C target-cpu=native -C opt-level=3` on the NEON dev
machine; in-cache and out-of-cache runs identical, so **compute-bound**; all strategies
return identical checksums) isolates the per-product accumulate cost:

| strategy (per product)            | full-width `a,b < 2^31` | small `a < 2^8`, `b < 2^31` |
|-----------------------------------|:----------------------:|:--------------------------:|
| eager (reduce every product)      | 1.19 ns                | 1.54 ns                    |
| chunked `u64`, reduce every 3     | 0.73 ns                | 0.80 ns                    |
| `u128` lane, reduce once          | **0.29 ns**            | **0.29 ns**                |
| `u64` lane, reduce once (deferred)| (overflows)            | **0.31 ns**                |

Two facts overturn the earlier "`u128` doesn't vectorize → avoid it" claim:

1. **`u128`-per-lane single-reduce is the *fastest* full-width path** (0.29 ns; 2.5× over
   chunked-`u64`, 4× over eager) — *even autovectorized on NEON*. On aarch64 the `u128`
   multiply-accumulate is `mul`/`umulh` + `adds`/`adcs`, cheap and branch-free; it wins
   because it carries **no modular reduction in the loop**. The earlier rejection was wrong.
2. **The lever is reduction frequency, not lane type.** Both single-reduce strategies
   (`u128` full-width; `u64` deferred small) land at ~0.3 ns; everything that reduces
   in-loop is 2.5-4× slower. `u128`'s capacity is irrelevant for us: a per-lane sum of
   `2^29` full-width products at nv32 is `< 2^91 ≪ 2^128` (safe past nv60).

**Magnitude-aware chunk size (your round-0 point, quantified).** The safe number of products
between reductions, `K`, is set by the operand *magnitudes*, not a fixed 2-4:

- **Full × full** (rounds ≥1, folded by random challenges): product `< 2^62`, a `u64` lane
  overflows after `K ≈ 3`. This is the expensive chunked regime → use a **`u128` lane**
  (single reduce per round, fastest, capacity-safe).
- **Small-balanced digit × full factor** (round-0 dense witness, `|d| < 2^b`): product
  `< 2^(31+b)`, so `K < 2^(63−31−b) = 2^(32−b)`. At nv32 a lane sums ≈ `2^29` terms, so for
  `b = 8` you reduce ≈ `2^29 / 2^24 = 32` times *per whole round* — negligible, effectively
  a **`u64` single-reduce**.
- **One-hot witness ∈ {0,1} × full factor** (round-0 one-hot): the accumulate is `Σ factor`
  over the support; product `< 2^31`, a `2^29`-term lane sums to `< 2^60 < 2^64` → the
  **entire round fits one `u64` reduce**, zero in-loop reductions.

So the fold-/accumulate-heavy hotspots (round 0 + partials) run in cheap `u64` lanes with
O(tens) reductions per round — exactly the small-magnitude advantage you flagged — and only
the geometrically-smaller full-width tail rounds use `u128` lanes. Neither hits the slow
"reduce every 3" regime.

**Decision (locked): build a packed `ProductAccum` that reduces once per round (or per
magnitude-bounded chunk), choosing lane width by regime.** `u64` lanes for the small-operand
round-0/partials accumulate (reduce only when a lane nears overflow, `K ≈ 2^(32−b)`, or never
for one-hot); `u128` lanes for the full-width later rounds (single reduce per round).
Concretely:

- `PackedHasUnreducedOps::ProductAccum` carries lane-wise partial sums plus a reduce
  threshold; packed `mul_base_to_product_accum` / `add_constant_product` /
  `add_quadratic_product` accumulate and reduce on threshold (a packed `mul_pmersenne31`-style
  reduce, PR #142), honoring `DELAYED_PRODUCT_SUM_IS_EXACT`. One horizontal reduce per round
  emits the scalar message.
- Akita's edge over the references is twofold: (i) it already ships the scalar full-sum
  accumulator and wires it into EOR (`mul_base_to_product_accum`), whereas leanMultisig added
  trait hooks for *exactly this* (base×ext sumcheck delayed reduction) but left them unwired
  (prior art below); (ii) pseudo-Mersenne `mod p` accepts a full `u64` input, so the
  small-operand regime gets the maximal `K` (more amortization than a tight `monty_reduce`
  bound) — verify against the exact bound, do not assume.

**Caveat (acceptance gate, not a re-litigation).** The table is autovectorized scalar, not
hand-tuned NEON intrinsics. A `vmull_u32`-based `u64` kernel with a vectorized reduce could
narrow the full-width chunked gap — but it cannot beat single-reduce, and `u128` also benefits
from `vmull`. Slice 2 must re-measure `u128`-lane vs `u64`-chunked with the *real* packed
types/intrinsics before locking the full-width-round lane choice; the round-0/partials `u64`
single-reduce result is robust regardless. Microbench source and raw numbers ship with the
slice so the colleague can re-run on AVX2/AVX-512.

**Sequencing (de-risk; not a re-litigation of the decision):** land the packed *fold*
first (Slice 1) — the linear interpolation `a + r·(b−a)` needs only packed
add/sub/mul-by-broadcast and **no** accumulator, so it captures the fold-dominated bulk of
the measured win immediately and validates the packed-table plumbing. Then land the
chunked-reduce accumulator (Slice 2) for the univariate accumulate. Packing the fold first
is a stepping stone *toward* the accumulator, not an alternative to it.

**Rejected:** an **eager per-product reduce** — it throws away the deferred-reduction win
akita depends on and measured slowest (1.2-1.5 ns). **No longer rejected:** the
`u128`-per-lane accumulator — it was the measured-*fastest* full-width option and is the
committed full-width-round form (pending hand-intrinsic confirmation, above). The fixed
"chunk every 2-4" is not a separate committed path; it is what the magnitude-aware `u64`
threshold degenerates to only in the full-width regime, where `u128` lanes beat it.

### D1 prior art (delayed reduction, including in Montgomery form)

Precise public references for the chunked-reduce design — and for the (correct) point that
Montgomery composes cleanly with delayed reduction:

- **Plonky3** (`Plonky3/Plonky3`, commit `3dc870c2`):
  - Fused packed dot, AVX2 — the headroom argument verbatim ("all inputs `< P < 2^31`, so
    `l0*r0 + l1*r1 < 2P² < 2^32 P`, so the Montgomery reduction algorithm can be applied to
    the sum of the products instead of to each product individually"):
    [`monty-31/src/x86_64_avx2/packing.rs#L510-L512`](https://github.com/Plonky3/Plonky3/blob/3dc870c2adff2591f2377b214f5166c5a66d9eb3/monty-31/src/x86_64_avx2/packing.rs#L510-L512).
    NEON / AVX-512 mirror it; the kernels are `dot_product_2/4/5/8` in
    `monty-31/src/{aarch64_neon,x86_64_avx2,x86_64_avx512}/packing.rs`.
  - Scalar `Sum` passes through `u64` "allowing for delayed reductions" for `N > 7`:
    [`monty-31/src/monty_31.rs#L250-L255`](https://github.com/Plonky3/Plonky3/blob/3dc870c2adff2591f2377b214f5166c5a66d9eb3/monty-31/src/monty_31.rs#L250-L255).
- **leanMultisig** (`leanEthereum/leanMultisig`, commit `a2efa4f3`):
  - **Clean Montgomery full-sum delayed reduction, in production:**
    `MontyField31::dot_product` accumulates products in a `u128` and applies one
    `monty_reduce_u128`:
    [`crates/backend/koala-bear/src/monty_31/monty_31.rs#L342-L361`](https://github.com/leanEthereum/leanMultisig/blob/a2efa4f35ccea70884ba77b417c1fd9ca2933559/crates/backend/koala-bear/src/monty_31/monty_31.rs#L342-L361).
  - **Direct precedent for akita's exact scenario:** Tom Wambsgans's commit
    [*"delayed modular reduction for base x ext product sumcheck (first round of whir)"*](https://github.com/leanEthereum/leanMultisig/commit/ab19b44863d41a841dc280006afa431742769b7a)
    added `reduce_product_sum(u128)` / `reduce_signed_product_sum(i128)` —
    [trait `field.rs#L897-L908`](https://github.com/leanEthereum/leanMultisig/blob/a2efa4f35ccea70884ba77b417c1fd9ca2933559/crates/backend/field/src/field.rs#L897-L908),
    [Monty impl `monty_31.rs#L628-L639`](https://github.com/leanEthereum/leanMultisig/blob/a2efa4f35ccea70884ba77b417c1fd9ca2933559/crates/backend/koala-bear/src/monty_31/monty_31.rs#L628-L639).
    These hooks are **currently uncalled** (the packed sumcheck accumulates in reduced
    `EFPacking`): leanMultisig built the scalar/trait machinery for delayed base×ext
    reduction but did **not** wire the *packed* version — consistent with the lane-width
    constraint above.
  - Broader delayed reduction is listed as open work:
    [`TODO.md#L8`](https://github.com/leanEthereum/leanMultisig/blob/a2efa4f35ccea70884ba77b417c1fd9ca2933559/TODO.md#L8).

Takeaway: chunked `u64`-lane reduce (Plonky3 fused dots `dot_product_2/4/5/8`) is the
full-width fallback; full-sum single-reduce is clean in Montgomery (leanMultisig scalar
`dot_product` over `u128`) and — per the microbench above — is the *faster* shape; the
base×ext sumcheck delayed reduction akita targets was prototyped by leanMultisig but not
taken to the packed path. Akita's packed `ProductAccum` picks the reduce frequency from the
operand magnitude (single-reduce in round-0/partials and full-width tail; never the slow
reduce-every-3) — that is the generalization the references stop short of.

### D2 — Fold order / lane alignment

Natural LSB-first folding drops into the SIMD lane immediately on a contiguously-packed
table. Mirror leanMultisig's **bit-reversed chunk storage + right-to-left fold** (or an
equivalent permutation) so the bound variable stays above the lane boundary for the early
(largest) rounds — that is where the SIMD time is. Verify the permutation is verifier-
invisible (it is in leanMultisig: storage-only).

### D3 — Transition + tail

Unpack and finish scalar for the final `log2(WIDTH)` rounds; handle non-`WIDTH`-multiple
tables with `pack_slice_with_suffix` + scalar epilogue or hypercube padding. Both are
mechanical (reference recipe moves 4 and 6).

### D4 — Sparse one-hot path (SIMD-amenable; kernel detail deferred)

The one-hot EOR univariate plateau (`eor-streamed-prover.md`, 1.25 s) is a sparse
`O(d_ext)`-per-query loop over a `2^24` support. It is **not excluded** — it can benefit
from SIMD, but the kernels are arch-sensitive and are designed in a later slice:
- **Factor fold** of a materialized (`SparseFactor::Dense`) residual is a dense `Vec<E>`
  fold → **packs exactly like the dense path** (no gather). Free once D1/D2 land.
- **`factor_pair`'s `O(d_ext)` dot** over `suffix_tables[t][suffix_index]` is a tiny dense
  dot product — vectorizable per query, and **batchable across `WIDTH` support entries**
  if their suffix indices are gathered.
- **Materialized-factor accumulate** (`m = 0` region, flat factor + O(1) reads) can pack
  `WIDTH` support entries by **gathering** their factor values, then packed-multiplying.
  Gather is the catch: AVX-512 has fast gather, AVX2's is slow, NEON has none — so the
  one-hot packed accumulate is **arch-gated** and may stay scalar on NEON. Work this out
  in Slice 4 against measured numbers; do not block the dense pilot on it.

## Pilot: EOR (all of it)

Order within the pilot:
1. **Dense fold** (`fused_fold_and_accumulate` fold half) — packed fold, no accumulator
   yet. Highest-confidence win; validates the plumbing.
2. **Dense accumulate** — packed unreduced accumulator (D1, the committed end-state).
3. **Factor fold** of the materialized residual — same packed fold; free after step 1.
4. **Partials** (`tensor_column_partials_split_fold`) — packed base×ext contraction; the
   `SplitEqEvals` tables become packed-aware (`Vec<PackedRingSubfieldFp4>` views).
5. **Sparse one-hot** accumulate/query — D4, arch-gated, deferred kernel.

## Rollout: stage1 / stage2

Same recipe applied to `compute_norm_round_eq_poly_from_s*` (stage1),
`compute_round_*_dense_terms` (stage2), and the shared `fold_evals_in_place`
(`akita-algebra/src/poly.rs:225`). Stage1's eq-factored structure already separates an
`e_in`/`e_out` split that maps onto the packed-eq pattern leanMultisig uses
(`split_eq.rs`); reuse the same packed eq tables.

## Trait surface (what to add to akita-field / akita-algebra)

- A **packed multilinear table view**: pack/unpack a `&[E]` into `&[E::Packing]` with a
  scalar tail, mirroring `PackedValue::pack_slice_with_suffix`. (akita-field already has
  the packed types; add the slice-cast helpers if missing.)
- A **packed fold** primitive: `E::Packing` linear interpolation with `broadcast(r)`
  (the Slice 1 fold needs only this).
- A **packed unreduced accumulator** (`PackedHasUnreducedOps::ProductAccum`) + packed
  `mul_base_to_product_accum` / `add_constant_product` / `add_quadratic_product`, honoring
  the scalar `DELAYED_PRODUCT_SUM_IS_EXACT` bound, with a **magnitude-aware reduce threshold**
  (`u64` lanes single-reduce for small-operand round-0/partials; `u128` lanes single-reduce
  for full-width tail rounds) — the committed D1 end-state.
- A **horizontal reduce** at round boundary (`to_ext`-style lane sum); no persistent
  trait method needed, just a helper.
- **`NoPacking` (WIDTH = 1) fallback** must keep non-SIMD builds and `fp128` byte-identical
  and within noise of today (the packed path degenerates to the current scalar loop).

## Invariants

- **Byte-identical proofs and transcript** for every mode/`num_vars` (packing is a
  representation; results are field-exact). The packed fold/accumulate equals the scalar
  one element-for-element; the only ordering freedom (associativity of the reduction) must
  preserve the field-exact sum — verify with the equality oracle.
- **Verifier untouched.** Prover-only. No storage-permutation (D2) leaks to the wire.
- **`NoPacking` parity.** `WIDTH = 1` builds reproduce today's scalar path bit-for-bit.
- **Delayed-reduction exactness preserved.** The packed `ProductAccum` (D1) must honor the
  same `DELAYED_PRODUCT_SUM_IS_EXACT` bound as the scalar path.
- **Composable with `eor-streamed-prover.md`.** The streamed witness round-0 fold and the
  budget-driven factor fold are *the same loops* this spec packs; they must be implemented
  in packed-ready shape (see that spec's "Packing readiness" note).

## Non-Goals

- No protocol, soundness, degree, schedule, or transcript change.
- No GPU / Metal path (separate workstream).
- No claim of base-field-like (8×) speedup on extension multiplies.
- No new extension field; reuse `PackedRingSubfieldFp4` and friends.

## Evaluation

### Acceptance criteria

- [ ] Packed table view + packed fold in `akita-field`/`akita-algebra`, with `NoPacking`
  fallback; `WIDTH = 1` builds byte-identical to today.
- [ ] EOR dense fold runs packed (Slice 1); proof bytes byte-identical on `dense_fp32_d32`,
  `onehot_fp32_d32`, `onehot_fp16_d32`, `onehot_fp64_d32`; `fp128` unaffected.
- [ ] Packed unreduced accumulator (`PackedHasUnreducedOps`) implemented and exact, with the
  magnitude-aware reduce threshold (D1); EOR dense accumulate + factor fold + partials run
  packed through it. Re-run the D1 microbench with the real packed types/intrinsics on
  NEON + AVX2 (+ AVX-512) and record whether `u128`-lane or `u64`-chunked wins the full-width
  rounds before locking that lane choice.
- [ ] stage1 + stage2 folds packed; full byte-identical proof/transcript suite.
- [ ] Sparse one-hot: factor fold packed; accumulate/query packed where the target arch
  supports it (gated), scalar fallback otherwise; byte-identical either way.
- [ ] Tests per field family (fp32 `RingSubfieldFp4`, fp16 `RingSubfieldFp8`, fp64 `Fp2`,
  fp128 identity): `packed_fold_matches_scalar`, `packed_round_univariate_matches_scalar`,
  `packed_eor_proof_byte_identical`, `noPacking_parity`.
- [ ] `cargo fmt -q`; `cargo clippy --all --all-targets -- -D warnings`;
  `cargo test -p akita-prover --test extension_opening_reduction`; cross-arch
  (NEON + AVX2, AVX-512 if available).

### Testing strategy

The scalar path is the oracle: assert packed fold/round/accumulate equal the scalar result
field-exactly, and that end-to-end proof bytes are byte-identical (the strongest guard).
Reuse PR #142's `field_arith/kernel/packed_macc` micro-bench (`acc += eq[i]*poly[i]` via
`pack_slice`) and add an EOR-prover-level bench so perf is gated on the real loop, not just
`field_arith` rows (per `cross-repo-field-microbench.md:252-257`).

### Performance

Validate on the profile harness (`sumcheck_round_{univariate,fold}` per-round spans, added
in PR #136) for `dense_fp32_d32` / `onehot_fp32_d32` / `onehot_fp16_d32` at nv26/30/32,
across NEON + AVX2 (+ AVX-512). Expectation: **≈ 2-4× on fold-dominated rounds**, smaller
on mul-heavy univariate; net EOR prove-time reduction concentrated in the measured
`sumcheck_round_fold` hotspot. No proof-size, schedule, or transcript effect.

## Implementation plan (sliced for review)

- **Slice 0 — packed table view + fold primitive.** Add the `pack_slice`-style table view,
  the packed fold primitive, and the `NoPacking` parity test. No behavior change yet.
- **Slice 1 — EOR dense fold.** Pack the fold half of `fused_fold_and_accumulate`; scalar
  accumulate for now. Byte-identical proofs; first measured win; validates the plumbing.
- **Slice 2 — packed unreduced accumulator (D1) + EOR dense accumulate + factor fold +
  partials.** Build `PackedHasUnreducedOps`; route the univariate accumulate, the factor
  fold, and the `SplitEqEvals`/partials contraction through it.
- **Slice 3 — stage1 / stage2 folds.** Roll the recipe into stage1/stage2 round+fold.
- **Slice 4 — sparse one-hot (D4).** Packed factor fold (free), then arch-gated packed
  accumulate/query with gather; scalar fallback on NEON.
- **Slice 5 — fold-order / lane alignment (D2), if measured to matter.** Storage
  permutation to keep early rounds lane-aligned; only if Slice 1-3 show lane-drop overhead.

First milestone: confirm the Slice 1 packed fold moves the measured `sumcheck_round_fold`
span, then build the packed unreduced accumulator (Slice 2) for the univariate accumulate.

## References

- **Plonky3** ([github.com/Plonky3/Plonky3](https://github.com/Plonky3/Plonky3), main):
  `p3-sumcheck` `sumcheck/src/product_polynomial.rs` (packed round + horizontal reduce `:335-387`,
  `:354-358`; transition `:283-300`), `sumcheck/src/strategy.rs` (`mixed_dot_product`
  tiles `:52-68`, `:116-173`); packed traits `field/src/packed/packed_traits.rs`
  (`PackedValue` `:19`, `PackedField` `:275`, `PackedFieldExtension` `:364`, layout
  `:356-363`); `Field::Packing` / `ExtensionPacking` `field/src/field.rs:962-981`,
  `:1139-1140`; packed-loop idiom `field/src/helpers.rs:34-38`,
  `dft/src/butterflies.rs:52-62`; multilinear pack/fold `multilinear-util/src/poly.rs`
  (`:476-479`, `:513-537`, `:611-624`); split-eq packed kernel
  `multilinear-util/src/split_eq/packed_kernel.rs`.
- **leanMultisig**
  ([github.com/leanEthereum/leanMultisig](https://github.com/leanEthereum/leanMultisig),
  main): field crate
  `crates/backend/field/src/packed/packed_traits.rs` (`PackedField` `:220`,
  `PackedFieldExtension` `:329`); sum-check `crates/backend/sumcheck/src/prove.rs`
  (round loop `:117-144`), `sc_computation.rs` (packed univariate `:428-474`, fold+compute
  `:504-553`, `packing_unpack_sum` `:117-120`, split-eq `:557-639`); `split_eq.rs:5-103`;
  storage-permutation design `crates/sub_protocols/src/air_sumcheck.rs:9-30`,
  `quotient_gkr/mod.rs:19-26`; `pack` reinterpretation `mle_group_ref.rs:81-92`.
- **PR #142** (`quang/plonky3-field-microbench`): `specs/cross-repo-field-microbench.md`
  (numbers `:216-243`, deferred `mul_add` for EOR `:252-257`), `bench-data/field-microbench.md`,
  `crates/akita-pcs/benches/field_arith/kernel.rs:8-57` (`packed_macc`).
- **akita-field packed** (this branch): traits `crates/akita-field/src/fields/packed.rs`
  (`PackedValue`, `PackedField`, `HasPacking`, `NoPacking` `:273`, arch select `:412-441`);
  packed extension `crates/akita-field/src/fields/packed_ext.rs` (`PackedRingSubfieldFp4`
  `:558-637`, `HasPacking` wiring `:702-710`, layout doc `:1-6`).
- **D1 microbench**: `specs/packed-accumulator-microbench.rs` (standalone `rustc`; the
  reduction-frequency-vs-lane-width numbers in D1; re-run on each target arch).
- **Companion spec**: `specs/eor-streamed-prover.md` (the algorithm this packs).
