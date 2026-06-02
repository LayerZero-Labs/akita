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

### D1 — Packed unreduced accumulator (DECIDED: build it)

Akita's EOR/sum-check speed rests on the **wide scalar** `HasUnreducedOps::ProductAccum`
(defer modular reduction across a sum, e.g. `MulBaseUnreduced::mul_base_to_product_accum`
added in PR #136). Plonky3/leanMultisig get their speed from Montgomery + SIMD, **not**
delayed reduction — so out of the box the two strategies do not compose: a packed
`add_constant_product` needs a **packed unreduced accumulator** (a lane-wise wide type).

**Decision (locked): build the packed unreduced accumulator and keep both wins** — SIMD
lanes *and* deferred reduction. This is the end-state; do **not** settle for a packed path
that drops delayed reduction. Concretely:

- Add a **`PackedHasUnreducedOps`** with a lane-wise wide `ProductAccum` (an arch-native
  wide vector, transposed the same way `PackedRingSubfieldFp4` packs the reduced element),
  plus packed `mul_base_to_product_accum` / `add_constant_product` / `add_quadratic_product`
  that honor the same `DELAYED_PRODUCT_SUM_IS_EXACT` bound as the scalar path. One
  horizontal reduce per round turns the packed accumulator into the scalar round message.
- This is the highest-payoff piece: it makes the *accumulate* (not just the fold)
  lane-parallel while **preserving the deferred-reduction win that makes akita's scalar EOR
  fast in the first place**. The pseudo-Mersenne reductions akita already vectorizes
  (`mul_pmersenne31_vec`, PR #142) are the building blocks for the packed wide reduce.

**Sequencing (de-risk; not a re-litigation of the decision):** land the packed *fold*
first (Slice 1) — the linear interpolation `a + r·(b−a)` needs only packed
add/sub/mul-by-broadcast and **no** accumulator, so it captures the fold-dominated bulk of
the measured win immediately and validates the packed-table plumbing. Then land the packed
accumulator (Slice 2) for the univariate accumulate. Packing the fold first is a stepping
stone *toward* the packed accumulator, not an alternative to it.

**Rejected:** an eager per-lane reduce on the packed path (reduce every product instead of
deferring). Simpler, but it throws away the delayed-reduction win akita already depends on;
the packed unreduced accumulator is strictly better and is the committed path.

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
  the scalar `DELAYED_PRODUCT_SUM_IS_EXACT` bound — the committed D1 end-state.
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
- [ ] Packed unreduced accumulator (`PackedHasUnreducedOps`) implemented and exact; EOR
  dense accumulate + factor fold + partials run packed through it.
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
- **Companion spec**: `specs/eor-streamed-prover.md` (the algorithm this packs).
