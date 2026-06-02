# Spec: EOR streamed prover (stream witness `g`, stage the transparent weight `A_η`, relax the eq-table cap)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao (spec) → hand-off for implementation |
| Created     | 2026-06-02                     |
| Status      | ready for implementation       |
| PR          | follow-on to [#136](https://github.com/LayerZero-Labs/akita/pull/136) |

## Summary

The root and recursive extension-opening-reduction (EOR) prover proves the
degree-two sum-check for `Σ_x g(x)·A_η(x)`, where `g` is the degree-`d_ext`
extension packing of a base-field witness `f` (`g(w) = Σ_{v<d_ext} f(v,w)·γ_v`)
and `A_η` is the transparent FRI-Binius tensor-equality weight. Both `g` and
`A_η` are currently **materialized as full `2^num_vars` tables** on the paths
that dominate small-field prove time, and the `A_η` materialization also trips a
1 GiB eq-table allocation cap.

This spec is the follow-on to PR #136 (`specs/eor-sumcheck-prover-acceleration.md`).
It is **one spec with two coupled halves of a single idea** — never materialize
the big tables; stream/stage from the structured inputs — plus a cap fix:

1. **Witness side (`g`):** stream `g` from the base-field slots `f(v,·)` so the
   fold never allocates the full packed `Vec<C>`, and pack at most once across the
   EOR sum-check and the ring-switch transform (today dense packs `g` twice).
2. **Factor side (`A_η`):** make the cutoff **budget-driven against a RAM-scale
   budget** (replacing the fixed `materialize_at = 12`). When the flat `2^tail`
   factor fits the budget — the common case at realistic `num_vars` (≈ through nv32-
   33 on 64 GB) — the cutoff is **`m = 0`: materialize and use the dense / flat
   O(1) path** (no prefix-suffix). Only when the flat table exceeds the budget
   (≈ nv34+ on 64 GB) does the single `d_ext`-term prefix-suffix cutoff kick in, so
   no path materializes a `width × 2^suffix` (or flat `2^tail`) table over budget at
   any `num_vars`. The recursive `O(√)`-space prover is an escape hatch, not the
   default (see "Operating point").
3. **Cap:** retarget `MAX_MATERIALIZED_EQ_TABLE_BYTES` (`eq_poly.rs:25`) to a
   verifier-only allocation ceiling so it no longer guards the wrong quantity, no
   longer pins the prover to a bad operating point, and no longer hard-fails valid
   prover shapes at large `num_vars`.

The math, notation, and the streamed staged-prover construction are taken directly
from the project writeup: `Research/lattice-jolt/sections/akita/b_implementation_details.tex`,
paragraphs "Dense packing versus the intended streamed path", "Tensor weight as a
prefix-suffix table", "Multilinear-extension caveat", "Small-space staged prover",
and "Arithmetic tradeoff", building on the prefix-suffix inner-product method of
`streamingjolt` (ePrint 2025/611, Appendix A) and the eq-poly optimizations of
`eqpolyeprint` (ePrint 2024/1210).

The streaming primitive (`MulBaseUnreduced::mul_base_to_product_accum`), the
column-source pattern (`akita_types::TensorColumnSource` / `FlatColumnSource` /
`DenseColumnSource`), and the single-cutoff lazy factor (`TensorEqualityFactor`)
all landed in PR #136 and are reused here.

### Honest framing up front (read before scoping)

The writeup's "Arithmetic tradeoff" paragraph is explicit, and this spec inherits
it: **materialize-vs-stream is a memory-hierarchy and arithmetic-reuse decision,
not a protocol distinction, and not a guaranteed speedup.** The staged/streamed
prover does `O(C·d_ext·N)` field work versus `O(N)` for full materialization — it
trades *more* loop-level work for *less* space (`O(C·d_ext·N^{1/C})`). So:

- The **witness-side** streaming is a genuine time *and* memory win, because it
  removes a redundant full pack and fuses round 0 into the fold.
- The **factor side** has two regimes. When the flat `A_η` fits the RAM budget
  (`m = 0`, ≈ through nv32-33 on 64 GB: 16 GiB at nv32), materializing it gives the
  one-hot path **O(1) factor reads → dense-like univariate**, collapsing the
  plateau; the counterweight is the full-factor fold (bandwidth-bound when RAM- but
  not cache-resident), so net time is the writeup's memory-hierarchy tradeoff, not a
  guaranteed speedup. When the flat table exceeds the budget (`m > 0`, only nv34+ on
  64 GB), the single cutoff is a **memory / robustness / cap** win, **not** a
  field-work reduction: those `m` prefix rounds pay the irreducible `O(d_ext)`-per-
  query plateau. So: do not promise dense-like univariate *from staging* (`m > 0`),
  but `m = 0` (the default at realistic nv on this box) does deliver it.

### Operating point: one cutoff by default, not √-space

**Do not blindly run the full `C`-stage `O(√)`-space prover.** The default is the
**lightweight single-cutoff** path:

- **Witness `g`:** decompose-and-stream **at round 0 only**, then continue with
  ordinary per-round folding on the half-size folded table. After round 0 the
  witness is no longer a base packing, so there is nothing to stage; one round is
  sufficient (and for one-hot the witness is already sparse — a non-issue).
- **Factor `A_η`:** the cutoff `m` is chosen from a **real memory budget (a
  fraction of available RAM), not the 1 GiB cap.** When the flat `A_η`
  (`2^{tail} × sizeof(C)`) fits the budget, **`m = 0`: materialize it outright and
  use the dense / `SparseFactor::Dense` fast path** (O(1) factor reads, no lazy
  query). Only when the flat table exceeds the budget do we keep the prefix-suffix
  form for a *single cutoff* of `m` early rounds, then materialize the residual once
  and fold per-round. This is `C = 1`, never recursive staging.

**Concrete budget (fp32_d32, `C = 16 B`):** flat `A_η` = `2^{tail} × 16 B`, so
16 GiB at nv32 (tail 30), 64 GiB at nv34 (tail 32). On a **64 GB** machine the
factor fits RAM with headroom through **≈ nv32-33** ⇒ `m = 0` is the default there
(materialize, fast O(1) queries — this is also what removes the one-hot univariate
plateau). A single small cutoff only starts to matter at **≈ nv34+**, where the
flat table approaches whole-RAM. The 1 GiB cap that previously forced `m ≈ 12` at
nv32 was the artificial blocker, not the table size.

Two distinct axes (don't conflate, as an earlier draft did):
- **Fits RAM = feasibility.** Sets the hard `m = 0` boundary (≈ nv33 on 64 GB).
- **Fits cache = speed.** The writeup's "Arithmetic tradeoff" is about *cache*:
  materializing wins cleanly when the table is cache-resident (one mul per tail
  point). A 16 GiB factor is RAM-resident but not cache-resident, so its full fold
  is memory-bandwidth-bound — a perf wash vs the lazy query, not a feasibility
  issue.

The full `C`-stage prover (`O(C·d_ext·N^{1/C})` space, streamingjolt App A) is an
**escape hatch only**, for tails so large that even one cutoff's residual will not
fit RAM for any acceptable `m` (well beyond nv34 on 64 GB). It is documented for
completeness; it is **not** the default and should not be built unless a concrete
nv target forces it.

## Smoking gun (measured)

Profile harness, PR #136's per-round `sumcheck_round_{univariate,fold}` spans,
native release, all cores, `AKITA_PROFILE_LOG=debug`.

### Witness side — dense packs `g` twice

`dense_fp32_d32 nv26`: `root_extension_dense_terms` and
`root_extension_transform_polys` each call `tensor_packed_extension_evals::<C>()`
on the same polys. That function takes **no point** (`dense.rs:324`, default
`lib.rs:331`) — it is a pure function of `(poly, C)`, so the two calls are
byte-identical full `2^num_vars` extension packings. Together they dominate
`prove_prepared_root_extension_opening_reduction` and
`root_extension_transform_polys`.

### Factor side — one-hot univariate plateau

`onehot_fp32_d32 nv32` vs `dense_fp32_d32 nv26` (same EOR step, 6 invocations):

| span                                   | onehot nv32 | dense nv26 |
|----------------------------------------|-------------|------------|
| `extension_opening_reduction_sumcheck` | **1683 ms** | 144 ms     |
| `sumcheck_round_univariate` (total)    | **1252 ms** | 52 ms      |
| `sumcheck_round_fold` (total)          | 719 ms      | 293 ms     |

The comparison is **confounded by `num_vars`** (a dense nv32 factor is
`2^30×16 B = 16 GiB`, infeasible — the reason one-hot goes sparse), so the point
is the *shape*, not "onehot slow vs dense". Per-round univariate µs for the
largest one-hot EOR sum-check (tail = 30):

```
round  table_len   uni_us
  0     2^30        257819   ┐
  1     2^29        111890   │ plateau: support ~constant at 2^24 while the table
  2     2^28        126131   │ halves, because folding low bits of a 2^-6-dense
  3     2^27        193773   │ witness rarely collides until density reaches 1
  4     2^26        171117   │
  5     2^25        174399   │
  6     2^24         96056   ┘
  7     2^23         30333   ← density ~1; entries now halve per round
  …                          (round ≥12: lazy factor materializes; univariate ~0)
```

**Key fact:** the one-hot witness is sparse — `tensor_packed_sparse_witness`
(`onehot/poly.rs:262-290`) emits one entry per hot chunk; the trace records
`chunks = 2^24`, so support is `2^24` in a `2^30` table (**density `2^-6`**).
`accumulate_round_with_factor` (`sparse.rs:386-425`) iterates only the support and
queries the `O(d_ext)` lazy factor per occupied pair. Folding low bits of a
`2^-6`-dense witness does not shrink the support until density hits 1 (~round 6),
so the univariate is `~6 rounds × 2^24 × O(d_ext) ≈ 1.2 s`. This is the irreducible
structured-weight cost (see "Honest framing").

### Memory / cap pathology

An older trace (`materialize_at = 4`) showed `suffix_tables = width × 2^26 = 4 GiB`
and **9.8 s (83% of an 11.9 s prove)** in `TensorEqualityFactor::new`. Current
`materialize_at = 12` shrinks it to `4 × 2^18 = 16 MB`, but the blow-up returns at
larger `num_vars`, and the cap turns it into a hard failure (P-F2 below).

## Background: the structured EOR inner product

(Notation from the writeup, §"Implemented prover paths" and Appendix B.)

`K = F_q` base, `L = F_{q^{2^κ}}` challenge field, `d_ext = [L:K] = 2^κ`. The
packed witness is **coordinate** packing in the implementation basis `(γ_v)`:
`g(w) = Σ_v f(v,w)·γ_v`. Small balanced representatives of `f` are the `K`-
coordinates of `g(w)`; after packing the dense prover sees one opaque `L` element,
losing that structure. The row-batching functional is
`τ_η(Σ_u a_u γ_u) := Σ_u η_u a_u` with `η_u = eq(u,η)` (`2_preliminaries.tex:487`),
and the transparent weight is `A_η(w) = Σ_u η_u·coord_u(eq(r_tail, w))`.

### The `d_ext`-term prefix-suffix weight (Eq. `eor-prefix-suffix-weight`)

Fix the coordinate functionals `λ_t : L → K` (`x = Σ_t λ_t(x)·γ_t`). For any
cutoff `w = (y, z)` of the tail, `eq(r_tail, w) = E_pre(y)·E_suf(z)` in `L`, and on
**Boolean** inputs:

```
A_η(y, z) = τ_η(E_pre(y)·E_suf(z))
          = Σ_{t<d_ext}  λ_t(E_pre(y)) · τ_η(γ_t · E_suf(z)).
```

This is a **`d_ext`-term** prefix-suffix representation at every cutoff. The naive
multiplication-table expansion is `d_ext²` terms; expanding only the prefix factor
in the implementation basis collapses it to `d_ext` terms. (This is why the
factor-side design must use the `d_ext`-term form, **not** a `width×width = d_ext²`
bilinear matrix.)

### Multilinear-extension caveat (correctness-critical)

Eq. `eor-prefix-suffix-weight` is a **Boolean-table identity**. During sum-check
the prefix/suffix tables are evaluated by MLE. **One must not** evaluate
`λ_t(E_pre(r))` by applying the coordinate map `λ_t` to an extension-valued folded
eq value at non-Boolean challenges, because `λ_t` is only `K`-linear, not
`L`-linear. Operationally, the prover extracts coordinates on the **Boolean**
prefix table and **folds those `K`-coordinate tables** under the challenges, exactly
as ordinary sum-check tables:

```
P_t(y) := MLE_{b,y}[ λ_t(E_pre(b,y)) ](r_prev, y).
```

This is precisely what PR #136's `TensorEqualityFactor` already does via its
`transitions` + `prefix_state`/`low_states` (it folds the Boolean coordinate
tables); the suffix side `suffix_tables[t][z] = τ_η(γ_t·E_suf(z))` is the same
`Q`-style suffix. So the **current lazy factor is the correct single-cutoff
instance of the paper's construction.** The work below generalizes the cutoff, it
does not re-derive the weight.

### The streamed staged prover (streamingjolt Appendix A)

With already-bound prefix `r_prev`, stage variables `y`, suffix variables `x`:

```
P_t(y) = MLE[ λ_t(E_pre(b,y)) ](r_prev, y)                              (prefix coords, folded)
Q_t(y) = Σ_x  g̃(r_prev, y, x) · τ_η(γ_t·E_suf(x))                       (packed-data form)
       = Σ_x Σ_v  f̃_v(r_prev, y, x) · γ_v · τ_η(γ_t·E_suf(x))           (fully streamed form)
```

and each stage's round messages are the ordinary degree-two sum-check messages for
`Σ_y Σ_t P_t(y)·Q_t(y)`. The fully streamed `Q_t` avoids storing `g`, exposes the
small balanced `f_v`, and with `C` stages over `N = 2^m` uses
**`O(C·d_ext·N^{1/C})` working space** + streamed witness access and
**`O(C·d_ext·N)` field work**. The current lazy factor is the `C`-stage path
truncated to one cutoff (prefix-coordinate state + one suffix table for a bounded
number of early rounds, then materialize the rest).

### Arithmetic tradeoff (the operating-point decision)

From the writeup: if the combined factor table fits the relevant cache level,
**materializing `A_η`** saves repeated prefix-suffix arithmetic and uses one
multiply by a packed witness value per tail point. If the packed witness or factor
table is too large, or the small balanced structure of `f` matters, **streaming**
wins despite more loop work. The right end-state is **adaptive**: materialize when
the combined table fits cache; otherwise stage. This is a memory-hierarchy choice,
byte-identical either way.

## Problems (precise)

- **P-W1 — `g` packed twice, fully.** `root_extension_dense_terms`
  (`root_extension.rs:347-358`) and `root_extension_transform_polys`
  (`root_fold.rs:324-329`, `:699-705`) both materialize the same full `g`
  (`tensor_packed_extension_evals` / `tensor_packed_extension_root_poly`).
  Recursive levels pack fully at `recursive.rs:645-653`.
- **P-F1 — `width × 2^suffix` suffix blow-up.** `TensorEqualityFactor::new`
  (`sparse.rs:530-546`) builds `suffix_tables = width × 2^(tail−materialize_at)`.
  At `materialize_at = 12`, `tail = 30`: 16 MB; at `tail = 38` (≈ nv40): 4 GiB.
- **P-F2 — the cap guards the wrong quantity and hard-fails at large nv.**
  `MAX_MATERIALIZED_EQ_TABLE_BYTES = 1<<30` lives only in
  `EqPolynomial::check_element_budget` (`eq_poly.rs:42-53`). The akita-types
  `checked_table_len` that the factor calls (`sparse.rs:505-506`,
  `extension_opening_reduction.rs:792-799`) has **no byte budget**. So the cap
  bounds the scalar `suffix_eq`, not the real `width × suffix_eq` allocation (a
  1 GiB cap silently permits 4 GiB), forces `materialize_at ≥ tail−26`, and makes
  `EqPolynomial::evals` **return `Err`** once `tail−12 > 26` (`num_vars ≳ 41` for
  onehot fp32_d32) — a hard prover failure. It is an allocation-safety ceiling, not
  a correctness requirement.
- **P-F3 — `materialize_at` U-curve + `low_states` churn (irreducible floor).**
  Small `materialize_at` → big flat/suffix tables (the 4 GiB corner); large
  `materialize_at` → many `O(d_ext)` lazy rounds (the 1.2 s plateau) with a
  `rebuild_low_states` (`sparse.rs:643-664`) per fold. Staging fixes the memory
  corner and the churn, but the `O(d_ext)`-per-query field work over the support is
  the floor (see "Honest framing").

## Intent

### Goal

Route the root and recursive EOR prover through the **lightweight single-cutoff**
streamed path (see "Operating point"):

1. **Witness:** stream `g` from `f(v,·)` for **round 0 only**, fold, then continue
   per-round on the half-size folded table; pack at most once across the EOR
   sum-check and the transform.
2. **Factor:** keep `A_η` in the `d_ext`-term prefix-suffix form for a **single
   bounded prefix block** of `m` early rounds, then materialize the residual once
   and fold per-round. Choose `m` from a cache/memory budget so the residual fits;
   keep the MLE caveat (fold Boolean coordinate tables). This is `C = 1`, not
   recursive staging.
3. **Budget-driven cutoff, not a hard-coded constant.** Replace the fixed
   `SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS = 12` with `m` derived from `(tail, d_ext,
   budget)` so the prefix-block suffix `d_ext × 2^{tail−m}` and the materialized
   residual `2^{tail−m}` both fit the budget at every `num_vars` (including nv40,
   where the constant currently hard-fails). At small tail, `m = 0` (materialize
   immediately).
4. **Cap:** verifier-only ceiling; prover never depends on it.
5. **Escape hatch (not default):** if no acceptable `m` makes one cutoff's residual
   fit at an extreme nv target, fall back to the `C`-stage prover (streamingjolt
   App A). Out of scope unless a concrete target forces it.

### Invariants

- **Never materialize the big tables.** No fold-side path allocates a `Vec<C>` of
  length `2^packed_num_vars`; no factor path allocates the flat `2^tail` weight or a
  `width × 2^{tail−m}` suffix that exceeds the budget. Fold-side peak witness
  allocation is the half-size (`2^{n-1}`) folded table; factor peak is the
  budget-bounded single-cutoff residual `d_ext × 2^{tail−m} ≤ budget` (the
  `O(C·d_ext·N^{1/C})` bound applies only on the escape-hatch `C`-stage path).
- **At most one packing of `g`** across the EOR sum-check and the transform.
- **MLE caveat respected.** Prefix coordinates are extracted on the Boolean table
  and folded as `K`-coordinate tables; `λ_t` is never applied to an `L`-folded eq
  value at a non-Boolean challenge. (Verified by the flat-reference oracle below.)
- **Byte-identical proofs and transcript** for every benchmarked mode/`num_vars`
  (proof bytes, `proof_size_bytes`, transcript bytes), and the proof verifies. The
  weight equals the current factor field-exactly (Eq. `eor-prefix-suffix-weight` is
  an identity), and the streamed round-0 witness fold equals the materialized
  round-0 fold field-exactly.
- **Delayed-reduction exactness reused, not re-derived.** Witness×base products use
  `mul_base_to_product_accum`; factor products honor
  `DELAYED_PRODUCT_SUM_IS_EXACT` exactly as `factor_pair` / `accumulate_dense_round`
  do today (`sparse.rs:761-784`, `dense.rs:25-30`).
- **Cap not prover-binding.** `onehot_fp32_d32` proves + verifies at nv up to at
  least 40 (where it currently hard-fails) with bounded memory; verifier-reachable
  `EqPolynomial::evals` stays bounded (no unbounded verifier allocation).
- **Verifier untouched.** Prover-only change. The verifier-side factor eval
  `tensor_equality_factor_eval_at_point` (`extension_opening_reduction.rs:504`) and
  all transcript/serialization are unchanged.

### Non-Goals

- No protocol, soundness, extension-degree, schedule, or transcript change.
- No new field trait (`MulBaseUnreduced` already exists, `lift.rs`).
- **No claim of dense-like univariate** from the factor side; the `O(d_ext)`-per-
  query floor stands. A larger one-hot time win is a separate lever (one-hot
  unit-value round-0 kernel — see "Future / optional").
- No removal of the verifier-reachable eq-table allocation ceiling.

## Evaluation

### Acceptance Criteria

- [ ] `EorWitnessSource<F, C>` (fold-side streaming from `f`) in `akita-prover`;
  dense (recursive `recursive_witness_base_evals`, root `DensePoly::coeffs`) and
  one-hot source impls; round-0 streaming integrated into the EOR term/prover;
  `recursive.rs:645-653` and `root_extension.rs:347-358` re-pointed.
- [ ] `g` packed at most once: `root_fold.rs:324-329`/`:699-705` pull from the
  source; no remaining duplicate `tensor_packed_extension_evals` production caller.
- [ ] Budget-driven single-cutoff factor: keep `TensorEqualityFactor`'s
  `d_ext`-term `P_t` + suffix shape and MLE-correct Boolean-coordinate folding;
  replace the fixed `materialize_at` with `m = m(tail, d_ext, budget)` so the
  prefix-block suffix `d_ext × 2^{tail−m}` and residual `2^{tail−m}` fit the budget
  at every nv. No recursive sqrt-space staging. `factor_at_index`, `factor_pair`,
  `fold_in_place`, `claim`, `len`, and the `len == 1` final singleton
  (`sparse.rs:843-856`) intact.
- [ ] `SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS` (`mod.rs:26`) and the `materialize_at`
  plumbing (`root_extension.rs:301-334`, `new_sparse_tensor_factor` `sparse.rs:956`,
  bench `:128-133`, test `:495-501`) removed or replaced by the staged constructor.
- [ ] Cap (`eq_poly.rs:25`) verifier-only; no prover factor path depends on it;
  verifier-reachable `EqPolynomial::evals` still bounded.
- [ ] New tests pass per field family (fp32 `RingSubfieldFp4`, fp16
  `RingSubfieldFp8`, fp64 `Fp2`, fp128 identity):
  `streamed_round0_matches_materialized`, `staged_factor_matches_flat_reference`,
  `prefix_coord_fold_matches_boolean_mle` (the MLE-caveat guard),
  `streamed_transform_eval_matches_packed`.
- [ ] Large-nv robustness: `onehot_fp32_d32` nv40 prove + verify (currently
  hard-fails on the cap), peak factor RSS bounded by the single-cutoff budget
  (`d_ext × 2^{tail−m} ≤ budget`).
- [ ] Byte-identical `proof_size_bytes` for `onehot_fp32_d32`, `onehot_fp16_d32`,
  `onehot_fp64_d32`, `dense_fp32_d32`; each `exit_code 0`, proof verifies.
- [ ] `cargo fmt -q`; `cargo clippy --all --all-targets -- -D warnings` (±`zk`);
  `cargo test -p akita-prover --test extension_opening_reduction`.

### Testing Strategy

The byte-identicality guard is the existing EOR round-by-round equality suite
(`sparse_tensor_factor_matches_dense_factor_rounds` + per-prime), the
`dense_tensor_opening_methods_match_flat_reference` oracle, and the profile
proof-size gate. Keep the current materialized packings and the flat
`tensor_equality_factor_evals` as **oracles**:

- `streamed_round0_matches_materialized`: streamed source vs
  `tensor_packed_extension_evals` + dense round-0 `fused_fold_and_accumulate`;
  assert equal `(constant, quadratic)` and equal folded half-size table.
- `staged_factor_matches_flat_reference`: assert `factor_at_index(i)` equals the
  flat `tensor_equality_factor_evals` table at every `i`, and `factor_pair`/fold
  match across a full challenge sequence, for several cutoffs `m` (incl. `m = 0`
  immediate-materialize and a large `m`) and all `d_ext`.
- `prefix_coord_fold_matches_boolean_mle`: assert the folded `P_t` equals the MLE of
  the Boolean coordinate table at the challenge point — the direct guard against the
  forbidden "apply `λ_t` to the `L`-folded eq" shortcut.
- `streamed_transform_eval_matches_packed`: assert the transform consumer equals
  `tensor_packed_extension_root_poly().evaluate_and_fold(...)`.

Run release + native, all cores; also `--features zk` (the ZK prove loop in
`drivers/standard.rs` shares the round structure and the EOR pads path).

### Performance

- **Witness side:** drop one full `2^num_vars` extension packing (pack-once), fuse
  the round-0 packing into the round-0 fold; lower peak RSS, fewer memory writes;
  measurable prove-time + RSS improvement on the EOR-dominated small-field modes,
  no regression on `fp128`.
- **Factor side:** with the budget-driven cutoff, the materialized residual (and
  the prefix-block suffix when `m > 0`) stays `≤ budget` at every nv (vs the fixed
  `width × 2^(tail−12)`); the 9.8 s `TensorEqualityFactor::new` corner is
  structurally impossible. Two regimes:
  - **`m = 0` (flat factor fits the RAM budget — through ≈ nv32-33 on 64 GB).**
    The one-hot path queries a flat `SparseFactor::Dense` table with **O(1) reads**,
    so the univariate **collapses to dense-like** (the 1.2 s plateau disappears).
    The counterweight is the full-factor fold (`2^{tail}`/round, memory-bandwidth-
    bound when the table is RAM- but not cache-resident, as at nv32's 16 GiB), so
    net wall-clock is a memory-hierarchy tradeoff (writeup "Arithmetic tradeoff"):
    a clear win when the table is cache-resident, ~neutral when only RAM-resident.
  - **`m > 0` (flat factor exceeds the budget — only ≈ nv34+ on 64 GB).** A single
    small cutoff keeps the residual within budget; the prefix block pays the
    `O(d_ext)`-per-query floor for `m` rounds, then rejoins the flat O(1) path.
- No proof-size, schedule, or transcript effect (byte-identical).

Validate with the profile harness (`sumcheck_round_{univariate,fold}` per-round
spans) on `onehot_fp32_d32` / `onehot_fp16_d32` / `dense_fp32_d32` at nv30/32 plus
the new nv40 robustness run.

## Design

### Current state (precise)

```
prove_root_fold_with_params / prove_recursive_*_root_fold      (root_fold.rs:288, :663)
├─ prove_prepared_root_extension_opening_reduction             (root_extension.rs:174)
│   ├─ sparse (onehot)   root_extension.rs:267-341
│   │     tensor_packed_extension_sparse_linear_combination     (onehot/ops.rs:251)
│   │     lazy_rounds = min(tail, MAX_LAZY=12)                   (root_extension.rs:301)
│   │     ExtensionOpeningReductionTerm::new_sparse_tensor_factor (sparse.rs:956)
│   │       → TensorEqualityFactor::new(tail, eta, materialize_at) (sparse.rs:486)
│   │           suffix_eq = EqPolynomial::evals(tail[m_at..])     (sparse.rs:530)  capped scalar
│   │           suffix_tables = width × suffix_eq (projected)     (sparse.rs:531-546) × width blow-up
│   │           transitions + rebuild_low_states                  (P_t Boolean-coord fold; MLE-correct)
│   ├─ dense   root_extension.rs:342-361
│   │     per claim: tensor_packed_extension_evals::<C>()         (dense.rs:324)  FULL g
│   │              + tensor_equality_factor_evals(tail, eta)      (full A_η table)
│   └─ extension_opening_reduction_sumcheck                      (root_extension.rs:368)
├─ protocol point ← reduction.rho                               (root_fold.rs:307-322 / :682-697)
└─ root_extension_transform_polys                               (root_fold.rs:324-329 / :699-705)
      per poly: tensor_packed_extension_root_poly::<C>()  → tensor_packed_extension_evals (SAME g)
```

### Target: the lightweight single-cutoff streamed prover

This is the `C = 1` instance of the writeup's staged prover (a single prefix block,
then materialize-and-fold), **not** the recursive `C`-stage `Q_t` construction. Two
cooperating pieces:

- **`EorWitnessSource<F, C>` (fold side, witness — round 0 only).** Streams
  `g[i] = Σ_v f(v,i)·γ_v` once, fused into the round-0 fold (`even + r0·(odd−even)`)
  and the `(constant, quadratic)` accumulation via `mul_base_to_product_accum`;
  outputs the half-size `Vec<C>` and the term **collapses to
  `ExtensionOpeningTables::Dense` for rounds ≥1** (ordinary per-round folding).
  Mirrors `fused_fold_and_accumulate` (`dense.rs:90-191`) reading `d_ext`-wise base
  slices. Dense source reuses `DenseColumnSource`-style access; one-hot source wraps
  the existing non-materializing `new_sparse[_tensor_factor]` construction; both also
  serve the transform via `evaluate_packed_at_multiplier_point` (kills the second
  pack, `root_fold.rs:324-329`/`:699-705`).
- **Single-cutoff prefix-suffix factor (factor side, `A_η`).** Keep
  `TensorEqualityFactor`'s shape — `d_ext` Boolean-coordinate-folded prefix tables
  `P_t` + one suffix factor `suffix_tables[t][z] = τ_η(γ_t·E_suf(z))` — for a prefix
  block of `m` rounds, then materialize the residual (`2^{tail−m}`) and rejoin the
  dense fast path (`fused_fold_and_accumulate`). The only changes vs PR #136 are:
  (1) choose `m` from a **budget** (replacing the fixed
  `SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS = 12`) so both `d_ext × 2^{tail−m}` (prefix
  block) and `2^{tail−m}` (residual) fit at every nv, including nv40; (2) decouple
  from the eq-table cap. The `factor_pair` query stays `O(d_ext)` (paper's
  `d_ext`-term form), branch-light, using `mul_base`. No recursive suffix staging,
  no `O(√)`-space machinery.

Escape hatch (only if no `m` fits at an extreme nv target): recurse the suffix into
`C` stages (the writeup's fully streamed `Q_t`, streamingjolt App A), giving
`O(C·d_ext·N^{1/C})` space at `O(C·d_ext·N)` work. Build this **only** when forced.

### MLE caveat (implementation rule)

Always extract `λ_t` on the **Boolean** prefix table and fold the resulting
`K`-coordinate tables `P_t` under the challenges (as PR #136's `transitions`
already do). Never compute `coords(folded_L_eq)` at a non-Boolean challenge. The
`prefix_coord_fold_matches_boolean_mle` test is the guard.

### Cap relaxation

Keep the cap on **verifier-reachable** `EqPolynomial::evals` (the verifier replays
eq tables and must not OOM on a malformed proof), but (a) raise it to a realistic
ceiling — >1 GiB eq tables are legitimate and break nothing; the cap bounds
*unbounded* allocation, not large-but-finite tables — and (b) ensure no prover
factor path depends on it once staging lands (the lazy `suffix_eq` /
`checked_table_len` calls are deleted with the single-cutoff factor). A residual
prover `EqPolynomial::evals` for a per-stage suffix of size `2^{stage}` is `≪` any
ceiling and is fine.

### Memory tradeoff for the transform consumer

The transform runs *after* the EOR returns `rho`, so it cannot reuse the
folded-away table. Recommended: **stream the transform too**
(`evaluate_packed_at_multiplier_point`), so neither consumer holds a full `g`.
Fallback (state in worklog if chosen): pack once before the EOR and hold the full
table across the sum-check (higher peak RSS, removes the redundant pack only).

### Packing readiness (coordinate with `specs/packed-sumcheck.md`)

This spec is written in **scalar** terms (`Vec<E>`, `E::fold_one`,
`add_constant_product`), but the loops it touches — the `EorWitnessSource` round-0
fold, `fused_fold_and_accumulate`, and the budget-driven factor fold — are **exactly
the loops the packed-SIMD workstream rewrites** (`specs/packed-sumcheck.md`). Packing
is a field-exact representation change orthogonal to the streaming algorithm here, but
to avoid building scalar loops that get rewritten immediately after, implement the new
abstractions in **packing-ready shape**:

- Define `EorWitnessSource` and the fold/accumulate helpers over a **slice/word view**
  that admits a packed backing (`&[E]` ↔ `&[E::Packing]` with a scalar tail), not a
  hard-coded scalar `Vec<E>` element-at-a-time loop. The default impl can be the scalar
  (`NoPacking`, `WIDTH = 1`) path, so this is a *shape* constraint, not extra work now.
- Keep the round-0 streamed fold expressed as the linear interpolation
  `even + r₀·(odd − even)` over contiguous lanes — that maps directly onto the packed
  fold primitive (`packed-sumcheck.md` D1(b)) with no `ProductAccum` involvement.
- The delayed-reduction primitive used here (`mul_base_to_product_accum`) is the one the
  packing spec must reconcile with SIMD (`packed-sumcheck.md` D1); do not assume the
  scalar `ProductAccum` is the final shape on the packed path.

Net: the streaming algorithm lands first (scalar, byte-identical), and packing slots in
as a representation swap on the same abstractions rather than a second rewrite.

### Alternatives considered (and rejected)

- **`d_ext²` bilinear matrix `M` with split-eq `E_out`/`E_in` queried at folded
  challenges.** Rejected: it is the naive `d_ext²` expansion (the paper's
  `d_ext`-term form is strictly better), **and** it violates the MLE caveat — it
  applies the `K`-linear `λ_t` (via `M`) to `L`-folded eq values at non-Boolean
  challenges, which is not the MLE of the coordinate table. Use Boolean-coordinate
  folding (`P_t`) instead.
- **A persistent `Streamed` arm for all EOR rounds.** Rejected for the witness: only
  round 0 reads the full-size base packing; rounds ≥1 operate on the
  challenge-folded half-size table, which is no longer a base packing. The streamed
  state collapses to `Dense` after round 0.
- **Fixed `materialize_at = 12`.** Keep the single-cutoff shape but make `m`
  **budget-driven** so it scales with `tail`/`d_ext` and never hard-fails the cap.
  Not replaced by sqrt-space.
- **Full `C`-stage `O(√)`-space prover as the default.** Rejected: it does more
  field work (`O(C·d_ext·N)`) for space we do not need once `m` is budget-driven
  (one cutoff already keeps the residual `≤ budget` for realistic nv). Kept only as
  an escape hatch for extreme nv.

## Implementation Plan

Sliced for review; each slice keeps proofs byte-identical and is independently
testable.

### Slice 1 — Witness source, dense fold (recursive)
- `EorWitnessSource` (`extension_opening_reduction/witness_source.rs`);
  `fold_first_round` + `claim_with_factor` for the recursive dense source
  (`recursive_witness_base_evals`).
- Round-0 streaming integrated into the EOR term/prover (add a `Streamed` term
  state that collapses to `Dense` after round 0).
- Re-point `recursive.rs:645-653`; demote `tensor_packed_witness_evals` to oracle.
- Tests: `streamed_round0_matches_materialized` (recursive).

### Slice 2 — Witness source, dense + one-hot (root) and the transform
- Dense root source (`DensePoly::coeffs`, reuse `DenseColumnSource`); one-hot source
  wrapping `new_sparse[_tensor_factor]`.
- Re-point `root_extension.rs:347-358`; implement
  `evaluate_packed_at_multiplier_point`; re-point `root_fold.rs:324-329`/`:699-705`;
  remove the second pack.
- Tests: `streamed_round0_matches_materialized` (root) +
  `streamed_transform_eval_matches_packed`; assert no duplicate `g` pack.

### Slice 3 — Budget-driven single-cutoff factor (no sqrt-space)
- Keep `TensorEqualityFactor`'s single-cutoff `d_ext`-term shape; replace the fixed
  `SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS = 12` with `m = m(tail, d_ext, budget)` so
  the prefix-block suffix `d_ext × 2^{tail−m}` and the residual `2^{tail−m}` fit the
  budget at every nv (incl. nv40). Keep Boolean-coordinate folding (MLE caveat).
- Do **not** build the recursive `C`-stage suffix here; it is the escape hatch.
- Tests: `staged_factor_matches_flat_reference` (single cutoff, several `m`, all
  d_ext), `prefix_coord_fold_matches_boolean_mle`, full EOR suite, byte-identical
  proof-size on the four modes.

### Slice 4 — Cap relaxation + large-nv + gate
- Retarget `MAX_MATERIALIZED_EQ_TABLE_BYTES` to verifier-only; confirm no prover
  dependence; verifier bound intact.
- Add `onehot_fp32_d32` nv40 prove+verify robustness test (gate `#[ignore]` if too
  heavy for CI).
- Delete dead materialization helpers with no remaining production caller.
- Full gate: fmt, clippy (±`zk`), EOR tests, proof-size byte-equality, `cargo doc`,
  profile-harness sanity on nv30/32 + nv40.

Risks to resolve first:
- The round-0 fold quad-stride in `fused_fold_and_accumulate_with` (the `4*i`
  pattern) must map cleanly onto `d_ext`-wise base slices for all `d_ext ∈
  {1,2,4,8}`; prototype on the recursive dense source (Slice 1).
- The stage-boundary index split for the factor (`out = idx >> in_unbound`,
  `in = idx & mask`) must stay consistent when a fold crosses a stage boundary;
  verify against the flat oracle (`staged_factor_matches_flat_reference`) before
  deleting the single-cutoff factor.

## Future / optional (out of scope; note in worklog)

- **One-hot unit-value round-0 kernel.** Round-0 one-hot witness values are basis
  units (`coords[head]=1`, `onehot/poly.rs:285-287`), so `w·A_η = γ_head·A_η` is a
  cheap structured product, not a general ext-mul. A one-hot-specific round-0 kernel
  could cut the constant on the `2^24`-support plateau — the lever for an actual
  one-hot *time* win, independent of the factor representation.
- **Support density.** The `2^-6` density (plateau driver) is `onehot_k`-vs-`width`;
  reducing it is a witness-shape question.

## References

- Writeup (math + construction): `Research/lattice-jolt/sections/akita/b_implementation_details.tex`
  §§"Dense packing versus the intended streamed path", "Tensor weight as a
  prefix-suffix table" (Eq. `eor-prefix-suffix-weight`), "Multilinear-extension
  caveat", "Small-space staged prover", "Arithmetic tradeoff"; notation in
  `2_preliminaries.tex:404-540` (`sec:prelim-ext-opening`, `τ_η`, `η_u`).
- Prior art: `streamingjolt` (ePrint 2025/611, Appendix A, prefix-suffix
  inner-product / small-space staged prover); `eqpolyeprint` (ePrint 2024/1210,
  eq-poly sum-check optimizations).
- Parent spec (landed PR #136): `specs/eor-sumcheck-prover-acceleration.md`.
- Landed primitives: `MulBaseUnreduced::mul_base_to_product_accum`
  (`akita-field/src/fields/lift.rs`); `TensorColumnSource` / `FlatColumnSource`
  (`akita-types/src/extension_opening_reduction.rs`); `DenseColumnSource`
  (`akita-prover/src/backend/dense.rs`).
- Witness materialization to replace: `tensor_packed_witness_evals` (akita-types);
  `tensor_packed_extension_evals` (`dense.rs:324`, default `lib.rs:331`);
  `tensor_packed_extension_poly` (`lib.rs:406`); `tensor_packed_extension_root_poly`
  (`lib.rs:440`). Call sites: `recursive.rs:645-653`; `root_extension.rs:267-341`
  (sparse), `:342-361` (dense); `root_fold.rs:296,324-329`/`:671,699-705`; consumer
  `root_fold.rs:3,41-56`.
- Single-cutoff factor to generalize: `TensorEqualityFactor` (`sparse.rs:476-784`),
  `new` (`:486-559`), `suffix_tables` (`:530-546`), `transitions`/`rebuild_low_states`
  (`:561-664`), `factor_pair` (`:749-784`), `materialize_dense` (`:697-719`),
  `new_sparse_tensor_factor` (`:956-987`); `SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS`
  (`mod.rs:26`), call site `root_extension.rs:301-334`.
- Factor math: `project_tensor_factor_value` (`extension_opening_reduction.rs:437-459`),
  `tensor_equality_factor_evals` (`:468-491`), verifier-side
  `tensor_equality_factor_eval_at_point` (`:504`).
- Cap: `MAX_MATERIALIZED_EQ_TABLE_BYTES`, `check_element_budget`
  (`akita-algebra/src/eq_poly.rs:20-53`); no-budget akita-types `checked_table_len`
  (`extension_opening_reduction.rs:792-799`).
- Consumer + round structure: `accumulate_round_with_factor` (`sparse.rs:386-425`),
  `ExtensionOpeningTables` (`:806`), fold/claim/final arms (`:825-904`),
  `fused_fold_and_accumulate` (`dense.rs:90-191`), per-round spans
  `sumcheck_round_{univariate,fold}` (`drivers/standard.rs`).
- EOR construction: `ExtensionOpeningReductionProver::from_dense_tables`
  (`prover.rs:61`); `ExtensionOpeningReductionTerm::{new, new_sparse,
  new_sparse_tensor_factor}` (`sparse.rs:912/929/956`).
