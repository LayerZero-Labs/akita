# Phase D-full Design: Recursive `S` Opening + Tiered Commitments

**Date**: 2026-05-13
**Branch**: `feat/tensor-challenges` (post-security-baseline commit `9089d667`)
**Reference spec**: `Lattice_Jolt/sections/akita/5_fourth_root_verifier.tex` §§5.3-5.4
**Predecessors**: `specs/security_analysis.md` (security baseline; both MSIS and CWSS clear 128 bits)

This document specifies the protocol modifications and code changes for Phase D-full: replacing the per-level `SetupMatrixPolynomialView::mle(...)` closing oracle in `verify_setup_claim_reduction` with a recursive opening claim on the shared setup polynomial `S`, plus the tiered commitment design from book §5.4 that controls the cascade.

The doc is grounded in the existing code: every protocol modification cites the exact file:line that must change. The implementation plan in §6 is decomposed into independently shippable slices.

---

## 1 — Why this is needed (recap)

After the security baseline fix, the verifier's hot path inside `verify_setup_claim_reduction` is:

```127:131:crates/akita-verifier/src/protocol/setup_claim_reduction.rs
    let setup_view = setup
        .shared_matrix
        .setup_polynomial_view::<D>(row_count, max_stride);
    let setup_at_point = setup_view.mle(row_challenges, col_challenges, coeff_challenges)?;
```

`SetupMatrixPolynomialView::mle` walks the live `num_rows × num_cols` prefix of the shared matrix at every level (`crates/akita-types/src/layout/flat_matrix.rs:392-413`), which is `O(rows · cols · D)` per level — the same asymptotic order as the legacy fused stage-2 path. This is what keeps the per-level verifier work at `O(√N')` rather than `O(N'^{1/4})`. Phase D-full closes this gap.

Book §5.3-5.4 prescribes:
- A commitment `c_S` to `S` at setup time, bound via Ajtai (same machinery as witness commitments).
- The setup-claim-reduction sumcheck closes with a point claim `S(r_setup) = y_setup` rather than evaluating `S` directly.
- At the next recursive level, `S` enters as an additional polynomial alongside the folded witness, opened jointly via a multi-polynomial Hachi PCS step.
- A **tiered commitment** design controls the cascade: at level `L`, `S` is split into `k = f²` chunks committed under shared per-chunk matrices that are `1/f` the column width of the baseline; a tier-3 meta-commitment binds the collection of per-chunk commitments. Book §5.4 picks `f = 8`, `k = 64` as the production sweet spot.

---

## 2 — Existing architecture (what we have)

### 2.1 Single-poly recursive pipeline

`RecursiveProverState` carries one polynomial across recursion levels:

```38:50:crates/akita-prover/src/protocol/flow.rs
pub struct RecursiveProverState<F: FieldCore> {
    pub w: RecursiveWitnessFlat,
    pub commitment: FlatRingVec<F>,
    pub hint: RecursiveCommitmentHintCache<F>,
    pub log_basis: u32,
    pub sumcheck_challenges: Vec<F>,
}
```

Symmetric on the verifier:

```32:45:crates/akita-verifier/src/protocol/levels.rs
pub struct RecursiveVerifierState<'a, F: FieldCore> {
    pub opening_point: Vec<F>,
    pub opening: F,
    pub commitment: &'a FlatRingVec<F>,
    pub basis: BasisMode,
    pub w_len: usize,
    pub log_basis: u32,
}
```

The recursive prover hardcodes single-poly shape:

```648:664:crates/akita-prover/src/protocol/quadratic_equation.rs
        Ok(Self {
            ...
            opening_points: vec![ring_opening_point],
            claim_to_point: vec![0],
            ...
            claim_group_sizes: vec![1],
            gamma: vec![F::one()],
            num_eval_rows: 1,
```

Verifier mirrors:

```340:351:crates/akita-verifier/src/protocol/levels.rs
    let rs = ring_switch_verifier::<F, T, { D }>(
        std::slice::from_ref(&ring_opening_point),
        &[0usize],
        &stage1_challenges,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        lp,
        &[1usize],
        &[F::one()],
        1,
    )?;
```

### 2.2 Batched multi-poly is supported, but only at root

`QuadraticEquation::new_prover` (root constructor) takes `polys: &[&P]`, `claim_to_point`, `claim_group_sizes`, `gamma`, and `y_rings` (`crates/akita-prover/src/protocol/quadratic_equation.rs::new_prover`). It linearly combines the polynomials with `gamma` into one fused recursive witness. So the root already exercises the multi-poly path; the recursive suffix does not.

`ring_switch_verifier` correspondingly accepts the full batched API (`crates/akita-verifier/src/protocol/ring_switch.rs:310-354`) — the issue is purely that `verify_one_level` calls it with single-claim arguments.

### 2.3 `S` is uncommitted today

`AkitaExpandedSetup` carries only the raw shared matrix:

```37:42:crates/akita-types/src/proof/setup.rs
pub struct AkitaExpandedSetup<F: FieldCore> {
    pub seed: AkitaSetupSeed,
    pub shared_matrix: FlatMatrix<F>,
}
```

There is no commitment to `S`. The verifier currently "trusts the setup" — both parties expand the same PRG seed and assume the resulting `shared_matrix` is well-formed.

### 2.4 Setup-claim-reduction state today

After `verify_stage2_with_setup_claim_reduction` completes, the only thing returned upward is the main stage-2 challenge vector (the prospective witness opening point for the next level):

```196:196:crates/akita-verifier/src/protocol/setup_claim_reduction.rs
    Ok(challenges)
```

The `r_setup` challenges are sampled inside `verify_setup_claim_reduction` via `CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND`, used locally to evaluate `S(r_setup)`, and then thrown away. The `y_setup` value the prover sent (`payload.m_setup_eval` divided by the weight if you like, but practically just `m_setup_eval` is the carried value) is also consumed locally.

### 2.5 Two big things must change

1. **Setup-side**: add a commitment `c_S` and the digit-decomposed `S` material the prover needs to open it. Tiered structure if we go with book §5.4 directly.
2. **Verifier-side**: stop calling `setup_view.mle(...)` inside `verify_setup_claim_reduction`. Instead carry `(r_setup, y_setup)` forward and route into the next level's recursive open.
3. **Prover-side**: include `S` as a second polynomial in the next level's batched recursive open (using the existing multi-poly machinery, lifted to recursive levels).
4. **Schedule planner**: account for the cascade — `S` adds `|S| / f ≈ |S| / 8` ring elements to the next-level witness under the tiered design (book §5.4 Table 1, sweet spot at `f = 8`).
5. **SIS rank tables**: the cascade-aware schedule will hit different `(D, collision, width)` cells; we may need to extend `sis_floor.rs` coverage.
6. **Security re-audit**: re-run `specs/security_analysis.md` after each slice to confirm we stay ≥ 128 bits.

---

## 3 — Protocol design

This section pins down the on-the-wire shape and the algebraic invariants the verifier checks. The implementation in §6 lifts this design into code.

### 3.1 Setup-time commitment to `S` (book §5.3 + §5.4)

At setup expansion the prover and verifier agree on:

- `S` itself, as today (`AkitaExpandedSetup::shared_matrix`).
- A digit-decomposed view `ŝ` with `δ_commit,S = ⌈log₂ q / log₂ b⌉` digits per ring-element coefficient of `S`. For the production prime `q ≈ 2^128` and the production basis range `b ∈ {4, 8, 16, 32, 64}`, this gives `δ_commit,S ∈ {65, 43, 32, 26, 22}` digits.
- A commitment `c_S = D_S · ŝ` via an Ajtai matrix `D_S` of width `δ_commit,S · n_S` where `n_S` is the number of ring elements in `S`.
- Per book §5.3 (split commitment), `c_S` is constructed analogously to the witness D-commitment, with `B_S · t̂_S = u_S` precomputed during setup (the "B-commitments separate" rule).

Both `c_S` and the auxiliary `(t̂_S, u_S)` are stored in `AkitaExpandedSetup` (deterministic from the seed; live in the same serializable artifact as `shared_matrix`).

**Tiered design (book §5.4) — enabled from day 1**:

- Pick shrink factor `f = 8`, hence `k = f² = 64` chunks. `S` is split row-major into `k` blocks of `block_len_S / k` ring elements each.
- Each chunk uses **shared** per-chunk matrices `D_chunk`, `B_chunk` whose column width is `1/f` of the baseline. The setup matrix shrinks correspondingly (book Table 1: at `n_v = 32`, `f = 8` shrinks the setup matrix `36×` because `n_A` drops from 3 to 1 at the smaller chunk widths).
- A tier-3 **meta-commitment** `(c_meta, v_meta, u_meta)` binds the collection of per-chunk commitments via a standard Akita commitment. The meta-commitment uses its own `D_meta, B_meta, A_meta`.
- The proof carries `(c, c_meta, v_meta, u_meta)` — size independent of `k`.

The lazy-derived `TieredSetupCommitments<F, D>` carried in `AkitaVerifierSetup`'s `OnceLock` (per D-3) holds:
- `c_meta` and the per-chunk `c_j` (j = 0..k-1).
- The precomputed B-side material `t̂_S`, `u_S` and the meta tier `v_meta`, `u_meta`.

### 3.2 Setup-claim-reduction output (replace closing oracle)

At each fold level `L` that runs setup-claim-reduction (currently behind `lp.use_setup_claim_reduction`), the verifier completes the sumcheck and receives `(r_setup, y_setup)`:

- `r_setup` = `(r_row, r_col, r_coeff)` is the sumcheck-bound point of length `row_bits + col_bits + coeff_bits` per `setup_polynomial_padded_dims(...)`.
- `y_setup` = `m_setup_eval / weight_at_point` (or equivalently, the prover sends `y_setup` directly and the verifier checks `weight_at_point · y_setup = final_running_claim`).

Instead of evaluating `S(r_setup)` directly via `setup_view.mle(...)` (`crates/akita-verifier/src/protocol/setup_claim_reduction.rs:131`), the verifier **adds `(r_setup, y_setup)` as an opening claim on `S` that the next level must verify via the Hachi PCS**.

### 3.3 Next-level batched opening

At fold level `L+1`, the prover opens both:
- `w_L → ŵ_e` (the folded witness, as today) at the level's standard recursive opening point `r_w`.
- `S` at `r_setup` (received from level `L`'s setup-claim-reduction).

Both openings share the level-`L+1` machinery: ring switch, stage-1 sumcheck, stage-2 sumcheck. Inside `QuadraticEquation`-style aggregation, `S` enters as a second polynomial.

The cleanest realization: lift the multi-poly batched opening that root already supports (`crates/akita-prover/src/protocol/quadratic_equation.rs::new_prover`) to recursive levels. The recursive single-poly hardcoding at `:648-664` becomes a default, and the multi-poly path opens up for levels that need to open `(w, S)` jointly.

**Tensor structure of `r_setup` and the next-level point**: `r_setup` is sampled independently of `r_w`. The book §5.3 says `S` enters "unfolded" — i.e., the next level evaluates `S` at the literal `r_setup` point, not at the folded version. This is automatically a different opening point from `r_w`, so the multi-poly batched opening uses `claim_to_point: [0, 1]` and `opening_points: [r_w, r_setup]` (two distinct points).

### 3.4 Per-level proof-shape changes

Each `AkitaLevelProof` gains:

- An opening commitment `c_S` slot (only at levels that open `S` recursively, i.e. levels with `use_setup_claim_reduction = true` at level `L` AND a "next" recursive fold at level `L+1`).
- Inside `setup_claim_reduction: Option<SetupClaimReductionPayload<F>>`, the existing `m_setup_eval` stays; a new field `r_setup_pinning` may be needed if the next-level transcript can't re-derive `r_setup` deterministically (under inspection: it can, because `r_setup` is sampled inside the same transcript chain).

`AkitaBatchedRootProof` gains the same change at the root level.

**Tiered case (v2)**: the proof carries `c_meta`, `v_meta`, `u_meta` for `S`'s tier-3 binding plus references to the per-chunk commitments. Per book §5.4, this is "independent of `k`" in proof bytes — meta-commitment dominates.

### 3.5 Security analysis carryover

The recursive `S` opening is verified by the existing Hachi PCS machinery on the next level. Its soundness reduces to the standard Module-SIS argument on the shared-matrix commitment key (book Theorem 5 / §5.5 "Security analysis"):

> The setup-side claim `λ · S(r_i, r_x, r_k) = y_setup` is verified at the next level via Akita PCS; its soundness reduces to the standard Module-SIS assumption on the shared-matrix commitment key.

The MSIS rank floors that the security baseline established (`specs/security_analysis.md` §4) apply unchanged. The new setup commitment `D_S` is sized analogously to the witness `D` and routed through the same `sis_derived_*_params_for_layout` flow — meaning the cascade-aware schedule will pick rank floors for `D_S`, `B_S`, `A_S` that satisfy 128-bit MSIS just as for the witness commitments.

The claim-reduction sumcheck still adds `(log m_row + log d) · (deg + 1) / |F_q^k|` to per-level knowledge error, which is `≤ 2^-123` per level (`specs/security_analysis.md` §6). Unchanged.

The CWSS / ring-switch knowledge errors are unchanged because the protocol shape at each level is the same as today (one stage-1 sumcheck, one fused stage-2, one setup-claim-reduction).

**Net**: Phase D-full does not change the security analysis at the per-level granularity. Only the cascade-aware schedule planner and the new SIS rank floors for `D_S`, `B_S` need to be re-audited at the end (§6 slice F).

---

## 4 — Code surface area

Concrete file:line list of every place that must change, organized by the slices in §6.

### 4.1 Tiered setup-time data shapes + lazy commitment (Slice A)

**Add** new module `crates/akita-types/src/proof/tiered_setup.rs`:
- `TieredSetupParams { shrink: usize, k: usize, /* per-tier widths */ }`. Production constants `f = 8`, `k = 64`.
- `TieredSetupCommitments<F, D> { c_meta: RingCommitment<F, D>, c_chunks: Vec<RingCommitment<F, D>>, t_hat_s: ..., u_s: ..., v_meta: ..., u_meta: ... }`.

**Modify** `crates/akita-types/src/proof/setup.rs`:
- `AkitaExpandedSetup<F>` stays as-is (no new persisted fields per D-3).
- `AkitaVerifierSetup<F>` gains `tiered_s_cache: Arc<OnceLock<TieredSetupCommitments<F, D>>>`.
- `AkitaProverSetup<F>` gains the same field if the prover also needs cached access.

**New helper** `crates/akita-prover/src/api/tiered_setup.rs::derive_tiered_setup_commitments`:
- Inputs: `expanded_setup`, `LevelParams` for the level that opens `S`, `TieredSetupParams`.
- Outputs: `TieredSetupCommitments`.
- Reuses the witness commitment infrastructure (`commit_inner_witness`, `mat_vec_mul_ntt_single_i8`).

**Lazy accessor**:
```rust
impl AkitaVerifierSetup<F> {
    pub fn tiered_s_commitments<const D: usize>(&self, params: TieredSetupParams)
        -> Result<&TieredSetupCommitments<F, D>, AkitaError>;
}
```

**Update** `crates/akita-config/src/proof_optimized.rs::proof_optimized_max_setup_matrix_size`: account for the tiered per-chunk and meta widths in the envelope.

### 4.2 Recursive multi-poly opening (Slice B)

**Generalize** `crates/akita-prover/src/protocol/flow.rs`:
- `RecursiveProverState` adds optional `s_handle` data (commitment + hint + claim) for the next level to open.
- `prove_fold_level_from_quadratic` (lines 583-757): when `lp.use_setup_claim_reduction && next_level.opens_s == true`, the level needs to consume the `(r_setup, y_setup)` claim from level-`L` and produce a multi-poly batch for level `L+1`.
- `prove_recursive_fold_with_params` (lines 774-875): currently uses `QuadraticEquation::new_recursive_prover`. Replace the single-poly path with a multi-poly path when the level needs to open `S` jointly with `w`.

**Generalize** `crates/akita-prover/src/protocol/quadratic_equation.rs`:
- `new_recursive_prover` (around line 648): replace the hardcoded `claim_group_sizes: [1]` / `claim_to_point: [0]` / `gamma: [F::one()]` / `num_eval_rows: 1` with parameters from the caller. The caller passes either `[1]/[0]/[F::one()]/1` for the legacy single-poly path or `[1, 1]/[0, 1]/[gamma_0, gamma_1]/2` for the joint `(w, S)` case.

**Generalize** `crates/akita-verifier/src/protocol/levels.rs::verify_one_level` (lines 277-437):
- Replace the hardcoded `stage1_challenges = derive_stage1_challenges(..., num_claims: 1, lp)` (line 330) with `num_claims` from the level's `LevelProofShape`.
- Replace the hardcoded `ring_switch_verifier(..., [0usize], [1usize], [F::one()], 1)` (lines 340-351) with the multi-claim arguments.

**Update** `crates/akita-verifier/src/protocol/levels.rs::RecursiveVerifierState`:
- Add `s_opening_point: Option<Vec<F>>` and `s_opening: Option<F>` so the verifier can carry the `(r_setup, y_setup)` from level `L` into level `L+1`.

**Or alternatively**, a cleaner refactor: introduce a `RecursiveOpeningClaim<F>` struct that encapsulates `{ opening_point, opening, commitment_ref, basis, log_basis }` and have `RecursiveVerifierState` hold `Vec<RecursiveOpeningClaim<F>>` (variable-length). Symmetric `RecursiveProverState` change. This is the more invasive but more general refactor and will be needed for tiered commitments in slice D anyway.

### 4.3 Reroute setup-claim-reduction closing oracle (Slice C)

**Modify** `crates/akita-verifier/src/protocol/setup_claim_reduction.rs::verify_setup_claim_reduction`:

```120:135:crates/akita-verifier/src/protocol/setup_claim_reduction.rs
    let weight_at_point =
        prepared.eval_setup_weight_at_point::<D>(x_challenges, setup, alpha, &challenges)?;

    let (row_bits, col_bits, _coeff_bits) = prepared.setup_polynomial_padded_dims(max_stride);
    let row_challenges = &challenges[..row_bits];
    let col_challenges = &challenges[row_bits..row_bits + col_bits];
    let coeff_challenges = &challenges[row_bits + col_bits..];
    let row_count = prepared.setup_polynomial_row_count();
    let setup_view = setup
        .shared_matrix
        .setup_polynomial_view::<D>(row_count, max_stride);
    let setup_at_point = setup_view.mle(row_challenges, col_challenges, coeff_challenges)?;

    if weight_at_point * setup_at_point != final_running_claim {
        return Err(AkitaError::InvalidProof);
    }
    Ok(challenges)
```

Becomes:

```rust
let y_setup = payload.s_opening_value;  // new field; or compute from `final_running_claim / weight_at_point`
let r_setup = challenges;  // returned upward to the orchestrator
// Verifier defers the `S(r_setup) = y_setup` check to the next level's recursive PCS.
// The `weight_at_point * y_setup = final_running_claim` invariant is checked here.
if weight_at_point * y_setup != final_running_claim {
    return Err(AkitaError::InvalidProof);
}
Ok((challenges, y_setup))  // signature changes to return both
```

**Modify** `verify_stage2_with_setup_claim_reduction` to thread `(r_setup, y_setup)` to its caller, and update `verify_one_level` / `verify_root_level` to put them into the `RecursiveVerifierState` for the next level.

**Modify** `crates/akita-prover/src/protocol/setup_claim_reduction.rs::prove_setup_claim_reduction`:
- Currently the prover materializes the full `(setup_table, setup_weights)` and runs `WeightedTableProver`. The closing oracle naturally produces a final `S(r_setup)` value at the end of the sumcheck. The prover should send this value (`y_setup`) in `SetupClaimReductionPayload` as a new field, so the verifier knows what to expect.

**Modify** `SetupClaimReductionPayload<F>` in `crates/akita-types/src/proof/mod.rs:1006-1014`:
- Add `s_opening_value: F` (the `y_setup` the prover claims for `S(r_setup)`).
- The existing `m_setup_eval` can be derived as `weight_at_point * s_opening_value` so we don't need to send both; pick one canonical wire form.

### 4.4 Tiered protocol shape (folded into Slice C)

Per D-1, the tiered design is in from day 1. Slice C implements the protocol shape:
- `f = 8`, `k = 64`, `r_chunk = r - log₂ f = r - 3`.
- Per-chunk shared matrices `D_chunk`, `B_chunk` of column width `1/f` baseline (derived from the seed via `derive_public_matrix_flat` at narrower widths).
- Tier-3 meta-commitment `(c_meta, v_meta, u_meta)` via standard Akita commitment.
- Stage-2 relation has **10 check groups** (5 original + 5 meta) instead of the current 5.

The setup-side data shapes for these are in slice A; the protocol shape changes are in slice C.

### 4.5 Schedule planner / SIS tables (Slice E)

**Modify** `crates/akita-planner/src/schedule_params.rs`:
- The planner currently picks `(m_vars, r_vars, n_a, n_b, n_d, ...)` based on the witness length. Now it must also account for `S`'s contribution to the next-level witness: at level `L+1`, `|w_{L+1}|_ring = |fold_output_L|_ring + |S| / f` with `f = 8` per book Table 1.
- New objective term: cascade penalty.

**Modify** `crates/akita-types/src/generated/sis_floor.rs`: if the new schedules hit collision buckets not currently in the table, extend with `scripts/gen_sis_table.py` (which already supports adding cells via the `ALL_ENTRIES` list).

**Regenerate** the six fp128 schedule tables.

### 4.6 Security re-audit (Slice F)

Re-run the security analysis pipeline from `specs/security_analysis.md`:
- `extract_params.py` — update to extract `D_S, B_S, A_S` rank/widths too.
- `run_estimator_all.py` — same.
- Confirm every preset still clears 128 bits MSIS at the new schedule shape.

---

## 5 — Resolved design decisions

These were chosen via author sign-off on `2026-05-13` before implementation began.

### D-1: Tiered v1 from the start (chosen)

Ship the full book §5.4 design directly: shrink factor `f = 8`, `k = f² = 64` chunks, tier-3 meta-commitment, 10-check-group stage-2 relation. Production-ready immediately at all NV; no cascade-overflow at NV ≥ 22.

Total estimated effort: ~12 weeks. The slices in §6 are reorganized accordingly: there is no separate "Slice D (tiered)" because every code path is tiered from day 1.

### D-2: `Vec<RecursiveOpeningClaim<F>>` refactor (chosen)

Replace the hardcoded single-poly recursive state with a variable-length opening-claim vector. Generalizes for both the `(w, S)` joint open in slice C and the per-chunk claims in the tiered protocol (~k = 64 claims per level when fully tiered). Single-element vec is the legacy path; all current tests must still pass with `Vec.len() == 1` after slice B.

### D-3: `OnceLock`-lazy `c_S` derivation (chosen)

Do **not** persist `c_S` alongside `shared_matrix`. Compute it lazily on first use via an `OnceLock` field on `AkitaVerifierSetup` (and the parallel field on `AkitaProverSetup`). One-shot cost at first verify; no setup-file growth or versioning concerns. Cache fields:

```rust
pub struct AkitaVerifierSetup<F: FieldCore> {
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    pub schedule_cache: Arc<Mutex<HashMap<AkitaScheduleLookupKey, Schedule>>>,
    pub tiered_s_cache: Arc<OnceLock<TieredSetupCommitments<F>>>, // NEW
}
```

`TieredSetupCommitments` carries the per-chunk and meta commitments plus the precomputed `t̂_S` / `u_S` material the prover/verifier need to bind opening claims. The lazy initializer takes the chunk-shape parameters from the level's `LevelParams` (book §5.4 picks `f = 8` for production); since chunk shape is part of the public schedule, this is deterministic.

### D-4: Stage-1 challenges for multi-poly recursive open

The existing root path samples `num_claims = total claims across all polys` stage-1 challenges. Recursive levels currently sample 1. After the change, recursive levels that open `S` jointly with `w` sample `1 + k = 65`-many claims (one for `w`, `k = 64` for the chunk-decomposed `S`).

### D-5: `LevelParams` for `S` at level L+1

`S`'s commitment at level `L+1` uses level `L+1`'s `LevelParams` (specifically `log_basis`, `D`, `num_digits_*`). Inside the multi-poly batched recursive open, `S` enters as a polynomial of length `|S|_ring` (un-folded), with its own claim pinned to `r_setup`. `δ_{commit,S} = ⌈log₂ q / log₂ b⌉` per book §5.3.

### D-6: Fail-loud SIS table coverage (chosen)

When a planner search hits a `(D, collision_bucket)` cell not in `sis_floor.rs`, the planner errors with a clear `InvalidSetup` pointing at `scripts/gen_sis_table.py`. The operator extends the table, commits the new `sis_floor.rs`, and re-runs the planner. Forces explicit auditability; preserves the existing security baseline's defense-in-depth check.

---

## 6 — Implementation plan (slices)

Each slice ends with: `cargo fmt -q && cargo clippy --all -- -D warnings && cargo test --release`. Each slice is one focused commit on `feat/tensor-challenges`. The slices are sequential. The tiered design is enabled from slice A — there is no un-tiered staging point.

### Slice A: Tiered setup foundation + `OnceLock`-lazy commitments

**Goal**: define the tiered commitment data shapes; derive `c_S` lazily on first use.

**Changes**:
1. New module `crates/akita-types/src/proof/tiered_setup.rs` (or similar):
   - `TieredSetupParams` (per-tier shrink factor, chunk count, per-chunk matrix widths).
   - `TieredSetupCommitments<F, D>`: holds per-chunk commitments `c_j` (j = 0..k-1), meta-commitment `c_meta`, and the precomputed B-side material (`t̂_S`, `u_S`, plus the meta-tier `v_meta`, `u_meta` per book §5.3-5.4).
2. New helper `crates/akita-prover/src/api/tiered_setup.rs::derive_tiered_setup_commitments`:
   - Inputs: `expanded_setup` (with `shared_matrix`), level-`L+1` `LevelParams` (gives `log_basis`, `δ_{commit,S}`, etc.), production tier params (`f = 8`, `k = 64`).
   - Outputs: `TieredSetupCommitments`.
   - Uses the same Ajtai pipeline as witness commitments (`mat_vec_mul_ntt_single_i8`).
3. Add `tiered_s_cache: Arc<OnceLock<TieredSetupCommitments<F, D>>>` to `AkitaVerifierSetup` (and the parallel field on `AkitaProverSetup` if the prover also needs cached access).
4. Lazy accessor:
   ```rust
   impl AkitaVerifierSetup<F> {
       pub fn tiered_s_commitments<const D: usize>(&self, params: TieredSetupParams)
           -> Result<&TieredSetupCommitments<F, D>, AkitaError>;
   }
   ```
5. Tests:
   - `derive_tiered_setup_commitments` is deterministic from the seed.
   - `c_meta` is a function of `(c_j)_j` only (not of any prover randomness).
   - Per-chunk shape matches book §5.4 table 1 at `f = 8`.
   - Round-trip the setup; `tiered_s_commitments(...)` returns identical material pre- and post-serialization.

**Acceptance**: No protocol change yet. `cargo test -p akita-setup -p akita-types --release` green.

**Estimated**: ~1.5 weeks.

### Slice B: `Vec<RecursiveOpeningClaim>` refactor

**Goal**: replace the hardcoded single-poly recursive state on prover and verifier with variable-length opening-claim vectors.

**Changes**:
1. New `RecursiveOpeningClaim<'a, F>`:
   ```rust
   pub struct RecursiveOpeningClaim<'a, F: FieldCore> {
       pub opening_point: Vec<F>,
       pub opening: F,
       pub commitment: &'a FlatRingVec<F>,
       pub basis: BasisMode,
       pub w_len: usize,
       pub log_basis: u32,
   }
   ```
2. `RecursiveVerifierState` becomes `Vec<RecursiveOpeningClaim<'a, F>>`. The first entry is the current witness; later entries (in slice C) will be recursive `S`-chunk openings.
3. `RecursiveProverState` grows a parallel `Vec<RecursivePolyHandle<F>>` (each handle: `w`, `commitment`, `hint`, `sumcheck_challenges`, `log_basis`).
4. **Legacy compatibility**: when `Vec.len() == 1`, the path is bitwise-equivalent to today's single-poly recursive path. All existing tests must pass unchanged. The multi-claim machinery from the root-level `QuadraticEquation::new_prover` is lifted to the recursive path; the recursive `new_recursive_prover` gains parameters for `claim_group_sizes`, `claim_to_point`, `gamma`, `num_eval_rows`.

**Acceptance**: full workspace test suite green with `Vec.len() == 1` everywhere. No protocol change.

**Estimated**: ~1 week (mostly mechanical plumbing across ~10 functions).

### Slice C: Tiered recursive `S` opening protocol

**Goal**: replace `setup_view.mle(...)` with a tiered opening claim that's routed into the next level's `Vec<RecursiveOpeningClaim>`.

**Changes**:
1. `SetupClaimReductionPayload<F>` gains `s_opening_value: F` (`y_setup`) and any per-chunk decomposition needed for tiered binding. The wire shape depends on the final tier protocol pinned in this slice.
2. `verify_setup_claim_reduction` no longer calls `setup_view.mle(...)`. Instead it returns `(challenges, y_setup, tier_claim_decomposition)` to its caller.
3. `verify_stage2_with_setup_claim_reduction` threads the `S`-opening tier-decomposition forward.
4. `verify_one_level` / `verify_root_level` append the tier decomposition (`k = 64` per-chunk claims plus the meta-binding check) to the next level's `RecursiveVerifierState`. The meta-binding check is handled inside the level transition: the verifier checks that the per-chunk commitments hash to `c_meta` consistent with `AkitaVerifierSetup::tiered_s_commitments(...)`.
5. Prover-side: `prove_setup_claim_reduction` returns the tier decomposition. `prove_fold_level_from_quadratic` / `prove_root_fold_from_quadratic` pack it into the payload and the next-level's `RecursiveProverState`.
6. **The 10-check-group stage-2 relation** (book §5.4): the next-level stage-2 sumcheck takes 10 inputs (5 original + 5 meta). Implement the meta-rows in `prepare_m_eval` (extend the row layout) and in the prover-side sumcheck instance.
7. Schedule planner cost model accounts for the cascade — `w_{L+1}_ring = w_fold_L + |S| / f` (book Table 1 at `f = 8`). Cascade ratio target ≲ 1 per book.
8. Regenerate the six fp128 schedule tables under the cascade-aware cost model.

**Acceptance**:
- E2E proof verification passes for all six fp128 presets at production NV (≥ 25) with `lp.use_setup_claim_reduction = true` and tiering enabled.
- Tamper tests: corrupting `y_setup`, any per-chunk claim, or `c_meta` makes verification reject.
- `validate_stored_sis_ranks` still passes for every entry in the regenerated tables.
- Workspace test green.

**Estimated**: ~5 weeks. This is the bulk of Phase D-full.

### Slice D: SIS table extension (fail-loud handling)

**Goal**: extend `sis_floor.rs` to cover any new `(D, collision_bucket)` cells the cascade-aware planner reaches.

**Changes**:
1. After slice C regenerates the tables, any planner-fail-loud errors point at uncovered cells.
2. For each uncovered cell, append `(d, collision)` to `scripts/gen_sis_table.py::ALL_ENTRIES`, run the script (Sage + lattice-estimator).
3. Commit the extended `sis_floor.rs`.
4. Re-run the planner; tables stabilize.

**Acceptance**: `cargo test --release` green; no `InvalidSetup(missing supported B/D collision bucket ...)` errors.

**Estimated**: ~3 days (estimator runtime + manual review).

### Slice E: Security re-audit post Phase D-full

**Goal**: confirm every production preset still satisfies 128-bit MSIS and 128-bit Fiat-Shamir/CWSS after the protocol shape change.

**Changes**:
1. `scripts/security_analysis/extract_params.py` extracts the new meta-tier `(D_meta, B_meta, A_meta)` parameters too.
2. Re-run `run_estimator_all.py` over the expanded quadruple set (includes per-chunk D / B / A and meta D / B / A).
3. Update `specs/security_analysis.md` with the post-Phase-D-full bit tables and document the tiered-commit invariant.
4. Confirm every production preset clears 128 bits at every Ajtai role (witness AND per-chunk AND meta).
5. Update `audit.md` and `roadmap.md` to reflect that the verifier is now at `O(N'^{1/4})` per level.

**Acceptance**: Every preset clears 128 bits. Per-level verifier cost benchmark shows the asymptotic improvement vs `main`.

**Estimated**: ~1 week.

### Total estimated effort

| Slice | Effort | Cumulative |
|---|---|---|
| A: Tiered setup foundation + `OnceLock`-lazy | ~1.5 weeks | ~1.5 weeks |
| B: `Vec<RecursiveOpeningClaim>` refactor | ~1 week | ~2.5 weeks |
| C: Tiered recursive `S` opening + 10-check-group stage-2 | ~5 weeks | ~7.5 weeks |
| D: SIS table extension | ~3 days | ~8 weeks |
| E: Security re-audit | ~1 week | **~9 weeks** |

(Estimates are wallclock for one engineer including review cycles. A→B→C→D→E are sequential; A and B can in principle proceed in parallel but I'll keep them sequential to limit review surface per slice.)

---

## 7 — What this design does *not* change

Documenting non-goals so reviewers don't expect them:

- **MSIS rank floors**: unchanged. The security baseline established in this PR is the foundation; Phase D-full picks the same `min_rank_for_secure_width` for every role.
- **CWSS knowledge error per level**: unchanged. The challenge family + `4ω` extraction degradation are independent of the setup-side optimization.
- **Stage-1 sumcheck and main stage-2 sumcheck**: unchanged. Only the closing oracle of stage-2 (via setup-claim-reduction) changes.
- **Ring-switch protocol**: unchanged.
- **Tensor challenge transcript model** (book §5.2): unchanged; the two-round empty-prover Fiat-Shamir model is what we ship today.
- **The base-field prime / verifier-field**: unchanged.
- **The Fiat-Shamir transcript domain labels**: only `CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND` and existing labels are reused; no new top-level domain.

---

## 8 — Reference index

| Topic | File | Key lines |
|---|---|---|
| Current single-poly recursive prover state | `crates/akita-prover/src/protocol/flow.rs` | 38-50 |
| Hardcoded single-poly recursive QE | `crates/akita-prover/src/protocol/quadratic_equation.rs` | 648-664 |
| Batched multi-poly root QE | `crates/akita-prover/src/protocol/quadratic_equation.rs` | `new_prover` |
| Single-poly verify recursive | `crates/akita-verifier/src/protocol/levels.rs` | 277-437 |
| Setup-claim-reduction verifier closing oracle (the line to replace) | `crates/akita-verifier/src/protocol/setup_claim_reduction.rs` | 127-135 |
| `verify_stage2_with_setup_claim_reduction` | `crates/akita-verifier/src/protocol/setup_claim_reduction.rs` | 156-197 |
| `RecursiveVerifierState` | `crates/akita-verifier/src/protocol/levels.rs` | 32-45 |
| `AkitaExpandedSetup` (where `c_S` will live) | `crates/akita-types/src/proof/setup.rs` | 37-42 |
| Setup expansion (PRG) | `crates/akita-prover/src/api/setup.rs` | 45-61 |
| `SetupMatrixPolynomialView` / `materialize_table` / `mle` | `crates/akita-types/src/layout/flat_matrix.rs` | 260-415 |
| `prove_setup_claim_reduction` (prover side) | `crates/akita-prover/src/protocol/setup_claim_reduction.rs` | 35-61 |
| `SetupClaimReductionPayload` | `crates/akita-types/src/proof/mod.rs` | 1006-1014 |
| Witness `commit_inner_witness` (parallel for `S`) | `crates/akita-prover/src/backend/recursive_witness.rs` | 304-357 |
| `setup_matrix_envelope_for_shape` (sets max stride) | `crates/akita-config/src/proof_optimized.rs` | 371-470 |
| Book §5.3 setup opening | `Lattice_Jolt/sections/akita/5_fourth_root_verifier.tex` | 627-670 |
| Book §5.4 tiered commitment | `Lattice_Jolt/sections/akita/5_fourth_root_verifier.tex` | 672-799 |

---

## 9 — Approval status

Resolved on `2026-05-13`:

1. **Scope**: tiered v1 from day 1 (book §5.4 directly), no un-tiered staging point.
2. **State shape**: `Vec<RecursiveOpeningClaim<F>>` refactor up front (slice B).
3. **`c_S` storage**: `OnceLock`-lazy derivation on first use; not persisted.
4. **SIS coverage**: planner-fail-loud; operator extends `sis_floor.rs` via `gen_sis_table.py` when needed.

Implementation proceeds slice-by-slice per §6. Each slice lands as a focused commit on `feat/tensor-challenges` so the branch is reviewable incrementally. Slice A starts after this design doc lands.
