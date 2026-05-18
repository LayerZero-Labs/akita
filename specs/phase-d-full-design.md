---
**SUPERSEDES** the v1 design that pushed `S` through the existing
recursive multi-claim machinery as a second handle with `w = t̂_S`.
That attempt was algebraically wrong (digit-poly ≠ source poly) and
the fix is to **extend** the batched recursive path to admit
per-group `(m, r, B, digit_count)` and mixed witness types — NOT to
introduce a parallel "split commitment" construct alongside the
existing infrastructure. See §10 for the v1→v2 diff. Reading order:
§1 problem statement → §2 book references → §3 protocol shape → §4
implementation surface area → §5 decisions → §6 implementation
plan → §7 non-goals → §8 reference index → §9 approval status →
§10 v1 retraction notes.
---

# Phase D-full Design v2: Recursive `S` Opening via Multi-Group Batched Hachi

**Date**: 2026-05-13
**Branch**: `feat/tensor-challenges` (post-security-baseline commit `9089d667`)
**Reference spec**: book §§5.3 (claim-reduction sumcheck) and §5.4 (tiered commitment design)
**Predecessors**: `specs/security_analysis.md` (security baseline; both
MSIS and CWSS clear 128 bits), original v1 of this document (committed
at `95e79c54`, retracted by §10).

This document specifies the protocol modifications and code changes
for Phase D-full v2: replacing the per-level
`SetupMatrixPolynomialView::mle(...)` closing oracle in
`verify_setup_claim_reduction` with a recursive opening claim on the
shared setup polynomial `S`, discharged at the next fold level via
**multi-group batched Hachi** (the book's "split commitment" is
exactly this: multi-group batched commit with shared `D/A` and
per-group `B/(m, r)/digit_count`). For large `S`, the S-group is
optionally split into `k = f²` sub-chunks bound by a tier-3
meta-commitment per book §5.4.

The doc is grounded in the book's §5 (fourth-root verifier). Every
protocol modification cites the book section that defines it.
The implementation plan in §6 is decomposed into independently
shippable slices and deliberately scoped to extending existing
primitives, not introducing parallel ones.

---

## 1 — Why this is needed (recap)

After the security baseline fix, the verifier's hot path inside
`verify_setup_claim_reduction` is:

```127:131:crates/akita-verifier/src/protocol/setup_claim_reduction.rs
    let setup_view = setup
        .shared_matrix
        .setup_polynomial_view::<D>(row_count, max_stride);
    let setup_at_point = setup_view.mle(row_challenges, col_challenges, coeff_challenges)?;
```

`SetupMatrixPolynomialView::mle` walks the live `num_rows × num_cols`
prefix of the shared matrix at every level
(`crates/akita-types/src/layout/flat_matrix.rs:392-413`), which is
`O(rows · cols · D)` per level — the same asymptotic order as the
legacy fused stage-2 path. This is what keeps the per-level verifier
work at `O(√N')` rather than `O(N'^{1/4})`.

Phase D-full v2 closes this gap by deferring `S(r_setup) = y_setup`
as a recursive opening claim and discharging it at the next fold
level via multi-group batched Hachi.

---

## 2 — Book references (load-bearing)

All claims in this design trace back to specific passages in
book §5 (fourth-root verifier).

| Topic | Section | Lines |
|---|---|---|
| Claim-reduction sumcheck (Technique 2) | §5.3 | 511–669 |
| M-table additive decomposition `m̃ = m̃_alg + m̃_setup` | §5.3 | 519–538 |
| Stage-1 batched (range + relation), stage-2 setup-side | §5.3 | 558–625 |
| Setup-side closing claim `λ · S̃(r_i, r_x, r_k) = y_setup` | §5.3 | 619–625 |
| Setup opening — `S` enters L+1 unfolded as additional polynomial | §5.3 | 627–642 |
| Digit-decomposition asymmetry (`δ_commit,S = ⌈128/log_2 b⌉`) | §5.3 | 635–642 |
| **Multi-group commit** (joint D / separate B / combined z_pre) | §5.3 | 643–660 |
| Tiered commitment (`f` = shrink factor, `k = f²` chunks) | §5.4 | 672–700 |
| Per-chunk shared matrices `D_chunk, B_chunk` (1/f baseline width) | §5.4 | 701–708 |
| Tier-3 meta-commitment `(c_meta, v_meta, u_meta)` | §5.4 | 692–699 |
| **10 check groups** (5 original per-chunk + 5 meta) | §5.4 | 709–754 |
| Cascade trade-off `T2 ratio = |S|/(w_fold_L)` | §5.4 | 762–800 |
| Sweet spot `f = 8`, `k = 64` (T2 ratio ≈ 1) | §5.4 | 793–799 |
| Combined protocol (rounds 1–8) | §5.5 | 802–938 |
| Round 5: assemble `w := (ê, t̂, ẑ)` | §5.5 | 865–870 |
| Round 8: claim-reduction stage-2 + close deferred output | §5.5 | 900–919 |
| What feeds into next level | §5.5 | 940–953 |
| Security analysis (Theorem 5.4 / §5.5.1) | §5.5 | 955–1024 |
| Setup-side claim soundness reduces to MSIS on shared key | §5.5 | 981–986 |

`§` numbers refer to the book's section headers; line numbers refer
to the `.tex` source. When this doc says "the book's ...", read it
as a normative reference to those passages.

---

## 3 — Protocol shape (book-aligned)

### 3.1 Setup-time commitment to `S`

At setup expansion, the prover and verifier agree on:

- `S` itself, as today (`AkitaExpandedSetup::shared_matrix`).
- A digit-decomposed view `ŝ` with
  `δ_commit,S = ⌈log₂ q / log₂ b⌉` digits per ring-element coefficient
  of `S`. For the production prime `q ≈ 2^128` and the production
  basis range `b ∈ {4, 8, 16, 32, 64}`, this gives
  `δ_commit,S ∈ {65, 43, 32, 26, 22}` digits.
- For tiered (`f > 1`, `k = f²`): `S` is split row-major into `k`
  sub-chunks, each committed under shared per-chunk matrices
  `D_chunk, B_chunk` (column width `1/f` of the baseline). The
  collection of per-chunk B-side outputs is bound by a tier-3
  meta-commitment under its own `D_meta, B_meta, A_meta`.

The proof carries `(c, c_meta, v_meta, u_meta)` — size independent of `k`
(book §5.4 line 698).

The lazy-derived material carried in `AkitaVerifierSetup`'s `OnceLock`
holds:

- `c_meta` and the per-chunk `c_j` (j = 0..k-1).
- The precomputed B-side material `t̂_S`, `u_S` (per-chunk for tiered)
  and the meta tier `v_meta`, `u_meta`.

For the un-tiered case (`f = 1`, `k = 1`), the S-group is a single
chunk and the meta tier collapses (no chunks to bind).

### 3.2 Setup-claim-reduction output (book §5.3 Round 8)

At each fold level `L` that runs setup-claim-reduction (currently
behind `lp.use_setup_claim_reduction`), the verifier completes the
sumcheck and receives `(r_setup, y_setup)`:

- `r_setup = (r_i, r_x, r_k)` is the sumcheck-bound point of length
  `⌈log₂ m_row⌉ + log₂ d` per book §5.3 (line 616).
- `y_setup` = `λ · S̃(r_i, r_x, r_k)` per book equation (eq:setup-claim,
  line 619). The book's `λ := w_eval · α̃(r_y)` (line 601) is the
  scaling factor.

The closing equality (book §5.5 Round 8 step 4, line 912):

```
γ_range · s_claim
  + γ_rel · (λ · m̃_alg(r_x) + y_setup)
= (batched sumcheck output)
```

The verifier defers `λ · S̃(r_i, r_x, r_k) = y_setup` to the next
fold level (book §5.3 lines 627–642): "The matrix polynomial `S̃`
from level `L` is batched into level `L+1`'s witness for joint PCS
opening: it enters level `L+1` **unfolded** as an additional
polynomial alongside the folded witness."

### 3.3 Recursive S opening = multi-group batched Hachi at L+1

This is the v2 reframing of what v1 mis-named "split commitment".
The book's design (§5.3 lines 643–660) is exactly multi-group
batched Hachi where:

- The batch has TWO commitment groups: the `w`-group (the L-fold
  output, an `i8`-digit recursive witness) and the `S`-group (a
  fresh field-element polynomial).
- Each group has its OWN `(m, r)` split, `B`-matrix, and digit count
  (`δ_open,w` for `w`, `δ_commit,S` for `S`).
- `D` and `A` are SHARED across groups: a single `v := D · (ê_w ‖ ê_S)`
  and a single `c := A · z_pre_combined` per L+1 commit.
- `B` is per-group: `u_w := B_w · t̂_w` and `u_S := B_S · t̂_S` (or
  per-chunk for tiered).
- The Ajtai binding covers the combined `z_pre_combined` formed by
  concatenating the per-group `z_pre` blocks.

The "split commitment" name in the book reflects the per-group `B`;
the protocol shape is otherwise standard batched Hachi extended to
admit:

1. **Per-group LP shape**: each commitment group has its own
   `(m, r, B, digit_count)`; `D, A` are shared.
2. **Mixed witness types in the recursive batch**: the `w`-group's
   "polynomial" is a `RecursiveWitnessView` (i8 digits viewed as
   field-element coefficients); the `S`-group's "polynomial" is a
   `DensePoly<F, D>` (S itself).

These are two extensions to the existing batched commit
infrastructure. Slice D adds (1); slice E adds (2). Together they
realize the book's "split commitment" without introducing a parallel
"split" type system.

In code terms, the level-L+1 commit is:

```
v_joint := D · (ê_w ‖ ê_S)                           // single D-commitment
u_w     := B_w · t̂_w                                  // per-group B
u_S     := B_S · t̂_S        // or per-chunk for tiered
c_joint := A · (z_pre_w ‖ z_pre_S)                   // single Ajtai
```

For the tiered case (k > 1), the S-group is split into k sub-chunks
each committed under shared `(D_chunk, B_chunk)`, plus a tier-3
meta-commit binding the per-chunk commitments under
`(D_meta, B_meta, A_meta)`. The W-group is unaffected by tiering.

### 3.4 Stage-2 relation: 5 (un-tiered) or 10 (tiered) check groups

Book §5.4 (lines 715–754) describes the relation in terms of "check
groups". These map onto extensions of the existing 5-group stage-2
row layout in `prepare_m_eval`:

**Un-tiered case (`f = 1`, `k = 1`)**: 5 check groups extended for
per-commitment-group sub-rows:

1. D-check (`n_D` rows): `D · (ê_w ‖ ê_S) = Recompose(v_digits)`.
2. B-checks (per-group; `n_{B,w}` rows for `w` + `n_{B,S}` rows for
   `S`): `B_w · t̂_w = Recompose(u_w_digits)` and
   `B_S · t̂_S = Recompose(u_S_digits)`.
3. Evaluation check (per-group; 2 rows): links each `ê_g` to the
   stage-1 evaluation claim for group `g`.
4. Fold check (per-group; 2 rows): `z_pre,g = Σ_j c_j · block_j`
   for each group.
5. Ajtai binding (`n_A` rows): `A · z_pre_combined = c_joint`.

**Tiered case (`f = 8`, `k = 64`)**: groups 1, 2, 3, 4 of the S
contribution become per-chunk (k block-diagonal sub-relations with
SHARED `D_chunk / B_chunk`), and groups 6–10 add the meta-tier
relation:

6. `D_meta`-check (`n_{D,meta}` rows): `D_meta · ê_meta = v_meta`.
7. `B_meta`-check (`n_{B,meta}` rows): `B_meta · t̂_meta = u_meta`.
8. Eval-like check (1 row): links `ê_meta` to the per-chunk
   commitment collection.
9. Fold check (1 row): `z_pre,meta = Σ_j c_meta,j · block_meta,j`.
10. Ajtai binding (`n_{A,meta}` rows):
    `A_meta · z_pre,meta = c_meta`.

> Per-chunk rows (groups 1–2 of the S contribution) have
> block-diagonal structure with **shared** `D_chunk / B_chunk`, so
> the MLE evaluation cost is `O(|D_chunk|) + O(log k)`, **independent
> of k**. (Book §5.4 lines 751–754.)

The implementation in `prepare_m_eval` extends the existing 5-group
row layout to support per-commitment-group sub-rows; the tiered case
adds the meta-tier rows on top. No parallel "10-group" relation
type is needed — it's the same row-layout machinery with one extra
row family per commitment group plus the meta-tier rows when
`k > 1`.

### 3.5 Cascade formula (book §5.4 Table 1, line 762)

> T2 ratio: ratio of digit-decomposed `S` to the next-level witness
> (must be `≲ 1` for cascading viability).

The book's cascade formula is:

```
w_{L+1}_field_count = w_fold_L + |S| / f    (tiered, k = f²)
                    = w_fold_L + |S|         (un-tiered, k = 1)
```

Where `w_fold_L = w_ring_element_count(level_L_lp) * D` (the natural
fold output) and `|S|` is the setup polynomial size in ring elements.

The previous v1 cascade `max(natural, |S| * D)` is **wrong**: the
book's formula is **additive**, not max-based. The multi-group
commit naturally combines `w_fold_L` field elements (for the
`w` group) with `|S| / f` field elements (for the `S` group under
tiered shrinkage).

The per-group `(m, r)` splits at L+1 are independent: `(m_w, r_w)`
is sized for `w_fold_L`, `(m_S, r_S)` is sized for `|S| / f`. The
SIS width constraint is `n_A · δ_open · 2^{r_S} ≤ SIS_MAX_WIDTH[d]`.

### 3.6 Per-level proof shape changes

Each `AkitaLevelProof` gains, when `lp.use_setup_claim_reduction`:

- The existing `setup_claim_reduction: Option<SetupClaimReductionPayload<F>>`
  with `m_setup_eval, s_opening_value, sumcheck` (already wired from
  slice C.2.a).

At fold levels that **discharge** a deferred S-claim from the
previous level (i.e., the previous level emitted a
setup-claim-reduction payload AND this level is a fold), the level
proof additionally carries:

- For tiered (`k > 1`): `c_meta`, `v_meta`, `u_meta` (the tier-3
  meta-commitment material). Per book §5.4 line 698, this is
  independent of `k` in proof bytes.
- The L+1 commit produces a single `v_joint` and a single `c_joint`,
  plus per-group `u_g` (`u_w` and `u_S` for un-tiered;
  `u_w` and per-chunk `u_{S,j}` bound by the meta-tier for tiered).

Note: the per-chunk commitments `(c_j)_{j=0..k-1}` themselves are
**not** in the proof — they are bound via the meta-commitment and
re-derivable from the public setup matrix. Only the meta-tier
material is on the wire.

### 3.7 Security analysis carryover

Per book §5.5 Theorem 5.4 (line 959, lines 981–986):

> The setup-side claim `λ · S̃(r_i, r_x, r_k) = y_setup` is verified
> at the next level via Akita PCS; its soundness reduces to the
> standard Module-SIS assumption on the shared-matrix commitment key.

The book's "Akita PCS opening on `c_S`" is exactly multi-group
batched Hachi at L+1 with the S-group included. v1 attempted this
by pushing `t̂_S` as a second handle in the existing recursive
multi-claim path; that path requires homogeneous witness types AND
shared LP across all claims, neither of which holds for `(w, S)`
recursive opening. v2 extends the recursive multi-claim path to
admit per-group LP and mixed witness types (slices D, E), which is
the cleanest way to realize the book's design without introducing
parallel infrastructure.

The MSIS rank floors that the security baseline established
(`specs/security_analysis.md` §4) apply unchanged. The new setup
commitment matrices `D_S`, `B_S`, `A_S` (and their meta-tier
counterparts `D_meta`, `B_meta`, `A_meta`) are sized analogously to
the witness commitments and routed through the same
`sis_derived_*_params_for_layout` flow — the cascade-aware schedule
will pick rank floors for each role that satisfy 128-bit MSIS just
as for the witness commitments.

The claim-reduction sumcheck still adds
`(log m_row + log d) · (deg + 1) / |F_q^k|` to per-level knowledge
error, which is `≤ 2^-123` per level
(`specs/security_analysis.md` §6). Unchanged from v1.

The book's tensor-CWSS analysis (Theorem 5.4 Step 1, lines 992–1003)
is unaffected by the setup-side change; it operates on the folding
challenge structure independently.

**Net**: Phase D-full v2 does not change the per-level CWSS or
Module-SIS knowledge errors. The cascade-aware schedule planner and
the new SIS rank floors for `D_S, B_S, A_S, D_meta, B_meta, A_meta`
need to be re-audited at the end (slice I).

---

## 4 — Code surface area

Concrete file:line list of every place that must change, organized by
the slices in §6. The v2 plan deliberately reuses existing primitives
where possible; sections marked NEW add small extensions, not
parallel constructs.

### 4.1 Setup-time data shapes + lazy commitment (Slice A — DONE in working tree)

Existing:

- `crates/akita-types/src/proof/tiered_setup.rs` —
  `TieredSetupParams { shrink_factor: usize, num_chunks: usize }`,
  `TieredSetupCommitments<F, D>`, `TieredSetupProverExtras<F, D>`.
- `crates/akita-prover/src/api/tiered_setup.rs::derive_tiered_setup_full_commitments`
  — splits `S` into `k` chunks, commits each via
  `commit_with_params(chunk_params)`, accumulates the per-chunk
  `u_{S,j}` vectors into a tier-3 meta-commitment via
  `commit_with_params(meta_params)`.
- Cache hooks: `AkitaProverSetup::tiered_s_cache_get_or_init`,
  `AkitaVerifierSetup::tiered_s_cache_get_or_init`.

These are book-correct and form the foundation for v2.

### 4.2 Recursive multi-claim plumbing (Slice B — DONE in working tree)

Existing:

- `crates/akita-types/src/proof/recursive_opening_claim.rs::RecursiveOpeningClaim<'a, F>`
  (verifier-side per-claim carrier).
- `crates/akita-prover/src/protocol/flow.rs::RecursivePolyHandle<F>`
  (prover-side per-claim carrier).
- `RecursiveProverState::handles: Vec<RecursivePolyHandle<F>>`,
  `RecursiveVerifierState::claims: Vec<RecursiveOpeningClaim<'a, F>>`.
- `prove_recursive_multi_fold_with_params` (N-claim variant of
  `prove_recursive_fold_with_params`).
- `verify_one_level` `claims.len() > 1` branch with `gamma` sampling
  and per-point trace check.

**Scope today**: the multi-claim path requires (a) all claims share
ONE `LevelParams` and (b) all claims have the same witness type
(i8-digit recursive witness). Slices D and E lift these
restrictions; the path then becomes the right primitive for the
`(w, S)` recursive open.

### 4.3 Wire field for `s_opening_value` (Slice C.2.a — DONE in working tree)

Existing:

- `SetupClaimReductionPayload<F>` carries `s_opening_value: F` on the
  wire.
- `verify_setup_claim_reduction` returns `(challenges, s_opening_value)`
  and validates the closing-oracle equality
  `weight_at_point * s_opening_value == final_running_claim`.
- The transitional `setup_view.mle == s_opening_value` check is
  retained inside `verify_setup_claim_reduction` until slice F drops
  it for levels with a recursive next fold.

### 4.4 Per-group LP + multi-group commit (Slice D)

**Extend** `crates/akita-types/src/layout/params.rs::LevelParams` to
support multi-group commits. The existing single-group case stays
the dominant shape; multi-group is the L+1 case.

**Representation (chosen)**: add a `groups: Option<Vec<GroupSpec>>`
field to `LevelParams`, where `GroupSpec` carries
`(m_vars, r_vars, b_key, num_digits_open)`. `None` means the
existing single-group case (every existing call site is unaffected);
`Some(vec)` means multi-group with shared `D, A` from the outer
`LevelParams`. This avoids touching every `LevelParams` consumer
and keeps the multi-group case localized to the new code path.

**Extend** `crates/akita-prover/src/api/commitment.rs::commit_with_params`
to accept a multi-group input (or add a sibling `commit_multi_group`
that delegates to the same per-row machinery). The output is a
single `(v_joint, c_joint, per_group_u)` triple plus per-group
hints. For `groups.len() == 1`, the output collapses to the existing
shape.

**Extend** `crates/akita-verifier/src/protocol/ring_switch.rs::prepare_m_eval`
row layout to support per-commitment-group sub-rows in the B and
eval/fold groups. The D and Ajtai groups stay shared (one row family
each, sized for the union).

**Mirror** in `crates/akita-prover/src/protocol/sumcheck.rs::AkitaStage2Prover`
and `crates/akita-prover/src/protocol/setup_claim_reduction.rs::materialize_setup_claim_tables`
to compose sumcheck output across per-group sub-relations.

**Tests** at root (no recursive plumbing yet): batched root commit
with two polys at mismatched `(m, r)`. The relation closes
correctly; commit shape is `(v_joint, c_joint, per_group_u)`.

### 4.5 Mixed witness types in recursive multi-claim path (Slice E)

**Add** small wrapper to `crates/akita-prover/src/backend/`:

- `RecursiveWitnessAsPoly<'a, F, const D>` — a thin newtype around
  `RecursiveWitnessView<'a, F, D>` that implements the same trait
  surface as `DensePoly<F, D>` (the multi-claim recursive path
  consumes via this trait). Methods delegate to the existing
  `RecursiveWitnessView` ops where possible.

**Extend** `RecursivePolyHandle<F>` and `RecursiveOpeningClaim<'a, F>`
to carry per-handle/per-claim `LevelParams` (currently they share
ONE `LevelParams` per recursive level). This is the per-group LP
plumbing at the multi-claim level; slice D establishes the row
layout and commit kernel, slice E threads it through the recursive
state.

**Extend** `prove_recursive_multi_fold_with_params` to:

- Accept handles with mixed witness types (some
  `RecursiveWitnessView`, some `DensePoly`).
- Accept per-handle `LevelParams` (per-group `(m, r, B, digit_count)`
  with shared outer `D, A`).
- Dispatch through the multi-group commit kernel from slice D for
  the next-level commit.

**Mirror** in `verify_one_level` `claims.len() > 1` branch.

**Tests**: a recursive multi-claim with one `RecursiveWitnessView`
handle and one `DensePoly` handle, at mismatched `(m, r)`, completes
end-to-end on a small artificial transcript (no production caller
yet).

### 4.6 Recursive S opening protocol wiring (Slice F)

**Modify** `crates/akita-verifier/src/protocol/setup_claim_reduction.rs::verify_setup_claim_reduction`:

- Take a `routes_recursively: bool` flag (already plumbed in working
  tree). When `true`, **drop** the
  `setup_view.mle == payload.s_opening_value` check; the
  closing-oracle equality
  `weight_at_point * s_opening_value == final_running_claim` stays.
  Soundness is then anchored by the L+1 multi-group open via the
  extended row layout (slice D).

**Modify** `verify_one_level` / `verify_root_level` to:

- After a level emits a setup-claim-reduction payload AND the next
  step is a fold, push S as a second `RecursiveOpeningClaim` (with
  its own per-group `LevelParams` for the S-side `(m, r, B)`) into
  the next state's `claims` vector.
- The `S`-claim's polynomial is `DensePoly` over the precomputed
  `S` (resolved on the verifier side from the cached setup
  commitment material).

**Modify** prover side similarly: push an S `RecursivePolyHandle`
with mixed witness type (`DensePoly`) and per-group LP into the
next prover state's `handles`.

**Modify** `prove_fold_level_from_quadratic` /
`prove_root_fold_from_quadratic` to consume the per-group LP from
each handle when dispatching to the next level's commit.

### 4.7 Tiered (k = 64) extension for the S-group (Slice G)

**Extend** the multi-group commit kernel from slice D so the
`S`-group can be SUB-CHUNKED into `k` per-chunk commitments under
shared `(D_chunk, B_chunk)`, plus a tier-3 meta-commit binding the
per-chunk outputs.

**Extend** `prepare_m_eval` row layout to add the meta-tier rows
(book §5.4 groups 6–10). The per-chunk D and B sub-relations use
the block-diagonal MLE evaluator (cost `O(|D_chunk|) + O(log k)`,
independent of k).

The `w`-group is unaffected by tiering. The change is purely
internal to the S-group.

### 4.8 Schedule planner cascade formula update (Slice G)

**Modify** `crates/akita-planner/src/schedule_params.rs`:

- The cascade penalty formula is already updated to
  `w_fold_L + |S|/f` (this session, Slice C.2.c v2 amendment).
  Slice G activates the formula in production paths (when
  `routes_recursively` is on) and adds proof-byte cost
  contributions for the tier-3 meta-commitment
  `(c_meta, v_meta, u_meta)` and the per-group sub-relation rows.
- Add per-group `(m_w, r_w)` and `(m_S, r_S)` optimization at L+1:
  each group chooses its own optimal `(m, r)` subject to the SIS
  width constraint
  `n_A · δ_open · 2^{r_S} ≤ SIS_MAX_WIDTH[d]`.

**Modify** `crates/akita-config/src/lib.rs::WCommitmentConfig::PlannerConfig::planner_setup_polynomial_size`
to extend its return to also report `δ_commit,S` (digit count for `S`)
and the tiered shrink factor `f`.

### 4.9 SIS table extension (Slice H)

**Modify** `crates/akita-types/src/generated/sis_floor.rs`: if the
new schedules (with `|S|/f` cascade and the meta-tier roles) hit
collision buckets not currently in the table, extend with
`scripts/gen_sis_table.py`. Per the book's sweet spot `f = 8`,
the new SIS roles `(D_meta, B_meta, A_meta)` are sized analogously
to the witness commitments and may hit new buckets.

**Regenerate** the six fp128 schedule tables under the cascade-aware
cost model + multi-group + meta-tier proof-byte cost.

### 4.10 Security re-audit (Slice I)

Re-run the security analysis pipeline from
`specs/security_analysis.md`:

- `extract_params.py` — update to extract `D_S, B_S, A_S, D_meta,
  B_meta, A_meta` rank/widths too.
- `run_estimator_all.py` — same.
- Confirm every preset still clears 128 bits MSIS at the new schedule
  shape.

---

## 5 — Resolved design decisions

These were chosen via author sign-off on `2026-05-13`. The v2
amendments below reflect the user-confirmed reframing that the
recursive `(w, S)` open IS extended batched Hachi (per-group LP +
mixed witness types), not a parallel "split commitment" subsystem.

### D-1: Tiered v1 from the start (chosen, re-affirmed)

Ship the full book §5.4 design directly: shrink factor `f = 8`,
`k = f² = 64` chunks, tier-3 meta-commitment, multi-group stage-2
relation with per-chunk + meta-tier sub-rows. Production-ready
immediately at all NV; no cascade overflow at NV ≥ 22.

**v2 amendment**: implementation begins with the un-tiered
(`f = 1`, `k = 1`) S-group (slice F) to validate the multi-group
machinery from slices D + E end-to-end with the simpler 5-row-family
relation, then extends to tiered for production (slice G). The
un-tiered case is a code intermediate, not a protocol intermediate
— both versions use the same multi-group commit kernel and the
same recursive S-opening soundness story.

Total estimated effort (v2): ~4 weeks for slices D through I (plus
the ~3 weeks already spent on slices A through C in v1).

### D-2: `Vec<RecursiveOpeningClaim<F>>` for batched recursive opens (re-affirmed)

`Vec<RecursiveOpeningClaim<F>>` is the right primitive for the
joint `(w, S)` open at recursive levels, ONCE slice E extends the
recursive multi-claim path to admit per-claim `LevelParams` (the
"per-group LP" extension) and mixed witness types (the
`RecursiveWitnessAsPoly` wrapper).

**v2 amendment**: no separate `SplitRecursiveOpeningClaim` type is
introduced. The existing `RecursiveOpeningClaim<F>` gains a
`per_claim_lp: LevelParams` field (or analogous) so each claim
carries its own commitment-group spec; the next level's commit
dispatches to the multi-group kernel from slice D.

### D-3: `OnceLock`-lazy `c_S` derivation (chosen, re-affirmed)

Do **not** persist `c_S` alongside `shared_matrix`. Compute it
lazily on first use via an `OnceLock` field on `AkitaVerifierSetup`
(and the parallel field on `AkitaProverSetup`). One-shot cost at
first verify; no setup-file growth or versioning concerns.

### D-4: Per-group fold challenges (re-defined for v2)

The multi-group commit at L+1 has TWO commitment groups (`w` and
`S`), each with its own fold challenge vector when `(m_w, r_w) ≠
(m_S, r_S)`. The book's §5.5 Round 4 samples a single tensor
challenge `c_{p ‖ q} = α_p · β_q`; the multi-group commit extends
this so each group derives its own fold challenges of length
matching its `(m, r)` from a shared transcript-bound scalar.

For the un-tiered first cut, the planner may pick a degenerate
optimum where both groups share the same `(m, r)` (then there is a
single fold challenge vector). For tiered, the per-group splits
will differ in general.

### D-5: Per-group `LevelParams` shape (re-defined for v2)

`S`'s commitment at level `L+1` uses its own per-group
`(m_S, r_S, B_S, δ_commit,S)` distinct from the `w`-side
`(m_w, r_w, B_w, δ_open,w)`. The shared `(D, A)` matrices apply to
the joint `v` and `c_joint` outputs.

This is realized by extending `LevelParams` in place with an
optional `groups: Option<Vec<GroupSpec>>` field (per §4.4): `None`
preserves the existing single-group shape (every existing call
site is unaffected); `Some(vec)` activates the multi-group code
path with shared `D, A` from the outer `LevelParams` and per-group
`(m, r, B, digit_count)` from each `GroupSpec`.

### D-6: Fail-loud SIS table coverage (chosen, re-affirmed)

When a planner search hits a `(D, collision_bucket)` cell not in
`sis_floor.rs`, the planner errors with a clear `InvalidSetup`
pointing at `scripts/gen_sis_table.py`. The operator extends the
table, commits the new `sis_floor.rs`, and re-runs the planner.

### D-7: k = 1 first cut, k = 64 follow-up (re-affirmed)

The implementation lands the un-tiered S-group (`f = 1`, `k = 1`)
end-to-end first (slice F), then extends to tiered (`f = 8`, `k = 64`)
in slice G. The un-tiered case requires the multi-group commit
kernel + multi-group row layout (slices D, E); the tiered case adds
per-chunk sub-chunking + meta-tier on top. No protocol shape change
between the two cases — only the S-group's chunking changes.

This phasing keeps the first end-to-end cut focused on the
multi-group machinery without simultaneously introducing the
meta-tier complications.

---

## 6 — Implementation plan (slices)

Each slice ends with: `cargo fmt -q && cargo clippy --all -- -D
warnings && cargo test --release`. Each slice is one focused commit
on `feat/tensor-challenges`. Slices D through G are sequential;
slices H and I run after.

### Slice A (DONE in working tree)

Tiered setup foundation + `OnceLock`-lazy commitments. See §4.1.

### Slice B (DONE in working tree)

`Vec<RecursiveOpeningClaim>` carrier for batched recursive openings.
See §4.2.

### Slice C.1 (DONE in working tree)

Recursive multi-claim plumbing (used at root for batched openings;
extended for per-group LP + mixed witness types in slices D and E).

### Slice C.2.a (DONE in working tree)

Wire field `s_opening_value` + verifier extraction. See §4.3.

### Slice C.2.b (DONE in working tree)

Tiered types + multi-claim transcript + multi-ring shape check
infrastructure.

### Slice C.2.c partial (DONE in working tree)

Cascade-aware planner cost model with the book formula `w_fold_L
+ |S|/f` (additive). The planner currently always picks
`num_claims = 1` (cascade off in production callers); slice F
activates cascade by routing the S-group through the next level.

### Slice D (NEW — multi-group batched commit kernel)

**Goal**: extend the batched commit kernel and stage-2 row layout
to support per-commitment-group `(m, r, B, digit_count)` with
shared `D, A`. Single new primitive that subsumes both root batched
opens (when all groups share `(m, r)`) and the L+1 recursive open
(when groups differ).

**Changes**:

1. Extend `LevelParams` in place with an optional
   `groups: Option<Vec<GroupSpec>>` field per §4.4. `None`
   preserves the existing single-group shape; `Some(vec)`
   activates the multi-group code path.
2. Extend `commit_with_params` to handle multi-group inputs (or add
   a sibling `commit_multi_group` that calls into the same
   per-row machinery). Output: single `v_joint`, single
   `c_joint`, per-group `u_g`, per-group hints.
3. Extend `prepare_m_eval` row layout to support per-group sub-rows
   in the B and eval/fold groups (D and Ajtai stay shared).
4. Mirror in `AkitaStage2Prover` /
   `materialize_setup_claim_tables`.
5. Cleanup: revert the v1 routing seam additions in
   `prove_fold_level_from_quadratic` /
   `prove_root_fold_from_quadratic` (these were already reverted
   this session as part of the foundation).

**Tests**: a batched root commit with two polys at mismatched
`(m, r)`; the relation closes correctly; commit shape collapses to
the existing single-group shape when `groups.len() == 1`.

**Acceptance**: workspace test green. Multi-group root commit works;
no production caller yet.

**Estimated**: ~1 week.

### Slice E (NEW — mixed witness types in recursive multi-claim path)

**Goal**: extend `RecursivePolyHandle` / `RecursiveOpeningClaim` to
admit per-claim `LevelParams` and mixed witness types
(`RecursiveWitnessView` and `DensePoly` in the same recursive
batch). Use the multi-group kernel from slice D for the next-level
commit.

**Changes**:

1. Add `RecursiveWitnessAsPoly<'a, F, D>` newtype implementing the
   same trait surface as `DensePoly<F, D>` so both can be members
   of a recursive multi-claim batch.
2. Extend `RecursivePolyHandle<F>` and `RecursiveOpeningClaim<'a, F>`
   to carry per-handle/per-claim `LevelParams` for the per-group LP
   shape.
3. Extend `prove_recursive_multi_fold_with_params` and
   `verify_one_level` `claims.len() > 1` branch to dispatch through
   the multi-group commit kernel from slice D.

**Tests**: a recursive multi-claim with one `RecursiveWitnessView`
handle + one `DensePoly` handle at mismatched `(m, r)` completes
end-to-end on a small artificial transcript (no production caller
yet).

**Acceptance**: workspace test green. Mixed-witness recursive
multi-claim works; no production caller yet.

**Estimated**: ~3-5 days (smaller — rests on slice D primitives).

### Slice F (NEW — recursive S opening wiring, k = 1)

**Goal**: drop the per-level `mle` check for intermediate levels
with `lp.use_setup_claim_reduction` AND a recursive next fold;
soundness anchored by the L+1 multi-group open from slices D + E.

**Changes**:

1. `verify_setup_claim_reduction` — drop the `mle` check when
   `routes_recursively == true` (already plumbed in working tree).
2. `verify_one_level` / `verify_root_level` — at levels that emit
   setup-claim-reduction AND have a recursive next fold, build the
   S-side `RecursiveOpeningClaim` (with its own per-group LP for
   the S `(m, r, B, δ_commit,S)`) and push to next state's
   `claims`. Set the polynomial backing to the resolved `DensePoly`
   over the cached setup material.
3. Prover mirror: at the same levels, build the S-side
   `RecursivePolyHandle` (with `RecursiveWitnessAsPoly`-wrapped
   `DensePoly` over the cached `S`, per-group LP) and push to
   next state's `handles`.
4. Activate the planner's cascade formula
   (`planner_setup_polynomial_size > 0`) in production callers
   under `lp.use_setup_claim_reduction`.

**Acceptance**:

- E2E proof verification passes for at least one fp128 preset at
  NV ≥ 12 with `lp.use_setup_claim_reduction = true` and the per-
  level mle check disabled at intermediate levels.
- Tamper tests still reject: corrupting `s_opening_value` causes
  the recursive PCS step (slice E's relation) to reject.
- `cargo clippy --all -- -D warnings` clean; `cargo test --release`
  green.

**Estimated**: ~1 week.

### Slice G (NEW — tiered (k = 64) extension for the S-group)

**Goal**: extend slices D-F from `k = 1` (S-group is one chunk) to
the production `k = 64` tiered case per book §5.4 (S-group is `k`
sub-chunks bound by a tier-3 meta-commit).

**Changes**:

1. Extend the multi-group commit kernel from slice D so the
   S-group can be SUB-CHUNKED into `k` per-chunk commitments under
   shared `(D_chunk, B_chunk)` (column width `1/f`).
2. Add the tier-3 meta-commit step: bind the per-chunk B-side
   outputs under `(D_meta, B_meta, A_meta)`. The proof gains
   `(c_meta, v_meta, u_meta)` (independent of `k`).
3. Extend `prepare_m_eval` row layout for the meta-tier rows
   (book §5.4 groups 6–10). Per-chunk D and B sub-relations use
   the block-diagonal MLE evaluator
   (`O(|D_chunk|) + O(log k)`).
4. Extend the planner cost model to account for the meta-tier
   proof bytes and the per-chunk row count.
5. Per-group `(m_w, r_w)` and `(m_S, r_S)` optimization at L+1
   under tiered shrinkage.

**Acceptance**: E2E tests pass with `f = 8`, `k = 64` for
production configs.

**Estimated**: ~1 week.

### Slice H (NEW — SIS table extension)

**Goal**: extend `sis_floor.rs` to cover any new
`(D, collision_bucket)` cells the cascade-aware planner reaches.

**Changes**:

1. After slice G regenerates the tables, any planner-fail-loud
   errors point at uncovered cells.
2. For each uncovered cell, append `(d, collision)` to
   `scripts/gen_sis_table.py::ALL_ENTRIES`, run the script
   (Sage + lattice-estimator).
3. Commit the extended `sis_floor.rs`.
4. Re-run the planner; tables stabilize.

**Acceptance**: `cargo test --release` green; no
`InvalidSetup(missing supported B/D collision bucket ...)` errors.

**Estimated**: ~3 days.

### Slice I (NEW — security re-audit post Phase D-full v2)

**Goal**: confirm every production preset still satisfies 128-bit
MSIS and 128-bit Fiat-Shamir/CWSS after the protocol shape change.

**Changes**:

1. `scripts/security_analysis/extract_params.py` extracts the new
   meta-tier `(D_meta, B_meta, A_meta)` parameters too.
2. Re-run `run_estimator_all.py` over the expanded quadruple set
   (includes per-chunk D / B / A and meta D / B / A).
3. Update `specs/security_analysis.md` with the post-Phase-D-full-v2
   bit tables and document the multi-group + tiered invariants.
4. Confirm every production preset clears 128 bits at every Ajtai
   role.
5. Update `audit.md` and `roadmap.md` to reflect that the verifier
   is now at `O(N'^{1/4})` per level.

**Acceptance**: Every preset clears 128 bits. Per-level verifier
cost benchmark shows the asymptotic improvement vs `main`.

**Estimated**: ~1 week.

### Total estimated effort (v2 incremental)

| Slice | Effort | Cumulative (v2 incremental) |
|---|---|---|
| D: Multi-group batched commit kernel | ~1 week | ~1 week |
| E: Mixed witness types in recursive multi-claim | ~3-5 days | ~1.5 weeks |
| F: Recursive S opening wiring (k = 1) | ~1 week | ~2.5 weeks |
| G: Tiered (k = 64) extension for S-group | ~1 week | ~3.5 weeks |
| H: SIS table extension | ~3 days | ~4 weeks |
| I: Security re-audit | ~1 week | **~5 weeks** |

Slices A through C.1 + C.2.a + C.2.b + C.2.c partial (already in
working tree) carried roughly 3 weeks of foundational work that
slices D-I build on directly.

---

## 7 — What this design does *not* change

Documenting non-goals so reviewers don't expect them:

- **The recursive `(w, S)` open is NOT a novel "split commitment"
  subsystem parallel to existing batched Hachi**. It IS multi-group
  batched Hachi extended for per-group `(m, r, B)` and mixed
  witness types. Slices D and E lift those restrictions on the
  existing primitives; slices F-G use the extended primitives.
- **MSIS rank floors**: unchanged at the per-role level. The
  cascade-aware schedule with the new `(D_meta, B_meta, A_meta)`
  roles may pick different floor cells, which is the subject of
  slices H and I.
- **CWSS knowledge error per level**: unchanged. The challenge
  family + `4ω` extraction degradation are independent of the
  setup-side optimization.
- **Stage-1 sumcheck and main stage-2 sumcheck closing equality**:
  unchanged at the abstraction level. The per-group sub-row
  extension is a row-layout extension of the existing 5-group
  stage-2 inside `prepare_m_eval`; the closing equality from book
  §5.5 Round 8 step 4 is the same shape.
- **Ring-switch protocol**: unchanged.
- **Tensor challenge transcript model** (book §5.2): unchanged.
- **The base-field prime / verifier-field**: unchanged.
- **The Fiat-Shamir transcript domain labels**: only
  `CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND` and existing labels are
  reused; no new top-level domain.
- **The root-level multi-poly batched opening**: stays as-is; slice
  D's multi-group extension makes it a degenerate case (`groups
  share (m, r)`) of the same kernel.

---

## 8 — Reference index

| Topic | File | Key lines |
|---|---|---|
| Current single-poly recursive prover state | `crates/akita-prover/src/protocol/flow.rs` | `RecursiveProverState` |
| Hardcoded single-poly recursive QE | `crates/akita-prover/src/protocol/quadratic_equation.rs` | `new_recursive_prover` |
| Batched multi-poly root QE | `crates/akita-prover/src/protocol/quadratic_equation.rs` | `new_prover` |
| Single-poly verify recursive | `crates/akita-verifier/src/protocol/levels.rs` | `verify_one_level` |
| Setup-claim-reduction verifier closing oracle (the line to drop conditionally) | `crates/akita-verifier/src/protocol/setup_claim_reduction.rs` | 138-155 |
| `verify_stage2_with_setup_claim_reduction` | `crates/akita-verifier/src/protocol/setup_claim_reduction.rs` | 178-218 |
| `RecursiveVerifierState` | `crates/akita-verifier/src/protocol/levels.rs` | `pub struct RecursiveVerifierState` |
| `AkitaExpandedSetup` (where `c_S` cache lives) | `crates/akita-types/src/proof/setup.rs` | 37-42 |
| Setup expansion (PRG) | `crates/akita-prover/src/api/setup.rs` | 45-61 |
| `SetupMatrixPolynomialView` / `materialize_table` / `mle` | `crates/akita-types/src/layout/flat_matrix.rs` | 260-415 |
| `prove_setup_claim_reduction` (prover side) | `crates/akita-prover/src/protocol/setup_claim_reduction.rs` | 35-81 |
| `SetupClaimReductionPayload` | `crates/akita-types/src/proof/mod.rs` | (search) |
| `commit_with_params` (existing single-LP commit; extend in slice D) | `crates/akita-prover/src/api/commitment.rs` | 170-216 |
| `prepare_m_eval` (5-group M-eval table; extend in slice D) | `crates/akita-verifier/src/protocol/ring_switch.rs` | 376-510 |
| `derive_tiered_setup_full_commitments` | `crates/akita-prover/src/api/tiered_setup.rs` | 121-193 |
| **Book Technique 2 (claim-reduction sumcheck)** | `5_fourth_root_verifier.tex` | 511-669 |
| **Book setup opening + multi-group commit** | `5_fourth_root_verifier.tex` | 627-660 |
| **Book tiered commitment + 10 check groups** | `5_fourth_root_verifier.tex` | 672-754 |
| **Book combined protocol Round 5 (assemble w)** | `5_fourth_root_verifier.tex` | 865-870 |
| **Book combined protocol Round 8 (claim reduction)** | `5_fourth_root_verifier.tex` | 900-919 |

---

## 9 — Approval status

Resolved on `2026-05-13`:

1. **Scope**: tiered v1 from day 1 (book §5.4 directly) at the
   protocol level. Implementation phases through `k = 1` (un-tiered
   S-group) as a code intermediate per D-7 — same multi-group
   commit kernel, same recursive-S-opening soundness story, just
   the S-group's chunking differs.
2. **State shape**: `Vec<RecursiveOpeningClaim<F>>` for both
   batched root openings AND the recursive `(w, S)` open, ONCE
   slice E extends each claim to carry its own per-group
   `LevelParams` and admit mixed witness types via a
   `RecursiveWitnessAsPoly` wrapper.
3. **`c_S` storage**: `OnceLock`-lazy derivation on first use; not
   persisted.
4. **SIS coverage**: planner-fail-loud; operator extends
   `sis_floor.rs` via `gen_sis_table.py` when needed.
5. **Cascade penalty**: book formula `w_fold_L + |S|/f` (v2
   amendment, already in working tree).
6. **k = 1 first cut**: implementation lands un-tiered S-group
   end-to-end first (slice F) to validate the multi-group
   machinery, then extends to `k = 64` for production (slice G).

Implementation proceeds slice-by-slice per §6. Each slice lands as
a focused commit on `feat/tensor-challenges` so the branch is
reviewable incrementally.

---

## 10 — v1 retraction notes

The v1 of this document (committed at `95e79c54`) described the
recursive `S` opening as "lift the multi-poly batched opening that
root already supports to recursive levels". The implementation
attempt that followed pushed `S` as a second `RecursivePolyHandle`
with `w = t̂_S` and the existing `prove_recursive_multi_fold_with_params`
multi-claim machinery, treating the (w, S) open as if both shared
ONE `LevelParams` and the same i8-digit witness shape.

This was algebraically wrong, for two reasons documented for the
record:

### 10.1 Trace check disagreement

The multi-claim recursive infrastructure folds each handle's `w`
(a flat `i8` digit witness) as a polynomial via `F::from_i8`-lifted
coefficients. So pushing `w = t̂_S` makes the next level's per-point
trace check

```
trace(y_S · σ_{-1}(v_S)) = d · γ · s_opening_value
```

reject, because `mle(t̂_S)(r_setup_padded)` ≠ `mle(S)(r_setup) =
s_opening_value`. The "polynomial" the next level proves an opening
for is the digit-poly `t̂_S`, not `S` itself. This was confirmed
empirically with a debug-printed trace check failure during the
attempt.

### 10.2 Primitive needed extension, not replacement

The book's design (§5.3 lines 643-660) is multi-group batched Hachi
where `S` enters the L+1 commit:

- As a fresh `DensePoly` (NOT a digit witness like `w`).
- With its own per-group `(m_S, r_S, B_S, δ_commit,S)` distinct
  from the `w`-side.
- Sharing `D` and `A` matrices with `w` (joint `v`, joint `c`).

The existing recursive multi-claim path requires (a) all claims
share ONE `LevelParams` and (b) all claims have the same i8-digit
witness shape. The v1 attempt pushed `S` while violating both
invariants.

The v2 fix is to **extend** the multi-claim path to lift these two
restrictions (slice D adds per-group LP shape, slice E adds mixed
witness types via `RecursiveWitnessAsPoly`), NOT to introduce a
parallel "split commitment" subsystem. The book's "split
commitment" name reflects the per-group `B`-matrix; structurally
it IS multi-group batched Hachi.

### 10.3 Code disposition under v2

The working-tree additions that were attempts at the v1 routing
seam (`RecursiveSMaterial`, `build_s_recursive_material`, the
`prover_setup_for_s_routing` / `next_lp_for_s_routing` parameters,
the verifier-side `cascade_c_s` / `root_c_s` parameters, the
scheme-side `derive_c_s` closure) have been REVERTED this session.
The codebase is back to a clean baseline; slice D builds forward
from there using the extension approach above.

What stays from the working tree as foundations: tiered setup data
types (slice A), `Vec<RecursiveOpeningClaim>` for batched recursive
openings (slice B), multi-claim plumbing (slice C.1),
`s_opening_value` wire field (slice C.2.a), tiered types and
multi-claim infrastructure (slice C.2.b), the cascade-aware
planner cost-model with the book's additive formula
`w_fold_L + |S|/f` (slice C.2.c partial — production callers
activate it in slice F).
