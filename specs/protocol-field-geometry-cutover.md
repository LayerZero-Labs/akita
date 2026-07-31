# Spec: Protocol Field Geometry Cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-06-24 |
| Status        | implemented |
| PR            | [#336](https://github.com/LayerZero-Labs/akita/pull/336) |
| Depends-on    | [`specs/runtime-ring-cutover.md`](runtime-ring-cutover.md) (**satisfied**: implemented, PR #249) |
| Coordinates-with | [`specs/protocol-core-eor-consolidation.md`](protocol-core-eor-consolidation.md) (implemented #194; module-layout reference); [`specs/extension-field-opening-batching.md`](extension-field-opening-batching.md) (EOR math / wire baseline); [`specs/digit-innermost-layout.md`](digit-innermost-layout.md) |
| Supersedes    | *(orchestration ownership extended from EOR-consolidation; that spec remains the #194 module-layout record)* |
| Superseded-by | |
| Book-chapter  | how/proving/fold-path.md |

## Summary

Akita’s `CommitmentConfig` has two field roles ([`CommitmentConfig`](../crates/akita-config/src/lib.rs), book [rings and fields](../book/src/foundations/rings-and-fields.md#base-field-coefficients-vs-extension-evaluation-points)):

| Role | Type | Used for |
|------|------|----------|
| Coefficient / ring | `Field` | Committed witnesses, setup matrices, SIS |
| Claim / challenge | `ExtField` | Public opening points, claimed evaluations, Fiat–Shamir |

`EXT_DEGREE = [ExtField : Field]`.
The clean geometry gate is whether those roles coincide:

| Geometry | Criterion | Today’s presets | EOR |
|----------|-----------|-----------------|-----|
| **Single-field** | `EXT_DEGREE == 1` (`Field` plays both roles) | all `fp128::*` | never |
| **Extension-claim** | `EXT_DEGREE > 1` (claims live in a proper extension of the coefficient field) | `fp32::*`, `fp64::*` | root when `root_tensor_projection_enabled`; suffix always |

This is **not** “128-bit fields cannot use extensions.”
A hypothetical `Prime128` + `FpExt4<Prime128>` preset would have `EXT_DEGREE == 4` and take the extension-claim path.
Production fp128 presets simply choose `type ExtField = Field` (`proof_optimized/fp128.rs`).

**Current production is not broken.** Honest fp128 proofs work, EOR never runs, wire carries 0 EOR bytes.
The problems are packaging and predicate drift:

1. Single-field auditors still read a shared `prepare_fold_inner` / `verify_fold_eor` path that imports and branches through EOR.
2. “Is EOR required?” is answered differently at root prover, suffix prover, root verifier, and planner.
3. Root verifier hardcodes `requires_reduction: false`, so extension-claim roots that should require EOR fail open if EOR is omitted.

This cutover is **minimal**:

1. Gate module dispatch on `EXT_DEGREE == 1` (the claim/coefficient coincidence).
2. Direct canonical predicates for EOR presence: root uses `root_tensor_projection_enabled`, suffix uses `EXT_DEGREE > 1`, and planner root pricing uses the width-based root tensor gate.
3. Fold prep/verify split so the single-field path has no EOR imports.
4. Fail-closed wire validation; descriptor stays **v1**.

No `OpeningGeometry` enum (that would duplicate `EXT_DEGREE`), no proof-type enum variants, no trace/backend reorg, no wire-format redesign.

## Intent

### Goal

Make the single-field fold path (`EXT_DEGREE == 1`) a short, EOR-free module tree, and make “does this level run EOR?” use the same canonical predicates in prover, verifier, and planner whenever claims live in a proper extension.

### Diagnosis (current `main`)

#### Problem A: Shared prep with a boolean fork

Root prover computes a real predicate, then always calls one helper:

```rust
// crates/akita-prover/src/protocol/core/root_fold.rs (today)
let needs_extension_reduction =
    root_tensor_projection_enabled::<F, E>(root_ring_d, opening_num_vars);
prepare_fold_inner(..., needs_extension_reduction, ...)?;
```

`prepare_fold_inner` (`fold.rs`) either runs grouped/scalar EOR + root tensor projection or skips it.
`PreparedFold` always carries `Option<ExtensionOpeningReductionProof<E>>`.
When `EXT_DEGREE == 1` that option is always `None`, but the type bounds and imports still pull extension-claim machinery into the single-field read path.

#### Problem B: Fragmented EOR predicates

| Site | Rule today |
|------|------------|
| Root prover | `root_tensor_projection_enabled(ring_d, num_vars)` |
| Suffix prover | `EXT_DEGREE != 1` |
| Root verifier (single-group and multi-group) | Hardcoded `requires_reduction: false` |
| Suffix verifier | `EXT_DEGREE != 1` |
| Planner root byte budget | `extension_opening_width <= 1 → 0`, else always budget EOR (no root `num_vars` gate) |

All `EXT_DEGREE == 1` presets agree (never).
Extension-claim roots (`EXT_DEGREE > 1`) can disagree across prover / verifier / planner.

#### Problem C: Root verifier fail-open

```rust
// crates/akita-verifier/src/protocol/core/root_fold.rs (today, single-group)
let root_eor = verify_fold_eor::<F, E, T>(
    extension_opening_reduction,
    /* ... */,
    false, // requires_reduction hardcoded
    transcript,
)?;
```

`verify_eor_sumcheck` only rejects a missing EOR when `requires_reduction && EXT_DEGREE != 1`.
With `false`, an extension-claim root that should have EOR can omit it and still verify.

### Invariants

- Protocol math for both geometries unchanged (EOR degree, stages 1/2/3, ring switch, digit-innermost opening points).
- All current `EXT_DEGREE == 1` presets (today’s fp128 production set) remain byte-identical after wire hardening.
- Verifier no-panic contract unchanged.
- Preset/module dispatch uses only `EXT_DEGREE == 1` vs `> 1` (claim field coincides with coefficient field or not).
- When `EXT_DEGREE > 1`, root EOR presence follows `root_tensor_projection_enabled` and suffix EOR presence follows `EXT_DEGREE > 1`; planner root pricing uses the same width-based root tensor gate. The verifier enforces exact presence at the `verify_eor_sumcheck` boundary: a payload is present if and only if the relevant predicate holds, so both an omitted required reduction and an unsolicited one reject.
- `fold/single_field.rs` must not reference `extension_opening_reduction`, `tensor_root_projection`, or `RootTensorProjectionPoly`. Enforced by the compiler: the module uses explicit imports only (no `super::*` glob), so any EOR reference fails name resolution.

### Non-Goals

- Removing fp32/fp64 support, or forbidding future fp128+extension presets.
- Changing EOR sumcheck mathematics.
- Duplicating stage 1/2/3 prove/verify per geometry.
- `OpeningGeometry` enum on `CommitmentConfig` (duplicates `EXT_DEGREE`).
- Marker traits keyed off field bit-width or preset family name.
- `PreparedFold` geometry-tagged variants.
- Trace-weight or backend trait file splits.
- `FoldLevelProof` SingleField / ExtensionClaim enum variants.
- Per-level geometry tag bytes or descriptor version bump.
- Cargo feature `extension-opening-reduction`.
- Trait objects on fold hot paths.
- Public dual APIs (`batched_prove_single_field` / `batched_prove_extension_claim`); internal `const` dispatch only.

## Evaluation

### Acceptance Criteria

**Phase A — predicate + drift fixes**

- [x] Root prover, suffix prover, root verifier, suffix verifier, and planner root EOR bytes all use the canonical root/suffix predicates (no leftover hardcoded root `false` at verifier sites).
- [x] Unit table test: `EXT_DEGREE == 1` always false; `EXT_DEGREE > 1` suffix always true; root matches `root_tensor_projection_enabled`.
- [x] Extension-claim root missing EOR when required → `InvalidProof` (fail-closed).
- [x] Current fp128 multifold still has zero EOR wire bytes.

**Phase B — fold module split**

- [x] `prepare_single_field_fold` / `prepare_extension_claim_fold` share existing `FinishFoldArgs` / `finish_prepared_fold`.
- [x] Matching verifier prefix split; shared `prove_fold` / `verify_fold` after prep.
- [x] `const { E::EXT_DEGREE == 1 }` dispatch at `prove_root` / batched prove and verify entry (not preset-family or bit-width checks).
- [x] Compiler enforces: single-field fold module has no EOR / root-tensor-projection imports.
- [x] Existing fp128 and fp32/fp64 e2e round-trips green, including multi-group root.

**Phase C — wire harden + docs**

- [x] Deserialize rejects non-empty EOR when `extension_degree == 1` or shape says absent.
- [x] fp128 byte fixtures roundtrip identical.
- [x] Book stub `how/proving/fold-path.md` + SUMMARY entry; architecture/config describe claim-vs-coefficient coincidence, not “fp128 vs small field.”
- [x] Spec status → `implemented`; optional supersede note on EOR-consolidation orchestration ownership.

### Testing Strategy

Keep existing:

- fp128 multifold / recursive round-trips (0 EOR bytes).
- fp32 ext4 EOR presence tests (`akita-pcs` fp32_ext4).
- Multi-group root / recursive multi-group tests.

Add:

- Root tensor gate alignment table (`akita-types`).
- Root fail-closed: tamper omit EOR on a tensor-root schedule → `InvalidProof`.
- `fp128_multifold_proof_has_no_extension_opening_reduction` structural assert (if not already covered).

### Performance

No intentional single-field (`EXT_DEGREE == 1`) proof-size or prover-time change.
Phase A may correct extension-claim **planner** root EOR byte budgets when the root tensor gate is off (size estimate only; honest proofs already omit that EOR).

## Design

### Gating criterion (claim field vs coefficient field)

Two nested questions:

```text
1. Preset / module dispatch
   EXT_DEGREE == 1  ?
      yes → single-field modules (no EOR imports ever)
      no  → extension-claim modules

2. Inside extension-claim only: per-level EOR
   Root   → root_tensor_projection_enabled(ring_d, num_vars)
   Suffix → true whenever EXT_DEGREE > 1
```

**Why not gate on “fp128” / field bit-width?**
Nothing in the protocol forbids `ExtField = FpExt4<Prime128>`.
Production presets choose coincidence (`Field = ExtField`); that is a product choice, not an algebraic prohibition.
Gating on `EXT_DEGREE` preserves every current production preset and correctly classifies any future large-field + extension-claim preset.

**Why not an `OpeningGeometry` enum?**
It would be a third copy of the same fact already carried by `CommitmentConfig::EXT_DEGREE` and the instance descriptor’s `extension_degree`.

**Naming**

| Name | Meaning |
|------|---------|
| **Single-field** | Claim field coincides with coefficient field (`EXT_DEGREE == 1`). Keep this name (matches config docs). |
| **Extension-claim** | Claims/challenges live in a proper extension; EOR (and root tensor projection when enabled) bridge back to base-field witnesses. Replaces the older “tensor-projection geometry” label. |
| **Tensor projection** | Keep as the **mechanism** name for root packing (`RootTensorProjectionPoly`, `tensor_root_projection`), not as the geometry name. |

### Architecture

```text
CommitmentConfig::EXT_DEGREE          (preset fact = [ExtField:Field]; already on config + descriptor)
        │
        ├── == 1  →  fold/single_field.rs     (no EOR)
        └── > 1   →  fold/extension_claim.rs
                        │
                        ▼
              root/suffix EOR predicates
                        │
                        ├── prove_root / prove_suffix
                        ├── verify_root / verify_suffix
                        └── planner proof_size
                                │
                                ▼
                        shared prove_fold / verify_fold
```

### Canonical EOR predicates

Root EOR uses `root_tensor_projection_enabled::<F, E>(ring_d, opening_num_vars)`.
Suffix EOR uses `E::EXT_DEGREE > 1`.
Planner root pricing uses the same root tensor gate through the internal width-based helper, because schedule pricing has an extension width instead of concrete field types.

When `E::EXT_DEGREE == 1`, both predicates are false (`root_tensor_projection_enabled` already requires `width > 1`).
Single-field modules should not call the EOR path; `const` dispatch already excluded them.
Extension-claim modules call the direct predicate for the level they are preparing or verifying.

**Unit tests cover:**

| Case | expected |
|------|----------|
| `EXT_DEGREE == 1` (e.g. fp128 `Field = ExtField`), Root or Suffix | `false` |
| `EXT_DEGREE == 4` (e.g. fp32-ext4), Suffix, any | `true` |
| `EXT_DEGREE == 4`, Root, `ring_d`/`num_vars` that pass `root_tensor_projection_enabled` | `true` |
| `EXT_DEGREE == 4`, Root, `num_vars` too small for root gate | `false` |

### Proposed diffs (explicit)

#### Diff 1 — root prover uses root tensor gate

**File:** `crates/akita-prover/src/protocol/core/root_fold.rs`

**After:**

```rust
let needs_extension_reduction =
    root_tensor_projection_enabled::<F, E>(root_ring_d, opening_num_vars);
```

(Phase B deletes the bool argument entirely by calling the geometry-specific prep function instead.)

#### Diff 2 — suffix prover uses extension degree

**File:** `crates/akita-prover/src/protocol/core/suffix.rs`

**After (two sites):**

```rust
let needs_reduction = E::EXT_DEGREE > 1;
// ...
let needs_extension_reduction = E::EXT_DEGREE > 1;
```

#### Diff 3 — root verifier fail-closed

**File:** `crates/akita-verifier/src/protocol/core/root_fold.rs`

**Today:**

```rust
let root_eor = verify_fold_eor::<F, E, T>(
    extension_opening_reduction,
    &group_points,
    openings,
    &row_coefficients,
    opening_batch,
    basis,
    root_lp,
    false,
    transcript,
)?;
```

**After:**

```rust
let requires_reduction = root_tensor_projection_enabled::<F, E>(
    root_lp.role_dims().d_a(),
    opening_batch.max_num_vars(),
);
let root_eor = verify_fold_eor::<F, E, T>(
    extension_opening_reduction,
    &group_points,
    openings,
    &row_coefficients,
    opening_batch,
    basis,
    root_lp,
    requires_reduction,
    transcript,
)?;
```

#### Diff 4 — root verifier fail-closed (multi-group)

**File:** same `root_fold.rs` (`verify_root_inner`, reached via `verify_extension_claim_root_prefix`)

**Today:** `verify_fold_eor(..., false, transcript)?` when EOR is present (and the missing-EOR path never requires it).

**After:** compute `requires_reduction` with `root_tensor_projection_enabled` and the multi-group opening layout’s `max_num_vars()` / A-role `ring_d`, then:

- Always call `verify_fold_eor` with that flag (including when `extension_opening_reduction` is `None`), **or**
- If keeping the `if extension_opening_reduction.is_some()` branch, still reject when `requires_reduction && extension_opening_reduction.is_none()`.

Preferred: one call path, same as single-group.

#### Diff 5 — suffix verifier uses extension degree

**File:** `crates/akita-verifier/src/protocol/core/suffix.rs`

**After:**

```rust
let requires_extension_reduction = E::EXT_DEGREE > 1;
```

#### Diff 6 — planner root EOR bytes

**File:** `crates/akita-types/src/layout/proof_size.rs`

**Today:**

```rust
pub fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    fold_level: usize,
    key: PolynomialGroupLayout,
    input_witness_len: usize,
) -> Result<usize, AkitaError> {
    if extension_opening_width <= 1 {
        return Ok(0);
    }
    // ... always budgets EOR for width > 1
}
```

**After (recommended signature extension):**

```rust
pub fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    fold_level: usize,
    key: PolynomialGroupLayout,
    input_witness_len: usize,
    ring_d: usize, // NEW: A-role fold ring for this level
) -> Result<usize, AkitaError> {
    let opening_num_vars = if fold_level == 0 {
        key.num_vars()
    } else {
        padded_boolean_opening_vars(input_witness_len)?
    };
    let requires_eor = if fold_level == 0 {
        root_tensor_projection_enabled_for_width(
            extension_opening_width,
            ring_d,
            opening_num_vars,
        )
    } else {
        extension_opening_width > 1
    };
    if !requires_eor {
        return Ok(0);
    }
    // ... existing partials / opening_vars byte math unchanged
}
```

**File:** `crates/akita-planner/src/schedule_params.rs` (~552)

Pass root `ring_d` (from the candidate / policy A-role dimension) into the updated helper.

**File:** `crates/akita-planner/src/schedule_params/suffix_dp.rs` (~534)

Pass the suffix level’s A-role `ring_d` likewise.

The width helper is internal to `akita-types`; public protocol code keeps calling the typed root predicate directly.

#### Diff 7 — fold module split (Phase B)

**Files:**

- Split `crates/akita-prover/src/protocol/core/fold.rs` into:
  - `fold/mod.rs` — shared `PreparedFold`, `FinishFoldArgs`, `finish_prepared_fold`, `prove_fold`
  - `fold/single_field.rs` — `prepare_single_field_fold` (no EOR imports; used iff `EXT_DEGREE == 1`)
  - `fold/extension_claim.rs` — `prepare_extension_claim_fold` (owns EOR + `RootTensorProjectionPoly`)
- Mirror under `crates/akita-verifier/src/protocol/core/fold/`:
  - `fold/mod.rs` — shared `verify_fold` and the common `FoldPrefix` produced by both geometry prefixes
  - `single_field.rs` — root, terminal-suffix, and recursive-suffix prefixes that never reference EOR
  - `extension_claim.rs` — root and suffix prefixes plus EOR replay (`verify_eor_sumcheck`, exact presence) keyed off the direct root/suffix predicates
  - A scalar root is the one-group case of the same grouped `verify_root_inner` path; there is no separate scalar orchestration

**Delete** the `needs_extension_reduction: bool` parameter from the shared prep entry.
Dispatch at callers on **extension degree only**:

```rust
// prove_root (sketch)
if const { <E as ExtField<F>>::EXT_DEGREE == 1 } {
    prepare_single_field_fold(...)?
} else if root_tensor_projection_enabled::<F, E>(root_ring_d, opening_num_vars) {
    prepare_extension_claim_fold(..., /* run EOR */ true)?
} else {
    // EXT_DEGREE > 1 but root tensor gate off: extension-claim types, skip EOR
    prepare_extension_claim_fold(..., /* run EOR */ false)?
}
```

Alternatively keep a single extension-claim prep that takes `run_eor: bool` **only inside** `extension_claim.rs`, never in the single-field module.

**Reuse** existing `FinishFoldArgs` / `finish_prepared_fold` (`fold.rs` ~218+). Do not invent a second finish path.

**Import isolation:** `fold/single_field.rs` must not import EOR / root-tensor-projection symbols.
The compiler is the gate; no separate `rg` CI script.
Implemented via explicit imports in both `single_field.rs` files (no `use super::super::*` glob, which would silently re-expose the core module's EOR re-exports).

#### Diff 9 — wire fail-closed (Phase C)

**File:** `crates/akita-types/src/proof/wire.rs` (deserialize helpers for `extension_opening_reduction`)

**Today:** shape-driven `Option` deserialize.

**After:** after deserialize, if the instance descriptor (or caller-provided) `extension_degree == 1` and the option is `Some`, return `SerializationError` / `InvalidProof`.
If shape says EOR absent but bytes remain, reject (if not already).

No new preamble fields.
`AKITA_INSTANCE_DESCRIPTOR_VERSION` stays `1`.

#### Diff 10 — book stub (Phase C)

**Add** `book/src/how/proving/fold-path.md`:

- Single-field walk (`EXT_DEGREE == 1`): `prove_root` → `prepare_single_field_fold` → `prove_fold`
- Explicit “claim field = coefficient field; no EOR”
- Link foundations [base-field coefficients vs extension evaluation points](../foundations/rings-and-fields.md#base-field-coefficients-vs-extension-evaluation-points)
- Link `root-fold-ring-switch.md`, `sumcheck-stages.md`, `opening-points-layout.md`
- Link `extension-opening-reduction.md` as the extension-claim path only

**Edit** `book/src/SUMMARY.md` under “The proving protocol”.

**Revise** `how/proving/extension-opening-reduction.md` intro: used when `EXT_DEGREE > 1`, not “small-field only.”

### Alternatives Considered

| Alternative | Why rejected for this cutover |
|-------------|-------------------------------|
| Gate on fp128 / field bit-width | Wrong criterion. Extensions over 128-bit primes are allowed; production just does not use them. |
| `OpeningGeometry` enum on `CommitmentConfig` | Third authority beside `EXT_DEGREE` and descriptor `extension_degree`. |
| Keep “tensor-projection” as the geometry name | Tensor projection is a root mechanism; the geometry is claim/coefficient mismatch. |
| Proof enum variants (`IntermediateSingleField` / `...ExtensionClaim`) | Large churn; wire unchanged. Harden `Option` deserialize instead. |
| Trace / backend file splits | Single-field auditors following fold prep never need them for the EOR problem. Defer. |
| Delete `root_tensor_projection_enabled` | Still the correct root rule when `EXT_DEGREE > 1`; call it directly. |
| Descriptor v2 / per-level geometry tags | Fragility is caller convention, not missing wire fields. |

## Documentation

| Doc | Change |
|-----|--------|
| This spec | Promote as tracked `proposed` when approved |
| `how/proving/fold-path.md` | New stub (Phase C): single-field path when claim = coefficient field |
| `how/proving/extension-opening-reduction.md` | Used when `EXT_DEGREE > 1` |
| `how/architecture.md` / `how/configuration.md` | Table: `EXT_DEGREE` / claim-vs-coefficient coincidence; link fold-path |
| `foundations/rings-and-fields.md` | Already owns the conceptual framing; cross-link from fold-path |
| `protocol-core-eor-consolidation.md` | After Phase C: optional `Superseded-by` for orchestration ownership only |
| `extension-field-opening-batching.md` | Cross-link; do not full-supersede |
| `docs/doc-blast-radius.json` | Add region when Phase B module paths land |

## Execution

### Phases

| Phase | Scope | Exit |
|-------|-------|------|
| **A** | Diffs 1–7 (predicate + all call sites + planner) | Fail-closed root; aligned planner bytes; no module split yet |
| **B** | Diff 8 (fold split + const dispatch + CI grep) | Single-field (`EXT_DEGREE == 1`) audit path has no EOR symbols |
| **C** | Diffs 9–10 (wire harden + book stub) | Spec `implemented`; fixtures identical |

Phase A can ship alone as a correctness fix for extension-claim root fail-open.
Phase B is the auditor-path win.
Phase C is documentation + deserialize belt-and-suspenders.

### Risks

| Risk | Mitigation |
|------|------------|
| Multi-group root EOR regression when splitting prep | Keep `prove_grouped_extension_opening_reduction` / `verify_extension_claim_root_prefix` in `extension_claim` module; run multi-group e2e in Phase B |
| Planner signature churn for `ring_d` | Localize to `extension_opening_reduction_level_bytes` + two call sites |
| Over-splitting before predicate lands | Do Phase A first; Phase B deletes the bool |
| Someone gates on “fp128” / bit-width by accident | Spec + review: only `EXT_DEGREE == 1` for single-field module entry |

### Deferred backlog (not acceptance blockers)

- Trace-weight field vs ring module split
- Backend trait narrowing for single-field
- `FoldLevelProof` geometry-tagged Rust variants (wire-neutral)
- Full book auditor walkthrough and transcript golden matrix
- Broader `specs/` archive sweep per `PRUNING.md`

## References

- `crates/akita-prover/src/protocol/core/{fold,root_fold,suffix}.rs`
- `crates/akita-verifier/src/protocol/core/{fold,root_fold,suffix}.rs`
- `crates/akita-types/src/proof/batch.rs` (`root_tensor_projection_enabled`)
- `crates/akita-types/src/layout/proof_size.rs` (`extension_opening_reduction_level_bytes`)
- `crates/akita-types/src/proof/wire.rs`
- `crates/akita-planner/src/schedule_params.rs` (root EOR bytes)
- [`specs/runtime-ring-cutover.md`](runtime-ring-cutover.md) (prerequisite, implemented #249)
- [`specs/protocol-core-eor-consolidation.md`](protocol-core-eor-consolidation.md) (#194)
- [`specs/extension-field-opening-batching.md`](extension-field-opening-batching.md)
- PR #309 / #331 (multi-group / mixed-D EOR; must stay green)
)
