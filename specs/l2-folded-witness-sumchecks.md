# Proposal: Folded-Witness L2 Sumchecks

| Field       | Value |
|-------------|-------|
| Author(s)   | Cursor agent draft, on behalf of the user |
| Created     | 2026-06-05 |
| Status      | proposal, ready for implementation |
| Parent spec | [`specs/l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) |
| Scope       | Sumcheck integration scaffolding only |

## Goal

This proposal is about the **two new sumchecks** needed for the folded-witness
L2 certificate:

```text
sumcheck 1:  sum_x z_aug(x)^2 = B_l2
sumcheck 2:  z_aug = G' · w_next
```

The parent L2 cutover also involves challenge sampling, SIS tables, schedule
regeneration, rank pricing, proof sizing, and public security docs. Those are not
the focus here. This document defines only the minimum scaffolding needed to make
the two sumchecks:

- well-defined,
- batched in the existing stage-1 and stage-2 protocol phases,
- tied to the committed recursive witness,
- descriptor-bound so prover and verifier use the same claim order,
- safe to parse at the verifier no-panic boundary.

In short: this is the **sumcheck integration proposal**. `B_l2`, `ell_hat`,
`offset_ell`, certificate-mode gating, and the temporary bucket helper are
supporting pieces only because the sumchecks need them.

## What The Two Sumchecks Prove

### Sumcheck 1: certify the norm bound

The prover wants to prove:

```text
Z_SQUARED = sum_i z[i]^2 <= B_l2
```

where `z` is the centered integer folded witness from
`DecomposeFoldWitness.centered_coeffs`.

Sumcheck proves equalities, not inequalities, so the prover adds four slack
integers `ell_0..ell_3`:

```text
Z_SQUARED + ell_0^2 + ell_1^2 + ell_2^2 + ell_3^2 = B_l2
```

Define:

```text
z_aug = z || ell_0 || ell_1 || ell_2 || ell_3
```

Then sumcheck 1 proves:

```text
sum_x z_aug(x)^2 = B_l2
```

This is the actual L2 certificate. It says the recomposed folded witness is short
enough for the chosen bound `B_l2`.

### Sumcheck 2: tie the norm proof to the committed witness

`z_aug` itself is not committed. The protocol commits to `w_next`, which contains
the decomposed digit planes:

- `z_hat`, the balanced base-`2^lb` decomposition of `z`,
- `ell_hat`, the balanced base-`2^lb` decomposition of the four slack integers.

So sumcheck 1 alone is not enough. A dishonest prover could try to prove a norm
bound for one `z_aug` while committing a different `w_next`.

Sumcheck 2 prevents that by proving:

```text
z_aug = G' · w_next
```

Here `G'` is the public gadget-recomposition map:

- on `z_hat`, it recomposes `z[coeff] = sum_d 2^(lb·d) · z_hat_d[coeff]`;
- on `ell_hat`, it recomposes `ell_j = sum_d 2^(lb·d) · ell_hat_j,d`;
- on all other `w_next` segments, it is zero.

This reduces to the same kind of `w_next(rho')` opening that stage 2 already
uses. With batched placement, the virtualization claim is folded into the
existing stage-2 point and uses the existing `next_w_eval` opening.

## Why Two Sumchecks Are Necessary

The two checks serve different roles:

- **Sumcheck 1 proves shortness.** It proves the squared norm of the recomposed
  object is exactly `B_l2`.
- **Sumcheck 2 proves consistency.** It proves that the recomposed object whose
  norm was certified is the same object encoded inside the committed
  `w_next`.

Together they give the stage-2 consistency invariant from the parent spec:

```text
certified z_aug
    == gadget recomposition of committed w_next
    == the witness opened by the existing recursive-witness opening
```

Without sumcheck 2, the L2 certificate is not tied to the committed recursive
witness. Without sumcheck 1, the verifier has no Euclidean bound to price.

## Integration Decisions

- **Bundled PR.** The proof shape, prover assembly, the two sumchecks, and
  verifier replay land together. That is the smallest end-to-end-testable unit
  for this certificate.
- **Per-level scope.** The certificate is decided independently for every fold
  level, root or recursive. A single proof may therefore contain a mix of
  `Deterministic` and `Realized` levels. The first cut uses the same
  `certificate_mode` rule for every fold level that carries the inline
  stage-1/stage-2 proof shape. Paths that do not use that shape are out of scope
  for this proposal rather than a third implicit variant.
- **Stage-2 placement is batched.** Sumcheck 2's `w_next(rho')` claim is batched
  into the existing stage-2 point. There is one combined opening, not an adjacent
  second opening.
- **fp32 falls back.** The first cut only certifies levels whose public
  field-capacity gate passes. In practice that is fp64/fp128. fp32 levels use the
  deterministic `l2_bound_squared` path and do not emit these sumchecks.
- **No dependency on S5b for sumcheck correctness.** The sumchecks only need a
  valid `B_l2` with `Z_SQUARED <= B_l2 <= l2_bound_squared`. The future L2 SIS
  ladder from S5b makes `B_l2` tighter for rank pricing, but it is not required
  to prove or test the sumchecks.
- **Fixed proof shape over maximum coverage.** `certificate_mode` evaluates the
  no-wrap gate at `B_l2 = l2_bound_squared`, not at the realized prover-chosen
  bucket. This can conservatively skip levels whose realized `B_l2` would fit,
  but it makes proof shape public and non-circular.
- **No L∞ table reuse.** Until S5b lands, tests use a self-contained
  power-of-two bucket helper. That helper is not the old L∞ collision table and
  does not price the A-role rank.

## Minimal Scaffolding Required By The Sumchecks

### 1. Public certificate mode

The verifier must know whether a level contains the two L2 sumchecks **before**
it parses the level proof. The proof cannot be trusted to say "I certify" because
that would make deserialization circular.

Add a public, proof-free mode decision:

```rust
// crates/akita-types/src/sis/l2_certificate.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2CertMode {
    /// No realized L2 certificate. The level uses the deterministic
    /// `l2_bound_squared` path and keeps the existing eq-factored stage-1 proof.
    Deterministic,
    /// The level carries the L2 certificate: sumcheck 1 and sumcheck 2 are active.
    Realized,
}

/// Decides whether this level carries the L2 certificate.
///
/// This uses only public level parameters and the field modulus. It evaluates
/// the parent no-wrap gate at the worst-case bucket `B_l2 = l2_bound_squared`,
/// so proof shape is fixed before reading any certificate bytes:
///
///   coeffs · balanced_digit_max(lb, num_digits_fold)^2
///     + 4 · l2_bound_squared < q_eff.
pub fn certificate_mode(lp: &LevelParams, q_eff: u128)
    -> Result<L2CertMode, AkitaError>;
```

`certificate_mode` is not materialized from proof bytes. It is a pure function of
the public level schedule/layout, the field used to accumulate the stage-1 L2
sum, and the deterministic `l2_bound_squared`. The default descriptor strategy is
**derived binding**: if the schedule digest already binds all inputs to this
function, the descriptor binds the active claim lists by binding that schedule.
If any input is not already schedule-bound, the descriptor must bind the
resulting per-level mode explicitly.

The mode controls:

- whether `B_l2` is present,
- whether `ell_hat` has nonzero length,
- whether stage 1 uses the fused plain-message format,
- whether stage 2 includes the virtualization claim,
- which claim order is bound in the descriptor.

### 2. The `B_l2` scalar

The only new certificate scalar outside the sumcheck proofs and committed-witness
shape is `B_l2`.

```rust
// crates/akita-types/src/proof/levels.rs
pub struct AkitaLevelProof<F: FieldCore, L: FieldCore> {
    // ... existing fields ...
    /// Present exactly when `certificate_mode == Realized`.
    ///
    /// Headerless serialization places this after `v` and before `stage1`, so
    /// the verifier reads and transcript-binds the bucket before parsing the
    /// realized stage-1 payload.
    pub l2_b_l2: Option<u128>,
}
```

The canonical source is the proof bytes. The verifier deserializes `l2_b_l2`
only after public mode selection says this level is `Realized`, validates it, and
then absorbs that exact value into the transcript:

```rust
pub fn validate_realized_bucket(
    b_l2: u128,
    l2_bound_squared: u128,
) -> Result<(), AkitaError>;
```

First cut validation only requires `b_l2 <= l2_bound_squared`; the lower bound
`Z_SQUARED <= b_l2` is enforced by the four-square equality from sumcheck 1. S5b
can later add an "on the audited L2 ladder" check.

`B_l2 = 0` is allowed only as the degenerate all-zero statement: the sumcheck
equality then forces every `z_aug` entry to be zero over the no-wrap range. Any
nonzero folded witness must use a positive bucket.

### 3. First-cut bucket helper

Before S5b lands, the prover still needs a concrete `B_l2` to construct
sumcheck 1. Use a pure helper:

```rust
pub fn select_b_l2(
    z_squared: u128,
    l2_bound_squared: u128,
) -> Result<u128, AkitaError>;
```

First cut behavior:

```text
B_l2 =
  0                                      if Z_SQUARED = 0,
  smallest power of two >= Z_SQUARED     otherwise,
  rejected if that bucket exceeds l2_bound_squared or overflows u128.
```

This is a protocol-construction seam, not rank pricing. It is valid for the
sumchecks because the checks only need `Z_SQUARED <= B_l2`. S5b replaces this
with the audited L2 ladder when rank tightening is wired.

The current four-square helper takes a `u64` target. Therefore the first cut must
either reject with `AkitaError` when `B_l2 - Z_SQUARED > u64::MAX`, or extend the
helper to a `u128` path before accepting such a certificate. The verifier must
never assume the slack fits in `u64` unless the prover-facing constructor has
checked that bound and encoded it in the proof shape.

### 4. `ell_hat` as a trailing committed segment

`ell_hat` is not a standalone proof object. It is part of the committed
`w_next`, because sumcheck 2 must open it through the same `w_next(rho')` claim.

The parent spec places `ell_hat` last:

```text
m_vars >= r_vars:  z_hat | w_hat | t_hat | (zk) | r_hat | ell_hat
else:              w_hat | t_hat | (zk) | z_hat | r_hat | ell_hat
```

Add a trailing offset:

```rust
pub struct RingRelationSegmentLayout {
    pub offset_e: usize,
    pub offset_t: usize,
    pub offset_z: usize,
    pub offset_r: usize,
    pub offset_ell: usize,
    // zk offsets ...
}
```

When the mode is `Deterministic`, `ell_len = 0` and the offset is inert. When the
mode is `Realized`, `ell_hat` is appended after `r_hat`.

This is good scaffolding because it leaves all existing offsets unchanged:
`offset_e`, `offset_t`, `offset_z`, and `offset_r` keep their current meanings.
Only the new trailing segment is added.

The trailing segment uses the parent-spec packing:

```text
delta_ell = num_digits_for_bound(32, field_bits, lb)
ell_hat   = ell_hat_0 || ell_hat_1 || ell_hat_2 || ell_hat_3
ell_len   = 4 · delta_ell
```

The packing is slack-major: all digits of `ell_0`, then all digits of `ell_1`,
and so on. `ell_len` contributes to the next-level committed-witness length and
therefore to `next_commit_coeffs` / shape validation. The next-level hypercube
round-up is computed after adding `ell_hat`. In ZK builds, existing blinding
segments keep their current positions; `ell_hat` still trails after `r_hat` and
any existing blinding-related segments already accounted for by the current
layout.

### 5. Typed claim lists for batching

The verifier and prover must derive the same batching vector in the same order.
Make the active claim order a shared type:

```rust
pub enum Stage1Claim {
    Range,
    L2,
}

pub enum Stage2Claim {
    S,
    Relation,
    Virtualization,
}

pub fn stage1_claims(mode: L2CertMode) -> &'static [Stage1Claim];
pub fn stage2_claims(mode: L2CertMode) -> &'static [Stage2Claim];
```

Expected active sets:

```text
Deterministic stage 1: { Range }
Realized stage 1:      { Range, L2 }

Deterministic stage 2: { S, Relation }
Realized stage 2:      { S, Relation, Virtualization }
```

The descriptor binds the number and order of active claims. This is the main
guard against prover/verifier drift in the batched design.

Rust enum names may stay short (`Range`, `L2`, `Virtualization`), while prose and
descriptor comments should connect them to the parent-spec names:
`norm_claim`, `l2_claim`, and `virtualization_claim`.

## Proof Shape And Serialization

`certificate_mode` determines proof shape before deserialization. Realized mode
**does not replace the full stage-1 tree with one flat sumcheck**. It keeps the
existing range-check tree and fuses the L2 claim into the root stage only:

```rust
pub enum AkitaStage1RootShape {
    EqFactored {
        stage: AkitaStage1StageShape,
    },
    FusedRangeNormRoot {
        rounds: usize,
        // Plain sumcheck shape for the fused root {Range, L2} claim set.
        sumcheck: SumcheckProofShape,
        // Root tree stages still output child claims for the remaining tree.
        child_claims: usize,
    },
}

pub struct AkitaStage1PayloadShape {
    pub root: AkitaStage1RootShape,
    // All non-root stages remain today's eq-factored range-check stages.
    pub tail_stages: Vec<AkitaStage1StageShape>,
}
```

`LevelProofShape` should carry this stage-1 shape instead of assuming every stage
is eq-factored. Deterministic levels preserve the current tree shape: an
eq-factored root plus eq-factored tail stages. Realized levels use a plain fused
root with round count:

```text
n_stage1 = max(root_range_vars, ceil(log2(coeffs + 4)))
```

Here `root_range_vars` is the number of sumcheck rounds / hypercube variables for
the root stage that `stage1_tree_stage_shapes(rounds, b)[0]` would use in
`Deterministic` mode. For `b <= 8`, stage 1 has one stage and that stage is the
root; in `Realized` mode that single stage becomes the fused plain stage. For
larger bases, the root is the first product stage of the existing range-check
tree, and all child/leaf stages remain eq-factored.

This is a refactor of `AkitaStage1Proof`, not a second stage-1 wrapper beside it:
`AkitaLevelProof` still has one `stage1` field, but the internals of that field
become `root + tail_stages + s_claim`.

`level_proof_bytes` must account for:

- the optional `B_l2` scalar on realized levels,
- `ell_hat` through the augmented `next_commit_coeffs`,
- the realized-level root-stage message format, which sends the linear
  coefficient that the eq-factored format currently omits,
- the stage-2 proof with one extra active claim but the same final
  `next_w_eval` opening.

The expected proof-size delta for the fused root is about one extra field element
per root-stage round relative to the eq-factored omission, plus the `B_l2` scalar
and any committed-witness growth from `ell_hat`.

Serialization is headerless where the current proof format is headerless, so the
verifier must derive the same `AkitaStage1PayloadShape` from public mode data
before reading stage-1 bytes. A shape mismatch is a `SerializationError`, not a
panic.

## Stage-1 Sumcheck Scaffolding

### Existing shape

Stage 1 currently proves the digit-range/norm relation with an eq-factored
message. The eq-factored format omits the linear coefficient of each round
polynomial.

### Realized-certificate shape

When the L2 certificate is active, stage 1 keeps the range-check tree but fuses
two claims at the root:

```text
Range claim: existing digit range-check claim
L2 claim:    sum_x z_aug(x)^2 = B_l2
```

The L2 summand has no global `eq(tau0, ·)` factor. Therefore the fused root-stage
message cannot use the existing eq-factored omission. Certifying levels use a
plain batched root message, then continue with the existing eq-factored child and
leaf stages for the range-check tree.

Represent that explicitly:

```rust
pub enum AkitaStage1RootProof<F: FieldCore> {
    EqFactored(AkitaStage1StageProof<F>),
    FusedRangeNormRoot(/* plain SumcheckProof-based payload + child claims */),
}

pub struct AkitaStage1Payload<F: FieldCore> {
    pub root: AkitaStage1RootProof<F>,
    pub tail_stages: Vec<AkitaStage1StageProof<F>>,
    pub s_claim: F,
}
```

The fused root emits the same child-claim wire format expected by the existing
tree. Tail stages reuse today's interstage transcript flow and batching
(`CHALLENGE_SUMCHECK_INTERSTAGE_BATCH` / `stage1_interstage_batch_weights`) with
no L2-specific changes.

### Domain alignment

`z_aug` has:

```text
n_aug = ceil(log2(coeffs + 4))
```

The root range-check stage runs over its current root-stage domain. To fuse the
claims without relying on which domain is larger, use the common root domain:

```text
n_stage1 = max(root_range_vars, n_aug)
```

Zero-extend each summand into that common domain:

```text
range_ext(x) = root_range_term(x) for x in the original root-stage domain
range_ext(x) = 0             otherwise

z_aug_ext(x) = z_aug(x)      for x < len(z_aug)
z_aug_ext(x) = 0             otherwise
```

Then:

```text
sum_x z_aug_ext(x)^2 = sum_x z_aug(x)^2 = B_l2
```

The fused root-stage round count is `n_stage1`. If the current root range-check
domain is already larger, this is exactly the parent-spec intuition: only
`z_aug` is extended. If `z_aug` is larger in some layout, the root range claim is
extended instead. The remaining engineering question is cost: avoid scanning
mostly-zero extensions more than necessary.

The fused stage-1 sumcheck runs in the same field as the existing stage-1 claim
for that level. The structural no-wrap gate in `certificate_mode` must use the
characteristic of that accumulation field; extension fields do not raise the
integer wrap ceiling because embedded base-field integers still reduce modulo the
base characteristic.

### L2 output handed to stage 2

The fused root L2 reduction outputs a point/value claim:

```text
z_aug(rho_l2) = v_l2
```

This exact claim is consumed by stage 2. The virtualization summand verifies:

```text
v_l2 = (G' · w_next)(rho_l2)
```

`rho_l2` is the stage-1 L2 output point derived by the fused root sumcheck, and
`v_l2` is the claimed terminal evaluation from that same reduction. Both are
transcript-bound by stage 1 and must be passed unchanged into the stage-2
virtualization claim before it is batched with `{S, Relation}`. A verifier must
reject if the stage-2 virtualization proof is not anchored to the stage-1 L2
output claim.

`rho_l2` is generally distinct from `rho'`: `rho_l2` is fixed by the stage-1 L2
reduction, while `rho'` is the final stage-2 sumcheck opening point where
`next_w_eval` is checked.

## Stage-2 Sumcheck Scaffolding

### Existing shape

Stage 2 already batches:

```text
s_claim
relation_claim
```

and reduces to the existing recursive-witness opening:

```text
w_next(rho')
```

### Realized-certificate shape

When the L2 certificate is active, add a third claim:

```text
virtualization_claim: z_aug = G' · w_next
```

The active stage-2 claim set becomes:

```text
{ s_claim, relation_claim, virtualization_claim }
```

All three are batched into the same stage-2 point. The final opening is still the
existing `next_w_eval`; there is no second commitment and no adjacent opening.

The verifier evaluates the new structured term using the same family of gadget
machinery as the existing ring-switch relation:

- `offset_z` and the fold gadget recomposition for `z_hat`,
- `offset_ell` and scalar gadget recomposition for `ell_hat`,
- zero contribution from `w_hat`, `t_hat`, `r_hat`, zk blinding, and padding.

This keeps the consistency proof tied to the committed `w_next`.

## Transcript And Batching Contract

Both stages derive batching coefficients from transcript challenges in the
canonical active-claim order.

Use one stage-local scalar and expand it by powers:

```text
weights(gamma, k) = [1, gamma, gamma^2, ..., gamma^(k-1)]
```

Then:

```text
Realized stage 1:      weights(gamma1, 2) over {Range, L2}

Deterministic stage 2: weights(gamma2, 2) over {S, Relation}
Realized stage 2:      weights(gamma2, 3) over {S, Relation, Virtualization}
```

`gamma1` is squeezed **only for realized levels**. Deterministic stage 1 has one
active claim (`{Range}`), so it does not need a batching scalar and keeps today's
transcript unchanged. `gamma2` is squeezed for every level, because stage 2
already batches `{S, Relation}` today and realized mode extends that list to
three claims.

`gamma1` and `gamma2` must be independent transcript challenges with distinct
labels or label contexts. Reusing today's single stage-2
`CHALLENGE_SUMCHECK_BATCH` label is acceptable only for `gamma2`, and only if the
descriptor-bound active claim count/order is included in the transcript context.
`gamma1` needs a new stage-1 batching label because deterministic stage 1 has no
L2 batching challenge.

The descriptor binding should be derived by default:

- **Default: derived binding.** The descriptor binds the schedule/layout/algebra
  values from which `certificate_mode` and active claim lists are pure functions.
- **Fallback: explicit binding.** If any mode input is not already descriptor- or
  schedule-bound, the descriptor carries per-level certificate mode and active
  claim counts/order.

The implementation must pin descriptor bytes for the chosen path. In either case,
prover and verifier must absorb/squeeze the same transcript events for the same
active claim lists.

### Event order

For a realized level, the transcript order is:

```text
1. absorb ring-switch public wires and current-level commitments as today
2. assemble z, choose B_l2, compute ell, append ell_hat to w_next
3. commit/absorb next_w_commitment, which includes ell_hat
4. absorb B_l2 as the realized L2 certificate scalar
5. squeeze stage-1 challenges and stage-1 L2 batching scalar gamma1
6. prove/verify fused stage-1 root {Range, L2}, then existing tail stages
7. squeeze stage-2 batching scalar gamma2
8. prove/verify stage-2 {S, Relation, Virtualization}
9. absorb/check next_w_eval at the batched stage-2 point
```

The key ordering invariant is wire-before-squeeze: `ell_hat` is committed through
`next_w_commitment`, and `B_l2` is transcript-bound, before any sumcheck-1
challenge depending on the L2 claim is squeezed.

## Verifier Safety Rules

Verifier-reachable code must reject malformed inputs with `AkitaError` or
`SerializationError`, never by panicking.

For this sumcheck integration that means:

- compute `certificate_mode` from public data before parsing the certificate
  shape;
- reject if `B_l2` is present when mode is `Deterministic`, or absent when mode is
  `Realized`;
- validate `B_l2 <= l2_bound_squared` before using it;
- check the structural no-wrap gate before treating the field equality from
  sumcheck 1 as an integer equality;
- reject if the stage-2 virtualization claim is not anchored to the exact
  `rho_l2` / `v_l2` output by the stage-1 L2 reduction;
- validate `ell_hat` length, digit bounds, and segment offsets before evaluating
  the `G'` map;
- bind active claim count and order in the descriptor;
- never use `unwrap`, `expect`, `todo`, `unimplemented`, unchecked indexing, or
  unchecked shape arithmetic in verifier-reachable paths.

## ZK Builds

The preferred first cut keeps transparent and ZK builds in lockstep:

- realized stage-1 uses the same masked/plain-opening convention as the existing
  stage-1 proof path under `feature = "zk"`;
- realized stage-2 uses the existing `SumcheckProofMasked` branch in
  `AkitaStage2Proof`;
- the virtualization claim participates in the same masked batch as `S` and
  `Relation`;
- `ell_hat` is part of the committed witness and therefore follows the existing
  witness blinding/commitment rules.

If implementation cost forces ZK realized certificates to be deferred, the
feature must reject `Realized` mode under `feature = "zk"` with `AkitaError` and
the descriptor must bind that policy. It must not silently drop the L2 claim.

## Tests To Pin The Sumcheck Scaffolding

The tests should focus on prover/verifier agreement for the two new sumchecks:

- `certificate_mode` pins which field/level pairs are `Deterministic` vs
  `Realized`.
- Descriptor bytes change when the active stage-1 or stage-2 claim list changes.
- Stage-1 shape test: deterministic levels keep the eq-factored message;
  realized levels use the fused plain root message and eq-factored tail stages.
- Stage-1 claim-order test: deterministic levels batch `{Range}`; realized
  levels batch `{Range, L2}`.
- Stage-1 tree test: realized root round count equals
  `max(root_range_vars, ceil(log2(coeffs + 4)))`, and non-root stages preserve
  the existing eq-factored shapes.
- Stage-1 handoff test: tampering the stage-1 L2 output point or value breaks
  the stage-2 virtualization check.
- Stage-2 shape test: deterministic levels batch `{S, Relation}`; realized
  levels batch `{S, Relation, Virtualization}`.
- Round-trip test for sumcheck 1 using the zero-slack case
  `B_l2 = Z_SQUARED`.
- Round-trip test for sumcheck 1 using nonzero four-square slack.
- Round-trip test for sumcheck 2 tying `z_aug` to `w_next`.
- Transcript test: stage-1 uses a distinct L2 batching challenge and stage-2
  expands its batching scalar over the descriptor-bound active claim list.
- Tampering `z_hat` fails.
- Tampering `ell_hat` fails.
- Tampering `B_l2` fails.
- Logging transcript test enforces `ell_hat` is committed before sumcheck-1
  challenges are squeezed.

## Non-Focus / Deferred Work

These items matter for the full L2 cutover, but they are not the main focus of
this sumcheck scaffolding proposal:

- S5b audited L2 SIS ladder and `collision_l2_sq` rank pricing.
- Generated schedule table regeneration.
- Production challenge policy `(31, 11), T = 16` after the S2 support
  certificate.
- Proof-size optimization beyond accounting for the new sumcheck payloads.
- Small-field realized certificates for fp32 beyond the deterministic fallback.

## Why This Is Good Scaffolding

- It keeps the proposal centered on the two sumchecks and their verifier replay.
- It adds only the support needed by those sumchecks: `B_l2`, `ell_hat`,
  `offset_ell`, certificate mode, and typed claim lists.
- It avoids circular parsing by making certificate mode public and proof-free.
- It preserves the existing recursive-witness opening: sumcheck 2 joins the
  current `w_next(rho')` path instead of creating a separate opening.
- It makes batching order explicit and descriptor-bound, which is the highest
  risk area for prover/verifier drift.
- It keeps non-certifying levels on the existing eq-factored stage-1 path.
- It lets the sumchecks be built and tested before S5b, without using the old L∞
  table.

## Affected Files

- `crates/akita-types/src/sis/l2_certificate.rs` (new): `certificate_mode`,
  `validate_realized_bucket`, `select_b_l2`, active claim lists.
- `crates/akita-types/src/proof/levels.rs`: `l2_b_l2`,
  refactor `AkitaStage1Proof` into root + tail stages.
- `crates/akita-types/src/proof/ring_relation.rs`: trailing `offset_ell`.
- `crates/akita-types/src/proof/shapes.rs`, `proof_size.rs`: mode-keyed shape and
  size accounting.
- `crates/akita-prover/src/protocol/ring_relation.rs`: assemble `Z_SQUARED`,
  `ell`, `z_aug`, and the L2 claims.
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`: append trailing
  `ell_hat`.
- `crates/akita-prover/src/protocol/sumcheck/`: fused stage-1 proof and stage-2
  virtualization claim.
- `crates/akita-verifier/src/stages/stage2.rs`,
  `crates/akita-verifier/src/protocol/levels.rs`,
  `crates/akita-verifier/src/protocol/slice_mle/setup_contribution/`: replay the
  two sumchecks and evaluate the `G'` contribution.
- `crates/akita-config`: bind certificate mode and active claim lists in the
  instance descriptor.

## References

- [`specs/l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md)
- `crates/akita-types/src/sis/norm_bound.rs`
- `crates/akita-types/src/sis/four_square.rs`
- `crates/akita-types/src/proof/levels.rs`
- `crates/akita-types/src/proof/ring_relation.rs`
