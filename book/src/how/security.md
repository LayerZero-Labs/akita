# Security model

> **Status:** stub. Part of the initial Akita Book scaffold.

One canonical security narrative (not parallel truths): the hardness assumption,
how Ajtai ranks connect to security bits, the weak-binding fold-price story, and
a changelog footnote for the evolving norm regimes (anchored → committed-fold L∞
→ proposed Euclidean model). Keep the marketing claim separate from audited
reality (see [Introduction → Security status](../intro.md#security-status-honest)).

## SIS / MSIS and Ajtai sizing

The Module-SIS assumption, the SIS security-floor tables that map parameters to
a minimum secure rank, the modulus families, and how the two norm models
(ℓ∞ and ℓ₂) are priced.

**Sources to fold in**

- `crates/akita-types/src/sis/mod.rs`, `ajtai_key.rs`, `generated_sis_linf_table/`.
- Paper §2.2 `def:msis`, §3.12 `sec:batched-soundness` ("MSIS targets", "Two norm models").
- `docs/security-posture.md`, `specs/akita-sis-consolidation.md`.

## Norm bounds and weak binding

The fold-response bounds, the committed-fold price as relaxed binding, the
batched weak-opening definition, and why range checks do not lower the binding
norm. Keep the fold-reprice correction explicit.

**Sources to fold in**

- `crates/akita-types/src/sis/norm_bound.rs`, `layout/digit_math.rs` (`optimal_m_r_split`).
- Paper §3.12 `sec:batched-soundness` (`def:batched-weak-opening`, `lem:batched-weak-binding`, `prop:committed-fold-price`).
- `specs/weak-binding-norm-fix.md` (fold reprice — keep the correction section).

## Euclidean security model

The proposed Euclidean-model upgrade: operator-norm-capped folding challenges,
the four-square norm witness, and the accepted-support floor that keeps the
challenge set large. Cross-link the operator-norm certification background in
[Foundations](../foundations/operator-norm-certification.md) and the roadmap
page [Euclidean security model](../roadmap/euclidean-security.md).

**Sources to fold in**

- `specs/l2-msis-opnorm-folded-witness.md` (flagship active spec; S1/S4/S7 done, rest open).
- `crates/akita-challenges/src/sampler/op_norm.rs`.
- Paper §3.12 (fold-response ℓ₂ bounds), App C `sec:opnorm-support` (accepted-support floor).
