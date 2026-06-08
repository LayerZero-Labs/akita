//! Canonical vocabulary for the Akita level sumcheck protocol.
//!
//! Historical **stage 1 / 2 / 3** numbers overload *stage*. Prefer the named
//! level stages and norm nodes below. Full rename table:
//! `specs/akita-sumcheck-level-naming.md`.
//!
//! **Level stage** = norm, fold, or setup (a wire block on [`akita_types::AkitaLevelProof`]).
//! **Norm node** = one vertex in the norm decomposition tree (one eq-factored
//! sumcheck). **`StagePlan`** = scheduled sumcheck batch only, not a level stage.
//!
//! # Layer A — Fold level
//!
//! One step in the recursion schedule. Wire payload:
//! [`akita_types::AkitaLevelProof`].
//!
//! # Layer B — Level stages (wire blocks)
//!
//! | Canonical | Legacy | Role |
//! |-----------|--------|------|
//! | **Norm stage** | stage 1 block | Range check over \(S = w(w+1)\); eq-factored tree |
//! | **Fold stage** | stage 2 block | Fused virtual + relation (or relation-only at terminal) |
//! | **Setup stage** | stage 3 block | Optional setup product sumcheck |
//!
//! # Layer C — Norm tree nodes
//!
//! The norm stage contains one or more **norm nodes** (legacy:
//! `AkitaStage1Proof::stages` → `NormCheckProof::nodes`). Each node is one
//! eq-factored sumcheck plus optional child claims for the next node.
//!
//! # Layer D — Unified plan ([`crate::plan`])
//!
//! - [`crate::plan::LevelProtocolPlan`]: full per-level Fiat-Shamir schedule.
//! - [`crate::plan::StagePlan`]: **scheduled sumcheck batch** (not a level stage).
//! - **Sumcheck instance**: one descriptor + one proof object (engine/sink unit).
//!
//! # Layer E — Carried claims
//!
//! | Canonical | Legacy | Role |
//! |-----------|--------|------|
//! | `norm_point` | `stage1_point` | Output point \(\rho\) after norm rounds |
//! | `virtual_witness_claim` | `s_claim` | \(S(\rho)\) fed into the fold stage |
//! | `EqNormPoint` | `EqStage1Point` | Public weight `eq(norm_point, ·)` in fold descriptor |
//!
//! Terminal fold levels omit the norm stage structurally via
//! [`crate::plan::LevelRole::Terminal`], not via a numerical shortcut.
