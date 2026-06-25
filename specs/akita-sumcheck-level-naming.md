# Spec: Sumcheck level protocol vocabulary (norm / fold / setup)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-06-08                     |
| Status      | proposed                       |
| PR          | SC-MIGRATE (implementation)      |

## Summary

Historical names **stage 1 / stage 2 / stage 3** overload the word *stage* at
three different layers: wire blocks on [`AkitaLevelProof`], internal nodes inside
the norm range-check tree, and scheduled sumcheck batches in
[`LevelProtocolPlan`](crates/akita-protocol/src/plan.rs). That overload makes
the sumcheck unification cutover harder to reason about and collides with
[`StagePlan`](crates/akita-protocol/src/plan.rs), which already means
"Fiat-Shamir-scheduled batch of sumcheck instances."

This spec defines canonical vocabulary for the level protocol and a rename table
for **SC-MIGRATE**. Implementation is behavior-preserving: proof bytes,
transcript sponge bytes, and serialized layouts stay identical on deterministic
fixtures unless a future transcript version explicitly allows a label change.

Companion design docs: sumcheck unification spec (branch `quang/sumcheck-unification-spec`,
PR #163), [`specs/core-protocol-naming-cleanup.md`](core-protocol-naming-cleanup.md)
(relation / fold-chain naming).

## Intent

### Goal

Adopt a single, layer-aware vocabulary so engineers can describe:

1. **Level stages** — the three wire-shaped blocks on a fold level (norm,
   fold, setup).
2. **Norm tree nodes** — the one-or-many eq-factored sumchecks inside the norm
   stage.
3. **Scheduled batches** — entries in `LevelProtocolPlan::stages` (each a
   [`StagePlan`] of [`SumcheckInstanceDescriptor`]s).
4. **Carried claims** — points and scalar outputs passed between level stages
   (`norm_point`, `virtual_witness_claim`).

[`akita-protocol::naming`](crates/akita-protocol/src/naming.rs) is the in-crate
glossary. SC-MIGRATE applies the Rust rename table below while keeping wire
shapes byte-identical.

**Word choice:** *level stage* (norm / fold / setup) = a wire block on
[`AkitaLevelProof`]. *Norm node* = one vertex in the norm decomposition tree.
[`StagePlan`] = scheduled sumcheck batch only, not a level stage. The old
`AkitaStage1Proof::stages` field becomes `::nodes` so *stage* is not reused for
tree internals.

### Invariants

- **Proof bytes unchanged** on existing deterministic fixtures (stack guardrail).
- **Transcript sponge bytes unchanged** in production builds (diagnostic labels
  may keep legacy strings until a deliberate transcript bump).
- **`StagePlan` keeps its Rust name** but documentation must never equate it
  with a level stage (norm / fold / setup).
- **Terminal fold levels** have no norm stage structurally
  ([`LevelRole::Terminal`](crates/akita-protocol/src/plan.rs)); vocabulary says
  "norm stage absent," not "stage 1 skipped with gamma = 0."
- **One sumcheck instance** = one descriptor + one proof object (engine/sink
  unit). A norm stage is one or many instances plus inter-node child claims.

### Non-Goals

- Renaming every `stage1_*` symbol in one commit before SC-MIGRATE owns the
  files (SC-ENGINE stays stage-2-only).
- Changing eq-factored wire format or tree arity logic (`stage1_tree_stage_shapes`).
- Renaming transcript label constants without a transcript-hardening version
  decision ([`specs/transcript-hardening.md`](transcript-hardening.md)).
- Collapsing the norm tree into a single monolithic descriptor (orchestration
  stays in the plan layer).

## Vocabulary

### Layer A — Fold level

A **fold level** is one step in the recursion schedule. Its proof payload is
[`AkitaLevelProof`](crates/akita-types/src/proof/levels.rs).

### Layer B — Level stages (wire blocks)

| Legacy name | Canonical name | Rust type (today → proposed) | Proof format |
|-------------|----------------|------------------------------|--------------|
| Stage 1 block | **Norm stage** | `AkitaStage1Proof` → `NormCheckProof` | Eq-factored tree (0+ nodes) + `s_claim` |
| Stage 2 block | **Fold stage** | `AkitaStage2Proof` → `FoldProof` | One regular sumcheck + `next_w` |
| Stage 3 block | **Setup stage** | `SetupSumcheckProof` → `SetupProductProof` | One regular product sumcheck |

**Norm stage** proves the range obligation

\[
0 = \sum_z \mathrm{eq}(\tau_0, z)\, Q(S(z)), \quad S(z) = w(z)\,(w(z)+1),
\]

where \(Q\) is the range polynomial over valid digit values. For small basis
`b <= 8` this is a single eq-factored instance; for larger `b` a **norm
decomposition tree** expands \(Q\) into several nodes (see Layer C).

**Fold stage** proves the fused virtual + relation sumcheck at intermediate
levels, or relation-only at terminal levels ([`LevelRole`](crates/akita-protocol/src/plan.rs)).

**Setup stage** is optional per level; see
[`specs/setup-product-sumcheck.md`](setup-product-sumcheck.md).

### Layer C — Norm tree nodes

Inside the norm stage:

| Legacy name | Canonical name | Rust type (today → proposed) |
|-------------|----------------|------------------------------|
| `AkitaStage1Proof::stages` | **norm tree nodes** | field → `nodes` |
| One element | **norm node** | `AkitaStage1StageProof` → `NormNodeProof` |
| Shape row | **norm node shape** | `AkitaStage1StageShape` → `NormNodeShape` |

Each **norm node** is one eq-factored sumcheck plus optional **child claims**
(seeded into the next node). Product nodes and leaf LC nodes differ in
descriptor shape, not in wire role.

**Prover-only:** [`two_round_prefix`](crates/akita-prover/src/protocol/sumcheck/two_round_prefix/)
compresses the first two hypercube rounds for norm and fold sumchecks. It is not
a separate level stage; it emits ordinary eq-factored or regular round messages.

### Layer D — Unified plan (`akita-protocol`)

| Term | Meaning |
|------|---------|
| **`LevelProtocolPlan`** | Full per-level schedule: ordered scheduled batches, carried openings, transcript events. |
| **`StagePlan`** | **Scheduled sumcheck batch** — one or more instances batched with a [`BatchingScheme`], not a level stage. |
| **`SumcheckInstanceDescriptor`** | One instance: summand, kind (`Regular` / `EqFactored`), input/output claim slots. |

`plan_level` today emits only the fold stage's batch (baseline stage-2
schedule). SC-MIGRATE extends it to prepend one batch per norm node (or a single
batch when `b <= 8`), then append setup when gated.

### Layer E — Carried claims and public ids

| Legacy | Canonical | Role |
|--------|-----------|------|
| `stage1_point` | **`norm_point`** (\(\rho\)) | Random point after norm Fiat-Shamir rounds |
| `s_claim` | **`virtual_witness_claim`** | \(S(\rho) = w(\rho)(w(\rho)+1)\) carried into fold stage |
| `AkitaPublicId::EqStage1Point` | **`EqNormPoint`** | `eq(norm_point, ·)` in fold descriptor |
| `tau0` | **`norm_fold_challenges`** (prose) or keep `tau0` in code | Norm-stage fold challenge vector |

## Rename table (SC-MIGRATE)

Apply in dependency order. Wire serialization uses shape-driven layouts; renaming
structs is safe when field order and `AkitaSerialize` logic are unchanged.

### `akita-types` proof types

| Current | Proposed |
|---------|----------|
| `AkitaStage1Proof` | `NormCheckProof` |
| `AkitaStage1StageProof` | `NormNodeProof` |
| `AkitaStage1StageShape` | `NormNodeShape` |
| `AkitaStage1Proof::stages` | `NormCheckProof::nodes` |
| `AkitaStage2Proof` | `FoldProof` |
| `SetupSumcheckProof` | `SetupProductProof` |
| `stage1_tree_stage_shapes` | `norm_tree_node_shapes` |
| `stage1_tree_product_stage_arities` | `norm_tree_product_node_arities` |
| `validate_stage1_tree_basis` | `validate_norm_tree_basis` |
| `stage1_leaf_coeffs` | `norm_leaf_coeffs` |
| `stage1_interstage_batch_weights` | `norm_internode_batch_weights` |
| `reorder_stage1_coords` | `reorder_norm_table_coords` |

### `akita-protocol`

| Current | Proposed |
|---------|----------|
| `AkitaPublicId::EqStage1Point` | `AkitaPublicId::EqNormPoint` |
| (new) `stage2.rs` only | add `norm.rs` with `norm_leaf_descriptor`, `norm_product_descriptor`, `norm_tree_plan` helpers |
| `plan_level` docs | emit norm batches + fold batch (+ setup when implemented) |

### Prover modules

| Current path | Proposed path |
|--------------|---------------|
| `protocol/sumcheck/akita_stage1/` | `protocol/sumcheck/norm_check/` |
| `protocol/sumcheck/akita_stage1_tree.rs` | `protocol/sumcheck/norm_check/tree.rs` |
| `AkitaStage1Prover` (re-export) | `NormCheckProver` |

Keep `two_round_prefix/stage1.rs` file names until a dedicated rename pass, but
document modules as "norm-node prefix kernels."

### Verifier

| Current | Proposed |
|---------|----------|
| `stages/stage1.rs` | `stages/norm_check.rs` |
| `stage1_point` fields | `norm_point` |

### Do not rename (yet)

- Transcript labels (`ABSORB_*`, `CHALLENGE_*`) — diagnostics only, but keep
  stable unless transcript version bumps.
- `LevelParams::stage1_config` — couples to schedule tables; rename with planner
  touch or alias-only in SC-MIGRATE slice 0.

## SC-MIGRATE implementation slices

0. **Vocabulary landing** — this spec + `akita-protocol::naming` + doc pass (no
   proof behavior change).
1. **Norm descriptors + plan** — `norm.rs`, extend `plan_level`, verifier
   `try_evaluate` per node.
2. **Prover cutover** — registry + sink per node; tree orchestration in
   `norm_check/tree.rs`; round-equivalence + byte-identical `NormCheckProof`.
3. **Fold/setup/EOR** — remaining level stages (fold production hookup in `flow.rs`
   after PO-CUTOVER).
4. **Type rename sweep** — table above; delete `traits.rs` / `drivers/` when
   last consumer migrates.

## Evaluation

### Acceptance Criteria

- [ ] `akita-protocol::naming` module documents all layers A–E.
- [ ] SC-MIGRATE PR description links this spec and states stage/node vocabulary.
- [ ] New code in `akita-protocol` uses **norm stage / norm node / fold stage**
  in docs and identifiers; legacy names only where wire-compat requires.
- [ ] `plan_level` tests name batches by role (norm vs fold), not "stage 1/2."
- [ ] After full SC-MIGRATE: `rg 'stage1_point'` limited to transcript labels and
  serde aliases (or zero hits outside labels).

### Testing Strategy

- Byte-identical `NormCheckProof` (formerly `AkitaStage1Proof`) for `b = 8`
  single-node and `b = 16` tree fixtures.
- Descriptor `try_evaluate` matches legacy verifier final checks per node.
- No change to `cargo test` proof-size / wire roundtrip tests beyond type renames.

## Relationship to sumcheck unification

| Unification concept | Vocabulary |
|--------------------|------------|
| `SumcheckInstanceDescriptor` | One **instance** (one norm node or one fold/setup sumcheck) |
| `StagePlan` | One **scheduled batch** at a level |
| `LevelProtocolPlan` | Full **level schedule** (norm nodes + fold + optional setup) |
| `SumcheckEngine` / sink | Proves one instance; norm stage = many instances + child claims |

SC-ENGINE (#166) validated engine + registry + sink on the **fold stage** only.
SC-MIGRATE applies the same pattern to **norm nodes** first, using eq-factored
format and tree orchestration.
