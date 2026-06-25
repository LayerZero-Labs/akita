# Akita Planner

The `akita-planner` crate is responsible for computing the parameters of each fold level in the Akita PCS, with the goal of minimizing proof size for a given field, ring dimension, number of variables, and number of polynomials to be batched.

This module is independent of the `Cfg` trait because `Cfg` uses the planner; if the planner named concrete configs directly, the workspace would face a circular dependency. All inputs that the planner needs from `Cfg` are therefore passed through the plain-value `PlannerPolicy`.

The planner covers the parameter-selection features supported by Akita, including batching, tiered commitments, tensor challenges, extension fields, and `zk` proof-size accounting. For each case it resolves the fold parameters that minimize the modeled proof size.

The planner can also generate and cache schedule values when a preset wants a shipped table. Later runtime calls can fetch and expand those compact entries quickly instead of repeating the heavy dynamic-programming search. If no cached value is available, the same deterministic planner computes the schedule on demand.

## What The Planner Optimizes

Akita proofs can either ship a witness directly or recursively fold the witness through one or more levels before the terminal direct-send step. The planner chooses the cheapest sequence of those steps.

The output is an `akita_types::Schedule`: either a root-direct `Step::Direct`, or one or more `Step::Fold` entries followed by a terminal `Step::Direct`, plus the total modeled byte count.

## Inputs And Outputs

The public search entry point is `find_schedule(key, &policy, ring_challenge_config, fold_challenge_shape_at_level)`.

`key: AkitaScheduleLookupKey` describes the supported scalar same-point root
opening shape with two fields:

- `num_vars`: the number of Boolean variables in the opened polynomial domain
  (shared opening-point arity).
- `num_polynomials`: the number of polynomials in the single commitment group,
  opened at the shared point (one claim per polynomial).

Root witness multiplicities are not stored in the key. For the scalar
same-point batch, the `t` and `w` multiplicities are just `num_polynomials` and
the `z` multiplicity is always `1`; call sites pass those directly where the
width helpers need them.

`policy: PlannerPolicy` is the `Cfg`-free projection of a preset:

- Ring dimension and SIS modulus family.
- Decomposition parameters, including the basis search range.
- Claim and challenge extension degrees.
- Ring-subfield norm bound.
- One-hot chunk size.
- Whether tiered commitments are enabled.

The `ring_challenge_config` closure supplies the sparse challenge configuration for a ring dimension, and the `fold_challenge_shape_at_level` closure supplies the fold challenge shape for a level. They are closures instead of config methods so the planner stays independent of `CommitmentConfig`.

## Resolution Flow

Most runtime callers use `resolve_schedule`, not `find_schedule` directly. `resolve_schedule` is the planner's cache-then-generate entry point:

1. The caller passes the preset's optional `GeneratedScheduleTable` catalog.
2. If a catalog is supplied, `resolve_schedule` validates its embedded identity against the runtime policy and hook closures.
3. If the validated table contains the lookup key, it expands the compact `GeneratedScheduleTableEntry` with `schedule_from_entry`.
4. If there is no catalog or no matching entry, it calls `find_schedule` and regenerates the schedule from scratch.

Both paths are deterministic functions of the lookup key, `PlannerPolicy`, and the two closures. This is important because prover and verifier must resolve the same schedule before the Fiat-Shamir transcript is bound.

## Search Model

For a fixed field, ring dimension, decomposition policy, and opening shape, the planner mainly searches over:

- `log_basis`: the balanced-digit base used by the fold level.
- `r_vars`: the number of block-index variables, which determines `num_blocks = 2^r`.
- `m_vars`: the number of variables inside each block.

Once those values are chosen, the rest of the level is derived rather than independently searched. Digit counts, collision bounds, matrix widths, and SIS-secure ranks come from the shared `akita_types::sis` helpers. The planner builds the A, B, and D Ajtai key parameters from those derived values and then scores the resulting proof size.

Conceptually, a candidate level answers two questions:

- How many bytes does it cost to prove the next witness?
- How many field elements will the next witness contain?

The first question determines whether the current fold is worthwhile. The second question determines how expensive later recursive levels can be.

## Root Level Search

The root level starts from the original witness length `2^num_vars`. It is the only level that sees the full root batching shape from the lookup key.

At the root, the planner iterates over the configured `log_basis` range and over valid `r_vars` values. For each candidate it derives:

- `m_vars = reduced_vars - r_vars`, where `reduced_vars` accounts for the ring dimension.
- The A-role committed block width.
- The B-role opening/check width.
- The D-role prover witness width.
- The SIS-secure ranks `n_a`, `n_b`, and `n_d`.
- The next witness length with a D block and without a D block.

Batching is folded directly into the root B and D widths. A batched root does not first plan a singleton layout and scale it later; the matrix widths are sized for the actual `num_polynomials` count.

The planner also considers the root-direct case, where the schedule ships the original witness directly. That gives the DP a baseline: folding is selected only if the fold proof plus its suffix is smaller than direct shipping.

## Recursive Suffix Search

Recursive levels do not enumerate the full exponential tree of all possible `(log_basis, r_vars)` choices at every depth. That would make schedule search too expensive as the number of levels grows.

Instead, for each recursive `log_basis`, `derive_candidate_level_params` scans the valid `r_vars` choices and keeps the candidate that minimizes the next witness length. This is a local shrinking rule: recursive levels commit dense balanced-digit witnesses, and reducing the next witness length is the main driver for making the remaining suffix cheaper.

After that candidate is chosen, the suffix DP still performs the important global comparison:

- Terminate now and ship the current packed-digit witness with `Step::Direct`.
- Fold once more and pay the current level proof bytes plus the best suffix below it.

The memoized suffix state is `(level, current_witness_len, current_witness_len_terminal, current_log_basis)`. Two witness lengths are tracked because the level's outgoing layout depends on whether the successor folds again (`WithDBlock`) or terminates directly (`WithoutDBlock`).

The search is capped by `MAX_RECURSION_DEPTH`. Beyond that cap, the suffix returns the direct-send branch only. In the supported parameter ranges, schedules do not need deeper recursion, and the cap keeps verifier-reachable fallback work bounded.

## Proof-Size Accounting

The planner uses the same byte formulas that runtime schedule expansion uses:

- `level_proof_bytes` for a fold level.
- `direct_witness_bytes` for a direct-send step.
- `extension_opening_reduction_proof_bytes` for extension-field opening reductions.
- `w_ring_element_count_with_counts_for_layout_bits` to compute next witness sizes under `WithDBlock` and `WithoutDBlock`.

This keeps generated-table expansion and DP fallback aligned. A table hit and a table miss are two ways to produce the same runtime `Schedule` shape.

## SIS Layout Derivation

For each level candidate, the planner derives the SIS layout in the same order:

1. Compute the decomposition for the candidate `log_basis`.
2. Compute the relevant digit counts for commitment and opening.
3. Compute the collision norm bucket for each role.
4. Compute the decomposed matrix width for each role.
5. Ask the SIS floor table for the minimum secure rank.
6. Build `AjtaiKeyParams` with the audited rank, width, collision bucket, SIS family, and ring dimension.

The searched parameters are therefore small: mostly `log_basis` and the fold split. The matrix dimensions are consequences of those choices and of the fixed policy inputs.

One-hot roots use a sparse committed-witness norm when `log_commit_bound == 1`. Recursive levels and full-field roots use dense balanced-digit witness bounds.

## Generated Tables

The planner owns the generated schedule-table representation and expansion logic under `src/generated/`. Shipped table data lives in the `akita-schedules` crate. Compact entries store only the brute-forced values needed to reconstruct a full level:

- `ring_d`
- `log_basis`
- `m_vars`
- `r_vars`
- `n_a`
- `n_b`
- `n_d`
- optional tiering fields `tier_split` and `n_f`

Everything else in `LevelParams` is deterministically reconstructed by `GeneratedFoldStep::expand_to_level_params`.

The reusable generated-table emitter lives in this crate and accepts explicit `EmitSpec` values. The `gen_schedule_tables` binary lives in `akita-config` because only `akita-config` can name concrete preset `Cfg` types. The emitted modules are written into `akita-schedules/src/generated/`, where feature-gated table constructors return `GeneratedScheduleTable` values to opted-in presets.

To regenerate the non-`zk` tables:

```bash
cargo run --release -p akita-config --bin gen_schedule_tables -- crates/akita-schedules/src/generated
```

To regenerate the `zk` tables:

```bash
cargo run --release -p akita-config --features zk --bin gen_schedule_tables -- crates/akita-schedules/src/generated
```

The family list is in `akita-config::generated_families::ALL_GENERATED_FAMILIES`. It is shared by the emitter and drift-guard tests so shipped entries and regeneration hooks stay aligned.

## Supported Features

### Batching

The lookup key carries the root vector counts needed for batched openings. Root B and D widths are sized with the batch factor directly, and the root proof-size formula uses the root `z` vector count.

Recursive levels are always single-claim levels: after the root fold, the next witness is one packed object that is folded or shipped.

### Tiered Commitments

When `PlannerPolicy::tiered` is enabled, the planner may split an oversized first-tier B matrix into a smaller reused `B'` and a second-tier F commitment.

`multi_tiered_keys` checks whether the B matrix is larger than A. If so, it scans split factors up to `MAX_TIERED_SPLIT_FACTOR` and takes the smallest divisor that makes `B'` fit under A. The generated table stores the resulting `tier_split`, shrunk B rank, and F rank directly, so expansion does not re-run the tiering search.

### Tensor Challenges

Some presets use a tensor-shaped level-0 fold challenge. Catalog identity records the root fold shape, so tensor and flat tables cannot be accidentally interchanged when both policies otherwise have the same field and ring dimension.

Recursive levels use the flat fold shape in the current planner search.

### Extension Fields

`PlannerPolicy` carries both claim and challenge extension degrees. When the claim field is an extension, the planner adds the extension-opening reduction proof bytes at the root and recursive levels.

### ZK Accounting

The `zk` feature selects generated tables emitted with `zk` proof-size accounting. The resolver and DP use the same formulas under the active feature set, so table hits and fallback schedules agree.

## Crate Boundary

The dependency direction is:

```text
akita-config -> akita-planner -> akita-types / akita-challenges / akita-field
akita-config -> akita-schedules -> akita-planner
```

`akita-config` derives `PlannerPolicy` from concrete presets with `policy_of::<Cfg>()` and delegates `CommitmentConfig::runtime_schedule` to `akita_planner::resolve_schedule`. The planner never names a preset type.

This boundary avoids a circular dependency while keeping a single source of truth for preset policy. It also means the DP fallback is verifier-reachable through config, so planner code follows the verifier no-panic contract: malformed verifier-facing input must return `AkitaError` rather than panic.

## Source Map

- `src/lib.rs`: public planner surface and `PlannerPolicy`.
- `src/resolve.rs`: cache-then-generate resolution, catalog validation, compact entry expansion, and proof-byte estimation.
- `src/schedule_params.rs`: DP search, root enumeration, recursive suffix search, and tiering search.
- `src/generated/mod.rs`: generated table types and table lookup helpers.
- `src/generated/expand.rs`: compact `GeneratedFoldStep` to runtime `LevelParams` expansion.
- `src/emit/mod.rs`: reusable generated-table emitter.
- `crates/akita-config/src/bin/gen_schedule_tables.rs`: offline table emitter adapter for concrete presets.
- `crates/akita-config/src/generated_families.rs`: preset family list and regeneration hooks.
- `crates/akita-schedules/src/generated/`: feature-gated shipped schedule table data.
