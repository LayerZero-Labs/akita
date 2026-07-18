# Akita Planner

The `akita-planner` crate is responsible for computing the parameters of each fold level in the Akita PCS, with the goal of minimizing proof size for a given field, ring dimension, number of variables, and number of polynomials to be batched.

This module is independent of the `Cfg` trait because `Cfg` uses the planner; if the planner named concrete configs directly, the workspace would face a circular dependency. All inputs that the planner needs from `Cfg` are therefore passed through the plain-value `PlannerPolicy`.

The planner covers the parameter-selection features supported by Akita, including batching, tensor challenges, and extension fields. For each case it resolves the fold parameters that minimize the modeled proof size.

The planner can also generate and cache schedule values when a preset wants a shipped table. Later runtime calls can fetch and expand those compact entries quickly instead of repeating the heavy dynamic-programming search. If no cached value is available, the same deterministic planner computes the schedule on demand.

## What The Planner Optimizes

Akita proofs recursively fold the witness through at least two levels before the terminal direct-send step. The planner chooses the cheapest supported folded sequence.

The output is an `akita_types::Schedule`: an ordered `folds` vector, one explicit `terminal` handoff, and the total modeled byte count.

## Inputs And Outputs

The public search entry point is `find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_challenge_shape_at_level)`.

`key: AkitaScheduleLookupKey` describes the supported root opening shape. Scalar
same-point openings store one `PolynomialGroupLayout` in `final_group` and leave
`precommitteds` empty:

- `num_vars`: the number of Boolean variables in the opened polynomial domain
  (shared opening-point arity).
- `num_polynomials`: the number of polynomials in the single commitment group,
  opened at the shared point (one claim per polynomial).

Multi-group roots use the same lookup key with any earlier groups recorded as
`PrecommittedGroupParams` in `precommitteds`. For the scalar same-point batch,
the root `t` and `w` multiplicities are just `num_polynomials` and the `z`
multiplicity is always `1`; multi-group roots derive those counts from
`final_group` plus `precommitteds`.

`policy: PlannerPolicy` is the `Cfg`-free projection of a preset:

- The exact SIS modulus profile and table digest.
- The scalar SIS policy identifier.
- Decomposition parameters, including the basis search range.
- Claim and challenge extension degrees.
- Ring-subfield norm bound.
- One-hot chunk size.

The `ring_challenge_config` closure supplies the sparse challenge configuration for a ring dimension, and the `fold_challenge_shape_at_level` closure supplies the fold challenge shape for a level. They are closures instead of config methods so the planner stays independent of `CommitmentConfig`.

## Resolution Flow

Most runtime callers use `resolve_schedule` / `resolve_group_batch_schedule`, not the DP directly. Resolution is the planner's cache-then-generate entry point:

1. The caller passes the preset's optional `GeneratedScheduleTable` catalog.
2. If a catalog is supplied, `resolve_schedule` validates its embedded identity against the runtime policy and hook closures.
3. If the validated table contains the lookup key, it expands the compact `GeneratedScheduleTableEntry` with `schedule_from_entry`.
4. If there is no catalog or no matching entry, it calls `find_group_batch_schedule` and regenerates the schedule from scratch.

Both paths are deterministic functions of the lookup key, `PlannerPolicy`, and the two closures. This is important because prover and verifier must resolve the same schedule before the Fiat-Shamir transcript is bound.

## Search Model

For a fixed field, ring dimension, decomposition policy, and opening shape, the planner mainly searches over:

- `log_basis`: the balanced-digit base used by the fold level.
- `num_live_blocks`: the exact number `B` of folded blocks.
- `block_index_bits`: the number `r_blk = ceil(log2 B)` of Boolean block-index variables.
- `position_index_bits`: the number of variables inside each block.

Once those values are chosen, the rest of the level is derived rather than independently searched. Digit counts, coefficient-`L∞` bounds, matrix widths, and SIS-secure ranks come from the shared `akita_types::sis` helpers. The planner builds the A, B, and D Ajtai key parameters from those derived values and then scores the resulting proof size.

Conceptually, a candidate level answers two questions:

- How many bytes does it cost to prove the next witness?
- How many field elements will the next witness contain?

The first question determines whether the current fold is worthwhile. The second question determines how expensive later recursive levels can be.

## Root Level Search

The root level starts from the original witness length `2^num_vars`. It is the only level that sees the full root batching shape from the lookup key.

At the root, the planner iterates over the configured `log_basis` range and over valid `block_index_bits` values. For each candidate it derives:

- `position_index_bits = reduced_vars - block_index_bits`, where `reduced_vars` accounts for the ring dimension.
- The A-role committed block width.
- The B-role opening/check width.
- The D-role prover witness width.
- The SIS-secure ranks `n_a`, `n_b`, and `n_d`.
- The ordinary recursive witness length and the quotient-free terminal witness
  length.

Batching is folded directly into the root B and D widths. A batched root does not first plan a singleton layout and scale it later; the matrix widths are sized for the actual `num_polynomials` count.

The planner fails with `UnsupportedSchedule` when no candidate contains at least two folds. Degenerate inputs are rejected instead of producing a separate proof topology.

## Recursive Suffix Search

Recursive levels do not enumerate the full exponential tree of all possible `(log_basis, block_index_bits)` choices at every depth. That would make schedule search too expensive as the number of levels grows.

Instead, for each recursive `log_basis`, `derive_candidate_level_params` scans the valid `block_index_bits` choices and keeps the candidate that minimizes the next witness length. This is a local shrinking rule: recursive levels commit dense balanced-digit witnesses, and reducing the next witness length is the main driver for making the remaining suffix cheaper.

After that candidate is chosen, the suffix DP still performs the important global comparison:

- Terminate after this fold and ship the segment-typed cleartext witness.
- Fold once more and pay the current level proof bytes plus the best suffix below it.

The memoized suffix state is `(level, current_witness_len, current_witness_len_terminal, current_log_basis)`. Two witness lengths are tracked because an ordinary recursive fold uses the full `WithDBlock` witness, while the terminal direct witness has no D/quotient tail. The terminal receives transcript-bound inner `t` from its predecessor and has no commitment block (`WithoutCommitmentBlocks`).

The same topology selects the outgoing binding of the preceding intermediate
fold. Ordinary recursive edges ship outer `u`; the final edge into a suffix
terminal binds inner `t` and contributes no duplicate `u` bytes. This is a
schedule property, not a proof-derived layout guess.

The search is capped by `MAX_RECURSION_DEPTH`. Beyond that cap, the suffix may
terminate only if doing so still produces the required root-plus-suffix folded
topology. In the supported parameter ranges, schedules do not need deeper
recursion, and the cap keeps verifier-reachable fallback work bounded.

## Proof-Size Accounting

The planner uses the same byte formulas that runtime schedule expansion uses:

- `level_proof_bytes` for a fold level.
- `direct_witness_bytes` for a direct-send step.
- `extension_opening_reduction_proof_bytes` for extension-field opening reductions.
- `w_ring_element_count_with_counts_for_layout_bits` to compute witness sizes
  under the schedule-selected row layout.

`level_proof_bytes` is also schedule-shaped: it prices an outer commitment on
ordinary recursive edges and zero outgoing-commitment bytes for the
`TerminalInnerState` handoff. Terminal proof bodies contain only the grind
nonce plus any extension-opening reduction; their clear witness is priced by
`direct_witness_bytes`.

This keeps generated-table expansion and DP fallback aligned. A table hit and a table miss are two ways to produce the same runtime `Schedule` shape.

## SIS Layout Derivation

For each level candidate, the planner derives the SIS layout in the same order:

1. Compute the decomposition for the candidate `log_basis`.
2. Compute the relevant digit counts for commitment and opening.
3. Compute the coefficient-`L∞` bucket for each role.
4. Compute the decomposed matrix width for each role.
5. Ask the SIS floor table for the minimum secure rank.
6. Build `AjtaiKeyParams` with the audited rank, width, coefficient-`L∞` bucket, exact SIS profile, ring dimension, and security floor.

Production SIS lookups use explicit role cells and the scalar `SisTableKey`:

```text
(sis_security_policy, table_digest, sis_modulus_profile, role,
 ring_dimension, coeff_linf_bound)
```

The shipped policy is `Quantum128BitADPS16`: a single ADPS16 quantum LGSA rule
at a 128-bit target. The policy, table digest, exact profile, and role are part
of planner inputs, catalog identity, generated table expansion, and descriptor
bytes, so a schedule generated for one table cannot be silently reused under
another table or role.

The searched parameters are therefore small: mostly `log_basis` and the fold split. The matrix dimensions are consequences of those choices and of the fixed policy inputs.

One-hot roots use a sparse committed-witness norm when `log_commit_bound == 1`. Recursive levels and full-field roots use dense balanced-digit witness bounds.

## Generated Tables

The planner owns the generated schedule-table representation and expansion logic under `src/generated/`. Shipped table data lives in the `akita-schedules` crate. Compact entries store only the brute-forced values needed to reconstruct a full level:

- `ring_d`
- `log_basis`
- `position_index_bits`
- `block_index_bits`
- `n_a`
- `n_b`
- `n_d`

Everything else in `LevelParams` is deterministically reconstructed by `GeneratedFoldStep::expand_to_level_params`.

The reusable generated-table emitter lives in this crate and accepts explicit `EmitSpec` values. The `gen_schedule_tables` binary lives in `akita-config` because only `akita-config` can name concrete preset `Cfg` types. The emitted modules are written into `akita-schedules/src/generated/`, where feature-gated table constructors return `GeneratedScheduleTable` values to opted-in presets.

To regenerate schedule tables:

```bash
cargo run --release -p akita-config --no-default-features --bin gen_schedule_tables -- crates/akita-schedules/src/generated
```

The family list is in `akita-config::generated_families::ALL_GENERATED_FAMILIES`. It is shared by the emitter and drift-guard tests so shipped entries and regeneration hooks stay aligned.

## Supported Features

### Batching

The lookup key carries the root vector counts needed for batched openings. Root B and D widths are sized with the batch factor directly, and the root proof-size formula uses the root `z` vector count.

Recursive levels are always single-claim levels: after the root fold, the next witness is one packed object that is folded or shipped.

### Tensor Challenges

Some presets use a tensor-shaped level-0 fold challenge. Catalog identity records the root fold shape, so tensor and flat tables cannot be accidentally interchanged when both policies otherwise have the same field and ring dimension.

Recursive levels use the flat fold shape in the current planner search.

### Extension Fields

`PlannerPolicy` carries both claim and challenge extension degrees. When the claim field is an extension, the planner adds the extension-opening reduction proof bytes at the root and recursive levels.

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
- `src/schedule_params.rs`: DP search, root enumeration, and recursive suffix search.
- `src/generated/mod.rs`: generated table types and table lookup helpers.
- `src/generated/expand.rs`: compact `GeneratedFoldStep` to runtime `LevelParams` expansion.
- `src/emit/mod.rs`: reusable generated-table emitter.
- `crates/akita-config/src/bin/gen_schedule_tables.rs`: offline table emitter adapter for concrete presets.
- `crates/akita-config/src/generated_families.rs`: preset family list and regeneration hooks.
- `crates/akita-schedules/src/generated/`: feature-gated shipped schedule table data.
