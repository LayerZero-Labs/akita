# Akita Planner

The `akita-planner` crate computes the parameters of each fold level in the
Akita PCS. Uniform direct schedules minimize modeled proof bytes. Adaptive
direct schedules minimize first-direct padded setup capacity, then proof bytes,
total setup, and root output-witness length. Recursive schedules first minimize
the power-of-two capacity covering total setup, then first-direct capacity,
proof bytes, and first-direct output-witness length. Numeric ties go directly
to the canonical descriptor.

This module is independent of the `Cfg` trait because `Cfg` uses the planner; if the planner named concrete configs directly, the workspace would face a circular dependency. All inputs that the planner needs from `Cfg` are therefore passed through the plain-value `PlannerPolicy`.

The planner covers the parameter-selection features supported by Akita,
including batching and extension fields. For each case it resolves the fold
parameters under the selection policy bound into the generated catalog.

The planner can also generate schedule values when a preset wants a table-backed runtime path. Later runtime calls can fetch and expand those compact entries quickly instead of repeating the heavy dynamic-programming search.

## What The Planner Optimizes

Akita proofs fold the witness at the root, may fold it through more recursive
levels, and then send the terminal witness directly. The planner chooses the
best complete schedule under the configured selection policy.

The complete schedule orders are:

```text
uniform direct:  (proof bytes, total setup, root output witness, descriptor)
adaptive direct: (first-direct padded capacity, proof bytes,
                  total setup, root output witness, descriptor)
recursive:       (padded total-setup capacity, first-direct padded capacity,
                  proof bytes, first-direct output witness, descriptor)
```

For a direct schedule, the first direct edge is the root. For an offloaded
schedule, it is the first edge after the setup-prefix chain.

First-direct capacity is a verifier-cost proxy, not a complete runtime model.
Its power-of-two bucket permits proof-size improvements within a factor-two
setup-scan bound and avoids letting a later, unusually large suffix matrix
control the leading direct objective. The
[planner rationale](../../specs/setup-offloading-planner.md#why-adaptive-direct-planning-starts-with-first-direct-capacity)
defines the related setup quantities and records the limitations of this design
choice.

Recursive setup planning uses a different leading metric because offloading can
move setup cost into a committed prefix. It first fixes the power-of-two
capacity covering every setup object, then minimizes the remaining direct scan,
proof bytes, and first-direct output witness. The
[recursive-objective rationale](../../specs/setup-offloading-planner.md#why-recursive-planning-starts-with-padded-total-setup-capacity)
explains why exact setup inside the winning bucket is not another tie-break.

The output is an `akita_types::PlannedFoldSchedule`. Its protocol value is a
typed `FoldSchedule { root, recursive_folds, terminal }`; its non-protocol
`FoldScheduleEstimate` stores the modeled byte costs used for selection.
Estimates are neither serialized nor Fiat–Shamir bound.

## Inputs And Outputs

The public search entry point is
`find_schedule(&key, final_honest_fold_policy, &precommitted_honest_fold_policies, &policy, ring_challenge_config)`.

`key: AkitaScheduleLookupKey` describes the supported root opening shape.
Single-group openings store one `PolynomialGroupLayout` in `final_group` and
leave `precommitteds` empty:

- `num_vars`: the number of Boolean variables in that group's opened
  polynomial domain.
- `num_polynomials`: the number of polynomials in the single commitment group,
  opened at that group's point (one claim per polynomial).

Multi-group roots use the same lookup key with any earlier groups recorded as
`GroupCommitPhaseParams` in `precommitteds`. For a single-group batch,
the root `t` and `w` multiplicities are just `num_polynomials` and the `z`
multiplicity is always `1`; multi-group roots derive those counts from
`final_group` plus `precommitteds`.

`policy: PlannerPolicy` is the `Cfg`-free projection of a preset:

- The exact SIS modulus profile and table digest.
- The scalar SIS policy identifier.
- Decomposition parameters, including the basis search range.
- Claim and challenge extension degrees.
- Ring dimension mode and recursive setup capability.
- Selection policy and optional setup field budget.

Source laws are separate planner-only values. For example, the one-hot policy
owns its exact chunk size while runtime schedule keys retain only public group
geometry.

The `ring_challenge_config` closure supplies the sparse challenge configuration for an A-role dimension. It is a closure instead of a config method so the planner stays independent of `CommitmentConfig`.

`PlannerPolicy::ring_dimension_schedule_mode` is the only dimension-domain
authority. Uniform policies carry one A, B, and D value in that mode. Adaptive
policies carry separate bounded domains. The selected schedule records the
exact dimensions used at each level.

## Resolution Flow

Most runtime callers use `resolve_generated_catalog_row_for_key`, not the DP directly. Resolution is the strict table entry point:

1. The caller passes the preset's optional `GeneratedScheduleTable` catalog.
2. If a catalog is supplied, `resolve_generated_catalog_row_for_key` validates its embedded identity against the runtime policy and hook closures.
3. If the validated table contains the lookup key, it expands the compact
   `GeneratedFoldScheduleEntry` with `schedule_from_entry`.
4. If there is no catalog or no matching entry, the request is unsupported.

Table generation and table expansion are deterministic functions of the lookup key, `PlannerPolicy`, and ring-challenge closure. This is important because prover and verifier must resolve the same schedule before the Fiat-Shamir transcript is bound.

## Search Model

For a fixed field, ring dimension, decomposition policy, and opening shape, the planner mainly searches over:

- `log_basis`: the balanced-digit base used by the fold level.
- `num_live_blocks`: the exact number `B` of folded blocks.
- `block_index_bits`: the number `r_blk = ceil(log2 B)` of Boolean block-index variables.
- `position_index_bits`: the number of variables inside each block.

Once those values are chosen, the rest of the level is derived rather than independently searched. Digit counts, coefficient-`L∞` bounds, matrix widths, and SIS-secure ranks come from the shared `akita_types::sis` helpers. The planner builds the A, B, and D Ajtai key parameters from those derived values and then scores the resulting proof size.

Conceptually, a candidate level answers three questions:

- How many bytes does it cost to prove the next witness?
- How many field elements will the next witness contain?
- What padded setup capacity is exposed at the first direct edge, and what is
  the first direct output-witness length and total setup envelope?

The first question determines whether the current fold is worthwhile. The second question determines how expensive later recursive levels can be.
Adaptive direct planning retains the first-direct-first V2 objective. Recursive
setup planning compares total setup at next-power-of-two capacity, then
minimizes first-direct capacity and proof bytes within the winning bucket
before comparing first-direct output-witness length.

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

Root contraction is only a search ordering hint. The planner admits contractive
and noncontractive root candidates into one suffix search and selects a complete
schedule only with `SelectionPolicyId`. It returns `UnsupportedSchedule` only
when no valid complete schedule exists in the audited fold domain.

## Recursive Suffix Search

Recursive levels do not enumerate the full exponential tree of all possible `(log_basis, block_index_bits)` choices at every depth. That would make schedule search too expensive as the number of levels grows.

Instead, `derive_fold_candidates` scans the valid `block_index_bits` choices for
each recursive `log_basis`. `FoldCandidatePolicy::Best` keeps the best
contracting candidate under the local layout score. `Frontier` retains every
contracting split candidate needed by proof-first, adaptive-dimension, or
setup-offloading search.

After that candidate is chosen, the suffix DP still performs the important global comparison:

- Terminate after this fold and ship the clear terminal response.
- Fold once more and pay the current level proof bytes plus the best suffix below it.

The memoized suffix state tracks the level, current witness length, active
basis choices, and parent-visible geometry. Uniform direct search keeps its
proof-first frontier. Adaptive direct and recursive search share one projected
frontier: a setup-aware first-direct projection and a setup-aware proof-payload
projection. A candidate is pruned only when both projections make it irrelevant
to every parent transition. At a complete root, a level setup bound larger than
the best complete envelope rejects both its direct and offloaded branches.
Ordinary recursive folds construct the single canonical
consistency/A/B/D relation and produce another recursive witness. The typed
terminal fold constructs no relation matrix or quotient: it receives
transcript-bound inner `t` from its predecessor and checks raw `e`, `t`, and
folded response `z` directly.

The same topology selects the outgoing binding of the preceding intermediate
fold. Ordinary recursive edges ship outer `u`; the final edge into a suffix
terminal binds inner `t` and contributes no duplicate `u` bytes. This is a
schedule property, not a proof-derived layout guess.

The search is capped by `MAX_RECURSION_DEPTH`. Beyond that cap, the suffix may
terminate only when the current state can feed the terminal directly. In the
supported parameter ranges, offline schedule generation does not need deeper
recursion. Runtime verification never invokes this search.

## Proof-Size Accounting

The planner uses the same byte formulas that runtime schedule expansion uses:

- `level_proof_bytes` for a fold level.
- `terminal_response_bytes` for the terminal witness.
- `extension_opening_reduction_proof_bytes` for extension-field opening reductions.
- the canonical grinding plan for the one proof-level packed nonce stream.
- `w_ring_element_count_with_counts_for_layout_bits` to compute witness sizes
  under the schedule-selected row layout.

`level_proof_bytes` is also schedule-shaped: it prices an outer commitment on
ordinary recursive edges and zero outgoing-commitment bytes for the
`TerminalInnerState` handoff. Level bodies contain no nonce field. The exact
stream byte count is rounded once across the complete plan. Terminal proof
bodies contain only any extension-opening reduction; their clear witness is
priced by `terminal_response_bytes`.

This keeps generated-table expansion and offline DP regeneration aligned. A
generated table row and a fresh DP run are two ways to produce the same typed
`FoldSchedule` and the same separately held `FoldScheduleEstimate`.

## SIS Layout Derivation

For each level candidate, the planner derives the SIS layout in the same order:

1. Compute the decomposition for the candidate `log_basis`.
2. Compute the relevant digit counts for commitment and opening.
3. Compute the coefficient-`L∞` bucket for each role.
4. Compute the decomposed matrix width for each role.
5. Ask the SIS floor table for the minimum secure rank.
6. Build the role-typed `InnerCommitMatrixParams`,
   `OuterCommitMatrixParams`, or `OpenCommitMatrixParams` with the audited
   rank, input width, coefficient-`L∞` bucket, exact SIS profile, ring
   dimension, and security floor.

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

The committed-source class is declared by the preset, not inferred from a bound: a
one-hot root uses a sparse committed-witness norm, while recursive levels and every
balanced signed-digit root use dense balanced-digit witness bounds.

`log_commit_bound` is a separate knob — the declared source coefficient bound,
anywhere from `1` to the field width. It does not select the sizing rule; it sets
the A-role digit depth `ceil(log_commit_bound / log_basis_inner)`, and through that
the A input width, the SIS rank, and the next-level witness length. A bounded dense
family (`fp128::DenseBounded`) differs from its full-width sibling in that parameter
alone.

## Generated Tables

The planner owns the generated schedule table representation and expansion
logic. Deterministic generated table data is tracked in the `akita-schedules`
crate. Compact entries mirror the protocol topology:

- `GeneratedFoldCore` stores the new group, shared opening matrix, and witness
  chunk count used by every nonterminal fold.
- `GeneratedRootFold` adds the root inner digit depth and ordered frozen
  precommitted groups.
- `GeneratedRecursiveFold` adds the optional setup prefix, payload mode, and
  optional L2 response cap.
- `GeneratedTerminalFold` records only source geometry and the inner matrix
  choice; terminal B/D matrices and outer/open digit bases do not exist.

Generated matrices store ring dimension, digit basis, and slice count where
applicable. Expansion reconstructs widths, collision buckets, and minimum
SIS-secure output ranks from the shared security primitives.

The reusable generated-table emitter lives in this crate and accepts explicit
`EmitSpec` values. The `gen_schedule_tables` binary is enabled by the
`catalog-gen` feature, which is allowed to name concrete `akita-config` preset
`Cfg` types. The emitted family modules are written into
`akita-schedules/src/generated/`, where feature-gated table constructors return
`GeneratedScheduleTable` values to opted-in presets.

The repository tracks a compact stock catalog for the shapes exercised by the
checked-in tests, examples, and profiles. Downstream applications that need a
different fixed catalog should run the standalone planner binary with their
exact shapes instead of expanding the repository catalog for every possible
polynomial size.

To regenerate schedule tables:

```bash
scripts/generate-schedule-tables.sh
```

During planner development, pass one or more generated family module names to
plan and publish only those families. The generator validates the names against
the canonical family registry and reports the elapsed time and key counts for
each selected family:

```bash
scripts/generate-schedule-tables.sh fp32_dense
scripts/generate-schedule-tables.sh fp32_dense fp64_dense
```

Add `--row-progress` when one of those searches is slow. It reports start,
completion, elapsed time, and the selected objective, proof bytes, total setup,
first-direct capacity, root output-witness length, dimensions, and fold count
for each flattened row request. It is disabled by default.

`--check-catalog` is a same-revision drift guard. It compares the union of the
compiled and regenerated keys and labels those sides explicitly in its stable
tab-separated report. The report includes added, removed, changed, and equal
rows, with compiled and regenerated setup capacity, proof payload, fold count,
and row identity. This check requires the generator's `catalog-check` feature;
the repository script selects it automatically. Add `--catalog-report <path>`
to keep that report separate from live progress on standard error.

Revision audits are separate. `--catalog-snapshot <path>` writes one stable row
per regenerated family and logical catalog key. To compare another revision,
generate its snapshot before switching revisions, then pass that file through
`--catalog-baseline <snapshot>`. The resulting `--catalog-report` is the
complete baseline/current logical-key union. It includes exact lookup and row
digests, first-direct padded capacity, total setup fields, proof bytes, fold
counts, successor witness lengths, per-level EOR bytes, opening methods,
packing geometry, and A security routes.
The command writes the complete report, including intentional removals. This
repository permits catalog-breaking revisions, so baseline/current policy is
reviewed from the checked evidence. Same-head drift remains an automatic
failure under `--check-catalog`.

Targeted generation leaves the shared `mod.rs` wiring complete. Before a
planner change is committed, run the unfiltered command above so every tracked
family is regenerated.

One generator run reuses a successful scalar schedule when grouped key
construction and scalar row emission need the same producer configuration and
layout. The cache lasts for one run and stores the complete selected
`FoldSchedule`. Before parallel row planning starts, the generator copies the
cached results into the family specifications and drops the mutable cache.
The generator also prepares independent families in parallel. When two
families need the same producer schedule, one computes it and the other waits
for that exact result.

The remaining scalar and grouped requests from every selected family are
flattened into one ordered queue. `AKITA_SCHEDULE_GEN_JOBS` bounds the number of
complete planner searches in the batch generator and defaults to `2`.
Each search stays sequential. The generator puts each result back in its input
family and sorts each family by the runtime lookup order. Worker completion
order therefore does not affect generated bytes.

The generated-table drift guard uses the same worker bound while comparing
tracked rows with fresh DP results.

Generic CI and Jolt smoke jobs compile the tracked tables directly. The
dedicated all-schedules drift job is the sole CI regeneration owner and rejects
any byte difference from the tracked catalog.

The family list is in
`akita_planner::generated_families::ALL_GENERATED_FAMILIES`. It is shared by the
emitter and drift-guard tests so generated entries and regeneration hooks stay
aligned. The family name selects the catalog class and planner policy: field,
dense versus one-hot roots, canonical root sources, chunking, and direct versus recursive
verifier setup are encoded in the family row. Explicit custom rows are limited
to D64 families; the standalone custom-catalog path does not accept ring
dimension as an input.

To emit a custom catalog, pass a final group and, when needed, an ordered list
of precommitted groups:

```bash
cargo run --release -p akita-planner --features catalog-gen \
  --bin gen_schedule_tables -- crates/akita-schedules/src/generated \
  --final-group fp128_onehot:32:2 \
  --precommitted-group fp128_onehot:16:1 \
  --precommitted-group fp128_dense:15:2
```

The explicit flags are:

- `--final-group family:num_vars:num_polynomials` selects the generated catalog
  family and the final group shape.
- `--precommitted-group family:num_vars:num_polynomials` adds one ordered
  precommitted group. Repeat the flag for each precommitted group position.

Each numeric slot accepts either a single value or an inclusive range written as
`start..=end` (or `start..end`). For example:

```bash
cargo run --release -p akita-planner --features catalog-gen \
  --bin gen_schedule_tables -- crates/akita-schedules/src/generated \
  --final-group fp128_onehot:30..=32:2..=4 \
  --precommitted-group fp128_onehot:14..=16:1 \
  --precommitted-group fp128_dense:15:1..=2
```

The generator expands the cartesian product of the final-group range and every
precommitted-group range. With no `--precommitted-group` flags, it emits
final-only scalar rows. With precommitted groups, it emits grouped-root rows and
the required standalone precommit profile registry rows. Repeating a
precommitted group preserves its multiplicity in the lookup key, while the
standalone precommit profile registry remains deduplicated.

## Supported Features

### Batching

The lookup key carries the root vector counts needed for batched openings. Root B and D widths are sized with the batch factor directly, and the root proof-size formula uses the root `z` vector count.

Witness-only recursive levels open one claim. A level that consumes an incoming setup prefix opens two groups and two claims: the recursive witness and the attached prefix.

### Extension Fields

`PlannerPolicy` carries both claim and challenge extension degrees. Emitted folds
at levels 0 and 1 use subring coefficient packing and do not carry EOR. Later
`EvaluationTrace` suffixes and the terminal opening retain method-aware EOR
pricing when the claim field is an extension.

Recursive setup offloading may continue across several levels. A fold that
receives a setup prefix opens the prefix and recursive witness with the same
method. At levels 0 and 1 that method is subring coefficient packing. At later
levels it is evaluation trace.

The fp128 presets use extension degree one, so their evaluation-trace folds do
not run EOR. The fp32 and fp64 presets use proper extension fields. Their later
evaluation-trace folds run one EOR that batches the dense setup prefix with the
recursive suffix witness. The planner prices that combined opening shape.

## Crate Boundary

The dependency direction is:

```text
akita-config -> akita-schedules -> akita-types / akita-challenges / jolt-field
akita-planner -> akita-schedules
akita-planner --features catalog-gen -> akita-config
```

`akita-config` derives `PlannerPolicy` from concrete presets with `policy_of::<Cfg>()` and delegates `CommitmentConfig::resolve_catalog_row_for_key` to strict generated-row resolution. Runtime resolution never invokes planner search.

This boundary avoids a circular dependency while keeping a single source of truth for preset policy. The DP remains offline-only in `akita-planner`; verifier-reachable runtime code must return `AkitaError` rather than panic on malformed input.

## Source Map

- `src/lib.rs`: public planner surface and `PlannerPolicy`.
- `src/generated_families.rs`: offline generated-table family registry behind `catalog-gen`.
- `src/emit/`: generated table emission and wiring refresh helpers.
- `src/schedule_params.rs`: DP search, root enumeration, and recursive suffix search.
- `src/generated/mod.rs`: generated table types and table lookup helpers.
- `src/generated/expand.rs`: typed compact root/recursive/terminal expansion to
  runtime schedule parameters.
- `src/emit/mod.rs`: reusable generated-table emitter.
- `crates/akita-planner/src/bin/gen_schedule_tables.rs`: offline table emitter adapter for concrete presets.
- `crates/akita-config/src/generated_families.rs`: preset family list and regeneration hooks.
- `crates/akita-schedules/src/generated/`: feature-gated generated schedule table wiring and ignored family table output.
