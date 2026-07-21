# Spec: Distributed (multi-chunk) setup offloading

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Author(s)     |                                            |
| Created       | 2026-07-16                                 |
| Status        | implemented                                |
| PR            |                                            |
| Supersedes    |                                            |
| Superseded-by |                                            |
| Book-chapter  | book/src/how/proving/distributed-prover.md |

## Summary

Akita already ships two independent verifier/prover-cost techniques for `D = 64`
one-hot presets:

- **Recursive setup offloading** (`RecursiveCommitmentConfig<Cfg>`,
  `SetupContributionMode::Recursive`, Stage-3 setup-product sum-check, carried
  setup-prefix opening). Generated table: `fp128_d64_onehot_recursive`.
- **Multi-chunk (distributed-prover) witness layout** (`ChunkedWitnessCfg`,
  `LevelParams::witness_chunk`, per-chunk folded responses `zⱼ`, shared `r̂`
  tail). Generated tables: `fp128_d64_onehot_multi_chunk_w2r2`,
  `fp128_d64_onehot_multi_chunk_w4r2`, etc.

They have never been combined. The setup-offloading design record
(`specs/setup-offloading-planner.md`) explicitly lists *"Distributed or
multi-chunk setup offloading"* as a non-goal.

This spec covers the **mix**: `8` witness chunks over `2` activated fold levels
(`W8R2`) **and** recursive setup offloading, for the `fp128 D=64` one-hot preset
only. `W8R2` is the single production slice shipped here; other chunk widths
(`W2R2`, `W4R2`, …) follow the same recipe but are out of scope. It is split into
two parts:

- **Part 1: the planner + generated schedule table.** A new generated family
  `fp128_d64_onehot_recursive_multi_chunk_w8r2` is emitted by the config
  `RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>`. The planner DP already
  prices both techniques together; one latent gap was fixed (the chunked
  multi-group **root** fold did not skip candidates whose *main* group had fewer
  live blocks than `num_chunks`). See [Part 1: what landed](#part-1-what-landed).
- **Part 2: the end-to-end prover / verifier / setup path.** The runtime
  machinery *largely composes* already; the genuinely new interaction points are
  the two chunked+recursive leading fold levels, now exercised end to end by
  `crates/akita-pcs/tests/distributed_setup_offload_e2e.rs`. See
  [Part 2: end-to-end plan](#part-2-end-to-end-implementation-plan).

## Intent

### Goal

Make an evaluation opening prove and verify under
`AkitaCommitmentScheme<RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>>`
for the shipped recursive profiling key
(`final_group = (32, 2)`, `precommitteds = [(16, 1), (16, 1)]`), so that the
leading two fold levels are **both** chunked (`num_chunks = 8`) **and** run the
Stage-3 setup-product sum-check (`SetupContributionMode::Recursive`), with the
carried setup-prefix opening discharged into the next fold's grouped opening
batch — exactly as the generated table
`fp128_d64_onehot_recursive_multi_chunk_w8r2` prescribes.

### Invariants

- **Config selects the family.** Only
  `RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>` activates the mix:
  `recursive_setup_planning() == true` (from the adapter) and
  `chunked_witness_cfg() == W8R2` (delegated from the base
  `D64OneHotMultiChunk`). `D == SETUP_OFFLOAD_D_SETUP == 64`.
- **Cutover aligns recursion and chunking.** `num_activated_levels = R = 2`
  chunks fold levels `0, 1`; the recursion window (`level <= 1`) also covers
  levels `0, 1`. Level `2` and beyond are single-chunk and, once no later fold
  can absorb a carried prefix, direct. There is no chunked terminal fold
  (`try_terminal_direct_suffix_cost` / `make_terminal_direct_step` reject
  `num_chunks > 1`).
- **Uniform ring dimension.** Every role dimension is `D = 64`, so
  `reject_mixed_d_multi_chunk` (the one hard multi-chunk/roles guard) always
  passes. The mix is defined only for the uniform-D64 one-hot shape.
- **Setup-prefix size is chunk-independent; its block split is not.**
  `active_setup_field_len` computes the setup-matrix footprint from role
  dimensions, so `natural_len` (hence `n_prefix`) is identical to the
  non-chunked recursive schedule. The setup-prefix precommitted **group's**
  block geometry differs, because a chunked consuming fold requires
  `num_live_blocks >= num_chunks`.
- **Every group is chunked the same way.** The canonical `WitnessLayout` is
  group-major / chunk-minor: at a chunked fold, the main witness group **and**
  every precommitted group (including the carried setup-prefix group) get
  `num_chunks` units. The shared `r̂` tail keeps the single-machine relation-row
  count (`num_commitments = 1`); it does not scale with `num_chunks`.
- **No verifier panics.** Every new rejection path (wrong slot, chunk/blocks
  mismatch, mode/successor mismatch) returns `AkitaError` /
  `SerializationError`, per the verifier no-panic contract.

### Non-Goals

- Ring dimensions other than uniform `D = 64`.
- Chunk profiles other than `W8R2` (`num_chunks = 8`, `num_activated_levels = 2`).
  Other profiles (`W2R2`, `W4R2`, …) follow the same recipe but are out of scope
  for this rollout.
- Full-field (`fp128_d64_full`) or tensor-verifier companions.
- Distributing setup preprocessing across machines. Setup-prefix commitments are
  a single-node preprocessing artifact; only the *witness fold* is distributed.
- Changing the `SetupContributionMode` call-wide argument or the setup-prefix
  slot identity model.
- A new soundness proof; but see [Security](#security-and-open-questions) for the
  argument that must be written/reviewed before this ships.

## Background: where the two techniques already compose

Both features are per-level properties of `LevelParams`
(`crates/akita-types/src/layout/params.rs`): `witness_chunk: ChunkedWitnessCfg`
(chunking) and `setup_contribution_mode: SetupContributionMode` +
`setup_prefix: Option<SetupPrefixSlotId>` (offloading). They are threaded through
the same planner, walker, prover, and verifier, and are mostly orthogonal:

- **Planner DP** already prices the mix. `derive_candidate_level_params`
  (`crates/akita-planner/src/schedule_params/candidate.rs`) resolves
  `num_chunks = policy.chunks_at_level(fold_level)` and threads it through both
  the folded-witness sizing (`grouped_setup_prefix_next_witness_len`,
  `grouped_segment_rings` scale `z_hat` by `num_chunks`) and the setup-prefix
  group derivation (`derive_setup_prefix_group`, which skips splits with
  `num_live_blocks < num_chunks`).
- **Generated walker** stamps `lp.witness_chunk = policy.witness_chunk_for_level(fold_level)`
  on recursive folds too (`crates/akita-planner/src/generated/walk.rs`, both the
  scalar and multi-group walkers), and expands `FoldWithSetupMetadata` generically.
- **Catalog identity** commits both `witness_chunk` and `recursive_setup_planning`
  (`crates/akita-planner/src/catalog_identity.rs`), so the mix table cannot alias
  the plain recursive or plain multi-chunk tables.
- **Multi-group + multi-chunk already round-trips** at runtime
  (`crates/akita-pcs/src/scheme/tests/onehot.rs::multi_group_multi_chunk_fold_round_trips`,
  `fp128::D64OneHotMultiChunkW2R2`).
- **Recursive setup offloading + multi-group already round-trips**
  (`crates/akita-pcs/tests/recursive_setup_e2e.rs`,
  `RecursiveCommitmentConfig<OneHotCfg>`).
- **The carried setup-prefix group folds per-chunk via the generic mechanism.**
  `fold_probe_witness_kernel` (`crates/akita-prover/src/protocol/fold_grind.rs`)
  windows the fold challenges per chunk range over the group's `num_live_blocks`
  and dispatches each window to the group's fold kernel; for the setup-prefix
  group that is `setup_prefix_decompose_fold`
  (`crates/akita-prover/src/backend/recursive/setup_prefix_source.rs`).
- **Stage-3 splits cleanly.** The witness-carry term
  (`build_witness_carry_term`,
  `crates/akita-prover/src/protocol/sumcheck/akita_stage3/mod.rs`) re-expresses
  the *flat* next-witness opening `W(stage2_point)` from `logical_w` and is
  chunk-agnostic; the setup-product term (built by `build_setup_product_term`
  → `prepare_setup_sumcheck_terms`,
  `crates/akita-prover/src/protocol/sumcheck/akita_stage3/mod.rs`, over the
  `SetupContributionPlan` in `crates/akita-types/src/setup_contribution/plan/`)
  is chunk-aware and validates
  `chunk_layout.num_chunks_for_group(g) == lp.witness_chunk.num_chunks`
  ("multi-group witness layout does not match root group order"), with the
  matching verifier check in
  `crates/akita-verifier/src/protocol/ring_switch.rs`.

The only hard incompatibility guard is `reject_mixed_d_multi_chunk`
(`crates/akita-verifier/src/protocol/ring_switch.rs`): multi-chunk requires
uniform role ring dimensions. The mix is uniform `D = 64`, so it passes.

## Part 1: what landed

The following are already implemented and verified (the
`generated_schedule_tables_match_key_planner` drift guard passes with and without
`all-schedules`, and `cargo clippy -- -D warnings` is clean):

1. **Feature flags.** `fp128-d64-onehot-recursive-multi-chunk-w8r2` on
   `akita-schedules`; `schedules-fp128-d64-onehot-recursive-multi-chunk-w8r2` on
   `akita-config`; both added to the respective `all-schedules` aggregates.
2. **Config family.** A `GeneratedFamily` row for
   `RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>` in
   `crates/akita-config/src/generated_families.rs` (module
   `fp128_d64_onehot_recursive_multi_chunk_w8r2`, `emit_group_batch = true`,
   reusing `recursive_profile_group_batch_keys`). The capacity selector
   `recursive_group_batch_candidates_for_capacity` now also returns the profiling
   key(s) for the mix config.
3. **Catalog wiring.** `RecursiveCommitmentConfig::schedule_catalog`
   (`crates/akita-config/src/recursive_commitment.rs`) returns the mix table for
   the `D64OneHotMultiChunk` base under the new feature; the `TypeId` import
   gate widened accordingly.
4. **Drift-guard arms.** New match arms in
   `crates/akita-config/tests/generated_tables.rs`
   (`family_catalog_is_linked`, `family_catalog`,
   `assert_family_group_batch_table_hit`, `resolve_family_group_batch_schedule`).
5. **Planner fix.** `find_group_batch_schedule`
   (`crates/akita-planner/src/group_batch.rs`) now skips a chunked root fold
   candidate whose **main** group has `num_live_blocks < num_chunks` (previously
   only precommitted groups were checked; the main-group case was unreachable
   until a multi-group family started emitting chunked roots).
6. **Generated table.** `crates/akita-schedules/src/generated/fp128_d64_onehot_recursive_multi_chunk_w8r2.rs`
   plus `mod.rs` wiring. The multi-group profiling row
   (`final_group = (32, 2)`, two `(16, 1)` precommits) is:

   | Level | mode | `setup_prefix_group` | chunked? |
   |-------|------|----------------------|----------|
   | 0 | `Recursive` | `None` (produces prefix) | yes (`num_chunks = 8`) |
   | 1 | `Recursive` | `Some(..)` (consumes L0, produces its own) | yes (`num_chunks = 8`) |
   | 2 | `Direct`    | `Some(..)` (consumes L1) | no (single-chunk) |
   | 3+ | `Direct`   | `None` | no |

   Catalog identity carries
   `witness_chunk = { num_chunks: 8, num_activated_levels: 2 }` and
   `recursive_setup_planning = true`.

Regenerate with:

```bash
cargo run --release -p akita-config --no-default-features --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated fp128_d64_onehot_recursive_multi_chunk_w8r2
cargo run --release -p akita-config --no-default-features --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated --wiring-only
```

## Part 2: end-to-end implementation plan

The genuinely new runtime combinations are the two leading fold levels:

- **Level 0** — multi-group **root** fold that is *chunked* **and** runs Stage-3
  (`Recursive`, no incoming prefix). Chunked multi-group roots are tested
  (`multi_group_multi_chunk_fold_round_trips`); chunked multi-group roots *with
  Stage-3* are not.
- **Level 1** — *chunked suffix* fold that *consumes* an incoming setup-prefix
  group **and** produces its own (`Recursive`). This is the highest-risk step:
  the carried setup-prefix precommitted group is folded per-chunk inside a
  chunked fold, and its per-chunk folded responses must match the chunked
  `WitnessLayout`.

Level 2 (single-chunk `Direct` fold consuming a setup prefix) and beyond are the
existing recursive pattern and require no new work.

Implement in the order below; each step is gated by its acceptance check.

### Step 1 — Setup preprocessing materializes the mix prefix slots

**What.** Confirm that
`AkitaCommitmentScheme<RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>>::setup_prover(FINAL_NV=32, TOTAL_GROUP_SIZE=4)`
materializes exactly the setup-prefix slots the mix schedule references, at
`gen_ring_dim = SETUP_OFFLOAD_D_SETUP`.

**Where.**
- `crates/akita-setup/src/recursive_prefixes.rs::populate_required_setup_prefix_slots`
  (runs only when `Cfg::recursive_setup_planning()` and `gen_ring_dim == D64`)
  → `crates/akita-config/src/setup_prefix_slots.rs::setup_prefix_slot_ids_for_capacity`
  → `recursive_group_batch_candidates_for_capacity::<Cfg>` (Part 1 already adds
  the mix key) → `extract_setup_prefix_slot_ids_from_schedule` (walks the mix
  schedule's `Recursive` folds and records each successor's
  `params.setup_prefix` slot id).
- `crates/akita-prover/src/api/setup_prefix.rs::commit_setup_prefix` commits the
  flat prefix `S^flat[0..natural_len]` (zero-padded to `n_prefix`) using the slot
  id's frozen `PrecommittedLevelParams` geometry.

**Why it should work.** `natural_len` is chunk-independent (setup-matrix
footprint), and the slot id's block geometry is exactly the planner's
`derive_setup_prefix_group` output, which enforced `num_live_blocks >= num_chunks`
(the generated table shows the carried setup-prefix group with
`num_live_blocks = 2048` at L1 and `1024` at L2, both `>= 8`). The commitment is
chunk-independent — chunking only affects how the
*fold response* is later laid out, not the committed prefix rows.

**Risk to close.** Two prefix-geometry code paths exist:
`derive_setup_prefix_group` (planner, chunk-aware) and
`setup_prefix_precommitted_params` (`crates/akita-types/src/proof/setup_prefix.rs`,
"fits successor commit widths"). Verify the runtime slot selection
(`select_setup_prefix_slot`, used by both Stage-3 prover and verifier) matches
the schedule's `LevelParams::setup_prefix` slot id, and that
`setup_prefix_precommitted_params` is not silently used to build a *different*
geometry for the mix. If it is, unify on the planner's derived geometry.

**Acceptance.**
- `setup.prefix_slots` is non-empty and every id equals one returned by
  `extract_setup_prefix_slot_ids_from_schedule(mix_schedule, root_layout)`.
- `matrix_envelope::inflate_envelope_for_setup_prefix_slot` covers
  `n_prefix / d_setup` for every mix slot.
- Round-trip persistence: `save_prover_setup` / `load_prover_setup` +
  `validate_loaded_prefix_registry` accept the mix registry.

### Step 2 — End-to-end prove/verify test scaffolding

**What.** `crates/akita-pcs/tests/distributed_setup_offload_e2e.rs`, sharing the
`recursive_multi_group_round_trip` helper with `recursive_setup_e2e.rs`,
parameterized on `RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>`. Same key
(`final_group = (32, 2)`, two `(16, 1)` precommits), same flow: `setup_prover`,
`batched_commit` the two precommitted groups (conservative adapter),
`commit_final_group`, build `OpeningClaims::from_groups`, `batched_prove` (the
schedule selects `SetupContributionMode::Recursive`), serialize/deserialize,
`batched_verify`.

**Assertions (beyond `recursive_setup_e2e.rs`).** The `on_schedule` hook asserts
the exact W8R2 profile shape, so a W4R2/W8R2 regression or a mode/level swap
fails the test:
- Levels 0 and 1 are both chunked with `num_chunks == 8` and
  `num_activated_levels == 2`, and both `SetupContributionMode::Recursive`.
- Level 1 and level 2 carry a `setup_prefix` group (L1 consumes L0's prefix and
  produces its own; L2 consumes L1's), and level 2 is `Direct` and single-chunk.

**Why.** This is the single most valuable artifact: it turns "the machinery
composes" into "the machinery is proven to compose", and it is the harness for
Steps 3–5.

**Acceptance.** The test proves and verifies; the proof carries Stage-3 evidence
on the two leading levels and chunked witness on those levels.

### Step 3 — Chunked setup-prefix group folding (highest risk)

**What.** Verify (and fix if needed) that at level 1 the carried setup-prefix
precommitted group is folded into `num_chunks` per-chunk responses whose sizes
match the chunked `WitnessLayout`, on both prover and verifier.

**Where.**
- Prover fold: `fold_probe_witness_kernel`
  (`crates/akita-prover/src/protocol/fold_grind.rs`) windows the fold challenges
  over `params.num_live_blocks()` for *each* group via
  `WitnessLayout::resolve_chunk_block_ranges` and `window_sparse_challenges`. For
  the setup-prefix group, `params` is its frozen geometry and the fold kernel is
  `setup_prefix_evaluate_and_fold` / `setup_prefix_decompose_fold`
  (`crates/akita-prover/src/backend/recursive/setup_prefix_source.rs`).
- Witness assembly: `ring_switch_build_w` / `emit_group_witness_segments`
  (`crates/akita-prover/src/protocol/ring_switch/coeffs.rs`) must emit
  `num_chunks` `[zⱼ | eⱼ | t̂ⱼ]` units for the setup-prefix group and assert the
  emitted length equals `lp.next_w_len(...)`.
- Verifier row-MLE: `RelationMatrixEvaluator::eval_at_point`
  (`crates/akita-verifier/src/protocol/ring_switch.rs`) evaluates the
  setup-prefix group's `E`/`T` partitioned per unit and `Z` replicated per unit,
  through `prepare_relation_matrix_evaluator_multi_group`.

**Why it should work.** The setup-prefix fold source already accepts windowed
(zero-padded) challenges of length `num_live_blocks`, so folding under a chunk
window yields that chunk's partial response. `resolve_chunk_block_ranges`
requires `num_chunks <= num_live_blocks`, satisfied by the generated slot
geometry.

**Risks to close.**
- `setup_prefix_decompose_fold` validates
  `plan.challenges.len() == num_live_blocks` and
  `plan.num_positions_per_block == frozen`. Confirm the windowed plan preserves
  those (it windows challenge *values*, not the vector length) so the per-chunk
  fold does not error.
- The setup-prefix group is a **singleton** (`num_polynomials == 1`, enforced by
  `setup_prefix_fold_geometry`). Confirm the chunked layout for a singleton
  group is consistent between planner pricing
  (`grouped_segment_rings(num_polys = 1, num_chunks, …)`) and runtime emission.
- `decompose_fold_batch` for `RecursiveFoldBatchView` returns
  `FallbackPerPoly`; confirm chunked folding still routes through the per-poly
  `decompose_fold` path (it does — chunk windows are per-poly probes).

**Acceptance.** Prover `emitted == next_w_len` at level 1; verifier row-MLE
recomputes the same value; the Step-2 test passes with level-1 chunking enabled.
Add a unit test asserting `WitnessLayout::new` for the level-1 two-group layout
produces `2 * num_chunks` units (setup-prefix + witness, `num_chunks` each).

### Step 4 — Stage-3 correctness under a chunked current fold

**What.** Verify Stage-3 is correct when the *current* fold (levels 0, 1) is
chunked.

**Where.**
- `build_witness_carry_term`
  (`crates/akita-prover/src/protocol/sumcheck/akita_stage3/mod.rs`) reconstructs
  the carried next-witness opening from the flat `logical_w` i8 digits and
  asserts `term.input_claim() == stage2_next_w_eval`. Confirm `logical_w` at a
  chunked fold is the flat concatenation the carry term expects (it is the same
  flat next witness, whose *layout* is chunked but whose *values* are the folded
  responses), so the assertion holds.
- `build_setup_product_term` → `prepare_setup_sumcheck_terms`
  (`crates/akita-prover/src/protocol/sumcheck/akita_stage3/mod.rs`) with the
  multi-group chunk-consistency check (the "multi-group witness layout does not
  match root group order" guard) must accept both the level-0 chunked
  multi-group root and the level-1 chunked two-group suffix. The verifier's copy
  of that guard lives in `crates/akita-verifier/src/protocol/ring_switch.rs`.
- Verifier Stage-3: `verify_batched_stage3` /
  `SetupIndexWeightEvaluator::evaluate`
  (`crates/akita-verifier/src/stages/stage3.rs`,
  `crates/akita-types/src/setup_contribution/setup_index_weight_evaluator.rs`).
  The setup-index weight and `alpha`-power ladder are challenge-driven and
  chunk-independent; confirm the carried `setup_prefix_eval` is consumed only
  when `next_fold_level_params.setup_prefix.is_some()`.

**Acceptance.** Stage-3 prover/verifier accept at levels 0 and 1; the
`setup_prefix_eval` carried into levels 1 and 2 matches the verifier's
recomputed point (`BatchedStage3Geometry::shared_suffix_point`).

### Step 5 — Guard and negative tests

**What.** Lock the mix's invariants and rejection paths.

- Assert `reject_mixed_d_multi_chunk` still fires only for non-uniform role dims,
  and the uniform-D64 mix passes.
- Tamper tests (mirror `recursive_setup_e2e.rs` style): corrupt the carried
  `setup_prefix_eval`, swap the setup-prefix slot id, perturb a per-chunk `zⱼ`
  segment, and change `num_chunks` between prove and verify — each must reject
  with `AkitaError::InvalidProof` and no panic.
- Assert the schedule structural guards
  (`crates/akita-types/src/schedule.rs`): a chunked `Recursive` fold's successor
  still carries exactly the setup-prefix group; a chunked terminal fold is still
  rejected.

**Acceptance.** All negative cases reject without panic; positive case still
verifies.

### Step 6 — Profiling, docs, and companion profiles

- Add a profiling workload entry mirroring `profile/akita-recursion/` for the
  mix, reporting proof bytes per level by `(mode, num_chunks)` and verifier
  cycles.
- Fold durable behavior into `book/src/how/proving/distributed-prover.md` and
  `book/src/roadmap/verifier-offloading.md`; note the combination and its
  cutover alignment.
- Update the `Non-Goal` line in `specs/setup-offloading-planner.md`
  ("Distributed or multi-chunk setup offloading") to reference this spec once the
  e2e path lands.
- (Optional follow-up) Generalize to other profiles (`W2R2`, `W4R2`) by adding
  the corresponding families; the recipe is identical.

## Security and open questions

- **Soundness of chunking a setup-prefix-carrying (batched) root.** The
  distributed-prover argument (treat all workers + aggregator as one prover;
  everything but digit decomposition is linear, handled by per-part responses)
  and the setup-offloading argument (the carried setup opening is batched into
  the next fold) must be composed explicitly. The distributed design already
  states a distributed root may start from a batched root and be refined into
  ordered parts "without changing its frozen parameters"; the setup-prefix group
  is such a frozen precommitted group. Write/review the combined statement before
  shipping, since `specs/setup-offloading-planner.md` scoped multi-chunk setup
  offloading out.
- **Whether the setup-prefix group *should* be chunked at all.** Two consistent
  designs exist: (a) chunk every group including the setup prefix (what the
  planner prices and the runtime mechanism supports today), or (b) keep
  precommitted/setup-prefix groups single-chunk while only the main witness is
  chunked. Option (a) is the current trajectory and needs no pricing change;
  option (b) would require the planner to *not* scale the setup-prefix `z_hat` by
  `num_chunks` and the `WitnessLayout` to special-case precommitted groups. This
  spec assumes (a); flag if (b) is preferred on cost or security grounds.
- **Digit-depth cost of the carried prefix under chunking.** The setup prefix
  carries full-field coefficients (`ceil(log_b q)` planes) vs the witness's small
  digits; replicating its `z` per chunk multiplies that already-large segment by
  `num_chunks`. Confirm the planner's cost model still selects `Recursive` only
  where profitable (the generated table does recurse at levels 0, 1 for the
  profiling key, so it is profitable there).

## Evaluation

### Acceptance Criteria

- [ ] `RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>::setup_prover(32, 4)`
      materializes the mix setup-prefix slots (Step 1).
- [ ] An e2e test proves and verifies the mix for the profiling key, with level 0
      and level 1 both chunked (`num_chunks = 8`) and both `Recursive` (Step 2).
- [ ] The carried setup-prefix group folds per-chunk with `emitted == next_w_len`
      and a matching verifier row-MLE at level 1 (Step 3).
- [ ] Stage-3 prover/verifier accept at levels 0, 1 with the chunked current fold
      (Step 4).
- [ ] Tamper / guard negatives reject without panic (Step 5).
- [ ] `cargo fmt`, `cargo clippy --all -- -D warnings`, `cargo test`, and
      `./scripts/check-doc-guardrails.sh` pass.

### Testing Strategy

- **`akita-setup` / `akita-config`:** mix slot enumeration and envelope inflation
  (Step 1); `setup_prefix_slot_ids_for_capacity::<mix>()` is bounded and unique.
- **`akita-types`:** `WitnessLayout::new` unit count for the level-1 two-group
  chunked layout; `active_setup_field_len` parity between mix and plain-recursive
  schedules.
- **`akita-prover` / `akita-verifier`:** the e2e test (Step 2) plus targeted
  round-trips for the level-1 chunked setup-prefix fold (Step 3) and Stage-3
  under chunking (Step 4).
- **Negative:** Step 5 tamper matrix.

### Performance

Track proof bytes per level by `(mode, num_chunks)`, setup-prefix preprocessing
bytes for the mix, prover fold time (per-chunk vs single global), and verifier
cycles saved by offloading vs the extra chunked-witness bytes.

## References

- `specs/setup-offloading-planner.md`
- `specs/distributed-planner.md`
- `specs/multi-group-batching.md`
- `specs/batched-stage3-setup-opening.md`
- `crates/akita-planner/src/group_batch.rs`
- `crates/akita-planner/src/schedule_params/candidate.rs`
- `crates/akita-prover/src/protocol/fold_grind.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage3/mod.rs`
- `crates/akita-prover/src/backend/recursive/setup_prefix_source.rs`
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/stages/stage3.rs`
- `crates/akita-types/src/setup_contribution/plan/` (setup-contribution plan; the
  former `relation.rs`/`inputs.rs` were folded into `plan/` and
  `prepare_setup_sumcheck_terms` by refactor PR #305)
- `crates/akita-types/src/witness.rs`
- `crates/akita-config/src/setup_prefix_slots.rs`
- `crates/akita-setup/src/recursive_prefixes.rs`
- `crates/akita-pcs/tests/recursive_setup_e2e.rs`
- `crates/akita-pcs/src/scheme/tests/onehot.rs` (`multi_group_multi_chunk_fold_round_trips`)
- `crates/akita-schedules/src/generated/fp128_d64_onehot_recursive_multi_chunk_w8r2.rs`
