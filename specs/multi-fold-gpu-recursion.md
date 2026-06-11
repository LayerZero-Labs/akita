# Spec: Multi-Fold GPU Recursion

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Cursor                         |
| Created     | 2026-06-11                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Akita currently folds one level's block witnesses into a single logical
`z = sum_i c_i s_i` segment before building the next recursive witness. That is
natural on CPU and on one GPU, but it creates an expensive synchronization point
for a multi-GPU prover: every device can compute its local partial
`z_j = sum_{i in P_j} c_i s_i`, but the current protocol shape wants those
partials reduced into one `z` before the next level can start.

This spec introduces an opt-in `multi_fold` config family whose recursive witness
stores deterministic partitioned fold outputs `z_0, ..., z_{k-1}` instead of
first reducing them to one host-visible `z`. The verifier still sees one
commitment to the flattened next witness and one claimed next-witness
evaluation; the protocol change is in the shape and constraints of the witness
that is committed. A CPU backend remains the reference implementation, while a
future multi-GPU backend can keep the large `z_j` payloads on devices and use GPU
collectives only for small transcript/proof outputs.

## Intent

### Goal

Add an opt-in protocol/config shape that carries multiple partitioned folded
witness segments through recursion, selected by a `multi_fold` `Cfg`, with
generated schedule tables for shipped partition counts and runtime DP fallback
for other config-selected counts.

Key implementation surfaces:

- `CommitmentConfig`: add a hook such as `fn multi_fold_partitions() -> usize`
  with default `1`. The hook returns the config's fixed requested partition
  count; callers choose a `Cfg` to choose this value. The hook is
  protocol-affecting and must feed schedule resolution, descriptor binding,
  prover construction, verifier replay, and table selection.
- `akita_config::multi_fold`: new config module. The first concrete family should
  be fp128 one-hot, for example
  `multi_fold::fp128::D64OneHotK8`, with type aliases for the shipped table set
  (`K4`, `K8`, and `K12` are good initial coverage; here `K8` means eight
  fold partitions, not `onehot_chunk_size`). Any fixed partition count chosen by
  a `Cfg` should work through runtime DP when no shipped table exists.
- `PlannerPolicy`: add `multi_fold_partitions: usize`, derived only from
  `policy_of::<Cfg>()`. The planner stays `Cfg`-free.
- `LevelParams`: add `fold_partitions: usize`, with default `1`, and bind it in
  descriptor bytes. This stores the active per-level value, not the raw config
  request:

```text
fold_partitions = min(Cfg::multi_fold_partitions(), level.num_blocks)
```

  It must be at least `1`. No proof-provided partition count is trusted.
- `AkitaScheduleLookupKey` and generated table keys: no new public root key field
  is required if the partition count is part of `PlannerPolicy`. The same public
  opening shape can select different schedules under different `Cfg` families
  because the descriptor binds the resulting `LevelParams`.
- `akita-planner`: compute witness lengths and proof bytes using
  `fold_partitions`; reject malformed partition counts with `AkitaError`, never
  a panic.
- `akita-prover`: replace single-output fold construction with partition-aware
  fold construction in root and recursive ring-relation builders.
- `akita-verifier`: replay the same partitioned relation from `LevelParams`
  without naming GPUs or prover backends.
- `akita-types`: update proof-size, terminal-witness shape, relation layout, and
  descriptor helpers so prover, verifier, and planner share the same count math.

### Protocol Shape

For a level with `B = lp.num_blocks` fold blocks and active
`fold_partitions`, define a deterministic contiguous balanced partition of block
indices:

```text
P_j = { i | floor(i * fold_partitions / B) = j },
0 <= j < fold_partitions
```

Every `P_j` is non-empty because `fold_partitions <= B`. For each public row or
recursive claim row `r`, the prover computes:

```text
z_{r,j} = sum_{i in P_j} c_{r,i} s_{r,i}
```

The current `z_folded_rings` segment becomes partition-major:

```text
z[public_row][partition][inner_index]
```

where each partition owns only its local blocks:

```text
partition_inner_width(j) = |P_j| * lp.num_digits_commit
sum_j partition_inner_width(j) = lp.inner_width()
```

The flattened next witness stays a single `RecursiveWitnessFlat`, so
`AkitaStage2Proof` can keep one `next_w_commitment` and one `next_w_eval`.

This is deliberately not a vector-valued recursive carry. The verifier still
threads one `RecursiveVerifierState` through the suffix; the partitioned `z_j`
segments are internal regions of that one committed witness. A different design
that carries independent `z_j` witnesses as separate commitments or separate
stage-2 evaluation claims would need new proof types, transcript absorbs,
schedule shapes, and recursive multipoint soundness work.

The verifier relation must constrain each partitioned fold, not only the sum of
all partitions:

```text
sum_{i in P_j} c_{r,i} e_folded_{r,i}
  - cyclic_consistency_z_product(z_{r,j}) = 0
```

This is stronger than checking only `sum_j z_{r,j}` and prevents a prover from
moving mass between partitions while preserving the aggregate. It also matches
the intended GPU semantics: each device's `z_j` is a meaningful partial fold.

### Invariants

- `fold_partitions = 1` is the current protocol shape. Existing configs must
  continue to derive one active partition, the current logical `z` layout, the
  current single consistency row, and the current generated-table selection.
- `fold_partitions` is verifier-reachable. It is derived from `Cfg` and
  `LevelParams`, is descriptor-bound, and is never read from a proof payload.
- `LevelParams.fold_partitions` equals
  `min(Cfg::multi_fold_partitions(), lp.num_blocks)` on every fold step.
  Verifier entry points must validate this against `Cfg`, analogous to the
  current one-hot chunk-size schedule validation.
- Partitioning is by logical fold-block index, not by backend memory address.
  CPU, single-GPU, and multi-GPU backends must produce the same partition order.
- The proof wire still contains one next-witness commitment and one
  next-witness evaluation claim per non-terminal level. Multi-GPU placement is a
  prover backend concern, not a proof object concern.
- The partitioned `z` witness region is range-checked and decomposed exactly like
  the current `z` region. The digit depth may be tightened using each
  partition's block count, but it must never be smaller than the bound implied by
  the largest partition.
- Every M-row offset helper must account for partition consistency rows and the
  existing tiered `COMMIT | B_inner` geometry. Do not add open-coded offsets in
  prover or verifier paths.
- Generated tables for `multi_fold` configs must be drift-checked against the DP,
  just like existing tables in `generated_families.rs` and `regen_diff.rs`.
- Verifier-reachable arithmetic uses checked `usize` math and returns
  `AkitaError` on overflow, invalid partition count, malformed proof shape, or
  schedule/proof mismatch.

### Non-Goals

- No GPU kernels are implemented by this spec. It only defines the protocol and
  schedule shape that lets a future GPU backend avoid host all-reduce of the
  large `z` witness segment.
- No multiple next-witness commitments. A future distributed backend may compute
  the one commitment through GPU collectives, but the verifier should not learn
  device count or device placement.
- No vector-valued recursive state. The first version does not change
  `AkitaStage2Proof`, `RecursiveVerifierState`, or stage-2 into a multi-claim
  handoff.
- No verifier dependency on `akita-prover`, accelerator crates, CUDA, Metal,
  NCCL, or host-device runtime state.
- No attempt to make tensor folding challenges work at recursive levels. Current
  recursive code rejects tensor fold shapes; `multi_fold` is independent of that
  optimization.
- No backward compatibility with proofs or schedules generated by experimental
  branches. Existing non-`multi_fold` configs remain the compatibility boundary.

## Evaluation

### Acceptance Criteria

- [ ] `CommitmentConfig` exposes a default `multi_fold_partitions() -> usize`
      hook returning `1`.
- [ ] `akita_config::multi_fold::fp128` provides at least one concrete public
      config with prefix `multi_fold`, preferably `D64OneHotK8`, plus shipped
      aliases for `K4` and `K12` if table size is acceptable.
- [ ] `policy_of::<Cfg>()` maps the config hook into `PlannerPolicy`, and the
      planner rejects a requested partition count of `0`.
- [ ] `LevelParams` carries descriptor-bound `fold_partitions`; existing configs
      materialize `1`.
- [ ] `w_ring_element_count_with_counts*`, terminal witness layout, proof-size
      helpers, and setup envelope code include partition-aware `z` digit sizing
      and partition consistency rows.
- [ ] Verifier schedule validation rejects a proof whose materialized
      `fold_partitions` does not match the active value derived from `Cfg`.
- [ ] Root and recursive prover folds compute `z_{row,partition}` in canonical
      partition-major order.
- [ ] Verifier ring-switch replay reconstructs the same partition consistency
      rows from `LevelParams`.
- [ ] `AkitaStage2Proof`, `AkitaLevelProof`, and `AkitaBatchedRootProof` do not
      grow a vector of next-witness commitments for this feature.
- [ ] `gen_schedule_tables` emits `multi_fold_*` modules, and `regen_diff.rs`
      covers them.
- [ ] CPU e2e tests prove and verify at partition counts `1`, `4`, and `8`,
      including serialized proof round trips and at least one recursive suffix.

### Testing Strategy

Required baseline checks:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `cargo test -p akita-verifier --no-default-features`
- `scripts/check-crate-deps.sh akita-verifier`

Focused tests:

- `crates/akita-types`: `w_ring_element_count_with_counts_bits` preserves the
  existing `fold_partitions = 1` layout and computes partition-aware `z` digits
  and M-row counts for partition counts `2`, `8`, and `12`.
- `crates/akita-planner`: DP schedules with `multi_fold_partitions = 1` match the
  existing shape; schedules with larger partition counts have
  descriptor-distinct level params and partition-aware next-witness lengths.
- `crates/akita-prover`: one-hot `decompose_fold_partitioned` matches the sum of
  per-partition reference folds and its aggregate equals the legacy
  `decompose_fold` output.
- `crates/akita-prover`: recursive `new_recursive_multipoint` with more than
  one active partition produces partition-major `z_folded_rings` and rejects
  tensor fold shapes as it does today.
- `crates/akita-verifier`: partitioned M-row replay matches a materialized CPU
  reference for small dimensions.
- `crates/akita-pcs`: e2e proof/verify for
  `multi_fold::fp128::D64OneHotK4` and `D64OneHotK8`, including proof
  serialization and transcript replay.
- Negative tests: verifying an `8`-partition proof under a `1`-partition config
  rejects; tamper one partition's `z` segment or one partition consistency row
  and verification rejects.

### Performance

The feature intentionally increases recursive witness size: the `z` segment
changes from roughly:

```text
num_public_rows * inner_width * num_digits_fold
```

to:

```text
num_public_rows
  * sum_j (partition_inner_width(j) * num_digits_fold_partition(j))
```

The raw pre-decomposition `z` ring width is still `num_public_rows *
inner_width()`, because partitions split blocks instead of duplicating every
block. The size risk comes from per-partition digit depths and from the extra
consistency rows. If every partition conservatively reuses the current aggregate
digit depth, the `z` digit region is close to the current size; the verifier
relation and `r` tail still grow with the partition row count. The preferred
implementation should compute a tighter partition digit depth from each `|P_j|`,
which is required before treating profile results as final.

The intended prover-side win is avoiding a large host-visible all-reduce of
`z`. A future GPU benchmark should compare:

- CPU/reference `fold_partitions = 1`.
- Single-GPU `fold_partitions = 1`.
- Multi-GPU with device all-reduce to one `z`.
- Multi-GPU `fold_partitions = device_count`, no host all-reduce of the `z`
  payload.

The primary metric is wall time for root and first recursive fold at large
`AKITA_NUM_VARS`; secondary metrics are proof bytes, GPU memory residency, and
host-device transfer volume.

## Design

### Current Code Anchors

- `crates/akita-prover/src/protocol/ring_relation.rs` computes folded `z` in
  `RingRelationProver::new` for root proofs and
  `RingRelationProver::new_recursive_multipoint` for recursive proofs.
- `crates/akita-prover/src/backend/onehot/accumulate.rs` and
  `crates/akita-prover/src/backend/onehot/ops.rs` contain the one-hot
  decompose-fold kernels that currently emit one aggregate folded witness.
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` threads
  `z_folded_rings` into `build_w_coeffs`, which constructs the flattened
  recursive witness.
- `crates/akita-prover/src/protocol/ring_switch/commit.rs` commits exactly one
  flattened next witness through `commit_next_w`.
- `crates/akita-types/src/proof/levels.rs` stores exactly one
  `next_w_commitment` and one `next_w_eval` in `AkitaStage2Proof`.
- `crates/akita-verifier/src/protocol/levels/recursive.rs` threads a single
  `RecursiveVerifierState` through recursive levels.
- `crates/akita-types/src/schedule.rs` already has `num_z_vectors` in the root
  schedule key. `multi_fold` should not overload root public rows; it needs an
  explicit partition count in `LevelParams`.
- `crates/akita-planner/src/resolve.rs` and
  `crates/akita-planner/src/schedule_params.rs` currently materialize recursive
  levels with singleton public-row counts; their count helpers need
  partition-aware `z` and consistency-row math.
- `crates/akita-config/src/generated_families.rs`,
  `crates/akita-config/src/bin/gen_schedule_tables.rs`,
  `crates/akita-planner/src/generated/mod.rs`, and
  `crates/akita-planner/src/resolve.rs` are the generated-table registration and
  selection path.

### Witness Layout

Add a partition count to `LevelParams`:

```rust
pub struct LevelParams {
    // existing fields
    pub fold_partitions: usize,
}
```

The `z` region of `w` becomes:

```text
for public_row in 0..num_public_rows:
  for partition in 0..fold_partitions:
    for inner in 0..partition_inner_width(partition):
      emit decompose(z[public_row][partition][inner])
```

For `fold_partitions = 1`, this is byte-identical to the current logical order:
one partition covers every block and the emission order reduces to the existing
point-major/block-major order documented in `emit_z_folded_block_inner`.
For `fold_partitions > 1`, the aggregate legacy `z` is not materialized by the
protocol:

```text
z_legacy[public_row][inner] = sum_partition z[public_row][partition][inner]
```

This equality may be used in tests and CPU reference code, but not as the runtime
representation.

### Relation Layout

Generalize the M-row prefix from:

```text
consistency(1) | public(num_public_rows) | D | COMMIT | B_inner | A
```

to:

```text
fold_consistency(num_fold_consistency_rows)
| public(num_public_rows)
| D
| COMMIT
| B_inner
| A
```

where:

```text
num_fold_consistency_rows =
  if fold_partitions == 1 { 1 }
  else { num_public_rows * fold_partitions }
```

All row-start helpers in `LevelParams` should take
`num_fold_consistency_rows` or derive it from `num_public_rows` and
`fold_partitions`. The public y rows do not multiply by the partition count:
`y_rings` still has one row per public opening row, because the user-facing
polynomial opening claim has not been partitioned.

This preserves the current batched-root geometry when `fold_partitions = 1`.
Today batched roots may have multiple public `z` segments but still only one
global consistency row; `multi_fold` must not change that case.

### Digit Bounds

The first implementation can use the existing aggregate fold digit depth for all
partitions, which is conservative:

```text
num_digits_fold_partition = lp.num_digits_fold(num_t_vectors, field_bits)
```

The preferred implementation should add a SIS helper that accepts an explicit
block-count bound:

```text
fold_bound_blocks(j) = |P_j|
```

and computes the folded-witness bound from each partition's block count rather
than `2^r_vars`. This requires threading an explicit block-count bound into the
SIS folded-witness beta helpers instead of overloading
`LevelParams::num_digits_fold(num_claims, field_bits)`, which currently prices a
fold over all blocks. This is important for `8` or `12` partitions; otherwise
the proof-size tax may dominate the GPU transfer savings.

The descriptor must bind enough information for prover and verifier to agree on
which bound was used. The cleanest path is to store only the active
`fold_partitions` in `LevelParams` and derive every `P_j` from `num_blocks` and
`fold_partitions`. Append `fold_partitions` unconditionally in
`LevelParams::append_descriptor_bytes`, including the value `1`.

### Prover Changes

Add partition-aware fold APIs beside the existing representation-specific folds:

```rust
fn decompose_fold_partitioned(
    &self,
    challenges: &[SparseChallenge],
    block_len: usize,
    num_blocks: usize,
    num_digits_commit: usize,
    log_basis: u32,
    fold_partitions: usize,
) -> Result<PartitionedDecomposeFoldWitness<D>, AkitaError>;
```

For one-hot, `onehot_accumulate` should accept a block range or a partition plan
and emit partition-major accumulators without first producing the aggregate.
Dense and recursive witnesses can initially use a simple CPU reference path that
loops over partitions; GPU backends can override with device-local kernels.

`RingRelationProver::new` and `new_recursive_multipoint` should:

1. Read `fold_partitions` from `LevelParams` and validate it is active for the
   level.
2. Sample the same challenge stream as today.
3. Build `z_folded_rings` in partition-major order.
4. Build one relation row per `(public_row, partition)`.
5. Pass the partitioned `z` segment to `build_w_coeffs`.

`prove_root_fold_from_ring_relation`, `prove_suffix`, `commit_next_w`, and
`RecursiveProverState` should continue to carry one flattened logical witness and
one next-witness commitment. No API should expose device count to the transcript.
If a later design wants one commitment or one carried evaluation per partition,
that should be specified as a separate recursive multipoint protocol change.

### Verifier Changes

The verifier reconstructs the same partition plan from `LevelParams` and uses it
only for relation replay and witness-length checks:

1. Decode `LevelParams.fold_partitions` from the schedule/descriptor path.
2. Validate it equals the active value derived from `Cfg` and `lp.num_blocks`.
3. Check proof vector lengths against partition-aware `w_len` and M-row counts.
4. Derive the same folding challenges.
5. Build or lazily evaluate the partition consistency rows.
6. Keep recursive state unchanged:

```text
opening_point = stage2 challenges
opening       = proof.stage2.next_w_eval()
commitment    = proof.stage2.next_w_commitment
w_len         = schedule.next_w_len
```

This is why the proof wire does not need multiple next-witness commitments: the
committed object is one flattened partitioned witness, and the verifier's random
opening claim is an evaluation of that flattened object.

### Planner And Tables

Add `multi_fold_partitions` to `PlannerPolicy` and include it in every place that
computes root and recursive witness sizes. Recursive suffix planning currently
uses `(num_points, num_t_vectors, num_w_vectors, num_z_vectors) = (1, 1, 1, 1)`;
that public-row contract stays true for v1 recursive levels, because recursive
proofs are single-point today. What changes is the partition-aware helper math:

```text
num_fold_partitions = active_partitions_for_level(params)
z digit count       = num_public_rows
                    * sum_j(partition_inner_width(j) * digits_fold(j))
consistency rows    = if num_fold_partitions == 1 {
                        1
                      } else {
                        num_public_rows * num_fold_partitions
                      }
```

At the root, `num_public_rows` continues to come from `key.num_z_vectors`; do not
overload `num_z_vectors` as the partition count.

Generated table work:

- Add `akita_config::multi_fold` with concrete `Cfg` types.
- Add rows to `ALL_GENERATED_FAMILIES`, for example:
  - `multi_fold_fp128_d64_onehot_k4`
  - `multi_fold_fp128_d64_onehot_k8`
  - `multi_fold_fp128_d64_onehot_k12`
- Add corresponding modules and table constructors in
  `akita_planner::generated`.
- Extend `shipped_table` so `multi_fold_partitions == 1` selects existing tables
  and larger partition counts select only matching `multi_fold_*` tables. An
  `8`-partition config must never alias the `1`-partition table.
- Keep compact `GeneratedFoldStep` entries free of `fold_partitions`; expansion
  derives the active value from `PlannerPolicy`, like `onehot_chunk_size`.
- v1 should be explicit about composition: start with flat, non-tiered
  `multi_fold` configs. Tensor and tiered composition can be added later with
  separate table discriminators.
- Regenerate with:

```bash
cargo run --release -p akita-config --bin gen_schedule_tables -- crates/akita-planner/src/generated
```

and the `--features zk` variant if `multi_fold` is supported under `zk`.

### GPU Backend Shape

This spec does not require GPU code, but the intended backend boundary is:

- Host selects the schedule and broadcasts transcript challenges.
- Each GPU owns a deterministic block partition `P_j` or several partitions.
- Each GPU computes its `z_j` and keeps it in device memory in the canonical
  partition-major witness layout.
- A distributed commitment kernel computes the single `next_w_commitment` from
  the flattened logical witness. Only the small commitment/proof payloads need to
  return to the host transcript.
- Stage-2 final witness evaluation should be computed as a distributed MLE over
  the same flattened order. Devices reduce one scalar, not the full `z` vector.

The CPU backend remains the reference and should be able to run every
`multi_fold` config without accelerator crates.

### Alternatives Considered

- **Host all-reduce to one `z`**: simplest protocol, but it is exactly the
  bottleneck this feature targets.
- **Device all-reduce to one `z`**: avoids CPU transfer, but still forces a large
  synchronization before the next level. It should remain a benchmark baseline,
  not the protocol target.
- **One commitment per GPU**: exposes device count in the proof, complicates
  recursive state, increases transcript surface, and makes verification depend on
  prover placement. Rejected.
- **Carry `z_j` but check only `sum_j z_j`**: smaller relation change, but it does
  not bind each device-local partial to its assigned block range. Rejected for the
  first protocol version.
- **Use existing `num_z_vectors` as the partition count**: root schedule keys
  already use `num_z_vectors` for public row shape. Overloading it would confuse
  batching semantics and does not cover recursive levels. Rejected.

## Documentation

Update these docs when implemented:

- `AGENTS.md`: add the `multi_fold` config family and generated-table ownership.
- `specs/akita-compute-backend-metal.md`: link this spec as the protocol shape
  needed before multi-GPU recursive witness residency.
- `examples/profile.rs` / profile README text: add modes such as
  `onehot_fp128_d64_multi_fold_k8`.
- Crate docs in `akita-config` and `akita-planner`: document that
  `multi_fold_partitions` is a table discriminator and descriptor-bound policy.

## Execution

Recommended implementation order:

1. Add `CommitmentConfig::multi_fold_partitions()`, `PlannerPolicy` plumbing, and
   `LevelParams.fold_partitions`, defaulting every existing config to `1`.
2. Update descriptor binding and checked witness/M-row/proof-size count helpers.
   Land tests proving `fold_partitions = 1` is unchanged and larger partition
   counts have expected sizes.
3. Add CPU partitioned fold reference paths for dense, one-hot, and recursive
   witnesses. Keep these representation-aware; do not force all paths through a
   dense materialization.
4. Update ring-relation prover and verifier replay to use partition consistency
   rows.
5. Add `akita_config::multi_fold` configs and runtime DP tests.
6. Generate and register shipped tables.
7. Add e2e proof/verify tests and cross-config rejection tests.
8. Only after the CPU reference is green, add GPU backend planning/kernels that
   keep partitioned `z` on devices.

Risks to resolve early:

- The relation-row expansion can increase verifier work with the partition
  count; benchmark proof bytes and verifier time before deciding which partition
  counts deserve shipped tables.
- If partition digit depth is not tightened, the first table set may look much
  worse than the intended GPU tradeoff. Implement the explicit
  `max_partition_blocks` digit bound before treating profile results as final.
- ZK support touches hiding-witness allocation and cursor accounting for the
  larger partitioned `z` and row set. It should be either implemented with tests
  or explicitly gated off for the first `multi_fold` configs.

## References

- `specs/akita-compute-backend-metal.md`
- `specs/tensor-structured-folding-challenges.md`
- `crates/akita-prover/src/protocol/ring_relation.rs`
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`
- `crates/akita-types/src/schedule.rs`
- `crates/akita-types/src/layout/params.rs`
- `crates/akita-verifier/src/protocol/levels/recursive.rs`
- `crates/akita-config/src/generated_families.rs`
- `crates/akita-planner/src/resolve.rs`
