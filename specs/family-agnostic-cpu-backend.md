# Spec: Family-Agnostic CPU Backend

| Field         | Value                                                                  |
|---------------|------------------------------------------------------------------------|
| Author(s)     | Quang Dao                                                              |
| Created       | 2026-09-25                                                             |
| Status        | active                                                                 |
| PR            | [#76](https://github.com/LayerZero-Labs/akita/pull/76)                 |
| Supersedes    | `CpuBackend` configuration ownership; catalog-keyed prefix persistence |
| Superseded-by |                                                                        |
| Book-chapter  | book/src/usage/commitment-api.md                                       |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14 when,
and only when, they appear in all capitals.

## Summary

The CPU backend was typed by one commitment configuration and borrowed that
configuration's trusted schedule catalog at construction. Its owned resources
are the prepared setup, the commitment executor, and the NTT and setup-prefix
caches. They depend only on the base field, the extension field, and the setup
seed. A batch that opens a dense precommitted group with a one-hot final group
still needed one backend per family and an explicit commitment import between
them. PR #76 types the backend by `(F, E)` only. It passes a catalog to each
operation that resolves a row, records on each commitment handle the producer
contract that admitted its sources, and keys the persisted setup-prefix registry
by the slots it contains instead of by the catalog that requested them.

## Intent

### Goal

One `CpuBackend<F, E>` owns the prepared setup and private CPU resources for a
base field `F` and extension field `E`. It serves every schedule family over
that pair. Commitment receives the producer family's catalog, proof admission
receives the proving catalog, and setup persistence is keyed by the required
prefix slots.

### Invariants

1. **Producer catalog.** `CpuBackend::commit::<Cfg>(family, source, context)`
   resolves a scheduler context's profile from `family` and admits the sources
   under `Cfg::committed_source_contract()`. The returned handle MUST record
   that contract, exposed as `CommitmentHandleMetadata::producer_contract`.
   `CpuBackend::import_commitment` MUST carry the recorded contract over only
   after it recomputes the same source and commitment. Protected by
   `cpu_commit_matches_full_executor_state_after_outer_image_completion` in
   `crates/akita-cpu-backend/src/opaque/owned_commit.rs`.
2. **Proving catalog.** `ProofAdmission::begin_proof::<Cfg>` receives the
   proving catalog and requires the proof schedule to be one of its rows
   (`CpuBackend::validate_proof_configuration`). No backend state selects or
   narrows the row.
3. **Final-group contract.** The proving catalog plans the final group under
   the proving configuration's contract.
   `SelectedProverOpeningData::from_committed_claims::<Cfg>` MUST reject a
   final-group handle whose recorded contract differs from
   `Cfg::committed_source_contract()`, before any proof work. Protected by
   `final_group_admitted_under_another_producer_contract_is_refused` in
   `crates/akita-pcs/tests/akita_fp128_e2e/heterogeneous.rs`.
4. **Precommitted contracts.** A schedule row records each precommitted group's
   frozen commit profile, not its producer contract. The generic prover MUST NOT
   infer a contract for a precommitted handle. The application or planning code
   that chose the row MUST compare each precommitted handle's
   `producer_contract()` with the contract it planned that group under, which
   for planner requests is `PrecommittedProducer::source_contract`.
5. **Setup-prefix provenance.** Setup-prefix handles record a balanced-digit
   contract with the full field width as the commit bound. This matches the
   planner's sizing of setup-prefix groups as uniform field elements.
6. **Wire invariance.** Setup, commitment, proof, and transcript bytes are
   unchanged. The producer contract is local handle metadata. Proof bytes do not
   bind it, and the verifier continues to enforce the row's frozen caps.
7. **Slot-keyed persistence.** With `disk-persistence`, the setup-prefix
   registry file name MUST be a function of the field modulus, the setup-seed
   digest, and the sorted required slot ids only (`prefix_registry_cache_file_name`
   in `crates/akita-setup/src/lib.rs`). It MUST NOT depend on a catalog digest,
   the capacity bound, or union order. Protected by
   `combined_recursive_prefix_registry_persists_and_serves_each_family` in
   `crates/akita-setup/src/tests/disk_persistence.rs`.
8. **Combined requirements.** `SetupRequirements::union` MUST reject operands
   computed at different `(max_num_vars, max_num_batched_polys)` and otherwise
   yield the larger matrix capacity and the sorted union of prefix slots. A
   setup built from the union MUST let each constituent family commit and prove.

### Non-Goals

- Binding producer contracts into proof bytes, the transcript, or the verifier
  statement.
- Recording producer identity in schedule rows. Without it, the generic prover
  cannot check precommitted groups itself.
- Sharing one backend across different `(F, E)` pairs.
- Changing planner sizing, catalog contents, or catalog digests.

## Evaluation

### Acceptance Criteria

- [x] `CpuBackend<F, E>` has no configuration parameter and stores no catalog.
  `CpuBackend::new(expanded)` takes only the expanded setup.
- [x] `CpuBackend::commit::<Cfg>` takes the producer family's
  `&TrustedScheduleCatalog<Cfg>`. One backend commits groups from different
  families over the same `(F, E)` (`heterogeneous_group_types`,
  `bounded_dense_precommit_with_onehot_final_group`).
- [x] `ProofAdmission::begin_proof::<Cfg>` and
  `AkitaCommitmentScheme::batched_prove` use the proving catalog, and the proof
  schedule must be one of its rows.
- [x] Commitment handles record the admitting producer contract, readable
  through `CommitmentHandleMetadata::producer_contract`, and
  `import_commitment` preserves it.
- [x] `from_committed_claims::<Cfg>` rejects a final group committed under a
  different contract. The negative test commits one small-coefficient source
  under the dense and bounded-dense families, checks that the public
  commitments agree, and checks that only the dense handle is admitted under
  the dense proving catalog.
- [x] Precommitted-group contracts are the caller's comparison. The Book states
  this rule in `book/src/usage/commitment-api.md`.
- [x] The setup-prefix registry is keyed by field modulus, setup seed, and
  sorted slot ids, and one combined registry serves each family's slots.
- [x] `fp32_combined_setup_proves_recursive_and_onehot_families`
  (`crates/akita-pcs/tests/akita_small_field_e2e/combined_recursive.rs`) runs
  in default CI. It builds one setup from the union of
  `RecursiveCommitmentConfig<fp32::Dense>` and `fp32::OneHot` requirements at
  `nv = 24`, where the recursive row opens a setup prefix. It then proves and
  verifies an opening under each catalog on one backend.
- [x] Release gate: `combined_recursive_setup_proves_both_onehot_families`
  (`crates/akita-pcs/tests/recursive_setup_e2e.rs`) passes in an optimized
  build. It proves the fp128 recursive one-hot and recursive multi-chunk
  families at `(32, 4)` on one setup built from their union. Those two families
  require distinct setup-prefix slot sets. The test is production-sized and
  ignored in default CI. Run it before a release that relies on one setup for
  two recursive families:
  `cargo test --release -p akita-pcs --test recursive_setup_e2e --features profile-ci -- --ignored combined_recursive_setup_proves_both_onehot_families`

### Testing Strategy

Default CI covers every criterion except the release gate through the named
tests and the existing end-to-end suites, which now construct backends with
`CpuBackend::new` and pass catalogs per operation. The persistence test runs only with `disk-persistence`.
The release gate reuses the recursive multi-group round trip in
`crates/akita-pcs/tests/common/recursive.rs` on a caller-supplied setup, so it
exercises the same Stage 3 setup-prefix and tamper checks as the per-family
production-sized tests.

### Performance

The proving path is unchanged. The final-group check compares two small `Copy`
values once per proof. A combined setup materializes the union of the families'
prefix slots and the larger matrix capacity. Commitments read the same
seed-derived matrix prefix under the combined setup as under a per-family
setup.

## Design

### Architecture

`CpuBackend<F, E>` in `crates/akita-cpu-backend/src/opaque/backend.rs` owns the
expanded setup, the setup-prefix cache, and backend identity. Row resolution
moves to operation arguments. `commit` takes the producer catalog.
`begin_proof`, reached through `batched_prove`, takes the proving catalog.
`CommittedSource` in `crates/akita-cpu-backend/src/opaque/owned.rs` stores the
contract that `resolve_commit_params` admitted the sources under. External
backends expose the same fact through the `CommitmentHandleMetadata` trait in
`crates/akita-prover/src/backend/handles.rs`.

The final-group check lives in `from_committed_claims`, the one prover entry
point that has both the proving configuration and the ordered handles. The
planner freezes the final group's contract as the proving configuration's
contract. Precommitted rows store only frozen profiles, so their contracts
remain a planning-side comparison.

`akita-setup` builds a setup from `SetupRequirements` alone. Persistence keys
the setup-prefix registry by slot set. The prefix commitments are a pure
function of the seed-derived matrix and the slot id, so one registry serves
every catalog combination that requires the same slots.

### Alternatives Considered

- **Keep `CpuBackend<Cfg>` and import across backends.** Every family would
  duplicate setup preparation and caches, and every heterogeneous batch would
  need an import. Rejected.
- **Check every group's contract in the generic prover.** This needs producer
  identity in schedule rows, which changes the artifact format and catalog
  digests. Deferred; the final-group check needs no row change.
- **Bind the contract in proof bytes.** A contract mismatch cannot make a false
  claim verify, because the verifier enforces the row's frozen caps. It can
  make an honest proof fail, because completeness and the grinding budget
  assume the planned contract. That is a prover-side configuration error, so
  local rejection suffices. Rejected.
- **Key persistence by catalog digest.** A combined setup serves several
  catalogs, and its prefix commitments do not depend on any of them. Rejected.

## Documentation

- `book/src/usage/commitment-api.md` owns the producer-catalog and
  proving-catalog rule, the recorded contract, and the final-group check.
- `book/src/usage/setup-runtime.md` owns combined requirements and the shared
  setup.
- `docs/compute-backends.md` describes the family-agnostic backend.
- `specs/composable-commitment-execution.md`,
  `specs/external-schedule-catalog-ownership.md`, and
  `specs/opaque-prover-consumer.md` point to this record for backend ownership
  and prefix persistence.

After PR #76 merges, fold the remaining design rationale into
`book/src/usage/commitment-api.md`, set `Status` to `implemented`, and archive
this record according to `specs/PRUNING.md`.

## References

- PR [#76](https://github.com/LayerZero-Labs/akita/pull/76) and its review.
- `specs/external-schedule-catalog-ownership.md`: catalog ownership and exact
  setup capacity.
- `specs/heterogeneous-group-source-contracts.md`: source contracts and fold
  admission.
- `specs/composable-commitment-execution.md`: commitment execution inside the
  backend.
