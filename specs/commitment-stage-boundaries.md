# Spec: Commitment stage boundaries

| Field         | Value      |
|---------------|------------|
| Author(s)     |            |
| Created       | 2026-08-24 |
| Status        | active     |
| PR            |            |
| Supersedes    |            |
| Superseded-by |            |
| Book-chapter  |            |

## Summary

The root commitment implementation currently combines construction of the inner
commitments `t_i`, preparation of those commitments for the outer matrix, and
construction and compression of the outer image inside nested role-dimension
dispatches. This obscures the two commitment stages and keeps dimension-erased
compression inside macros whose dimensions it does not use. Split the path into
canonical `compute_inner_commitment` and `compute_outer_commitment` stages under
one dimension-free `commit_inner_outer` orchestration entry point, erase the
outer ring dimension at the end of the outer stage, and execute compression
through a separate `compute_compression_chain` entry point. The refactor changes
neither the commitment, hint, proof shape, validation behavior, nor the
computational work performed.

## Intent

### Goal

Make `commit_with_validated_geometry` compose exactly two named operations:
`commit_inner_outer` for all role-dispatched commitment work and
`compute_compression_chain` for dimension-erased compression.

### Invariants

- `compute_inner_commitment` performs exactly one same-shape batched
  `RootCommitKernel::commit_inner_group` invocation and returns its
  `CommitInnerWitness` values in source order.
- `compute_outer_commitment` validates the backend's group length and every
  inner witness shape before using its rows.
- `commit_inner_outer` owns both role dispatches, source admission,
  commit-view construction, and the ordered calls to the inner and outer stage
  functions. It returns only dimension-erased values.
- Commit views remain source-typed and borrowed until they are moved into
  `RootCommitKernel::commit_inner_group`. In particular, a recursive
  `SuffixWitnessView` continues to expose packed signed digits directly to
  `recursive_packed_witness_commit_rows`; orchestration must not predecode or
  materialize its coefficients.
- The outer stage preserves the existing polynomial-major digit layout,
  canonical slice order, zero padding, and `B`-matrix computation enforced by
  `decompose_commit_blocks_into` and `commit_outer_slices`.
- The outer stage returns the persistent A-native inner rows in source order for
  `AkitaCommitmentHint` and returns the outer image as a dimension-erased
  `RingVec<F>` for compression.
- Compression consumes exactly the coefficients of that outer `RingVec`; its
  plan, terminal payload, witness, and quotient images remain byte-for-byte
  unchanged.
- `compute_compression_chain` owns compression-plan derivation, executor input
  construction, compression execution, terminal ring-dimension recovery, and
  terminal payload construction.
- Inner work depends on `D_A` but not `D_B`. Outer decomposition and matrix work
  depend on `D_A` and `D_B`. Compression depends on neither role dimension.
- No new clones, full-size buffers, backend calls, or serial iteration are
  introduced.
- Error ordering stays unchanged within each stage: source admission precedes
  the inner kernel; group-length and inner-shape validation precede
  decomposition; outer-shape validation precedes compression.

### Non-Goals

- Do not change `RootCommitKernel`, `DigitRowsComputeBackend`, compression
  kernels, setup formats, schedules, transcript data, proof serialization, or
  verifier behavior.
- Do not change `RecursiveWitnessFlat`, `SuffixWitnessView`, packed signed-digit
  storage, or the packed recursive commitment kernel introduced on the target
  branch.
- Do not change the mathematical decomposition, slice geometry, matrix
  multiplication, compression policy, or compression map dimensions.
- Do not make `D_A` or `D_B` runtime values inside typed arithmetic kernels.
- Do not retain the current combined preparation helper as a compatibility
  alias. This repository provides no backward-compatibility guarantee, and
  retaining it would create a second entry point for the same operation.

## Evaluation

### Acceptance Criteria

- [x] The current combined preparation helper is removed.
- [x] `compute_inner_commitment` contains only the call that computes the
      same-shape group of inner witnesses; it does not validate, decompose, or
      compute outer rows.
- [x] `compute_inner_commitment` has no `D_B` parameter and is called before the
      outer-role dispatch.
- [x] `compute_outer_commitment` owns group-length validation, parallel witness
      validation and decomposition, outer-slice commitment, collection of the
      persistent inner rows, and conversion of typed outer rows to `RingVec<F>`.
- [x] `commit_inner_outer` owns the `D_A` and `D_B` dispatches and
      returns `(inner_rows, outer_source)` with no const-generic ring dimension
      in its return type.
- [x] `commit_with_validated_geometry` contains no inner-role or outer-role
      dispatch and calls `commit_inner_outer` exactly once.
- [x] `compute_compression_chain` lives in
      `crates/akita-prover/src/api/commitment/compression.rs` and owns the
      complete compression block.
- [x] `commit_with_validated_geometry` calls `compute_compression_chain` exactly
      once after `commit_inner_outer` returns.
- [x] The former `inner.rs` module is renamed to `inner_outer.rs`, and focused
      shape and slice unit tests live with that module.
- [x] The compression plan continues to use the outer matrix's SIS modulus
      profile and `outer_source.coeff_len()`.
- [x] Existing commitment and compression tests pass without fixture updates
      except for helper-name or helper-signature changes required by the
      refactor.
- [x] The implementation adds no clone of `CommitInnerWitness`, `RingVec`,
      `DigitBlocks`, typed outer rows, or their coefficient storage.
- [x] The generic inner orchestration moves each source-typed commit view
      directly into the backend call; it introduces no conversion of packed
      recursive witness data to an unpacked coefficient vector.
- [ ] Release-profile before/after measurement shows no statistically
      meaningful commitment-throughput regression on the same machine and
      fixture.

### Testing Strategy

Keep the existing shape and slice tests with their owning implementation in
`crates/akita-prover/src/api/commitment/inner_outer.rs`, including
`commit_inner_shape_accepts_expected_layout`, the bad-block and bad-row
rejections, the all-zero-block case, slice padding, buffer reuse, and the
independent block-diagonal oracle. Keep end-to-end composition tests in
`crates/akita-prover/src/api/commitment/tests.rs`. Adapt the independent
unsliced reference helper to call the two new stages. Preserve
`s1_matches_real_unsliced_commitment_pipeline`, which compares the production
commitment, compression witness, compression quotients, and complete hint with
an independent reference and also checks all supported slice counts.

The rebased target stores recursive witnesses as packed signed digits. Preserve
the focused kernel equivalence tests
`packed_recursive_commit_matches_predecoded_block_parallel_kernel` and
`packed_recursive_raw_commit_matches_predecoded_kernel` in
`crates/akita-prover/src/compute/cpu/kernel_tests.rs`. Add or reuse an
end-to-end `RecursiveWitnessFlat` commitment fixture when measuring this
refactor so the orchestration boundary cannot silently erase the packed-source
optimization.

Run the repository's cheap CI preflight first. Then run focused
`akita-prover` tests in the development profile, followed by the current
release test and Clippy invocations selected by `.github/workflows/ci.yml` for
the affected feature graphs. Because this changes a live spec, also run
`./scripts/check-doc-guardrails.sh`.

No new cryptographic test vector is required: this refactor does not change a
protocol relation. A focused stage-boundary unit test may be added if it can
verify ownership or call count without duplicating the existing end-to-end
oracle.

### Performance

The before and after paths must have the same asymptotic work and allocation
shape:

| Operation | Before | After |
|-----------|--------|-------|
| Commit-view collection | one `Vec` | one `Vec` |
| Packed recursive source decoding | backend-owned, block-local | unchanged |
| Inner backend invocation | one batched call | one batched call |
| Inner witness storage | one `Vec` | the same `Vec`, moved |
| Witness validation/decomposition | `cfg_into_iter!` | unchanged `cfg_into_iter!` |
| Prepared digit storage | one vector per source | unchanged |
| Outer backend invocation | one batched slice call | unchanged |
| Outer dimension erasure | one `RingVec::from_ring_elems` | unchanged |
| Compression execution | one chain execution | unchanged, outside role dispatch |

No intermediate representation may be cloned to cross a function or macro
boundary. Generic helpers remain statically dispatched and monomorphized.
Moving compression outside role dispatch reduces generated duplication and must
not add a runtime dispatch: the compression executor already dispatches on the
compression maps' own checked ring dimensions.

For performance acceptance, record multiple release-profile samples of the
same representative recursive packed-witness commitment fixture before and
after the change. A dense or one-hot fixture may be measured additionally but
does not replace the recursive measurement on this branch. Compare medians
after a warm-up and investigate any result outside ordinary run-to-run
variance. The refactor is not accepted with a reproducible throughput
regression, increased peak allocation, or eager unpacking of recursive source
digits.

## Design

### Architecture

The intended data flow is:

```text
polynomial sources
    |
    v
commit_inner_outer
    |
    | owns D_A dispatch
    v
compute_inner_commitment
    |
    | Vec<CommitInnerWitness<F>> (t_i)
    | owns nested D_B dispatch
    v
compute_outer_commitment
    |                         |
    | Vec<RingVec<F>>         | RingVec<F>
    | persistent t_i rows     | dimension-erased outer image
    |                         v
    |              compute_compression_chain
    |                         |
    +-------------------------+
                  |
                  v
       Commitment + AkitaCommitmentHint
```

`commit_inner_outer` lives in
`crates/akita-prover/src/api/commitment/inner_outer.rs`. It is called by
`commit_with_validated_geometry` in
`crates/akita-prover/src/api/commitment.rs` and is the only function called by
that caller for the uncompressed inner and outer commitment stages. Its
conceptual interface is:

```rust,ignore
fn commit_inner_outer<F, P, B>(
    polys: &[P],
    ctx: &OperationCtx<'_, F, B>,
    geometry: CommitmentGeometry<'_>,
    slice_geometry: &CommitmentSliceGeometry,
    contract: CommittedSourceContract,
) -> Result<(Vec<RingVec<F>>, RingVec<F>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RuntimeCommitSource<F>,
    B: RuntimeCommitBackendFor<F, P>;
```

The exact field bounds may include bounds already required by source admission
or the backend capability traits; the important API property is that this
function has no const-generic role dimension and returns no typed ring value.
It derives `CommitmentRingDims` and `CommitInnerPlan` from `geometry`, obtains
the backend and prepared setup from `ctx`, dispatches `D_A`, admits sources and
constructs their typed views, calls `compute_inner_commitment`, dispatches
`D_B`, and calls `compute_outer_commitment`. It performs no compression.

`compute_inner_commitment` lives in
`crates/akita-prover/src/api/commitment/inner_outer.rs` and has the following
conceptual interface:

```rust,ignore
pub(super) fn compute_inner_commitment<F, S, B, const D_A: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    sources: Vec<S>,
    plan: CommitInnerPlan,
) -> Result<Vec<CommitInnerWitness<F>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F> + RootCommitKernel<S, F, D_A>,
{
    backend.commit_inner_group(prepared, sources, plan)
}
```

This is deliberately the complete inner protocol-stage boundary. It has no
outer-ring generic parameter or outer-stage logic. The call still occurs inside
the `D_A` dispatch because `RootCommitKernel<S, F, D_A>` and each source's
commit view are typed by `D_A`. Moving this exact one-line operation outside
the `D_A` dispatch would require adding another runtime dispatch inside the
function and would make it more than the requested operation.

`compute_outer_commitment` also lives in `inner_outer.rs`. Its conceptual
interface is:

```rust,ignore
pub(super) fn compute_outer_commitment<F, B, const D_A: usize, const D_B: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    inners: Vec<CommitInnerWitness<F>>,
    expected_source_count: usize,
    n_a: usize,
    num_live_blocks: usize,
    num_digits_open: usize,
    log_basis: u32,
    n_b: usize,
    slice_geometry: &CommitmentSliceGeometry,
) -> Result<(Vec<RingVec<F>>, RingVec<F>), AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>;
```

The implementation first checks `inners.len()`, then retains the current
parallel map that validates each witness, borrows its typed blocks, decomposes
them for the B role, and pairs its consumed persistent rows with the resulting
digits. It passes borrowed digit blocks to `commit_outer_slices`, collects the
persistent rows without cloning them, and converts the resulting typed `u`
rows into one `RingVec<F>` before returning.

The outer stage includes `commit_outer_slices`; otherwise a helper named
`compute_outer_commitment` would only prepare digits and the call site would
still expose a third, unnamed part of the outer commitment stage.

`compute_compression_chain` lives in
`crates/akita-prover/src/api/commitment/compression.rs`. It accepts the
dimension-erased outer source and returns the terminal payload together with
the witness and quotient rows needed by the hint:

```rust,ignore
fn compute_compression_chain<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    modulus_profile: SisModulusProfileId,
    source: RingVec<F>,
) -> Result<CompressionChainOutput<F>, AkitaError>;
```

The conversion from typed outer rows to `RingVec<F>` stays at the end of
`compute_outer_commitment`, because typed `u` cannot cross the runtime `D_B`
dispatch. All subsequent compression logic belongs to
`compute_compression_chain`.

### Before

The current control flow nests all work, including dimension-erased
compression, under both role dispatches:

```rust,ignore
dispatch D_A {
    ensure_sources_fit_accepted_interval(...)?;

    dispatch D_B {
        let views = collect_commit_views(polys)?;
        let prepared_polynomials = current_combined_preparation::<F, _, _, D_A, D_B>(
            backend,
            prepared,
            views,
            plan,
            num_live_blocks,
            num_digits_open,
            log_basis,
        )?;
        let u = commit_outer_slices::<F, _, D_B>(...)?;
        let source = RingVec::from_ring_elems(&u);

        let compression_plan = CompressionChainPlan::for_complete_source(
            modulus_profile,
            source.coeff_len(),
        )?;
        let output = execute_compression_chains(..., source.into_coeffs())?;

        Ok((commitment, collect_inner_rows(prepared_polynomials), witness, quotients))
    }
}
```

### After

The orchestration helper owns all typed role work and returns an entirely
dimension-erased result. `commit_with_validated_geometry` contains no role
dispatch:

```rust,ignore
let (inner_rows, outer_source) = commit_inner_outer(
    polys,
    ctx,
    geometry,
    slice_geometry,
    contract,
)?;

let CompressionChainOutput { payload, witness, quotients } =
    compute_compression_chain(
    ctx,
    geometry.outer_matrix.sis_table_key().modulus_profile,
    outer_source,
)?;

// Final Commitment and AkitaCommitmentHint assembly remains here.
```

The body of `commit_inner_outer` contains the staged dispatch:

```rust,ignore
let (inner_rows, outer_source) = dispatch D_A {
    ensure_sources_fit_accepted_interval(...)?;
    let views = collect_commit_views::<D_A>(polys)?;
    let inners = compute_inner_commitment::<F, _, _, D_A>(...)?;

    dispatch D_B {
        compute_outer_commitment::<F, _, D_A, D_B>(..., inners, ...)
    }
}?;

Ok((inner_rows, outer_source))
```

### Why compression is outside the role dispatches

Compression does not use `D_A` or `D_B`:

- `CompressionChainPlan::for_complete_source` takes a modulus profile and a
  flat source coefficient count.
- `CompressionExecutionInput` owns a flat `Vec<F>` rather than typed
  `CyclotomicRing<F, D_A>` or `CyclotomicRing<F, D_B>` values.
- `execute_compression_chains` groups work by the checked dimensions stored in
  each `CompressionMapPlan` and performs its own
  `ProtocolDispatchSlot::Compression` dispatch.
- The terminal ring dimension is read from the last compression map, not from
  either commitment role.

The typed outer image must be flattened inside `D_B` via
`RingVec::from_ring_elems`; after that boundary there is no role-dimension type
left. The current placement appears to be control-flow locality inherited from
building `u` inside the nested dispatch, not a type or protocol requirement.
The setup-prefix commitment path already demonstrates the desired structure by
returning a `RingVec` from its outer dispatch and compressing afterward.

### Single-source-of-truth review note

The requested `compute_inner_commitment` is intentionally a one-operation
protocol-stage boundary over the backend kernel. Reviewers should decide
explicitly whether this named boundary is an acceptable exception to the
repository's general prohibition on pass-through aliases. If it is not, the
compliant alternative is to keep the direct
`backend.commit_inner_group(prepared, views, plan)` call at the staged call site
and introduce only `compute_outer_commitment`; retaining both names through an
additional alias is not acceptable.

### Alternatives Considered

1. **Split only the old preparation helper.** This would leave
   `commit_outer_slices` at the call site, so the proposed outer helper would
   prepare rather than compute the outer commitment. Rejected because it does
   not express the requested two protocol stages.
2. **Leave compression inside `D_B`.** This is type-correct but preserves a
   false dependency and repeats dimension-independent control flow in generated
   role branches. Rejected because `RingVec` is the existing erasure boundary.
3. **Keep role dispatch directly in `commit_with_validated_geometry`.** This is
   type-correct but leaves protocol-stage orchestration mixed with compression
   and final output assembly. Rejected because a dimension-free orchestration
   function gives the caller one clear boundary without hiding the two typed
   stage functions internally.
4. **Combine all work into one function and remove the two stage helpers.**
   Rejected because the orchestration entry point and the two mathematical
   stages have different responsibilities: runtime role selection versus the
   typed `t_i` and `u` computations.
5. **Keep the current combined preparation helper as an alias.** Rejected
   because the repository makes no compatibility guarantee and forbids
   duplicate wrapper APIs for one concept.

## Documentation

This proposed internal refactor is recorded here for team review. It changes no
user-visible API or protocol behavior, so no Book or `AGENTS.md` update is
required during the proposal stage. If the implemented names become durable
architecture vocabulary, fold the stage description into
`book/src/how/architecture.md`, set `Book-chapter`, and archive this spec under
the applicable quarter according to `specs/PRUNING.md`.

Keep this spec synchronized with `specs/PRUNING.md`,
`book/src/foundations/spec-index.md`, and
`scripts/check-spec-references.sh` while it remains live.

## Execution

1. Rename the commitment `inner.rs` module to `inner_outer.rs`, add the two
   typed stage functions there, and remove the current combined preparation
   helper.
2. Update imports and the independent commitment reference in
   `crates/akita-prover/src/api/commitment/tests.rs`.
3. Add `commit_inner_outer` to `inner_outer.rs`; move both role dispatches,
   source admission, commit-view construction, and the ordered stage calls into
   it.
4. Replace the nested dispatch in `commit_with_validated_geometry` with one
   `commit_inner_outer` call returning `(Vec<RingVec<F>>, RingVec<F>)`.
5. Add commitment-level `compression.rs` with `compute_compression_chain`, and
   replace inline compression with one call after `commit_inner_outer`.
6. Move shape and slice unit tests into `inner_outer.rs`; keep end-to-end
   composition tests in the parent commitment test module.
7. Run focused correctness tests, release performance comparison, repository
   preflight, applicable CI feature graphs, and documentation guardrails.
8. When implementation begins, change the status to `active`; when it lands,
   complete the acceptance list, record the PR, and follow the fold/archive
   policy.

## References

- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-prover/src/api/commitment/inner_outer.rs`
- `crates/akita-prover/src/api/commitment/compression.rs`
- `crates/akita-prover/src/api/setup_prefix.rs`
- `crates/akita-prover/src/compute/compression.rs`
- `crates/akita-types/src/compression/chain.rs`
- `crates/akita-prover/src/api/commitment/tests.rs`
- `crates/akita-prover/src/backend/recursive/witness.rs`
- `crates/akita-prover/src/compute/cpu/commitment.rs`
- `crates/akita-prover/src/compute/cpu/kernel_tests.rs`
- `docs/documentation.md`
- `specs/PRUNING.md`
