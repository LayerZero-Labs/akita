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
one dimension-free `compute_inner_outer_commitment` orchestration entry point,
erase the outer ring dimension at the end of the outer stage, and execute
compression through a separate `compute_commitment_compression` entry point.
Move validated parameter and geometry resolution behind
`get_commitment_geometry` and remove the intermediate
`commit_with_validated_geometry` layer. The refactor changes neither the
commitment, hint, proof shape, validation behavior, nor the computational work
performed.

## Intent

### Goal

Make `commit` read as four operations: `get_commitment_geometry`,
`compute_inner_outer_commitment`, `compute_commitment_compression`, and final
`CommitOutput` assembly.

### Invariants

- `compute_inner_commitment` performs exactly one same-shape batched
  `RootCommitKernel::commit_inner_group` invocation and returns its
  `CommitInnerWitness` values in source order.
- `compute_outer_commitment` validates the backend's group length and every
  inner witness shape before using its rows.
- `get_commitment_geometry` owns input-layout validation, explicit or scheduled
  parameter resolution, setup admission, committed-source class admission,
  commitment geometry validation, and source-encoding validation.
- `compute_inner_outer_commitment` owns both role dispatches, source admission,
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
- `compute_commitment_compression` owns compression-plan derivation, executor
  input construction, compression execution, terminal ring-dimension recovery,
  and terminal payload construction.
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
- Do not retain `commit_with_validated_geometry`; `commit` is the single owner
  of root commitment composition and final output assembly.

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
- [x] `compute_inner_outer_commitment` owns the `D_A` and `D_B` dispatches and
      returns `(inner_rows, outer_source)` with no const-generic ring dimension
      in its return type.
- [x] `get_commitment_geometry` owns parameter selection and all validation
      needed before commitment computation.
- [x] `commit_with_validated_geometry` is removed.
- [x] `commit` contains no inner-role or outer-role dispatch and calls
      `compute_inner_outer_commitment` exactly once.
- [x] `compute_commitment_compression` lives in
      `crates/akita-prover/src/api/commitment/compression.rs` and owns the
      complete compression block.
- [x] `commit` calls `compute_commitment_compression` exactly once after
      `compute_inner_outer_commitment` returns.
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
get_commitment_geometry
    |
    v
compute_inner_outer_commitment
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
    |           compute_commitment_compression
    |                         |
    +-------------------------+
                  |
                  v
       Commitment + AkitaCommitmentHint
```

`get_commitment_geometry` lives beside `commit` in
`crates/akita-prover/src/api/commitment.rs`. It returns a
`ResolvedCommitmentGeometry` containing owned copy-sized A/B geometry, checked
slice geometry, the admitted source contract, and the final group profile. The
owned matrices avoid cloning `CommittedGroupParams` while allowing a scheduled
row to be dropped before arithmetic begins.

`compute_inner_outer_commitment` lives in
`crates/akita-prover/src/api/commitment/inner_outer.rs`. It is called directly
by `commit` and is the only function called there for the uncompressed inner
and outer commitment stages. Its conceptual interface is:

```rust,ignore
fn compute_inner_outer_commitment<F, P, B>(
    polys: &[P],
    ctx: &OperationCtx<'_, F, B>,
    geometry: CommitmentGeometry,
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

`compute_commitment_compression` lives in
`crates/akita-prover/src/api/commitment/compression.rs`. It accepts the
dimension-erased outer source and returns the terminal payload together with
the witness and quotient rows needed by the hint:

```rust,ignore
fn compute_commitment_compression<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    modulus_profile: SisModulusProfileId,
    source: RingVec<F>,
) -> Result<CommitmentCompressionOutput<F>, AkitaError>;
```

The conversion from typed outer rows to `RingVec<F>` stays at the end of
`compute_outer_commitment`, because typed `u` cannot cross the runtime `D_B`
dispatch. All subsequent compression logic belongs to
`compute_commitment_compression`.

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

`commit` resolves checked geometry, invokes the two arithmetic stages, and
assembles the public result. It contains no role dispatch or inline compression
implementation:

```rust,ignore
let ResolvedCommitmentGeometry {
    geometry,
    slice_geometry,
    contract,
    profile,
} = get_commitment_geometry(polys, expanded, context)?;
let ctx = stack.commit();

let (inner_rows, outer_source) = compute_inner_outer_commitment(
    polys,
    ctx,
    geometry,
    &slice_geometry,
    contract,
)?;

let CommitmentCompressionOutput { payload, witness, quotients } =
    compute_commitment_compression(
    ctx,
    geometry.outer_matrix.sis_table_key().modulus_profile,
    outer_source,
)?;

// Assemble AkitaCommitmentHint and CommitOutput, then return.
```

The body of `compute_inner_outer_commitment` contains the staged dispatch:

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
3. **Retain `commit_with_validated_geometry` as an intermediate composer.**
   Rejected because it only sequences two already named, dimension-erased
   operations and hides the simple root commitment flow from `commit`.
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
3. Add `compute_inner_outer_commitment` to `inner_outer.rs`; move both role
   dispatches, source admission, commit-view construction, and the ordered
   stage calls into it.
4. Add commitment-level `compression.rs` with
   `compute_commitment_compression`.
5. Add `get_commitment_geometry` to own parameter selection and pre-computation
   validation, using owned copy-sized matrix geometry rather than cloning full
   parameters.
6. Remove `commit_with_validated_geometry` and compose
   `get_commitment_geometry`, `compute_inner_outer_commitment`, and
   `compute_commitment_compression` directly in `commit`.
7. Move shape and slice unit tests into `inner_outer.rs`; keep end-to-end
   composition tests in the parent commitment test module.
8. Run focused correctness tests, release performance comparison, repository
   preflight, applicable CI feature graphs, and documentation guardrails.
9. When implementation begins, change the status to `active`; when it lands,
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
