# Runtime-ring cutover: performance contract

Canonical design record: [`specs/runtime-ring-cutover.md`](../specs/runtime-ring-cutover.md).
Verifier safety contract (orthogonal): [`docs/verifier-contract.md`](verifier-contract.md).

Demoting `const D` from protocol **storage** must not regress prover or verifier throughput on
uniform-`D` workloads. `const D` stays on kernel and backend monomorphization sites behind
`dispatch_ring_dim_result!` until Wave 7.

## Rules

1. **D-erased storage, D at the boundary.** Types like `RingRelationInstance<F>`,
   `RingMultiplierOpeningPoint<F>`, and `PreparedOpeningPoint<F, L>` store `RingBuf<F>` (or
   scalar payloads). The schedule supplies `ring_d`; callers inside
   `dispatch_ring_dim_result!(ring_d, |D| { … })` pass `D` as a type parameter at use sites.
   This is not re-coupling `D` to the struct.

2. **Validate once, borrow hot.** Shape checks belong at construction, deserialization, or
   verifier entry. After a value is built at `D` (prover `new::<D>`, `from_ring::<D>`, or
   fallible `as_ring_slice::<D>` on wire decode), hot paths use trusted borrows:
   `as_ring_slice_trusted::<D>`, `as_single_ring_trusted::<D>`,
   `packed_inner_ring_trusted::<D>`, and infallible `RingMultiplierOpeningPoint` accessors.
   Release builds must not re-check `ring_dim` or coefficient divisibility on those paths.

3. **Fast path in fallible APIs.** `FlatRingVec::as_ring_slice::<D>` checks
   `self.ring_dim == D` once; on match it delegates to the trusted borrow (one compare, no
   alloc). Use fallible `as_ring_slice` on verifier and wire boundaries; use trusted borrows
   in prover fold, ring-switch, and stage-2 trace hot loops.

4. **No new hot-loop `Result` branches.** Do not replace infallible prover accessors
   (`a_rings`, `is_constant`, `a_constant_coeff`, etc.) with fallible ones. Index errors and
   transcript shape errors may still return `Result`; ring-dimension shape is not re-validated
   per access.

5. **Kernels stay monomorphized.** `RingSwitchProveBackend<F, D>`, NTT/matvec kernels, and
   `OpeningFoldKernel<…, D>` keep `const D`. Moving `dispatch_ring_dim_result!` between files
   is a refactor only; it must not add dynamic dispatch inside tight loops.

6. **No extra allocation on the fold path.** `RingBuf` replaces `Vec<CyclotomicRing<F, D>>`
   with the same flat coefficient layout. Trusted borrows are pointer casts over existing
   storage; no `to_vec`, `clone`, or fresh `Vec` in fold, ring-switch, or trace weight
   materialization hot paths. Suffix levels use `ProverOpeningBatch::carried_flat_commitment`
   so `FlatRingVec` is not rebuilt into `RingCommitment` and back per fold level.

7. **Profile gate for uniform-D.** Before merging a Wave 6+ storage demotion PR, run the
   profile preset called out in [`book/src/usage/profiling.md`](../book/src/usage/profiling.md)
   on uniform-`D` configs. Wall time must not regress beyond noise; proof wire bytes stay
   identical through Wave 5 and unchanged by storage-only demotion in Wave 6.

## API map (Wave 6a salvage)

| Layer | Fallible (boundary) | Trusted (hot path) |
|-------|---------------------|--------------------|
| `FlatRingVec` | `as_ring_slice`, `as_single_ring` | `as_ring_slice_trusted`, `as_single_ring_trusted` |
| `PreparedOpeningPoint` | `packed_inner_ring` | `packed_inner_ring_trusted` |
| `RingMultiplierOpeningPoint` | (construction only) | `a_rings::<D>`, `b_rings::<D>`, `is_constant::<D>`, `a_constant_coeff::<D>` |
| `RingRelationInstance` | `new::<D>`, `row_coefficient_rings::<D>`, `y::<D>`, `v_as_ring_slice::<D>` | same methods after successful `new::<D>`; fast path via tagged `ring_dim` |

## Out of scope for this contract

- Planner mixed-`D` DP search (Phase 4).
- `CommitmentProver<F, D>` / `AkitaCommitmentHint<F, D>` demotion (Wave 7).
