# The commitment API

The user-facing surface of `AkitaCommitmentScheme`: how to commit, prove, and
verify, plus the setup and transcript objects those calls thread through.

## Commit, prove, verify

The `batched_commit`, `batched_prove`, and `batched_verify` entry points operate
on ordered commitment groups.
Every `PolynomialGroupClaims` owns one complete opening point, its evaluations,
and its commitment.
Polynomials within one group share that point.
Polynomials opened at another point belong in another group.

```rust
let claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(point_a, evaluations_a, commitment_a)?,
    PolynomialGroupClaims::new(point_b, evaluations_b, commitment_b)?,
])?;
```

The group order is protocol-visible.
The descriptor and transcript bind each group's arity, evaluation count,
commitment, point coordinates, and evaluations in that order.
`OpeningClaimsLayout` contains only group arities and polynomial counts, so
setup and schedule selection do not depend on point values.

On the prover side, `ProverOpeningData` privately binds each commitment hint to
one `PreparedProverGroup<P>` in the same protocol-visible order as the public
claims. `SelectedProverOpeningData` pairs that material with the one exact
`OpeningScheduleSelection` returned by the final commit. Akita treats the
concrete polynomial representation as a caller contract: callers must supply
groups with the arity and shape claimed by the public statement. Bad prover
material is a completeness failure, not a verifier soundness input.

Applications that need dense and one-hot polynomials in one opening use one
application-owned enum as `P`. Akita does not add provider registrations or
recursive heterogeneous wrappers. The verifier receives a
`GroupBatchStatement` containing only the exact generated-row selection and
self-describing public claims.

`commit_group` takes a raw polynomial group. `commit_final_group` takes the
exact ordered precommitted profiles and atomically returns the final
`CommittedGroup`, its hint, and the `OpeningScheduleSelection` that must be used
for proving and verification.

There is no ambient shared point, global polynomial type for the batch, or
coordinate-routing object.

**Sources to fold in**

- `crates/akita-pcs/src/scheme/mod.rs`.
- `crates/akita-prover/src/api/scheme.rs` (`CommitmentProver`).
- `crates/akita-types/src/proof/scheme.rs` (`CommitmentVerifier`).
- `crates/akita-types/src/opening_claims.rs` (`OpeningClaims`, `OpeningClaimsLayout`).
- `crates/akita-prover/src/types/opening_data.rs` (`ProverOpeningData`,
  `SelectedProverOpeningData`).
- `crates/akita-prover/src/api/prepared_group.rs` (coarse prepared group
  carrier).
- `crates/akita-pcs/tests/single_poly_e2e.rs`, `batched_aggregated_e2e.rs`.

## Setup and caching

Building public parameters, the shared setup vector reused as A/B/D matrices at
every level, and the optional on-disk setup cache.

**Sources to fold in**

- `crates/akita-setup/src/lib.rs`.
- Paper §3.9 `sec:akita-setup` (packed shared setup), `Setup` in §3.8 `sec:akita-full-pcs`.
- `specs/setup-layout-repack.md` (broader packed-setup direction — roadmap).
- `specs/flat-public-matrix-and-exact-ntt-cache.md` (active stacked follow-up to
  PR #334: dimension-free field derivation, exact setup capacity, and derived
  NTT-cache contracts).

## Transcripts in practice

How callers obtain and thread an `AkitaTranscript`, the descriptor preamble that
gets bound first, and what the caller must keep identical between prove and
verify.

**Sources to fold in**

- `crates/akita-transcript/README.md`, `crates/akita-transcript/src/`.
- `crates/akita-pcs/examples/transcript_schedule.rs`.
- Deep dive in [How it works → Transcript and instance binding](../how/transcript.md).
