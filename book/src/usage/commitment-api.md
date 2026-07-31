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

Akita separates four concepts that used to be mixed together:

1. `PublicMatrixId` identifies an infinite, deterministic stream of base-field
   elements.
2. `SetupMatrixCapacity` says how many elements of that stream a host has
   materialized.
3. A schedule gives each A, B, and D matrix its own rows, active width, and ring
   dimension. Each matrix is an overlapping view of the same flat prefix.
4. A prepared compute backend owns derived NTT caches for the work actually
   executed on that backend.

The public stream has no ring dimension. `Shake256PagedV1` derives it in
independently addressable pages of 4096 field elements. The page length belongs
to that versioned derivation policy and is absorbed into each page's SHAKE256
input together with the policy tag, public seed, field modulus, and page index.
`F::random` uses exact rejection sampling, so every accepted coefficient is
uniform in the field. Changing the page length, sampling rule, or absorbed
fields requires a new `PublicMatrixDerivation` variant.

For a concrete matrix use, the required public capacity is

```text
rows * active_width * ring_dimension
```

All matrix uses start at flat index zero, so a schedule's capacity is the
maximum of those field-element counts, not their sum. A stored prefix may be
larger than a schedule needs. That is only local provisioning: it does not
change `PublicMatrixId`, transcript bytes, or proof validity. A proof made with
one materialized capacity can therefore be verified with a larger covering
prefix carrying the same public matrix identity.

Setup-prefix offloading follows the same rule. If Stage 3 needs
`natural_len` coefficients, setup storage covers exactly that many source
coefficients. The committed setup polynomial still has the power-of-two length
`n_prefix`; preprocessing constructs
`S[0..natural_len] || 0^(n_prefix-natural_len)` explicitly. Later random stream
coefficients are never used as padding. The prefix commitment's A and B matrix
dimensions are ordinary planner-owned commitment parameters, independent of
the dimensions used by the producing or consuming fold.

NTT caches are reproducible backend state, not setup data. Preparation starts
with an empty cache. A matrix-consuming kernel derives an exact
`NttPrefixRequirement` from the same row count and active width it passes to
the arithmetic kernel. Negacyclic and cyclic representations have separate
keys and can have different prefix lengths. Equal requirements join by maximum
because the matrices overlap. An initialized larger prefix covers a smaller
request with the same field profile, ring dimension, and transform domain.
Concurrent construction of the same key is single-flight.

The terminal verifier keeps its separate exact-negacyclic cache and adds the
i16 tail only when the checked CRT bound requires it. Compression diagnostics
also use a separate cache and dimension policy. Neither namespace changes
public setup identity or ordinary prover cache sizing.

Optional disk persistence serializes the public matrix identity, provisioning
limits, flat field-element count, matrix coefficients, and setup-prefix
artifacts. It does not serialize NTT caches. Cache filenames use a versioned
flat-matrix namespace, so setup files from the former generation-dimension
format are not accepted as current entries.

**Sources to fold in**

- `crates/akita-setup/src/lib.rs`.
- Paper §3.9 `sec:akita-setup` (packed shared setup), `Setup` in §3.8 `sec:akita-full-pcs`.
- `specs/flat-public-matrix-and-exact-ntt-cache.md` (detailed derivation,
  capacity, setup-prefix, and NTT-cache contracts).

## Transcripts in practice

How callers obtain and thread an `AkitaTranscript`, the descriptor preamble that
gets bound first, and what the caller must keep identical between prove and
verify.

**Sources to fold in**

- `crates/akita-transcript/README.md`, `crates/akita-transcript/src/`.
- `crates/akita-pcs/examples/transcript_schedule.rs`.
- Deep dive in [How it works → Transcript and instance binding](../how/transcript.md).
