# The commitment API

The user-facing surface of `AkitaCommitmentScheme`: how to commit, prove, and
verify, plus the setup and transcript objects those calls thread through.

## Commit, prove, verify

The `commit`, `batched_prove`, and `batched_verify` entry points operate on
ordered commitment groups. Every commit call creates exactly one homogeneous
polynomial group and supplies its complete parameter context:

```rust
let output = AkitaCommitmentScheme::<Cfg>::commit(
    &setup,
    &polynomials,
    &stack,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

`GroupContext::scheduler_without_precommitted_groups()` selects the *scalar row*, the
generated row for a group that has no precommitted groups. The resulting committed
group may be opened alone or retained as a precommitted group for a later grouped
opening; both uses have exactly the same parameters.
`GroupContext::scheduler_with_precommitted_groups(&prior)` selects the exact *grouped
row*, the generated row keyed on that ordered prefix.

All polynomials inside one group must have the same `num_vars`. A commit call
rejects a mixed-arity bundle rather than padding it to the widest polynomial.
Polynomials of different arities belong in separate groups, and separate groups
may have different arities.

For a grouped lifecycle, construct one ordered `PrecommittedGroupProfiles` and borrow
it for the final commit:

```rust
let prior = PrecommittedGroupProfiles::from_ordered_groups(prior_commitments.iter())?;
let final_output = AkitaCommitmentScheme::<Cfg>::commit(
    &setup,
    &final_polynomials,
    &stack,
    GroupContext::scheduler_with_precommitted_groups(&prior),
)?;

let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
    opening_claims,
    hints,
    polynomial_groups,
)?;
let selection = prover_data.selection();
```

`PrecommittedGroupProfiles` is non-empty by construction, so both grouped constructors
are infallible; the "no precommitted groups" case has its own spelling. Batch assembly
derives the prefix from the ordered claims, so it is not passed twice.

The constructor derives the exact batch profile and selects its schedule before
stripping commitment profiles from prover-owned opening data. The same
`selection` is placed in `GroupBatchStatement` for verification.

Callers that already own audited root parameters use the same `commit` method
with `GroupContext::explicit_without_precommitted_groups(&params)` or
`GroupContext::explicit_with_precommitted_groups(&prior, &params)`. Explicit mode
does not select a catalog row. It validates the supplied parameters, while
the supplied opening method also selects and authenticates the committed source
encoding. `EvaluationTrace` may use `TensorSubfieldProjection` when the field
and root geometry admit it. `SubringCoefficientPacking` uses
`CanonicalCoefficientTable`. The encoding is commitment identity and cannot be
reinterpreted by a later opening plan.

### Precommitting under a recursive configuration

A `RecursiveCommitmentConfig<Cfg>` catalog ships no row without precommitted groups at
a precommit layout. It carries only the grouped root. Committing a precommitted group
under the recursive configuration therefore fails with `UnsupportedSchedule`.

Build the setup under the recursive configuration, commit each precommitted group
under the base configuration `Cfg`, then commit the final group and prove under
the recursive one. Both configurations share the same setup:

```rust
let setup = AkitaCommitmentScheme::<RecursiveCommitmentConfig<Cfg>>::setup_prover(nv, k)?;

// Precommitted groups use the base configuration, which owns the scalar rows.
let pre = AkitaCommitmentScheme::<Cfg>::commit(
    &setup,
    &prior_polynomials,
    &stack,
    GroupContext::scheduler_without_precommitted_groups(),
)?;

// The grouped root uses the recursive configuration.
let root = AkitaCommitmentScheme::<RecursiveCommitmentConfig<Cfg>>::commit(
    &setup,
    &final_polynomials,
    &stack,
    GroupContext::scheduler_with_precommitted_groups(&prior),
)?;
```

The frozen precommitted descriptor a recursive grouped row carries is exactly
`Cfg::profile_without_precommitted_groups(group)`, which is what makes the split
sound.

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
claims. `SelectedProverOpeningData` privately pairs that material with the one
exact `OpeningScheduleSelection` derived during batch assembly. Akita treats the
concrete polynomial representation as a caller contract: callers must supply
groups with the arity and shape claimed by the public statement. Bad prover
material is a completeness failure, not a verifier soundness input.

Applications that need dense and one-hot polynomials in one opening use one
application-owned enum as `P`. Akita does not add provider registrations or
recursive heterogeneous wrappers. The verifier receives a
`GroupBatchStatement` containing only the exact generated-row selection and
self-describing public claims.

There is no ambient shared point, global polynomial type for the batch, or
coordinate-routing object.

**Sources to fold in**

- `crates/akita-pcs/src/scheme/mod.rs`.
- `crates/akita-prover/src/api/commitment.rs` (`GroupContext`, `CommitOutput`).
- `crates/akita-types/src/proof/scheme.rs` (`CommitmentVerifier`).
- `crates/akita-types/src/opening_claims.rs` (`OpeningClaims`, `OpeningClaimsLayout`).
- `crates/akita-prover/src/types/opening_data.rs` (`ProverOpeningData`,
  `SelectedProverOpeningData`).
- `crates/akita-prover/src/api/prepared_group.rs` (coarse prepared group
  carrier).
- `crates/akita-pcs/tests/akita_fp128_e2e.rs`, `batched_aggregated_e2e.rs`.

## Setup and caching

Akita separates four concepts that used to be mixed together:

1. `AkitaSetupSeed` identifies an infinite, deterministic stream of base-field
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
change `AkitaSetupSeed`, transcript bytes, or proof validity. A proof made with
one materialized capacity can therefore be verified with a larger covering
prefix carrying the same public matrix identity.

Setup-prefix offloading follows the same one-stream rule, but the committed
object is now the actual power-of-two setup prefix. If Stage 3 has active
support `natural_len`, setup storage for the selected prefix covers
`n_prefix = next_power_of_two(natural_len)` source coefficients, and
preprocessing commits `S[0..n_prefix]`. The tail after `natural_len` is real
public setup data; it contributes zero because the setup-index weight is zero
there. The prefix commitment's A and B matrix dimensions are ordinary
planner-owned commitment parameters, independent of the dimensions used by the
producing or consuming fold.

After a concrete schedule is selected, callers may use
`setup_verifier_for_schedule` to keep only the public-matrix prefix that proof
verification can read directly. A producer followed by an incoming setup-prefix
commitment is offloaded, so the verifier does not scan that producer's active
source prefix during Stage 3; it authenticates the carried full-prefix opening
against the setup-prefix commitment in the successor opening. The retained
matrix is the maximum of the terminal A matrix and the active setup fields of
every producer that remains direct. This is a
schedule-derived capacity, not necessarily the length of any stored setup
prefix. The complete setup-prefix commitment registry remains in the verifier
artifact.

This narrowing concerns the per-proof hot path. Loading an externally supplied
setup-prefix registry still needs a provenance policy. A deployment must
recompute the commitments, validate a future derivation-certificate chain, or
authenticate a package that was validated earlier. Structural decoding alone
does not prove that a stored commitment came from the named public stream.

NTT caches are reproducible backend state, not setup data. Preparation starts
with an empty cache. A matrix-consuming kernel derives an exact
`NttPrefixRequirement` from the same row count and active width it passes to
the arithmetic kernel. Negacyclic and cyclic representations have separate
keys and can have different prefix lengths. Equal requirements join by maximum
because the matrices overlap. An initialized larger prefix covers a smaller
request with the same field profile, ring dimension, and transform domain.
Concurrent construction of the same key is single-flight.

The execution plan describes every matrix operation. The selected backend then
decides which operations retain NTT slots. CPU ring switch operations above the
streaming threshold compute transform chunks from the public matrix and do not
prewarm a complete slot. Memory reporting uses the same decision, so its total
matches the slots that prewarming can leave resident.

Prepared state stays warm across proofs by default. A caller may use
`ReleaseRootNttAfterFold` when it owns an isolated root cache and wants to free
it before the recursive suffix. Release removes built shared matrix keys and
deduplicates clusters that share one physical cache owner. Small compression
NTT entries stay resident. Existing readers remain valid through their `Arc`.
Release does not cancel construction already in progress, so a caller that
needs an empty shared matrix cache must prevent new construction during the
release boundary.

A normal lifecycle is:

```text
prepare empty backend state
prewarm retained requirements and skip streamed requirements
run the proof and retain built slots for reuse
optionally release shared matrix state at an exclusive root boundary
reuse the prepared setup; released shared slots rebuild at the next exact extent
```

The terminal verifier keeps its separate exact-negacyclic cache and adds the
i16 tail only when the checked CRT bound requires it. Compression execution
uses its compression-aware cache path and dimension policy. Neither namespace
changes public setup identity or ordinary prover cache sizing.

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
