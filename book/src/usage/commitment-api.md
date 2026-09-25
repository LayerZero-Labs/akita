# Commitment groups and opening claims

Akita commits to polynomials in groups. Each group has a public commitment and
an immutable private handle retaining its exact source. One backend can reuse
these handles across independent and concurrent proofs.

## Commit one group

Construct a backend with the setup and trusted catalog, then transfer the
polynomials into its source storage:

```rust
let backend = std::sync::Arc::new(CpuBackend::<Config>::new(
    setup.expanded.clone(),
    scheme.schedules(),
)?);
let source = backend.import_source(polynomials)?;
let CommitOutput {
    committed_group,
    private_handle,
} = backend.commit(
    &source,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

All polynomials in a source group must have the same number of variables and
one supported source representation. Akita rejects a mixed group instead of
padding smaller tables. Put polynomials with another size or representation in
another group.

`committed_group` carries the public commitment and its frozen profile.
`private_handle` retains the exact source, parameters, and private commitment
material. It remains usable after the application drops `source`. Clone the
handle when opening the same commitment in another proof.

## State one group's opening claims

Every public group claim contains one complete point, one claimed value for
each polynomial, and the group commitment:

```rust
let group = PolynomialGroupClaims::new(
    opening_point,
    evaluations,
    committed_group.clone(),
)?;
let claims = OpeningClaims::from_groups(vec![group])?;
let prover_data = SelectedProverOpeningData::from_committed_claims::<Config>(
    claims,
    vec![private_handle.clone()],
    scheme.schedules(),
)?;
let selection = prover_data.selection();
```

If the group contains four polynomials, `evaluations` contains four values in
the same order. The point has one coordinate for each polynomial variable.
The constructor selects one exact catalog row. The backend checks that each
handle belongs to it and matches the supplied public commitment and profile.
The proof checks the claimed evaluations against the retained source.

The returned selection is public and travels with the verifier statement.
Opening accepts only the handles and public claims; the application cannot
substitute a different polynomial table at proof time.

## Open several groups in one proof

Separate groups may have different numbers of variables and different points.
Akita keeps each point with its own group:

```rust
let claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(point_a, values_a, commitment_a.clone())?,
    PolynomialGroupClaims::new(point_b, values_b, commitment_b.clone())?,
    PolynomialGroupClaims::new(point_c, values_c, commitment_c.clone())?,
])?;
```

The final group is the last item. Every earlier item is a precommitted group.
Build the commitment handle vector in exactly the same order. Each group binds
its variable count, polynomial count, frozen profile, commitment, point, and
claimed evaluations.

Before committing the final group, derive the ordered profiles of the earlier
commitments:

```rust
let prior = PrecommittedGroupProfiles::from_ordered_groups(
    prior_commitments.iter(),
)?;
let final_source = backend.import_source(final_polynomials)?;
let final_output = backend.commit(
    &final_source,
    GroupContext::scheduler_with_precommitted_groups(&prior),
)?;
```

The grouped context selects a trusted catalog row keyed by the complete ordered
prefix and final group. Earlier independent commitments remain reusable in
later batches. `PrecommittedGroupProfiles` is nonempty by construction; the
independent case uses its own constructor.

## Recursive grouped openings

A `RecursiveCommitmentConfig<BaseConfig>` uses setup offloading for supported
large verifier workloads. Construct the backend with the recursive catalog.
Earlier groups use the reviewed independent commitment profile from the base
catalog, supplied through `GroupContext::explicit(&base_profile)`. The final
group uses `GroupContext::scheduler_with_precommitted_groups(&prior)`.

This keeps one backend owner for the entire proof and preserves the independent
public profiles of earlier groups. The proof's recursive catalog checks the
complete ordered batch.

## Dense and one hot groups in one batch

Import dense and one hot polynomials as separate homogeneous source groups.
Their commitments have the same opaque handle type, so their handles can be
placed in one ordered opening batch. Concrete CPU source operations remain
inside the backend, and one hot data retains its compact representation.

A handle from another backend is rejected during admission. When transferring
an existing commitment deliberately, use `backend.import_commitment(&handle)`.
The receiving backend checks the source and commitment material against its
own setup before issuing a fresh handle owned by that backend.

## Groups from another schedule family

Admission reads the backend's own configuration. A `CpuBackend<Cfg>` accepts
only sources that fit `Cfg`'s declared source class and coefficient interval,
because its catalog prices the honest response for exactly those sources. A
grouped opening can still contain a precommitted group from a different family
over the same field and extension, for example a dense group inside a one hot
batch, or a group with a wider coefficient bound. The planner prices such a
group under its producer's contract.

Commit that group on the opening backend under its producer family:

```rust
let source = backend.import_source(wide_polynomials)?;
let output = backend.commit_in_family(
    wide_scheme.schedules(),
    &source,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

`commit_in_family` selects the profile from the family's catalog and admits the
source under that family's contract. It computes the commitment once, with the
opening backend's setup and caches, and returns a handle owned by the opening
backend. The result is the handle that committing on a `CpuBackend<FamilyCfg>`
and importing it would produce. `backend.commit(..)` is
`backend.commit_in_family(own_catalog, ..)`.

Use `import_commitment` for a commitment that another backend produced, for
example one reused from an earlier proof. Its recomputation checks the foreign
setup against the receiving one.

## Explicit commitment parameters

Normal applications use trusted catalog selection. A caller that already owns
a reviewed commit profile may use `GroupContext::explicit(&profile)`.
Explicit mode commits the supplied `GroupCommitPhaseParams`. The opening
schedule validates its compatibility when the commitment is consumed. Use
explicit mode when the application owns the reviewed parameter distribution.

## What the verifier receives

The verifier rebuilds the same ordered claims with borrowed commitments and
joins them to the public schedule selection:

```rust
let statement = GroupBatchStatement::new(
    selection,
    OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(point, values, &committed_group)?,
    ])?,
)?;
```

The statement contains no source tables or private handles. The
[verifier only guide](./verifier-only.md) continues from this object. The
[proof artifacts guide](./proof-artifacts.md) explains how to carry it across a
process boundary.
