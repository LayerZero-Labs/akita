# Commitment groups and opening claims

Akita commits to polynomials in groups. Each group has a public commitment and
an immutable private handle retaining its exact source. One backend can reuse
these handles across independent and concurrent proofs.

## Commit one group

Construct a backend from the public setup, then transfer the polynomials into
its source storage. Each commit names the trusted catalog that selects the
group's profile:

```rust
let backend = std::sync::Arc::new(CpuBackend::new(setup.expanded.clone())?);
let source = backend.import_source(polynomials)?;
let CommitOutput {
    committed_group,
    private_handle,
} = backend.commit(
    scheme.schedules(),
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
    scheme.schedules(),
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
large verifier workloads. Commit the final group under the recursive catalog.
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

The backend stores no catalog, so one backend commits every group. Pass each
group's producer catalog to its own `commit`, then commit the final group and
open the batch under the proving catalog:

```rust
let dense_output = backend.commit(
    dense_scheme.schedules(),
    &backend.import_source(dense_polynomials)?,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
let prior = PrecommittedGroupProfiles::from_profiles(vec![
    dense_output.committed_group.profile.clone(),
])?;
let final_output = backend.commit(
    onehot_scheme.schedules(),
    &backend.import_source(onehot_polynomials)?,
    GroupContext::scheduler_with_precommitted_groups(&prior),
)?;
```

The catalog passed to `commit` is the producer contract for that group: it
fixes the commit profile and the source class and coefficient bound that
admission enforces. The handle records that contract, and
`CommitmentHandleMetadata::producer_contract` returns it. Opening data built
with `SelectedProverOpeningData::from_committed_claims::<Cfg>` rejects a final
group whose recorded contract differs from `Cfg::committed_source_contract()`,
since the proving catalog plans its final group under that contract.

Schedule rows carry no producer identity for precommitted groups, so opening
does not check them. The application must commit each precommitted group under
the configuration whose contract the proving catalog's row assumes for that
group, and it can compare each handle's `producer_contract()` with the contract
it planned. A mismatch cannot make a false claim verify, since the verifier
enforces the row's frozen caps. The proof's completeness and grinding budget,
however, no longer follow from the planner's model, so an honest proof can
fail. Proof bytes do not bind the producer contract.

The setup must cover every family that commits on the backend. When the
proving catalog's rows already include each precommitted profile, its own
setup suffices. Otherwise build the setup from combined requirements, as in
[Share one setup across families](./setup-runtime.md#share-one-setup-across-families).

A handle from another backend is rejected during admission. When transferring
an existing commitment deliberately, use `backend.import_commitment(&handle)`.
The receiving backend checks the source and commitment material against its
own setup before issuing a fresh handle owned by that backend.

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
