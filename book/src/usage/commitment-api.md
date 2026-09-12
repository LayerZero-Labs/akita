# Commitment groups and opening claims

Akita commits to polynomials in groups. A group is the unit of commitment,
private prover state, and public opening claims. This model gives applications
one interface for a single polynomial, a batch opened at one point, or several
earlier commitments combined into a final proof.

## Commit one group

Every call to `commit` supplies the setup, a slice of polynomials, a commitment
executor, and the complete context for that group.

```rust
let output = scheme.commit(
    &setup,
    &polynomials,
    stack.commitment(),
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

All polynomials in `polynomials` must have the same number of variables. Akita
rejects a mixed group instead of padding smaller tables. Put polynomials with
another size in another group.

The result has two parts:

```rust
let CommitOutput {
    committed_group,
    prover_state,
} = output;
```

`committed_group` is public and self describing. It carries the commitment and
the frozen profile that produced it. `prover_state` is private prover data.
Store it beside the exact polynomial group and preserve their order.

## Independent commitments stay reusable

`GroupContext::scheduler_without_precommitted_groups()` selects the trusted
catalog row for an independent group. The resulting commitment can be opened by itself
or used as an earlier group in a later batched proof. The commitment does not
need to predict that later proof.

This property is important for long lived application state. A host can commit
to a table once, keep its commitment and prover state, and decide later which other
groups to open beside it.

## State one group's opening claims

Every public group claim contains one complete point, one claimed value for
each polynomial, and the group commitment.

```rust
let group = PolynomialGroupClaims::new(
    opening_point,
    evaluations,
    committed_group.clone(),
)?;
let claims = OpeningClaims::from_groups(vec![group])?;
```

If the group contains four polynomials, `evaluations` contains four values in
the same order. The point has one coordinate for each polynomial variable.

The prover then pairs the public claims with private material:

```rust
let polynomial_refs: Vec<&DensePoly<F>> = polynomials.iter().collect();
let prover_data = SelectedProverOpeningData::from_committed_claims::<Config>(
    claims,
    vec![prover_state],
    vec![&polynomial_refs],
)?;
let selection = prover_data.selection();
```

This constructor checks the complete batch shape and selects one exact catalog
row. The returned selection is public and must travel with the verifier
statement.

## Open several groups in one proof

Separate groups may have different numbers of variables and different points.
Akita keeps each point with its own group.

```rust
let claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(point_a, values_a, commitment_a.clone())?,
    PolynomialGroupClaims::new(point_b, values_b, commitment_b.clone())?,
    PolynomialGroupClaims::new(point_c, values_c, commitment_c.clone())?,
])?;
```

The final group is the last item. Every earlier item is a precommitted group.
This order is visible to the protocol. Build the prover-state vector and polynomial
group vector in exactly the same order.

Akita binds the following facts for each group:

- The number of variables.
- The number of polynomials.
- The commitment profile and commitment value.
- The opening point.
- The claimed values.

There is no shared global point. There is also no hidden routing object that
reorders coordinates between groups.

## Commit the final group beside earlier groups

Before committing the final group, derive the ordered profiles of the earlier
commitments.

```rust
let prior = PrecommittedGroupProfiles::from_ordered_groups(
    prior_commitments.iter(),
)?;

let final_output = scheme.commit(
    &setup,
    &final_polynomials,
    stack.commitment(),
    GroupContext::scheduler_with_precommitted_groups(&prior),
)?;
```

The grouped context selects a trusted catalog row keyed by the complete ordered
prefix and final group. It also checks that every earlier commitment profile is
the independent profile that the corresponding base configuration produces.

`PrecommittedGroupProfiles` is nonempty by construction. The independent case
has its own explicit constructor, so an empty list cannot accidentally select a
grouped path.

## Recursive grouped openings

A `RecursiveCommitmentConfig<BaseConfig>` uses setup offloading for supported
large verifier workloads. Earlier groups still commit under `BaseConfig`. The
final group and opening proof use the recursive configuration.

```rust
let recursive_catalog = TrustedScheduleCatalog::<
    RecursiveCommitmentConfig<BaseConfig>,
>::from_artifact_bytes(&recursive_artifact_bytes)?;
let recursive_scheme = AkitaCommitmentScheme::<
    RecursiveCommitmentConfig<BaseConfig>,
>::new(recursive_catalog);
let setup = recursive_scheme.setup_prover(max_num_vars, max_total_batched_polys)?;

let earlier = base_scheme.commit(
    &setup,
    &earlier_polynomials,
    stack.commitment(),
    GroupContext::scheduler_without_precommitted_groups(),
)?;

let prior = PrecommittedGroupProfiles::from_ordered_groups([
    &earlier.committed_group,
])?;
let final_group = AkitaCommitmentScheme::<
    RecursiveCommitmentConfig<BaseConfig>,
>::commit(
    &setup,
    &final_polynomials,
    stack.commitment(),
    GroupContext::scheduler_with_precommitted_groups(&prior),
)?;
```

Both configurations use the same public setup. The split keeps earlier
commitments reusable and applies setup offloading to the complete grouped
opening where it is useful.

## Dense and one hot groups in one batch

Every commitment group is homogeneous: all of its polynomials use one concrete
source type and one selected commitment representation. Commit dense and one
hot data as separate groups. They can still be opened together in one proof.

Build each `ErasedPreparedProverGroup` from a nonempty slice of references to
one concrete type, then pass the ordered erased groups to
`SelectedProverOpeningData::from_prepared_groups`. Erasure occurs once per
group. Akita continues to call the concrete dense or one hot batch kernels, and
one hot data stays compact throughout commitment and opening. Claims, retained
states, and erased groups must use the same group order.

Akita rejects a commitment request unless all members share one external
operation or one exact standard representation supported by the selected
backend. It does not split a mixed group automatically because that would
change commitment identities and opening claims.

## Explicit commitment parameters

Normal applications should use trusted catalog selection. A caller that
already owns a reviewed commit profile may use
`GroupContext::explicit(&profile)`.

Explicit mode commits the supplied `GroupCommitPhaseParams` but does not select
a catalog row. The profile fully determines the A/B commitment; any opening
schedule the caller intends to use later — including a grouped root over
precommitted groups — is validated where the opening consumes it, not at commit
time. Use explicit mode only when the application has its own reviewed parameter
distribution process.

## What the verifier receives

The verifier rebuilds the same ordered claims with borrowed commitments and
joins them to the public schedule selection.

```rust
let statement = GroupBatchStatement::new(
    selection,
    OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(point, values, &committed_group)?,
    ])?,
)?;
```

The statement contains no polynomial tables or private hints. The
[verifier only guide](./verifier-only.md) continues from this object. The
[proof artifacts guide](./proof-artifacts.md) explains how to carry it across a
process boundary.
