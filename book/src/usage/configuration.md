# Choosing a configuration

An Akita configuration defines the field, committed data shape, security
policy, and expected schedule-artifact family. The application chooses the
family and supplies its approved artifact. Akita selects the exact row from the
validated catalog.

This split is one of Akita's strengths. Applications make decisions about their
own data. They do not tune ring dimensions or search for cryptographic
parameters at runtime.

## Start with the host field

Each production family has a base field for commitments and an opening field
for evaluation points and claimed values.

| Family | Commitment field | Opening field | Common use |
| --- | --- | --- | --- |
| `fp32` | A 32 bit prime field | A degree 4 extension | Hosts built around small prime fields |
| `fp64` | A 64 bit prime field | A degree 2 extension | Hosts that want 64 bit base arithmetic |
| `fp128` | A 128 bit prime field | The same field | Hosts that already work over the full challenge field |

All three families target 128-bit quantum security under Akita's concrete
lattice attack model. The smaller base fields use extension fields for opening
points and claimed values. The host should choose the family whose opening
field matches the values it needs to prove. The [rings and
fields](../foundations/rings-and-fields.md) chapter explains the construction.

## Match the committed data

The next choice is the representation of the table.

| Representation | Use it when | Main benefit |
| --- | --- | --- |
| `Dense` | Entries may be arbitrary field elements | General purpose commitment path |
| `OneHot` | Each fixed size chunk contains one selected position | Preserves the compact source and skips work for zero entries |
| `DenseBounded` | Every centered coefficient fits the declared signed bound | Uses fewer commitment digits than full width dense data |

The direct fp128 types are `fp128::Dense` and `fp128::OneHot`.
`fp128::DenseBounded` accepts every `u64` value and the corresponding negative
range. It uses the `fp128_dense_bounded` artifact family.

The bounded configuration enforces its range during commitment. This is useful
because the tighter range becomes part of the commitment profile and the
security calculation. Use ordinary `Dense` when the application cannot prove
the bound before calling Akita.

One hot is more than a label on a dense table. The prover stores selected
indices and uses kernels written for that representation. Applications should
keep structured data in this form instead of expanding it into a dense vector.

## Describe the opening batch

A schedule is selected for the complete ordered batch. The selection depends
on the following public shape:

- The number of commitment groups.
- The number of variables in each group.
- The number of polynomials in each group.
- The commitment profile carried by each group.
- The opening method fixed by the configuration and commitment profiles.

Polynomials in one group have the same number of variables and share one
opening point. Polynomials with another arity or point belong in another group.
The order of those groups is part of the proof statement.

The common direct case uses one group and
`GroupContext::scheduler_without_precommitted_groups()`. A grouped opening
commits earlier groups first, then commits the final group with
`GroupContext::scheduler_with_precommitted_groups(&prior)`.

## Direct and recursively offloaded setup checks

Every Akita proof reduces a large opening through a sequence of folds. The word
recursive in `RecursiveCommitmentConfig<Cfg>` refers to an additional setup
checking path. It lets the proof carry the large public setup contribution into
a later fold instead of asking the verifier to scan it directly.

Use the direct configuration first. It has the simplest setup and is the right
choice for ordinary local verification. Use a shipped recursive configuration
when the verifier environment makes direct setup evaluation expensive. The
[setup offloading](../how/setup-offloading.md) chapter explains the complete
protocol and its supported families.

Precommitted groups in a recursive grouped opening still use the base
configuration. The final group and the proof use
`RecursiveCommitmentConfig<Cfg>`. This preserves the commitment identity of the
earlier groups while selecting the recursive adapter for the complete opening.

## Partition large prover work

The fp128 configuration module also provides multi chunk companions for
selected production workloads. These configurations divide the witness
relation into exact chunks during the first folds. They let the prover spread
large work across a controlled partition without changing the public claim.

Multi chunk configurations are deployment choices for measured large
workloads. Begin with `Dense` or `OneHot`. Move to a multi chunk companion after
profiling the complete host application and confirming that its trusted
external catalog contains the required row.

## Let the trusted catalog choose dimensions

Akita may use different ring dimensions for the commitment matrix, later
relations, and different fold levels. The offline planner searches these
choices under the same security rules that the verifier enforces. The selected
row records the result.

Application code should not choose a fixed ring dimension. It should describe
the field, data representation, and opening batch. This lets Akita improve
offline schedules without changing the high level integration model.

Normal proving and verification never run the planner. They resolve a row from
the catalog supplied to the scheme instance. `AkitaError::UnsupportedSchedule`
means that catalog does not contain the exact request. The
[troubleshooting guide](./troubleshooting.md#the-requested-schedule-is-unsupported)
lists the information needed to diagnose that result.

## Practical choices

Use these starting points:

| Application data | Starting configuration |
| --- | --- |
| Arbitrary values in the fp128 field | `fp128::Dense` |
| One selected entry in each fixed size chunk | `fp128::OneHot` |
| Values known to fit the full `u64` range | `fp128::DenseBounded` |
| A host using a small prime field | The matching `fp32` or `fp64` family |
| A verifier that must avoid a large direct setup scan | A supported `RecursiveCommitmentConfig<Cfg>` |

The [first proof](./quickstart.md) uses `fp128::Dense`. The
[integration guide](./integration.md) develops grouped openings and prepared
state. The [profiling guide](./profiling.md) shows how to measure the selected
configuration at the host's real workload size.
