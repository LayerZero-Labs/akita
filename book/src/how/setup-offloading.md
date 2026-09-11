# Setup offloading

Setup offloading reduces the amount of public setup data that the verifier must
scan during an opening. Akita prepares commitments to selected setup prefixes
in advance. During the proof, the prover shows that a claimed setup contribution
comes from one of those prefixes. A later fold authenticates the resulting
opening against the prepared commitment.

This feature is part of the current recursive setup path. It is selected by an
expanded schedule row loaded from a trusted external `.aks` artifact. Once that
row is resolved, a caller cannot add or remove an offloaded edge.

## The setup work inside a fold

Each nonterminal fold checks relations that use the public A, B, and D matrices.
At the verifier's sampled point, those matrices contribute one weighted sum of
setup coefficients. Akita calls this value the setup contribution.

The verifier can obtain the value in two ways.

| Mode | What happens during verification | Extra proof work |
| --- | --- | --- |
| Direct | The verifier scans the active public setup prefix and computes the weighted sum. | None. The value is checked inside Stage 2. |
| Offloaded | The proof supplies the setup contribution. Stage 3 proves that it is the correct weighted sum, then leaves one opening claim for a committed setup prefix. | One setup product sumcheck and one carried setup prefix opening. |

Direct mode keeps the proof smaller, but its scan grows with the setup used by
the fold. Offloaded mode replaces that scan with preprocessing and proof work.
It is useful only when the later recursive opening costs less than the direct
scan it removes.

## What a setup prefix is

The public setup is one flat vector of field elements. A fold reads an initial
part of that vector as its A, B, and D matrices. The active part may not have a
power of two length, while a multilinear commitment needs a power of two
domain.

Akita therefore records two lengths for each prepared prefix:

- `natural_len` is the number of setup coefficients that can receive nonzero
  weight in the fold;
- `n_prefix` is the complete power of two prefix covered by the commitment.

The setup builder commits to all `n_prefix` coefficients. Coefficients after
`natural_len` are real setup values, not zero padding. The Stage 3 weight is
zero outside `natural_len`, so those extra coefficients do not change the
claimed setup contribution.

The commitment block geometry covers exactly `n_prefix / d_A` setup rings.
Artifact admission, serialized slot registries, and prover setup all
reject a prefix profile that covers only the natural support or leaves a
partial final block.

A `SetupPrefixSlotId` binds the active length and the commitment profile. That
profile fixes the commitment domain, inner and outer matrices, ring dimensions,
and decomposition parameters. The scheduled incoming prefix also fixes how its
opening will be proved. Prover and verifier registries must contain the exact
set of slots required by the admitted schedule rows.

## The recursive handoff

An offloaded fold produces two independent claims:

1. Stage 2 produces an opening claim for the next folded witness.
2. Stage 3 produces an opening claim for the selected setup prefix.

The successor fold receives both claims as ordinary commitment groups. The
setup prefix is a precommitted group. The folded witness is the new witness
group. Each group keeps its own opening point, and the grouped opening protocol
authenticates both against their commitments.

```text
prepared setup prefix S
          |
          v
fold i: Stage 2 checks the witness relation
        Stage 3 proves the claimed setup contribution
          |
          +--> opening of the next witness
          +--> opening of S at the Stage 3 point
                         |
                         v
fold i + 1: authenticate both openings in one grouped fold
```

The witness claim and setup claim never become one point or one value. Stage 3
does not change the witness point produced by Stage 2. A fold may consume a
setup opening from its predecessor while producing another setup opening for
its successor.

The terminal fold cannot offload its setup contribution. It has no successor
that could authenticate a new setup opening, so it always performs its final
checks directly.

## What Stage 3 proves

Stage 2 prepares one checked `SetupContributionPlan`. The plan describes the A,
B, and D setup rows, their witness weights, and the powers of the ring challenge
used at the sampled relation point.

In direct mode, the verifier uses this plan to scan the public setup. In
offloaded mode, Stage 2 accepts a claimed value and saves the same plan for
Stage 3. The prover and verifier then run a degree two sumcheck over two
coordinates:

- one coordinate selects a coefficient inside a setup ring; and
- one coordinate selects the setup ring within the flat prefix.

At the final point, the verifier checks the product of three evaluations:

```text
setup prefix value
    × setup index weight
    × power of the ring challenge
```

This is a setup only proof. It does not include the next witness claim. The
detailed polynomial identity is in [Sumcheck stages](./proving/sumcheck-stages.md#stage-3-recursive-setup-contribution).
The exact mixed dimension evaluation rules are in
[Setup contribution and Stage 3](./verifying/setup_contribution.md).

## How the planner chooses offloading

The ordinary configuration catalogs use direct setup evaluation. A supported
`RecursiveCommitmentConfig<Cfg>` expects a separate external catalog in which
the planner may use setup offloading.

For each nonterminal edge, the planner retains a direct successor. At producer
levels admitted by the catalog's recursive setup search policy, it also
compares an offloaded successor:

```text
Direct successor:    [folded witness]
Offloaded successor: [setup prefix, folded witness]
```

The planner keeps an offloaded edge only when all of the following are true:

- setup construction can prepare the exact prefix commitment;
- the successor can open the prefix and witness together;
- the complete folded witness contracts by at least a factor of three; and
- the offloaded suffix reduces the power of two capacity of the first setup
  scan that still runs directly.

Among feasible schedules, the production policy first minimizes that remaining
direct setup capacity. It then compares exact estimated proof bytes, including
every Stage 3 proof. Later tie breaks prefer a smaller total setup envelope and
then a smaller root output witness before the canonical schedule order.

The shipped recursive catalogs consider offloaded edges produced by the root
and its direct child. This `RootAndFirstChildV1` domain is part of the catalog
identity; direct traversal remains available at every level. Exhaustive search
is available for audit workloads. Within the selected domain, the planner may
select no offloaded levels, one level, or several levels. A prefix size does
not force the choice. The successor's `incoming_setup_prefix` field records the
selected edge. There is no second producer-side mode bit that can disagree
with it.

Planner search happens offline. The artifact row records the exact choices,
and the verifier resolves and audits that row. It never searches for a cheaper
schedule while checking a proof.

## Current supported configurations

The recursive catalog is intentionally narrower than the ordinary Akita
catalogs. The current build can expose recursive setup schedules for:

- the fp128 one hot configuration; and
- the fp128 one hot multi chunk configuration with eight chunks and two leading
  distributed levels.

Setup offloading currently uses the supported uniform $D = 64$ shape. Other
setup ring dimensions do not expose a recursive offloading catalog.

Support depends on supplying the matching recursive family artifact. Other base
configurations have no recursive catalog and are rejected rather than silently
falling back to a direct schedule under the recursive adapter.

When a grouped proof includes commitments formed earlier, those commitments are
created under the base configuration. The later grouped opening selects
`RecursiveCommitmentConfig<Cfg>`. This lets commitments made at different times
enter the same opening without guessing a future setup offloading schedule. A
scalar recursive profile can select the recursive adapter directly because it
has no earlier commitment groups to preserve.

## What the verifier rejects

The schedule, setup, proof, and carried claims must agree exactly. The verifier
rejects the proof when any of the following occurs:

- a required setup prefix slot is missing or has the wrong identity;
- the proof contains Stage 3 data for a direct edge;
- an offloaded edge omits its Stage 3 proof;
- the committed prefix is shorter than the active setup support;
- the successor has the wrong group order or opening point;
- the setup contribution plan does not match the scheduled A, B, and D
  geometry; or
- any checked size or offset overflows.

The transcript binds the admitted schedule, setup identity, prefix slot, group
layout, and claims before the challenges that depend on them. Malformed input
must return `AkitaError` or `SerializationError`. It must not cause a verifier
panic.

## Where the implementation lives

| Responsibility | Primary implementation |
| --- | --- |
| Enable the recursive catalog | `crates/akita-config/src/recursive_commitment.rs` |
| Search direct and offloaded suffixes | `crates/akita-planner/src/schedule_params/suffix_dp/` |
| Build recursive candidates and prefix requirements | `crates/akita-planner/src/schedule_params/candidate/recursive.rs` and `candidate/setup_prefix.rs` |
| Define prefix identities and proof data | `crates/akita-types/src/proof/setup_prefix.rs` |
| Build the shared setup contribution plan | `crates/akita-types/src/setup_contribution/` |
| Materialize required prefix commitments | `crates/akita-setup/src/recursive_prefixes.rs` |
| Prove the setup product | `crates/akita-prover/src/protocol/sumcheck/akita_stage3/` |
| Verify Stage 3 | `crates/akita-verifier/src/stages/stage3.rs` |
| Enforce the recursive fold handoff | `crates/akita-prover/src/protocol/core/` and `crates/akita-verifier/src/protocol/core/` |

The live planner contract is
[`specs/setup-offloading-planner.md`](../../specs/setup-offloading-planner.md).
It records the exact feasibility rules and external-artifact invariants. This
chapter owns the reader explanation of the implemented feature.
