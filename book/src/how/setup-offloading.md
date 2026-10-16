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
offloaded mode, Stage 2 accepts a claimed value, and Stage 3 prepares the same
plan again from the Stage 2 challenge point. The prover and verifier then run a degree two sumcheck over two
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

Every configuration uses the same offloading-enabled planner. At each
nonterminal fold it retains a direct successor. At producer levels zero and
one it also compares an offloaded successor:

```text
Direct successor:    [folded witness]
Offloaded successor: [setup prefix, folded witness]
```

The offloaded successor must support opening the exact setup prefix alongside
the witness, contract the combined witness by at least a factor of three, and
reduce the power-of-two capacity of the first setup scan still evaluated
directly. The suffix must contain at least two folds. Ring-relation, source,
security, and prefix-geometry checks must also pass.

The complete schedule is ordered lexicographically by:

1. Padded total setup-envelope capacity.
2. First directly evaluated setup capacity.
3. Exact modeled proof-and-work score.
4. Modeled proof bytes, including every Stage 3 proof.
5. First-direct output witness length.
6. Canonical schedule descriptor.

The exact score is `2^18 * proof_bytes + witness_work + direct_setup_scan_work`.
A direct scan of L setup field elements costs `2L + 64 * ceil(L / d)`, where d
is that fold's common-base setup ring dimension. This is an offline cost model, not a
measurement of verifier latency. Padding lets the planner trade raw setup
size within a power-of-two bucket for a smaller remaining direct scan.

The production search domain is `RootAndFirstChildV1`: **zero, one, or two setup
offloads**, with two a maximum. Witness reduction can still contain more than
two folds. Zero-offload schedules remain eligible and commonly win for small
inputs, where carrying the prefix and proving Stage 3 adds overhead. There is
no fixed root-size threshold and no forced offload at any size. Each selected
consumer-owned setup-prefix group records one offloaded predecessor edge.

A zero-offload result uses the same direct evaluation mechanism as before,
but its layout can differ from the former direct-only planner: the surviving
objective prioritizes the total setup-envelope bucket. For example, the
unchunked fp128 one-hot nv14 row changes A dimension from 128 to 256 and outer
slicing from one to two while retaining zero offloads. Setup, commitments,
proof bytes, and grinding-plan fixtures must be regenerated together.

Planning happens offline. The verifier loads and audits the exact trusted row;
it does not run this search or accept a proof-selected schedule.

## Observed counts and non-monotone cases

The original baseline survey, before the packing eligibility changes in
PR #245, covered 584 inputs spanning fp32/fp64/fp128, dense/one-hot sources,
and ten witness chunk layouts. It produced 561 admitted schedules and 23 search
timeouts. Sixteen results came from existing catalogs. All 60 combinations had
both zero and positive observations; 46 had a two-offload observation.

For one unchunked polynomial with 2^nv entries, the successful sampled points
were as follows. Ranges include every integer; separated points leave gaps.

| Source | Zero offloads: nv | One offload: nv | Two offloads: nv |
| --- | --- | --- | --- |
| fp32 dense | 16–22 | 23–32, 40 | 44, 48 |
| fp32 one-hot | 16–24 | 25–32, 39–40 | 44, 48 |
| fp64 dense | 16–21 | 22–32, 36 | 38 |
| fp64 one-hot | 16–24 | 25–32, 39 | 40, 48 |
| fp128 dense | 16–24 | 25–32 | 36, 40 |
| fp128 one-hot | 16–24, 26, 28 | 25, 27, 29–32, 36 | 40, 42, 48 |

The first five series are monotone on their successful sampled points. The
fp128 one-hot scalar series is non-monotone: nv 24 through 29 selects
`0, 1, 0, 1, 0, 1` under one policy digest. Discrete ring dimensions, splits,
padding, and suffix choices change the winning layout; the survey did not
attribute every reversal to a specific losing candidate.

Witness partitioning changes the boundaries. For fp128 one-hot W8R2, every
integer nv=16…48 was checked: zero at 16–28, one at 29–34, two at 35–48.
W means witness chunk count; it is distinct from the one-hot density K=256.

These are historical schedule choices, not current catalog guarantees, universal
thresholds, or verifier benchmarks. fp64 dense timed out at nv40/48 and fp128 dense at nv48; a timeout
establishes no count or infeasibility. Unmeasured gaps and larger sizes are not
extrapolated. Modeled proof bytes and planner wall time do not measure proof
size or verifier latency in an end-to-end run.

## Configuration and catalog loading

Use the ordinary field/source preset, including multi-chunk presets when
needed, and load its matching `TrustedScheduleCatalog<Cfg>`. There is one
catalog family per preset. Small and large rows share the same policy; the
presence of setup-prefix groups determines each row's offload count.

Each artifact row fixes the native A, B, and D dimensions for its folds.
Offloading rows can use different dimensions for these matrices and can change
dimensions between folds. A ring dimension counts field coefficients in one
ring element; it is not a matrix row count.

For example, the `fp128_onehot.aks` row with two earlier singleton nv16 groups
and a final group of two nv34 polynomials opens a setup-prefix group in its
first recursive fold with

$$
d_A=512,\qquad d_B=128,\qquad d_D=128.
$$

The A and B dimensions belong to that group. The consuming fold owns the shared
D dimension. The [ring-dimension contract](./architecture.md#ring-dimension-ownership)
explains these roles. The matching catalog must contain an admitted row for the
requested workload and dimensions.

Regenerate the catalogs with:

```bash
scripts/generate-schedule-artifacts.sh --row-progress
```

Earlier independently committed groups retain their frozen public profiles.
The final grouped opening uses the same configuration and resolves the complete
ordered batch, including those earlier profiles. Grouped results can differ
from scalar results with the same final-group arity because their complete
input shapes differ.

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
| Derive the single catalog policy | `crates/akita-config/src/lib.rs` |
| Search direct and offloaded suffixes | `crates/akita-planner/src/schedule_params/suffix_dp/` |
| Build recursive candidates and prefix requirements | `crates/akita-planner/src/schedule_params/candidate/recursive.rs` and `candidate/setup_prefix.rs` |
| Define prefix identities and proof data | `crates/akita-params/src/layout/setup_prefix_slots.rs` and `crates/akita-types/src/proof/setup_prefix.rs` |
| Build the shared setup contribution plan | `crates/akita-types/src/setup_contribution/` |
| Evaluate the setup contribution at the Stage 2 point | `crates/akita-verifier/src/setup_contribution/` |
| Materialize required prefix commitments | `crates/akita-setup/src/recursive_prefixes.rs` |
| Prove the setup product | `crates/akita-cpu-backend/src/opaque/sumcheck/stage3/` |
| Verify Stage 3 | `crates/akita-verifier/src/stages/stage3.rs` |
| Enforce the recursive fold handoff | `crates/akita-prover/src/protocol/prove/` and `crates/akita-verifier/src/fold/` |

[Configuration and schedule planning](configuration.md) explains the sole
planner policy, bounded setup-offload search, and trusted catalog admission.
This chapter explains the implemented setup-offloading protocol.
