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

The same check fixes the inner digit count of the prefix commitment. A
commitment first splits each coefficient into `num_digits` digits of
`log_basis` bits. A recursive witness holds small values, so its schedule can
use few digits. Setup coefficients are different: they are uniform field
elements and use every bit of the field. A prefix profile must therefore use

```text
num_digits = ceil(field_bits / log_basis)
```

inner digits. For a 128-bit field and `log_basis = 8`, that is 16 digits. A
smaller count would drop the high bits, and the commitment would bind a
truncated value in place of the setup coefficient. A profile with any other
count is rejected with `InvalidSetup`. The check is
`GroupCommitPhaseParams::validate_setup_prefix_geometry` in
`crates/akita-params/src/schedule/profiles.rs`, and the test
`setup_prefix_uses_full_field_digits_for_a_tight_recursive_consumer` covers it.

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

A recursive application can attach setup claims from inner proofs to one
offloaded fold. Stage 3 then proves those claims together with the fold's own
claim. See
[Stage 3 with an inherited batch](./verifying/setup_contribution.md#stage-3-with-an-inherited-batch).

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

Among feasible recursive schedules, the production policy first minimizes the
power of two capacity that covers all setup objects. It then minimizes the
first setup capacity that the verifier still scans directly. Within those
capacity buckets, it compares the exact proof-and-work score, then estimated
proof bytes, including every Stage 3 proof, and the first direct output-witness
length. The canonical descriptor breaks remaining ties.

This order has a name, and so does the order that direct catalogs use. Both
are variants of `SelectionPolicyId` in `crates/akita-schedules/src/runtime.rs`.
The planner compares candidates field by field, and a later field matters only
when all earlier fields are equal.

| Policy | Used by | Compared in order |
| --- | --- | --- |
| `MinFirstDirectSetupThenExactProofAndWorkV5` | Direct catalogs | First direct setup capacity, exact proof-and-work score, proof bytes, total setup size, root output witness length, descriptor |
| `MinPaddedSetupEnvelopeThenFirstDirectThenExactProofAndWorkV6` | Recursive setup catalogs | Power-of-two setup envelope, first direct setup capacity, exact proof-and-work score, proof bytes, first direct output witness length, descriptor |

The setup envelope is the total setup size rounded up to a power of two. The
policy identity is bound into the catalog artifact, so a catalog generated
under one policy is not admitted as a catalog of the other.

Two rows of `artifacts/schedule-catalog.tsv` for one dense polynomial with 24
variables show the effect:

| Family | Setup field elements | First direct setup capacity | Proof bytes | Fold levels |
| --- | --- | --- | --- | --- |
| `fp128_dense` | 1,048,576 | 1,048,576 | 76,204 | 7 |
| `fp128_dense_recursive` | 1,048,576 | 524,288 | 80,986 | 8 |

Both rows fit the same $2^{20}$ envelope. The recursive row halves the setup
that the verifier scans directly. It pays 4,782 more proof bytes and one more
fold.

### Searching under a widening setup budget

A search under the V6 policy without a limit examines many candidates whose
setup is far larger than the winner's. The planner avoids most of that work
with a setup budget, which is an upper bound on a schedule's total setup
field elements for one search pass.

1. The first pass uses a budget of one field element.
2. Each pass starts from an empty memo table and rejects every candidate
   whose setup exceeds the budget. It records the smallest setup size that it
   rejected.
3. If the pass selects a schedule, the search stops and returns it.
4. If the pass selects nothing, the next budget is the smallest rejected
   size rounded up to a power of two.
5. If the pass selects nothing and rejected nothing, no schedule exists.
6. If the rounding overflows, a last pass runs without a budget.

The result equals the result of one pass without a budget. The V6 policy
compares the power-of-two setup envelope first, so a candidate over the
budget can only displace candidates that are also over the budget. The
winner of the first pass that finds any schedule is therefore the overall
winner. The starting budget changes only the number of passes.

This search is an implementation choice of the planner and affects only its
running time. It runs only under the V6 policy and only when the caller gave
no explicit setup budget; an explicit budget gets a single pass. The verifier
never runs it. The loop is in `crates/akita-planner/src/planner.rs`, and the
test `padded_setup_budget_passes_match_one_unbounded_pass` checks that the
passes select the same schedule as one unbounded pass.

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

- the fp32, fp64, and fp128 dense configurations;
- the fp128 one hot configuration;
- the fp128 one hot multi chunk configuration with eight chunks and two leading
  distributed levels (`OneHotMultiChunk`); and
- the fp128 one hot multi chunk configurations with two, four, or eight chunks
  and three leading distributed levels (`OneHotMultiChunkW2R3`,
  `OneHotMultiChunkW4R3`, and `OneHotMultiChunkW8R3`).

`RecursiveCommitmentConfig<Cfg>` inherits the base configuration's ring-dimension
policy. Each artifact row fixes the native A, B, and D dimensions for its folds.
Supported offloading rows can use different dimensions for these matrices and
can change dimensions between folds. A ring dimension is the number of field
coefficients in one ring element; it is not a matrix row count.

For example, one row in `fp128_onehot_recursive.aks` has two earlier commitment
groups, each with one 16-variable polynomial. Its final group contains two
34-variable polynomials. The first recursive fold opens a setup-prefix group
with

$$
d_A=512,\qquad d_B=128,\qquad d_D=128.
$$

The A and B dimensions belong to that group. The consuming fold owns the shared
D dimension. The [ring-dimension contract](./architecture.md#ring-dimension-ownership)
explains these roles. The matching recursive catalog must contain an admitted
row for the requested workload and dimensions.

Support depends on supplying the matching recursive family artifact. Other base
configurations have no recursive catalog and are rejected rather than silently
falling back to a direct schedule under the recursive adapter.

The dense artifacts cover singleton polynomials with these variable counts:

| Base configuration | Recursive artifact | Variable counts |
| --- | --- | --- |
| `fp32::Dense` | `fp32_dense_recursive.aks` | 20, 22, 24, 26, 28, 30 |
| `fp64::Dense` | `fp64_dense_recursive.aks` | 21, 23, 25, 27, 29 |
| `fp128::Dense` | `fp128_dense_recursive.aks` | 20, 22, 24, 26, 28 |

Load the matching artifact into
`TrustedScheduleCatalog<RecursiveCommitmentConfig<Cfg>>`. A recursive family
allows offloading; individual rows may still select only direct edges when
that is the planner's preferred feasible schedule.

Regenerate these artifacts with the current planner:

```bash
scripts/generate-schedule-artifacts.sh --row-progress \
  fp32_dense_recursive fp64_dense_recursive fp128_dense_recursive
```

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
- the prefix profile's inner digit count does not cover the full field width;
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
| Define prefix identities and proof data | `crates/akita-params/src/layout/setup_prefix_slots.rs` and `crates/akita-types/src/proof/setup_prefix.rs` |
| Build the shared setup contribution plan | `crates/akita-types/src/setup_contribution/` |
| Evaluate the setup contribution at the Stage 2 point | `crates/akita-verifier/src/setup_contribution/` |
| Materialize required prefix commitments | `crates/akita-setup/src/recursive_prefixes.rs` |
| Prove the setup product | `crates/akita-cpu-backend/src/opaque/sumcheck/stage3/` |
| Verify Stage 3 | `crates/akita-verifier/src/stages/stage3.rs` |
| Enforce the recursive fold handoff | `crates/akita-prover/src/protocol/prove/` and `crates/akita-verifier/src/fold/` |

The live planner contract is
[`specs/setup-offloading-planner.md`](https://github.com/LayerZero-Labs/akita/blob/main/specs/setup-offloading-planner.md).
It records the exact feasibility rules and external-artifact invariants. This
chapter owns the reader explanation of the implemented feature.
