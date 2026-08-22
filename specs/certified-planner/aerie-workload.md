# Supporting Design: Aerie Falcon v1 Planner Workload

Parent specification:
[Certified Planner Architecture](../certified-planner-architecture.md)

This file is normative support for the parent specification. It inherits the
parent's status, pull request, ownership, acceptance process, and retirement.
It is not an independent live specification.

## Source pin and precedence

This workload is pinned to Aerie commit
[`e102ee371d08770b6fdfcebc33abe169e60e2756`](https://github.com/a16z/aerie/tree/e102ee371d08770b6fdfcebc33abe169e60e2756).
The source set is:

- the [Falcon version 1 protocol specification](https://github.com/a16z/aerie/blob/e102ee371d08770b6fdfcebc33abe169e60e2756/specs/falcon-v1.md);
- the [frozen production profile decision](https://github.com/a16z/aerie/blob/e102ee371d08770b6fdfcebc33abe169e60e2756/specs/falcon-v1-profiles.md);
- the executable [commitment manifest](https://github.com/a16z/aerie/blob/e102ee371d08770b6fdfcebc33abe169e60e2756/crates/aerie-falcon/src/protocol/manifest.rs);
- the [transcript state machine](https://github.com/a16z/aerie/blob/e102ee371d08770b6fdfcebc33abe169e60e2756/crates/aerie-falcon/src/protocol/transcript.rs);
- the [Akita PCS integration](https://github.com/a16z/aerie/blob/e102ee371d08770b6fdfcebc33abe169e60e2756/crates/aerie-falcon/src/protocol/pcs.rs);
- merged [Aerie PR #51](https://github.com/a16z/aerie/pull/51), which introduced the complete Falcon version 1 path; and
- merged [Aerie PR #52](https://github.com/a16z/aerie/pull/52), which adopted the JL4-final layout.

The protocol document defines the witness boundary, logical shards, extraction
families, and opening claims. Some inventory and transcript prose at the pinned
commit still lists the three semantic JL shards as `JlFour`, `JlTwo`,
`JlOne`. PR #52 and the executable manifest, transcript state machine, and
PCS integration at the same commit enforce the adopted wire order
`JlTwo`, `JlOne`, `JlFour`. This planner fixture uses that executable
order. It does not reinterpret the stale prose order as a second supported
variant.

Open [Aerie PR #53](https://github.com/a16z/aerie/pull/53) adds adversarial
coverage for the digit alphabets, JL source and group order, commitment
lengths, shard layout, and minimum batch zero extension. Open
[Aerie PR #54](https://github.com/a16z/aerie/pull/54) adds timing spans around
the same protocol. Neither pull request proposes a different commitment
layout.

## Why this workload matters

Aerie cannot be described as one semantic main group with several unrelated
precommits. It commits protocol objects in two epochs separated by a
Fiat-Shamir challenge. Three groups do not exist until that challenge is
known. All eight groups are later opened by one Akita multi-group proof.

The planner must choose the complete schedule before the first commitment. It
must preserve the challenge boundary and canonical group order while comparing
all group profiles and the shared opening schedule together.

This case separates four concepts which older planner APIs tend to combine:

1. the protocol meaning of a committed object;
2. the time at which its values become available;
3. the extraction family required by its source contract; and
4. its mechanical position in an Akita grouped opening.

## Protocol object boundary

Falcon version 1 commits the following logical sources.

- `S1` is a committed Falcon intermediate with 512 cells per padded record.
- `Epsilon` is the four square slack source with 4 cells per padded record.
- `S2SourcesFour` packs `a0`, `a1`, `Hcom`, and `zS`. Each source
  has 512 cells per padded record.
- `BudgetTwo` packs budget digits `e0` and `e1`.
- `BudgetOne` contains budget digit `e2`.
- `JlFour` packs projection digits `D0` through `D3`.
- `JlTwo` packs projection digits `D4` and `D5`.
- `JlOne` contains projection digit `D6`.

The JL projection source is the joint vector `S1 | V | Epsilon`. `V` is
virtual. The decoded `S2` is also virtual and is absent from the JL source
because the encodability proof supplies its required bound.

The production fixture uses the centered high plane with `kappa_H = 8` and
`Hcom` in `[-8, 7]`. It uses the signed sign plane with `zS` in `{-1, 1}` and
the direct decoder. These choices are bound into the transcript before the JL
seed. The other implemented profile variants are not alternate defaults for
this fixture.

The target JL profile has group size 8, projection count `m = 128`, seven
digit planes, and container bound `2^26`. For a padded batch smaller than 8,
the effective group size is the padded batch size. `D0` through `D5` use radix
16. `D6` uses radix 8.

## Exact commitment manifest

Let `N*` be the padded record count and let `g* = min(8, N*)` be the effective
JL group size. The canonical manifest and Akita batch roles are:

| Position | Group | Logical contents | Logical cells | Epoch | PCS family | Akita batch role | Opening suffix |
|---:|---|---|---:|---|---|---|---|
| 1 | `S1` | `S1` | `512 N*` | before JL seed | bound 18 | frozen | `r_F` |
| 2 | `Epsilon` | `Eps` | `4 N*` | before JL seed | bound 18 | frozen | `r_F,epsilon` |
| 3 | `S2SourcesFour` | `a0, a1, Hcom, zS` | `2048 N*` | before JL seed | bound 6 | frozen | `r_F` |
| 4 | `BudgetTwo` | `e0, e1` | `2 N*` | before JL seed | bound 6 | frozen | `r_F,row` |
| 5 | `BudgetOne` | `e2` | `N*` | before JL seed | bound 6 | frozen | `r_F,row` |
| 6 | `JlTwo` | `D4, D5` | `256 N* / g*` | after JL seed | bound 6 | frozen | `u_J` |
| 7 | `JlOne` | `D6` | `128 N* / g*` | after JL seed | bound 6 | frozen | `u_J` |
| 8 | `JlFour` | `D0, D1, D2, D3` | `512 N* / g*` | after JL seed | bound 6 | closing | `u_J` |

For `N* >= 8`, the three JL lengths simplify to `32 N*`, `16 N*`, and
`64 N*`, and the logical total is `2679 N*` cells. For a smaller padded batch,
the exact total is `2567 N* + 896 N* / g*`. Reordering the three JL rows does
not change either sum. It does change the transcript and the Akita lookup key,
so the planner must use the canonical order shown above.

The terms `frozen` and `closing` describe the compiled Akita batch. They
are not protocol semantics. In particular, `JlFour` is not the main Aerie
object. It is the last group passed to the grouped root.

## Commitment epochs and transcript

The planner request contains these fixed epochs:

~~~text
epoch 0
    S1
    Epsilon
    S2SourcesFour
    BudgetTwo
    BudgetOne
    challenge_after_epoch = JL seed

epoch 1
    JlTwo
    JlOne
    JlFour
    challenge_after_epoch = none
~~~

The transcript first binds the protocol version, record count, commitment
layout, high plane profile, sign profile, complete JL profile, public input
digest, salts, messages, and exact shard manifest. It then follows this order:

1. Compute the five epoch 0 commitments, possibly in parallel, and absorb them
   in manifest order.
2. Derive the 32 byte JL seed.
3. Derive the joint ternary matrix from that seed and the bound public profile.
4. Compute the projection values and seven digit planes.
5. Compute the `JlTwo` and `JlOne` commitments, possibly in parallel, then
   absorb `JlTwo` followed by `JlOne`.
6. Commit `JlFour` as the final Akita group with the first seven profiles
   frozen, then absorb it.
7. Run the proof blocks and the batched PCS opening.

The seed changes the post-seed values. It does not change the digit count,
packing, logical lengths, extraction families, group order, or schedule
selection. These facts are enough to plan both epochs offline.

The request sets group reordering to forbidden. A planner may study other
orders as an explicit protocol design experiment, but it may not silently
choose one while compiling this workload.

## Logical and physical geometry

Each row in the table gives a logical polynomial length. Akita currently
requires a power of two physical length and has a `2^13` cell floor for the
single-polynomial bound 18 and bound 6 schedules used here. For a logical
length `L`, the current physical length is

\[
\operatorname{pcslen}(L)=2^{\max(13,\log_2 L)}.
\]

When `pcslen(L) > L`, the committed polynomial contains the logical table
followed by zeros. The full opening point appends zero pad coordinates. This
ties the physical polynomial to the logical table rather than treating the
extra cells as witness data.

The planner must price the physical length. It must retain the logical length,
zero extension rule, and opening point construction in the plan so that the
prover and verifier derive the same shape. This is especially important at the
minimum batch size, where several shards hit the floor.

## Joint planning and compilation

The planning request contains all eight group requirements, both epochs, and
one opening batch:

~~~text
opening batch
    ordered groups =
        S1, Epsilon, S2SourcesFour, BudgetTwo, BudgetOne,
        JlTwo, JlOne, JlFour
    closing group = JlFour
    reorder policy = forbidden
~~~

The planner selects each commitment profile and the shared opening schedule as
one workload decision. It may not select five schedules, derive the seed, and
then run a second planner for the JL groups.

After selection, compilation produces the current Akita representation:

~~~text
AkitaScheduleLookupKey
    precommitteds =
        S1, Epsilon, S2SourcesFour, BudgetTwo, BudgetOne,
        JlTwo, JlOne
    final_group = JlFour
~~~

This is a lowering step. The public planner request should not require Aerie to
call any of those seven groups semantic precommits or to call `JlFour` the
main route.

The verifier derives all eight profiles and the opening selection from the
statement shape and generated catalogs. The prover does not supply the
selection.

## Opening contract

Each physical commitment reaches the PCS at one complete opening point. Every
group owns its point. The points do not need to be prefixes of one shared
point, and different groups may use different suffixes.

One `PolynomialGroupClaims` value represents each physical commitment. One
Akita multi-group proof may batch the eight groups. The opening policy belongs
to that batch, not to the `JlFour` source family.

This contract is why all eight profiles must be planned jointly even though
the values are committed in two epochs.

## Why JL4 closes the batch

The original integration used `JlOne` as the final Akita group. Aerie PR #52
compared another valid lowering. It precommits `JlTwo` and `JlOne`, then
uses the four-plane `JlFour` shard as the final group. The adopted row had a
lighter recursive ladder and reduced the measured PCS proving time without
changing proof bytes.

This is optimization evidence for one fixed protocol order. It does not prove
that the largest shard is always the right closing group. Future planner
guidance may reproduce the choice quickly, but an exhaustive diagnostic run
must still compare every protocol-permitted lowering unless a certified
dominance result removes it.

## Required planner fixtures

The planner test suite must contain at least these fixtures.

1. The target eight group workload at a representative power of two batch.
2. The minimum batch workload with every required PCS zero extension.
3. A transcript order rejection which swaps `JlTwo` and `JlFour`.
4. A challenge boundary rejection which moves a JL group before the seed.
5. A compiled key check with seven frozen profiles and `JlFour` as the
   closing group.
6. A source check with bound 18 only for `S1` and `Epsilon`.
7. A verifier derivation check which does not accept a prover supplied plan.
8. Oracle and guided planner runs which return the same canonical descriptor.

The performance report records compilation time separately. The guided target
applies after compilation, as defined by the parent specification.

## Non-inferences

The planner must not infer any of the following:

- the closing group is the semantic main object;
- the closing group's extraction family owns the opening policy;
- post-seed groups can be planned only after the seed is sampled;
- JL shard names determine their commitment order;
- equal logical cell totals make different orders transcript equivalent;
- a local profile winner for one shard is the global batch winner; or
- a successful generated catalog lets runtime code invoke planner search.
