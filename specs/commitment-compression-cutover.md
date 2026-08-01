# Spec: Commitment Compression Cutover

Status: protocol activated; final validation pending.

Implementation base: PR 341 at `f0710986f93c2dd9a81acce4bee3d1ee2ae2211d`.

Paper authority: `sections/akita/6_commitment_and_fold.tex`,
`sections/akita/7_checking_a_fold.tex`, and
`sections/akita/13_parameter_selection.tex` in the FTTA paper.

## Decision

Akita uses compressed standalone and root commitments, followed by a
planner-selected monotone cutover from compressed recursive payloads to raw
recursive payloads:

```text
compressed, compressed, ..., compressed, raw, raw, ..., raw
```

The root fold and first recursive fold are compressed whenever they exist. At
every later level in the compressed prefix, the planner evaluates both another
compressed fold and the start of a raw suffix. A fold that consumes a recursive
setup prefix remains compressed. Compression cannot resume after raw mode
begins. The schedule descriptor binds every level's mode; the proof carries no
mode tag.

Each source image must contain at most 8 KiB of canonical field elements. The
bound is inclusive. The protocol rejects a larger source during schedule
validation and again before prover or verifier allocation.

Every compression chain has exactly two rank one maps. Both maps use the
negative binary alphabet `{-1, 0}`. The transmitted terminal payload is exactly
128 bytes.

The protocol does not support three map chains and does not slice a large
source. Raw mode is not a compression fallback for an oversized source. It is
an independently validated recursive payload mode selected by the planner to
avoid carrying a fixed compression-witness tax through the small tail of the
recursion.

This is a breaking protocol change. The complete cutover must merge as one
working protocol. Intermediate development commits may add internal parts, but
no release may expose a mode not bound by the selected schedule.

## Why The Cap Is 8 KiB

The standard two map ladder remains above the current SIS cutoff beyond 8 KiB.
Its exact profile independent limit is 14,144 bytes. However, 8 KiB is the
largest power of two below that limit.

The 8 KiB cap has four useful properties.

1. Every profile uses one fixed pair of compression dimensions.
2. Every chain has exactly two maps.
3. A compressed level has no map-shape choice or third-map state.
4. The cap has a large margin below the certified SIS width.

The PR 341 schedule census covers 5,604 supported schedules from 17 generated
families. Its largest current source is 3,072 bytes. The 8 KiB cap therefore
does not reject any current schedule.

## Compression Geometry

Let `y` be one flat source image. Its coefficients are stored canonically in
the base field. For every coefficient `y_j`, define its negative binary digits
by

```text
y_j = -sum_k bit_k(q - y_j) * 2^k mod q
bit_k(q - y_j) in {0, 1}
```

A stored digit is `-1` when the corresponding bit is one. It is zero
otherwise. Digits use bit major order across source coefficients. Within each
compression ring value, consecutive coefficient positions hold consecutive
digits.

For a B image, the two maps are called `F_1` and `F_2`:

```text
u_1   = F_1 * negbin(u)
p_F   = F_2 * negbin(u_1)
```

For a D image, the two maps are called `H_1` and `H_2`:

```text
v_1   = H_1 * negbin(v)
p_H   = H_2 * negbin(v_1)
```

The protocol transmits `p_F` and `p_H`. It never transmits `u`, `u_1`, `v`, or
`v_1`.

The dimensions are fixed by the field profile:

| Profile | First map dimension | First image | Second map dimension | Payload |
| ------- | ------------------- | ----------- | -------------------- | ------- |
| q128 | 16 | 256 bytes | 8 | 128 bytes |
| q64 | 32 | 256 bytes | 16 | 128 bytes |
| q32 | 64 | 256 bytes | 32 | 128 bytes |

These are compression-only dimensions. They are admitted through the exact
two-entry ladder for the selected modulus profile and do not enter
`CommitmentRingDims`, role projection, or ordinary A/B/D matrix admission.
Every A, B, and D matrix dimension is at least 64. The shared NTT layer still
supports the first compression dimension where it overlaps its checked
profile band; the smaller terminal dimension enters only through the
compression-aware NTT path.

All maps have output rank one. For a source with `s` field coefficients and
field bit width `k`, the first map width is

```text
ceil(s * k / D_1)
```

The second map width is fixed because the first image is always 256 bytes.

| Profile | First width at 8 KiB | Second width | Payload coefficients |
| ------- | -------------------- | ------------ | -------------------- |
| q128 | 4,096 | 256 | 8 |
| q64 | 2,048 | 128 | 16 |
| q32 | 1,024 | 64 | 32 |

Padding through the last compression ring row is zero. Padding participates in
the relation and in the negative binary check.

## Security Contract

Each map is a standalone rank one Module SIS instance under the
`Quantum128BitADPS16` policy. Its coefficient infinity bound is one. This bound
is exact because two negative binary witnesses differ coefficientwise by at
most one.

The production compression authority contains only the six cells used by the
two map protocol:

| Profile | Dimension | Required width | Certified maximum width |
| ------- | --------- | -------------- | ----------------------- |
| q128 | 16 | 4,096 | 7,077 |
| q128 | 8 | 256 | 508 |
| q64 | 32 | 2,048 | 3,538 |
| q64 | 16 | 128 | 254 |
| q32 | 64 | 1,024 | 1,769 |
| q32 | 32 | 64 | 127 |

The current Rust local minimum attack search reports 162.352 bits for the
first map at the 8 KiB cap and 168.776 bits for the second map. These are
estimates, not the protocol guarantee. The checked table guarantees that every
listed width clears the 128 bit quantum floor.

For reference, the quick Rust estimates for the first map are:

| Source size | Estimated quantum bits |
| ----------- | ---------------------- |
| 8 KiB | 162.352 |
| 9 KiB | 157.388 |
| 10 KiB | 153.300 |
| 11 KiB | 149.504 |
| 12 KiB | 146.292 |
| 13 KiB | 143.372 |
| 14 KiB | 140.744 under local search, but rejected by exhaustive certification |

The 14 KiB row shows why runtime selection must use the certified table. The
local optimizer can miss a cheaper attack.

## Canonical Plan And Cutover

For a compressed level, the compression plan is derived and is not a planner
choice. The cutover policy determines the level payload mode, while the planner
chooses the surrounding fold and setup geometry. Each recursive schedule row
stores `CommitmentPayloadMode::{Compressed, Raw}`.

The source matrix profile and output shape determine the source coefficient
count. The field profile determines both compression dimensions. The policy
constant fixes the cap, alphabet, map count, output rank, and terminal byte
count.

The protocol defines one cutover policy identifier:

```text
NegativeBinaryTwoMapExactMonotoneCutover8KiBV3
```

The instance descriptor binds this identifier once. Standalone committed group
profiles also bind the compressed commitment format through their version.
The effective schedule descriptor additionally binds each recursive mode. Any
change to the cap, alphabet, dimensions, map count, terminal size, minimum
compressed prefix, or monotone-cutover rule requires a new policy identifier
and protocol epoch.

Recursive setup-prefix offload is permitted only in the compressed prefix. A
consumer of that prefix is compressed even at level two or later. After raw
mode begins, setup contribution is direct. This keeps precommitted setup
commitments uniformly compressed and prevents an offload edge from hiding a
raw-to-compressed transition.

Equal matrix shapes use the same canonical prefix of the universal flat setup.
`F` and `H` are semantic row names. They do not create separate random setup
matrices when their profile, dimension, rank, and width agree.

## Public Types And Wire Format

Akita does not use Protocol Buffers for these objects. It uses custom
shape directed serialization. The cutover changes that wire format as follows.

### Standalone group commitment

`CommittedGroup.commitment` changes from the raw B image to `p_F`.

The frozen group profile keeps the original B matrix description. The verifier
needs that description to derive the F plan and later check the B and F
relations. Validation expects the fixed terminal coefficient count for the
field profile instead of `n_B * d_B` coefficients.

`CommittedGroupProfile::VERSION` increases from 1 to 2. Version 1 commitments
are rejected.

### Commitment hint

`AkitaCommitmentHint` remains prover data. It is not part of the public proof.
It retains:

1. The existing inner A rows.
2. The packed negative binary digits for `F_1`.
3. The packed negative binary digits for `F_2`.
4. The cyclic quotient for each F map.

The hint does not retain the raw B image or the intermediate F image. Both are
recomposed from the retained digits when the group is opened. Retaining the F
quotients avoids repeating the commitment-time ring division during every
opening. The H digits and quotients remain fold-time witnesses and are not
persisted in the commitment hint.

Hint persistence derives the expected digit and quotient shapes from the frozen
group profile. Decoded byte lengths, quotient counts, and quotient ring
dimensions must match those shapes exactly. The serialized hint does not carry
an independent compression plan.

### Fold proof

At a compressed level, `FoldLevelProof.opening_payload` carries the 128 byte
`p_H` payload. At a raw level, it carries the native D image.

`NextWitnessBinding::OuterPayload` carries the next level's B encoding: 128
byte `p_F` when the next level is compressed and the native B image when it is
raw. This permits the cutover boundary to carry compressed `p_H` for the
current level while binding its child through raw B.

The terminal inner state variant is unchanged. Terminal folds have no B or D
payload.

Proof decoding remains headerless. The schedule-selected mode supplies the
expected payload coefficient count and ring dimension. No payload length,
mode, or compression plan is serialized inside a fold proof.

### Protocol epoch

`AKITA_INSTANCE_DESCRIPTOR_VERSION` increases from 2 to 3. The schedule row
hash domain also increases from `v1` to `v2` because witness and proof size
semantics change. Generated schedule identities and catalog digests are
regenerated.

## Transcript Order

The positional transcript order remains the same where possible.

1. Opening claims absorb each group profile and its `p_F` payload in canonical
   group order.
2. A nonterminal fold absorbs its schedule-selected opening payload: `p_H` in
   compressed mode or raw `v` in raw mode.
3. The fold absorbs the outgoing schedule-selected B payload for its child.
4. A compressed level samples the negative binary batching challenge after the
   range image claim and before the fused relation sumcheck. A raw level omits
   this challenge and follows the pre-compression transcript sequence.

Production labels are logging names and are not sponge domain separators. The
The logging label is `ABSORB_OPENING_PAYLOAD`, and
`CHALLENGE_COMPRESSION_BINARY` names the new batching challenge.

The descriptor version and policy identifier provide cross protocol domain
separation. A raw payload proof cannot replay as a compressed payload proof.

## Witness Layout

At a compressed level, the witness keeps the current physical chunk units
first and adds the global compression suffix below. At a raw level, the witness
ends after the ordinary relation quotient digits; it has no compression spans,
compression quotient rows, alignment tail, or negative-binary support.

The canonical order is:

```text
[all chunk units Z | E | T]
[ordinary relation quotient digits R for consistency, A, B, and D]
[derived zero alignment padding]
[F_1 digits for each relation group]
[H_1 digits]
[F_1 quotient digits for each relation group]
[H_1 quotient digits]
[derived layer alignment padding, when needed]
[F_2 digits for each relation group]
[H_2 digits]
[F_2 quotient digits for each relation group]
[H_2 quotient digits]
[derived zero suffix padding, when needed]
```

Within one F layer, groups use the same order as the relation layout. The final
group comes first. Precommitted groups follow in their canonical order. The H
chain is shared by the level and appears once.

Compression is the final physical witness tail because its native dimensions
are smaller. Each layer keeps its F/H negative-binary digits together with the
quotient digits for the same map. The first layer precedes the lower-dimension
second layer. The shared-tail envelope is contiguous, but compression quotient
rows are intentionally not a separate physical block. Padding is derived by
`WitnessLayout` at the ordinary-to-compression boundary, between layers when
needed, and at the suffix. It aligns the flat witness length to the unchanged
A, B, and D common coefficient block. It is witness data, not serialized
metadata. This avoids every small-dimension to large-dimension reset in the
physical layout.

Digits remain the innermost coordinate. The layout stores the complete padded
ring rows for every map. `WitnessLayout` is the only authority for all F, H,
and R coefficient addresses.

The negative binary support is the union of all F and H digit ranges. Each
layer contributes one sorted interval; its adjacent quotient digits are not in
the support. The intervals are derived from `WitnessLayout`, not materialized
as a witness sized bitmap. Alignment padding is outside this support and is
fixed to zero by witness construction.

## Relation Layout

Raw mode uses the ordinary relation unchanged: the public B and D images occupy
their native right-hand-side rows, and no F/H rows exist.

Compressed mode uses the following extended relation.

The raw public B and D images leave the right hand side. Their rows now bind the
first compression digits.

For every relation group `g`:

```text
B_g * t_hat_g - G_bin * xi_F[g,1] = 0
F_1 * xi_F[g,1] - G_bin * xi_F[g,2] = 0
F_2 * xi_F[g,2] = p_F[g]
```

For the level shared opening image:

```text
D * e_hat - G_bin * xi_H[1] = 0
H_1 * xi_H[1] - G_bin * xi_H[2] = 0
H_2 * xi_H[2] = p_H
```

`G_bin` is the fixed binary recomposition gadget. Its powers are
`1, 2, 4, ...` in source coefficient order. The witness digits themselves are
negative, so this gadget recomposes the original field value.

The canonical row order is:

```text
[consistency, A, B] for each relation group
[D]
[F_1 for each relation group, H_1]
[F_2 for each relation group, H_2]
[evaluation row]
```

Each row family uses its native ring dimension. B rows use `d_B`. D rows use
`d_D`. F and H rows use their map dimensions. The evaluation row remains a
field row and has no quotient.

Every ring row gets one quotient row in the same order. The prover computes
the quotient from the cyclic and negacyclic products. Compression setup caches
must therefore contain both transforms before the relation path is active.

F and H use an independent compact relation-address geometry. They never lower
the common coefficient block used by the existing A, B, and D roles. The two
address geometries share the same flat witness domain and are combined only at
the semantic relation evaluation boundary.

The right hand side contains zero for the B, D, first F, and first H rows. It
contains `p_F` and `p_H` only on the terminal F and H rows. The evaluation row
contains the public evaluation claim.

## Negative Binary Proof

The existing shared range check only proves that every witness coordinate lies
in the level alphabet. Compression needs the stronger statement that every F
and H digit lies in `{-1, 0}`.

At compressed levels, the protocol follows the paper and adds a support
restricted term to the fused stage 2 sumcheck:

```text
rho_bin * eq_restricted(r_virt, x) * w(x) * (w(x) + 1)
```

`eq_restricted` is the multilinear extension of the equality table restricted
to the compression intervals. It is not the product of two separately
extended tables.

The prover stores equality weights only on the live compression intervals. It
folds and merges those sparse weights after every challenge. The verifier
evaluates the same interval sum directly. Neither side allocates a dense table
for the compression support. Raw levels construct neither this oracle nor its
support and reuse the existing optimized two-round Stage-2 batching path
without an empty compression term.

## Setup And Compute Backend

The universal setup remains one flat coefficient stream. Each compression map
uses the exact prefix selected by its rank, width, and native dimension.

Prepared proving setup contains one cache slot for each active compression
dimension. A slot contains both cyclic and negacyclic transforms and covers the
largest required prefix at that dimension. If an existing role slot at the same
dimension covers a longer prefix, compression reuses that slot.

Commitment creation needs the negacyclic F products. Opening later needs the
cyclic F products for quotient construction. H execution needs both products
during the fold. The backend may defer the cyclic F pass until opening, but the
proof, quotient, and transcript must be identical under eager and deferred
execution.

The diagnostic backend trait and the `compression-diagnostics` feature are
deleted after production execution is connected. Compression becomes a
required operation of every backend that supports the affected field profile.

## Planner And Generated Schedules

The planner does not search over compression maps or arbitrary per-level mode
strings. It carries one monotone phase bit. Levels zero and one are compressed.
At every later compressed-prefix state, suffix DP prices a compressed edge and
a raw edge. A raw edge permanently enters the raw suffix. A setup-prefix
consumer is compressed and cannot occur after that transition. The same DP
jointly optimizes fold geometry, fold count, setup offload, terminal geometry,
and the cutover.

For every compressed candidate it must:

1. Reject a source above 8 KiB.
2. Derive the two map shapes.
3. Add F and H digit spans to the successor witness.
4. Add compression relation rows and quotient rows.
5. Add the 128 byte public payloads to proof size accounting.
6. Add both transform cache prefixes to setup accounting.
7. Include the negative binary sparse support cost.
8. Include every compression map in the security inventory.

For every raw candidate it instead prices the native B and D wire images and
derives the successor witness from the ordinary relation layout. The raw
candidate has no compression setup, witness, relation, quotient, or binary
check cost. A source above 8 KiB invalidates only compressed mode; it does not
silently force raw mode.

Schedule generation runs only after these quantities use the same canonical
types as the prover and verifier. Generated schedule files are never edited by
hand.

Direct per-level proof pricing charges either one 128 byte `p_H` or the native D
image according to the current mode. An `OuterPayload` successor edge charges
the child mode's B encoding. A `TerminalInnerState` edge charges no duplicate B
payload. Successor witness length comes only from `WitnessLayout`. Setup
capacity includes compression prefixes only for compressed levels.

The planner uses the exact monotone cutover search with proof bytes as its
primary objective. Direct complete schedules use the strict lexicographic
order

```text
(exact proof bytes, physical setup field elements, canonical descriptor)
```

The direct suffix frontier retains the same first two coordinates. Grouped,
standalone, exhaustive, and generated catalog paths use the same complete
selector, so equivalent searches cannot disagree because of traversal order.
This avoids paying setup, prover, verifier, or memory cost that buys no proof
reduction.

This policy does not use a proof byte tolerance and does not rank fold count.
A bounded proof byte window, fold count preference, or unified multiobjective
frontier is a separate planner policy change. It requires a new policy
identifier, regenerated schedules, and its own proof-size and performance
census.

The measured optimum retained four compressed folds for fp128 dense and five
for fp128 onehot. Performance work must preserve the selected proof bytes and
schedule-bound monotone compressed-prefix/raw-suffix contract. A future change
to this rule requires a new policy identifier and regenerated schedules.

## Incremental Implementation Sequence

The work proceeds in the following order. Every slice must format, compile, and
pass its focused tests before the next slice starts. The public protocol remains
the current protocol until the activation slice.

### Slice 0: Security and census gate

1. Replace the nine cell diagnostic table with the six production cells.
2. Check every production width against the existing per-instance 128 bit SIS
   authority.
3. Keep the schedule census as a checked planner example or test.
4. Assert that every current generated source is at most 8 KiB.

This slice changes no proof bytes.

### Slice 1: Canonical compression types

1. Change `MAX_COMPRESSION_INPUT_BYTES` to 8 KiB.
2. Require exactly two maps.
3. Remove the three map dimensions and selection branch.
4. Add the fixed policy identifier.
5. Keep packed negative binary digits and terminal payload validation in
   `akita-types`.

This slice changes no proof bytes.

### Slice 2: Production matrix execution

1. Promote the bounded executor out of diagnostics.
2. Prepare paired cyclic and negacyclic cache views.
3. Return the products needed for image construction and quotient construction.
4. Keep equal shape batching as an execution detail.
5. Test every field profile against schoolbook cyclic and negacyclic products.

This slice changes no proof bytes.

### Slice 3: Witness and relation geometry

1. Extend `WitnessLayout` with the layer major F and H spans.
2. Extend `RelationRhsLayout` with F and H row families and native dimensions.
3. Add an independent compact F/H address geometry and extend quotient row
   layout without changing the A/B/D coefficient block.
4. Add the support interval projection for all compression spans.
5. Add pure layout tests for scalar, multi group, mixed dimension, and chunked
   witnesses.

This slice changes internal derived geometry but does not activate new wire
payloads.

### Slice 4: Planner and setup accounting

1. Apply the fixed plan to every candidate B and D source.
2. Recompute successor witness lengths and relation row counts.
3. Recompute proof sizes and setup envelopes.
4. Generate compression-aware schedules inside planner tests.
5. Check that schedule replay derives the same shapes without planner code.

This slice prepares the authoritative schedules for activation.

### Slice 5: Prover witness and relation

1. Create `p_F` and retain packed F digits and cyclic quotients during
   standalone commitment.
2. Recompose F sources from the hint when the group is opened, reusing the
   retained quotients.
3. Create `p_H` and both H digit vectors during each nonterminal fold.
4. Insert every F and H digit vector through `WitnessLayout`.
5. Build the B, D, F, and H relation rows.
6. Compute the matching H cyclic quotients at fold time.
7. Add the support restricted negative binary term to stage 2.

Focused internal tests compare every produced relation against direct
coefficient arithmetic before any wire cutover.

### Slice 6: Verifier relation

1. Derive the same fixed plans from public profiles and level params.
2. Validate every payload length before allocation.
3. Assemble the new relation right hand side.
4. Evaluate B, D, F, and H row weights through the unified relation evaluator.
5. Evaluate the restricted equality term from support intervals.
6. Reject malformed layouts, payloads, and transcript order without panicking.

An internal prove and verify harness must pass before activation.

### Slice 7: Atomic wire activation

1. Change standalone commitments to `p_F`.
2. Change fold `v` to `p_H` and rename the field.
3. Change outgoing outer commitments to `p_F` and rename the variant.
4. Update proof shapes and headerless decoding contexts.
5. Bump the committed group version, instance descriptor version, and schedule
   row domain.
6. Regenerate the checked schedule tables and catalog identities.
7. Update transcript logging labels.
8. Bind the monotone per-level cutover and retain both raw and compressed
   recursive payload handling.

This is the only slice that changes public proof bytes. Prover and verifier
activate together.

### Slice 8: Cleanup and full validation

1. Delete diagnostic traits, feature flags, reports, and hooks.
2. Delete three map code and tests.
3. Update proof size reports, examples, benches, the book, and architecture
   documentation.
4. Add tampering and cross protocol rejection tests.
5. Run every repository gate and every affected workflow command.

## Required Tests

The final cutover must include:

1. Exact 8 KiB acceptance and next field coefficient rejection for all three
   profiles.
2. Exact two map geometry and 128 byte payload tests.
3. Packed digit round trips and zero padding checks.
4. Schoolbook equivalence for both transforms at every production dimension.
5. B and D base row checks against the first digit vector.
6. Intermediate and terminal F and H row checks.
7. Compression quotient checks at every native dimension.
8. Negative binary rejection for one tampered digit in every span position.
9. Scalar, multi group, mixed dimension, and chunked end to end proofs,
   including a compressed-to-raw boundary.
10. Tampered `p_F`, tampered `p_H`, group reordering, and truncated payload
    rejection.
11. Version 1 commitment and protocol epoch rejection.
12. Default, no default feature, parallel, and disk persistence builds.
13. Proof size and setup envelope agreement between planner, prover, and
    verifier for both modes.
14. Schedule rejection for a raw root, raw first recursive fold, compressed
    mode after raw mode, and setup-prefix offload inside the raw suffix.

## Completion Condition

The cutover is complete only when standalone/root commitments are compressed,
every schedule obeys the minimum two-fold compressed prefix and monotone raw
suffix, both recursive payload modes prove and verify end to end, every emitted
compression digit is bound by both the relation and negative-binary proof, and
the planner selects the smallest modeled proof schedule. Prover performance is
secondary but must not retain avoidable duplicate work.
