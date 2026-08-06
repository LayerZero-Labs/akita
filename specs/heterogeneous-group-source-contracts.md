# Spec: Source-free groups and honest fold sizing

| Field | Value |
|-------|-------|
| Author(s) | Quang Dao |
| Created | 2026-07-30 |
| Revised | 2026-07-31 |
| Status | active |
| PR | [#338](https://github.com/LayerZero-Labs/akita/pull/338) |
| Supersedes | Earlier source-provider and fold-admission revisions of this specification |
| Superseded-by | |
| Book-chapter | how/architecture.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita supports opening batches whose commitment groups use different concrete
polynomial representations. The protocol does not need to name those
representations. It needs the exact public geometry and the exact schedule row
that the verifier checks.

This specification defines two boundaries.

First, runtime protocol state is source-free. A committed group carries its
exact commitment profile. A selected schedule row carries the exact fold,
matrix, recursive, terminal, and wire parameters. Runtime code MUST NOT carry a
dense, one-hot, lookup, or application source tag.

Second, each group owns an offline honest fold sizing policy. That policy uses
the group distribution and the candidate fold geometry to select
`num_digits_fold`. The core planner checks the selected digit depth and prices
its exact protocol consequences. The planner MUST NOT reinterpret or reduce a
policy result.

The existing balanced signed digit sizing and snap behavior MUST remain
unchanged. The new unit one-hot policy uses the exact one-coordinate moment
generating function. Its initial snap calibration is `1/1`, which disables
snap. The unit one-hot result MUST NOT be worse than the existing sizing result
for the same row.

Intermediate folds and terminal raw responses have different contracts.
Intermediate folds are accepted through balanced digit decomposition, so their
schedule rows store `num_digits_fold` and do not store a separate honest
infinity norm cap. Terminal responses are raw integers. Their verifier-visible
shape stores the response admission cap and the exact Golomb-Rice wire
parameters.

## Intent

### Goals

This cut MUST do the following:

1. Keep commitments, public profiles, setup, schedule lookup, transcripts,
   proofs, and verification free of source identity.
2. Replace planner-facing `FoldWitnessNorms` with a group-owned offline fold
   sizing policy.
3. Make `num_digits_fold` the only honest sizing output for an intermediate
   digitized fold.
4. Preserve the existing balanced signed digit outputs, including the current
   field-specific snap ratios.
5. Add an exact unit one-hot sizing policy that can tighten existing rows but
   cannot make them larger.
6. Move terminal admission and Golomb-Rice ownership into the terminal response
   shape.
7. Remove all residual ZK grind probe behavior. Akita has one sequential probe
   rule.
8. Complete the cutover without compatibility wrappers, deprecated aliases, or
   parallel legacy paths.

### Non-goals

This cut does not try to prove a tight model of honest prover behavior.

This cut does not introduce per-group acceptance targets or allocate a miss
budget across groups. All policies use the same fixed protocol sizing
convention.

To preserve current behavior, that convention uses the existing global
`p_grind = 1/8` value. This value is a coarse generator cutoff. It is not a
per-group knob, a runtime field, or a verifier claim about observed acceptance.
The policy architecture does not claim a joint acceptance probability for all
groups that share one nonce.

This cut does not tune the unit one-hot snap ratio from benchmark data. Its
initial value is `1/1`.

This cut does not change the challenge sampler, the shared fold nonce, or the
nonce attempt limit.

This cut does not change balanced signed digit proof size, terminal response
admission, or terminal Golomb-Rice budgets. Any such drift is a regression.

## Ownership

### Runtime protocol state

The verifier MUST know:

- the exact commitment profile for every group;
- the ordered selected schedule row;
- each intermediate `log_basis` and `num_digits_fold`;
- the exact matrices and their security parameters;
- each challenge configuration;
- the shared nonce range and transcript position;
- each terminal response admission cap;
- each terminal Golomb-Rice remainder width and payload byte budget.

The verifier MUST NOT know:

- a source family name;
- a provider registration;
- dense coefficient bits;
- a one-hot chunk size;
- honest witness norms;
- an analytic tail cap;
- a snap calibration input;
- a planner cost model;
- a target acceptance probability;
- a prover probe order choice.

Two source implementations that produce valid witnesses for the same exact
profile and schedule row are protocol equivalent.

### Offline group policy

Each group configuration MUST select its own honest fold sizing policy before
the core planner evaluates a candidate row. The policy MAY use facts about the
honest source that do not appear in runtime state.

The policy result is authoritative. The core planner MUST NOT apply another
snap, safety factor, discount, or source-specific correction to it.

The core planner MUST still reject a result that fails a hard protocol check,
including arithmetic capacity, matrix capacity, dimension validity, or SIS
security.

### Generated schedule rows

A generated intermediate row MUST store the selected `num_digits_fold`.

A generated intermediate row MUST NOT store `fold_witness_linf_cap` or another
field with the same meaning.

The row MUST freeze every downstream consequence of `num_digits_fold`,
including matrix widths, ranks, setup use, proof shape, and the row digest.

The generator MAY report the analytic cap, the unsnapped digit depth, the snap
ratio, and the final digit depth for audit. These diagnostics MUST NOT enter
runtime types or protocol identity.

## Honest fold sizing contract

### Minimal interface

The intended interface is:

```rust
pub struct HonestFoldSizingQuery<'a> {
    pub ring_dimension: usize,
    pub num_claims: usize,
    pub num_live_blocks: usize,
    pub num_chunks: usize,
    pub num_fold_coeffs: usize,
    pub log_basis: u32,
    pub challenge_config: &'a SparseChallengeConfig,
}

pub trait HonestFoldPolicy {
    fn num_digits_fold(
        &self,
        query: HonestFoldSizingQuery<'_>,
    ) -> Result<usize, AkitaError>;
}
```

The final names MAY change. The ownership and information content are
normative.

The trait returns a scalar because an intermediate fold has one honest sizing
decision. An `HonestFoldPlan` wrapper with only `num_digits_fold` SHOULD NOT be
introduced.

The trait itself SHOULD NOT require `Sync`. A caller that evaluates policies in
parallel MAY require `HonestFoldPolicy + Sync` at that call site.

### Query fields

`ring_dimension` is REQUIRED because the challenge occupancy law depends on the
ring dimension.

`num_claims` and `num_live_blocks` are REQUIRED because they determine the
number and structure of fold contributions.

`num_chunks` is REQUIRED because the prover emits and the verifier admits one
physical response window per chunk. It MUST be positive and no greater than
`num_live_blocks`.

`num_fold_coeffs` is REQUIRED because the policy sizes the maximum over every
actual emitted coefficient in every chunk response. The caller MUST pass the
total physical coefficient count, not a single logical window or a padded
allocation width. The count MUST divide evenly by `num_chunks` because every
chunk response has the same physical width.

The preserved balanced signed digit policy MAY reconstruct its historical
single-window coefficient count by dividing `num_fold_coeffs` by `num_chunks`.
This is an explicit compatibility rule for frozen balanced schedules, not the
physical geometry used by new sizing policies.

`log_basis` is REQUIRED because the policy selects a balanced digit depth and
snap acts on digit boundaries.

`challenge_config` is REQUIRED because it defines the challenge law.

`field_bits` MUST NOT appear in this query. A field-specific policy MUST carry
its calibration when the group configuration constructs it. Hard field
capacity remains a core planner check.

`inner_width` MUST NOT appear under that ambiguous name. If the implementation
can derive `num_fold_coeffs` from checked geometry without losing information,
it MAY remove `num_fold_coeffs` from the query and use that one canonical
derivation. It MUST NOT carry both independent values without validating their
equality.

### Policy result

The policy MUST return the final `num_digits_fold` after it has applied its own
analytic model and snap calibration.

The policy MUST NOT return an analytic infinity norm cap for an intermediate
fold. That cap is an internal planning value. Once the policy has selected the
digit depth, the cap has no independent protocol meaning.

The planner MUST compute the accepted negative and positive coefficient bounds
from `log_basis` and `num_digits_fold` through the canonical balanced digit
functions.

## Snap calibration

### Meaning

The analytic tail calculation is a conservative sizing baseline. It uses
inequalities and a maximum over many coordinates. Akita does not treat it as a
tight prediction of honest prover behavior.

Snap is an explicit policy calibration. It lowers the digit depth when the next
smaller balanced digit interval retains a configured fraction of the analytic
cap.

Snap MUST be applied inside the group policy. It MUST NOT be a generic planner
operation.

The implementation SHOULD use one validated value type:

```rust
pub struct DigitSnapCalibration {
    pub retain_num: u32,
    pub retain_den: u32,
}
```

The type MUST reject a zero denominator, a zero numerator, and a numerator
greater than its denominator.

A calibration of `1/1` means no snap. A policy with `1/1` MUST return the
minimum digit depth that covers its unsnapped cap.

### Balanced signed digit policy

The balanced signed digit policy MUST retain the behavior present before this
architecture cutover.

In particular:

- Fp32 policies MUST retain the existing `3/4` snap ratio.
- All existing wider field policies MUST retain the existing `1/2` snap ratio.
- The signed-sparse tail formula MUST remain unchanged.

The policy object owns the field-specific calibration. The query does not carry
`field_bits`.

For every existing balanced signed digit candidate, the new policy MUST return
the same `num_digits_fold` as the pre-cutover implementation.

### Unit one-hot policy

The unit one-hot policy MUST accept a `DigitSnapCalibration`. Shipping policies
MUST initially construct it with `1/1`.

The implementation MUST NOT apply the balanced signed digit `1/2` or `3/4`
snap ratio to the new exact unit one-hot estimate.

The calibration remains explicit so a later protocol change can tune it after
benchmark and grind data exist. Such a change requires schedule regeneration,
proof size review, and a protocol identity change.

## Exact unit one-hot model

### Applicability

The exact unit one-hot model applies when every logical
witness block has at most one nonzero coefficient and that coefficient has
absolute value one.

The group configuration establishes this fact offline when it selects the
policy. Runtime profiles and the verifier MUST NOT carry or validate a one-hot
tag.

If the group configuration cannot establish the unit one-hot condition, it
MUST use the balanced signed digit policy or another valid group-owned policy.

### One-coordinate moment generating function

Let a challenge of ring dimension `D` contain `k_a` coefficients of magnitude
`a`, with independent symmetric signs and uniformly sampled support without
replacement. For a fixed unit one-hot witness location, one contribution `X`
has moment generating function

\[
M_X(\lambda)
=
1+
\sum_{a\ge 1}
\frac{k_a}{D}\left(\cosh(a\lambda)-1\right).
\]

For the shipping `D = 64` challenge with 31 coefficients of magnitude one and
10 coefficients of magnitude two, the policy MUST use

\[
M_X(\lambda)
=
1+
\frac{31}{64}(\cosh\lambda-1)
+
\frac{10}{64}(\cosh2\lambda-1).
\]

The constants 31 and 10 are specific to `D = 64`. Other ring dimensions MUST
derive their counts from the selected `SparseChallengeConfig`. They MUST NOT
reuse the `D = 64` constants.

At most

\[
m = \mathtt{num\_claims}\left\lceil
\frac{\mathtt{num\_live\_blocks}}{\mathtt{num\_chunks}}
\right\rceil
\]

independent unit contributions enter one coordinate of a physical chunk
response. The ceiling prices the largest response window when blocks do not
divide evenly across chunks. Then

\[
M_Z(\lambda)=M_X(\lambda)^m.
\]

For `N = num_fold_coeffs`, where `N` counts coefficients across every physical
chunk response, the policy computes the smallest integer threshold `t` for
which the fixed protocol cutoff is met:

\[
2N\inf_{\lambda>0}
\exp(-\lambda t)M_X(\lambda)^m
\le 1-p_{\mathrm{grind}}=\frac{7}{8}.
\]

The protocol uses one fixed convention for all groups. The query MUST NOT carry
a per-group target probability or miss allocation.

The numeric procedure MUST be deterministic for schedule generation. It MUST
not understate its computed upper bound because of floating point rounding.

### Dominance guard

The unit one-hot cutover is allowed to tighten sizing. It is not allowed to
increase proof size.

For every candidate row, the generator MUST also compute the pre-cutover sizing
result using the preserved balanced signed digit path and its existing snap
behavior. The selected unit one-hot digit depth MUST be no greater than that
result.

The exact MGF candidate itself uses snap calibration `1/1`. Choosing the better
of the exact candidate and the preserved pre-cutover result is a regression
guard. It is not an additional snap applied to the MGF estimate.

The policy MUST also clamp any analytic threshold by the deterministic
worst-case ring product bound before converting it to a digit depth.

If the exact model is unavailable for a ring dimension or challenge
configuration, the policy MUST use the preserved pre-cutover result.

## Intermediate fold admission

For an intermediate fold, the accepted coefficient interval is exactly the
interval represented by `log_basis` and `num_digits_fold`.

The prover MUST accept a candidate nonce only if every centered fold
coefficient fits that interval and all other existing fold checks pass.

The prover MUST NOT apply a second check against an analytic or snapped
`fold_witness_linf_cap`.

The verifier MUST continue to enforce the balanced digit decomposition and
range relations. It MUST NOT evaluate the honest fold policy.

The shared fold nonce remains a fixed `u32`. The prover MUST probe nonces in
ascending order starting at zero and MUST publish the first accepting nonce.
The verifier MUST reject a nonce outside the fixed global attempt range.

Akita MUST remove transcript-seeded shuffle constants, descriptor fields,
preview labels, permutation helpers, branches, and tests that exist only for ZK
probe order. There is no ZK-specific fold grind behavior in this protocol.

## Terminal response and Golomb-Rice contract

### Why the terminal is separate

A terminal response carries raw centered integers. No later balanced digit
decomposition constrains those integers. The verifier therefore needs an
explicit raw response admission cap.

The terminal response cap is not an intermediate honest sizing cap. It is a
verifier admission and wire decoding parameter. The scheduled cap MUST fit the
terminal matrix security capacity.

### Exact terminal shape

Each terminal response group shape MUST own these exact values:

```rust
pub struct TerminalResponseGroupShape {
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    pub z_admission_linf_cap: u128,
    pub z_rice_low_bits: u32,
    pub z_payload_bytes: usize,
}
```

The final name MAY remain `TailSegmentGroupLayout`. The field ownership is
normative.

`z_admission_linf_cap` is the maximum raw absolute coefficient accepted by the
verifier.

`z_rice_low_bits` is the actual Golomb-Rice remainder width used by both the
encoder and decoder. It MUST NOT be a planning proxy that runtime code converts
again.

`z_payload_bytes` is the maximum encoded payload length accepted for that
group.

The canonical zigzag width MAY be derived from `z_admission_linf_cap`. It need
not be stored when one total derivation exists.

### Offline versus runtime use

An offline terminal planner MAY use an honest distribution estimate to select
the Rice remainder width and payload budget. It MUST emit the exact selected
wire values into the terminal response shape.

Runtime encoding and decoding MUST consume the terminal shape directly. They
MUST NOT call `fold_witness_linf_cap_for_claims`, rebuild a fold tail model, or
read intermediate `num_digits_fold` to recover terminal wire parameters.

The prover MUST check that every raw terminal coefficient is within
`z_admission_linf_cap`. It MUST encode with `z_rice_low_bits`. It MUST reject a
candidate whose encoded payload exceeds `z_payload_bytes`.

The verifier MUST decode with the same `z_rice_low_bits`, reject values outside
`z_admission_linf_cap`, and reject a payload longer than `z_payload_bytes`.

The terminal response shape and all these fields MUST be bound into the exact
schedule row and transcript descriptor.

### Behavior preservation

For every existing balanced signed digit schedule, this cut MUST preserve:

- the raw terminal coefficient admission cap;
- the actual Golomb-Rice remainder width;
- the terminal payload byte budget;
- the serialized proof size produced by the same fixture.

The implementation MAY rename fields and move their owner. It MUST NOT change
these values as an incidental effect of the ownership cutover.

One-hot rows MAY produce smaller intermediate digit depths through the exact
MGF policy. Any resulting terminal wire change MUST be an intentional
consequence of that tighter one-hot result and MUST be reported by the profile
benchmark.

## Protocol identity

Protocol identity MUST bind exact accepted and wire parameters. It MUST NOT bind
offline model inputs that have no runtime meaning.

The descriptor MUST bind:

- the fixed global nonce limit and nonce wire width;
- the fact that probe order is sequential;
- each intermediate digit depth and basis through the selected row;
- each terminal response admission cap;
- each terminal Rice remainder width;
- each terminal payload byte budget.

The descriptor MUST NOT bind:

- balanced witness norms;
- unit one-hot tags;
- exact MGF coefficients as a separate runtime policy identity;
- snap ratios;
- field-specific snap selection;
- analytic caps;
- a terminal average-case planner model identifier;
- a cap-to-Rice conversion rule or delta;
- ZK probe order.

Changing an offline policy requires regenerated rows. If the exact generated
row changes, its digest changes through those exact consequences.

## Required cutover

The implementation MUST complete these changes in one pass:

1. Introduce the minimal group-owned `HonestFoldPolicy` boundary.
2. Move the current balanced signed digit formula and snap behavior behind that
   policy without changing its outputs.
3. Add the exact unit one-hot policy with `1/1` snap and the dominance guard.
4. Make planner candidate construction consume only the returned
   `num_digits_fold`.
5. Remove planner-facing `FoldWitnessNorms` and all source model fields from
   core planner keys and cache identities.
6. Remove intermediate `fold_witness_linf_cap` from generated rows, runtime
   group parameters, setup descriptors, schedule digests, and grind contracts.
7. Make intermediate grind acceptance use the canonical balanced digit interval
   only.
8. Add the terminal response admission cap to the terminal group shape and make
   exact Rice parameters authoritative there.
9. Rewire terminal builders, decoders, proof sizing, schedule validation, and
   verifier admission to consume the terminal shape.
10. Delete ZK probe shuffle code and descriptor fields. Delete terminal planner
    model and cap-to-Rice rule fields once the exact terminal shape replaces
    them.
11. Regenerate all affected schedule tables and pinned descriptors.
12. Delete obsolete helpers, wrappers, aliases, tests, and documentation.

## Invariants

1. **Source-free runtime.** Runtime and verifier types contain no source or
   honest distribution identity.
2. **Policy ownership.** Each group policy selects its final digit depth. The
   core planner does not revise it.
3. **Exact intermediate admission.** Intermediate prover acceptance and
   verifier range checks use the same balanced digit interval.
4. **Accepted-range security.** Matrix pricing uses the full coefficient range
   admitted by the verifier.
5. **Exact terminal contract.** Terminal encoder, decoder, proof sizing, and
   verifier admission use one schedule-bound response shape.
6. **Balanced behavior preservation.** Existing balanced signed digit rows and
   proof fixtures do not drift.
7. **One-hot non-regression.** The exact unit one-hot policy never selects more
   digits than the preserved pre-cutover result.
8. **One probe rule.** Every fold uses the same sequential shared nonce rule.
9. **Planner-free verification.** The verifier does not execute honest sizing
   policies or planner search.
10. **No verifier panic.** Malformed dimensions, caps, codec parameters,
    payloads, and nonces return `AkitaError` or `SerializationError`.
11. **Full cutover.** No legacy cap path or compatibility layer remains.

## Evaluation

### Required regression fixtures

Before replacing the old implementation, tests MUST record the existing
balanced signed digit outputs for every shipped family and relevant candidate
geometry.

After the cutover, tests MUST prove exact equality for:

- `num_digits_fold`;
- accepted negative and positive digit bounds;
- matrix widths and ranks affected by the fold depth;
- terminal response admission caps;
- terminal Rice remainder widths;
- terminal payload byte budgets;
- serialized proof sizes for fixed balanced fixtures.

Descriptor bytes and row digests are expected to change because obsolete
fields are removed and terminal fields move. Tests MUST pin the new values.

### Unit one-hot tests

Tests MUST cover the following:

- the `D = 64` moment generating function uses counts 31 and 10;
- other ring dimensions derive their counts from their challenge config;
- the exact MGF agrees with direct enumeration of the one-coordinate law;
- the optimized tail expression is no larger than the deterministic bound;
- shipping unit one-hot rows use snap calibration `1/1`;
- every selected one-hot digit depth is no greater than the preserved
  pre-cutover result;
- at least one row tightens when the exact model supports a smaller depth;
- unsupported source conditions use the preserved fallback.

### Intermediate admission tests

Tests MUST prove that:

- every coefficient inside the balanced digit interval is accepted by the
  intermediate admission predicate;
- either endpoint outside that interval is rejected;
- no intermediate analytic cap check remains;
- the prover and verifier replay the same sequential nonce;
- the attempt limit rejects an out-of-range nonce;
- no transcript shuffle symbol or branch remains.

### Terminal wire tests

Tests MUST prove that:

- encoding and decoding use the exact `z_rice_low_bits` from the terminal shape;
- coefficients above `z_admission_linf_cap` are rejected;
- payloads above `z_payload_bytes` are rejected before an unbounded allocation;
- the terminal admission cap does not exceed matrix security capacity;
- runtime terminal code does not call an honest fold sizing policy;
- fixed balanced fixtures preserve their old cap, Rice width, byte budget, and
  proof size.

### Schedule and end-to-end tests

The generated schedule drift guards MUST pass after regeneration.

Dense, one-hot, extension field, mixed group, recursive, terminal, and
setup prefix end-to-end tests MUST pass.

The profile benchmark report MUST compare pre-cutover and post-cutover values
for each affected mode. A balanced mode with changed proof size MUST fail the
regression check. A one-hot mode MAY stay equal or become smaller. It MUST NOT
become larger.

All repository documentation guardrails and CI commands in `AGENTS.md` MUST
pass.

## Alternatives rejected

### Return both cap and digit depth

An intermediate analytic cap has no independent protocol use after the policy
selects the digit depth. Returning both values creates two facts that can drift.
The policy returns the digit depth only.

### Let the planner snap every policy result

This makes the planner silently distrust group-owned results. It also makes
source calibration a hidden global behavior. Each group policy owns its snap
calibration.

### Store the intermediate cap for Golomb-Rice

This gives one field two unrelated owners. Intermediate admission uses balanced
digits. Terminal coding uses raw response and wire parameters. The terminal
shape stores the latter directly.

### Apply the existing snap ratio to the exact one-hot estimate

The new estimate has not yet been compared with production grind behavior.
Shipping unit one-hot policies start with `1/1`. Later tuning is a separate
protocol change.

### Carry `field_bits` in every sizing query

Only the current calibration selection needs field width. The group
configuration can construct the correct policy once. The planner separately
checks hard field capacity.

### Bind analytic policy inputs into protocol identity

The verifier accepts exact digit and terminal envelopes. It does not verify the
honest distribution model. Binding model inputs would enlarge runtime identity
without strengthening soundness.

## Documentation

The implementation MUST update:

- [`fold-linf-rejection.md`](fold-linf-rejection.md) to mark its old cap and ZK
  probe ownership as superseded;
- [`tail-wire-encoding.md`](tail-wire-encoding.md) to name the terminal response
  shape as the wire authority;
- [`book/src/how/architecture.md`](../book/src/how/architecture.md) to describe
  the offline group policy and source-free runtime boundary;
- [`book/src/how/verification.md`](../book/src/how/verification.md) to distinguish
  intermediate digit admission from terminal raw response admission.

## References

- [BCP 14](https://www.rfc-editor.org/info/bcp14)
- [`fold-linf-rejection.md`](fold-linf-rejection.md)
- [`tail-wire-encoding.md`](tail-wire-encoding.md)
- [`multi-group-batching.md`](multi-group-batching.md)
- [`schedule-catalog-ownership.md`](schedule-catalog-ownership.md)
