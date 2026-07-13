# Spec: SIS ADPS16 Quantum 128-Bit Policy and Role Driven Scalar Table

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-13 |
| Status        | implemented |
| PR            | |
| Supersedes    | the SIS policy and table specs deleted in this cutover |
| Superseded-by | |
| Book-chapter  | how/security.md |

## Summary

Production SIS sizing uses one hard security rule. The scalar infinity norm
LGSA estimator must report at least 128 bits under the ADPS16 quantum cost
model. The model and its exact estimator revision are part of the policy
identity. It is an attack cost model, not a physical resource estimate.

The estimator receives scalar SIS parameters:

```text
n = rank * d
m = width * d
length_bound = B
```

The generated artifact stores scalar cutoffs with key
`(modulus_profile, B, n)`. Runtime callers still provide the matrix role and
ring dimension. The role determines which ring dimensions, coefficient bounds,
and ranks the generator must cover.

A, B, D, and F do not share one forced geometry. B and D often use the current
ring dimension or a smaller dimension. This includes dimension 32 for a 128 bit
field. A currently uses dimension 64 or larger and may use dimensions above the
other matrices. The generator takes the union of all required role cells. It
deduplicates two requests only when they produce the same scalar SIS key.

Policy identifier:

```text
Quantum128BitADPS16
```

The old policy identity and scalar `min_security_bits` identity are removed in
the same cutover. Unsupported policy and table identities fail closed.

## Intent

### Goal

Make the security rule, estimator, generated artifacts, planner, and runtime
lookup agree on these points:

1. The hard security target is 128 bits under the ADPS16 quantum cost model.
2. The estimator uses scalar infinity norm SIS parameters.
3. Each matrix role has its own required ring dimensions and coefficient bound
   cells.
4. The generated artifact stores one copy of each identical scalar SIS cell.
5. Every accepted cutoff has a complete and reproducible certificate.

### Hard acceptance

For a candidate scalar instance `(modulus_profile, B, n, m)`, run the infinity
norm LGSA optimizer under the dedicated ADPS16 quantum cost model.

Generation may use the `local-minimum` search to find a candidate boundary. It
must certify the accepted boundary and the first rejected successor with an
exhaustive search over the configured beta and zeta domain. The certificate
domain is part of the policy identity.

A candidate passes only when the certified estimate returns a finite score or
an explicit above-target lower bound. A finite score or represented lower bound
must be at least 128 bits:

```text
score.log2() >= 128.
```

The generator must not treat a generic `CostValue::Infinity` as secure.
Numeric underflow, unsupported input, a failed search, or an unclassified
infinite result stops generation. If the estimator can prove that a cost is
above the target without representing the full value, it returns the distinct
`CostValue::ProvenAboveTarget` result with a supporting lower bound. That result
may pass only when its bound is at least 128 bits.

For each scalar key `(modulus_profile, B, n)`, store the largest certified `m`
within the search range. Security cannot increase as `m` grows because an
attacker can pad a shorter witness with zeros. The generator must check that
estimator output follows this order at every probe and in a fixed neighborhood
around the boundary. It must stop if the output breaks the expected prefix
shape.

### Policy identity

The policy ID names the complete acceptance rule. It includes:

- the hard target;
- the reduction cost model and exponent;
- the norm and shape model;
- the estimator revision;
- the boundary certificate domain;
- the meaning of finite, classified, and failed estimates.

Any change that can change whether the same scalar SIS cell passes requires a
new policy ID and regenerated artifacts. This includes an optimizer change.

The table digest is separate. It commits to the exact modulus profiles, role
coverage, coefficient bound cells, rank limits, search caps, certificates, and
generated cutoffs. A coverage change that leaves the acceptance rule unchanged
may keep the policy ID, but it must change the table digest and every dependent
catalog identity.

### Claim language

After the tables land, accurate public language is:

> Akita's generated SIS table targets at least 128 bits under a scalarized
> infinity norm LGSA estimate that uses the ADPS16 quantum cost model.

Do not shorten this to an unqualified post quantum security claim. The table
prices one known attack family on a scalarized instance. It does not prove that
every quantum attack costs at least `2^128`.

### Structured attack boundary

The scalar estimator does not model every property of the production Module SIS
instance. It does not price attacks that use ring or module structure, CRT
splitting, subfield projection, or role specific matrix structure.

The policy provenance must state this limit. A table update must include a
written review of known structured attacks. That review may conclude that no
separate adjustment is needed, but the scalar table must not be presented as a
complete proof of Module SIS security.

## Table geometry

### Scalar embedding

```text
n = rank * d
m = width * d
q = exact modulus selected by modulus_profile
length_bound = B
norm = infinity
```

Inside this estimator, security depends on `(q, n, m, B)`. Matrix role and ring
dimension determine how runtime parameters map to those values. They do not
change the scalar estimate after the mapping is fixed.

Equivalent role requests share a cutoff only when all four scalar values agree.
The generator must not merge cells based only on field bit length, ring
dimension, or module rank.

### Exact modulus profiles

The table key uses an exact modulus profile, not a caller supplied size class.
The initial profiles are:

```rust
pub enum SisModulusProfileId {
    Q32Offset99,
    Q64Offset59,
    Q128OffsetA7F7,
}
```

Each variant maps to one exact integer `q`. Runtime configuration must verify
that the field modulus equals the modulus in the selected profile. The table
digest includes the exact integer values.

Adding a field with another modulus requires a new profile and generated cells.
It must not reuse a profile because the modulus has the same bit length.

### Runtime role key

Runtime callers use this canonical key:

```rust
pub enum SisMatrixRole {
    A,
    B,
    D,
    F,
}

pub struct SisTableKey {
    pub policy: SisSecurityPolicyId,
    pub table_digest: SisTableDigest,
    pub modulus_profile: SisModulusProfileId,
    pub role: SisMatrixRole,
    pub ring_dimension: u32,
    pub coeff_linf_bound: u128,
}
```

The role is part of runtime validation and descriptor identity. It tells the
lookup which dimensions, coefficient bound cells, and rank limit are allowed.
The role is not part of the internal scalar estimator key.

### Role coverage

One canonical coverage declaration is shared by the planner, generator, tests,
and runtime validation. It is a list of required role cells:

```rust
pub struct SisRoleCell {
    pub role: SisMatrixRole,
    pub modulus_profile: SisModulusProfileId,
    pub ring_dimension: u32,
    pub coeff_linf_bound: u128,
    pub max_module_rank: u32,
    pub required_max_width: u64,
}
```

The initial coverage follows these rules:

- B and D include every ring dimension that the planner may assign to those
  matrices. For the 128 bit field this includes dimension 32.
- A includes every ring dimension that the planner may assign to A. Its current
  minimum is 64. Its cells may use larger dimensions than B and D.
- F has its own list. It does not inherit A, B, or D coverage without an
  explicit equality in the planner contract.
- A new mixed dimension planner choice must update the matching role coverage
  and generated cells in the same change.

The spec does not force the four role cell sets to be equal. The implementation
must use the actual planner domain as the source of truth. It must not form an
extra product of all dimensions and bounds within one role.

### Stored scalar shape

The generator expands every required role cell into rank requests:

```text
(role, modulus_profile, d, B, rank, required_width)
```

It maps each request to:

```text
n = rank * d
m_need = required_width * d
scalar_key = (modulus_profile, B, n)
```

It then takes the union of the scalar keys. If two role requests map to the
same scalar key, the generator estimates that cell once and records both role
origins in provenance.

The generated table has shape:

```text
(modulus_profile, B, n) -> ScalarCutoff
```

```rust
pub enum ScalarCutoff {
    Exact(u64),
    AtLeast(u64),
}
```

`Exact(m)` means that `m` passes and `m + 1` fails. `AtLeast(m)` means that `m`
passes and the search reached its cap. Runtime may accept only `m_need <= m` in
both cases.

### Reachable row dimensions

The generator does not create a dense base 32 grid. It derives the required set
from role coverage:

```text
REACHABLE_N = union { rank * d }
```

where the union ranges over every role, modulus profile, allowed ring dimension,
and allowed rank.

The generator must prove that every runtime role lookup maps to a generated
scalar cell. A missing required cell fails generation. A missing scalar value
that no supported role can request is not an error.

### Coefficient bound cells

The four roles do not share one forced coefficient ladder.

B and D use exact gadget anchors when their formulas produce
`2^log_basis - 1`. The initial anchors for `log_basis` from 2 to 8 are:

```text
3, 7, 15, 31, 63, 127, 255
```

A uses the raw bounds produced across the supported planner domain. The table
builder chooses upper buckets from those bounds. Bucket selection minimizes the
resulting rank or proof byte increase under a fixed table budget. A fixed
geometric ratio is not the definition of the A ladder.

F uses the bounds required by its own formula and planner domain.

Each role helper rounds a raw bound up within that role's allowed cells. The
generator stores the union of the resulting `(B, n)` requests. It does not
generate the full product of every role's bounds and every role's row
dimensions unless those cells are actually reachable.

Changing role bounds requires regenerated scalar cells and affected schedules
in the same change. The table digest and catalog identity must change.

### Search caps

Each scalar cell records the largest required `m_need` among its role origins.
The generator searches beyond that demand by the configured review margin, up
to the policy table cap:

```text
DEFAULT_M_SEARCH_CAP = 6_400_000_000_000
```

The cap is a generation limit, not an exact security boundary. A cap hit emits
`ScalarCutoff::AtLeast` and a review record. The table digest includes every
cell cap.

Generation must fail if a required runtime demand exceeds its cell cap.

### Runtime lookup

Given an audited `(policy, table_digest, modulus_profile, role, d, B, width)`:

```text
validate exact modulus profile
validate role permits d and B
m_need = checked_mul(width, d)
for rank in 1 ..= role_max_rank:
    n = checked_mul(rank, d)
    require scalar cell (modulus_profile, B, n)
    if m_need <= scalar_cutoff_value:
        return rank
reject
```

All arithmetic is checked. Overflow returns `AkitaError`. A missing required
cell, unsupported role geometry, unsupported policy, or table digest mismatch
also returns `AkitaError`.

`min_secure_rank` is the single canonical rank chooser. `AjtaiKeyParams`
constructors call it directly.

### Provenance

Generation provenance includes:

- policy ID and table digest;
- estimator revision and certificate domain;
- exact modulus values;
- norm, shape model, exponent, and target;
- every role origin for each scalar cell;
- accepted and rejected boundary scores with beta and zeta witnesses;
- monotonicity checks;
- exact or cap hit status;
- coefficient cell rules and role coverage;
- search caps and review margins.

The checked in table and audit artifact must have a shared digest.

The digest is SHA3-256 over the fixed UTF-8 domain tag
`akita-sis-table-digest-adps16-quantum-128bit\0`, followed in this order by the generated files
`q32.rs`, `q64.rs`, `q128.rs`, `policy_audit.csv`, and `policy_review.txt`.
Each file is encoded as an unsigned little endian 64-bit byte length, its
UTF-8 filename, a NUL byte, and its exact bytes. This encoding is independent
of host word size, map iteration order, and parallel generation order.

## Invariants

- The hard gate is the ADPS16 quantum score at 128 bits.
- A generic infinite estimate never passes.
- Every accepted boundary has a complete certificate.
- The estimator key contains the exact modulus, coefficient bound, scalar row
  count, and scalar column count.
- Runtime role keys may have different dimension and bound coverage.
- Identical scalar cells are generated once.
- Unreachable scalar cells are not required.
- Exact modulus profiles are checked against the configured field.
- Policy identity changes whenever acceptance semantics change.
- Table identity changes whenever coverage or generated data changes.
- Missing required cells and arithmetic overflow fail closed with `AkitaError`.
- Estimator work is offline. Verifier reachable code uses static tables and does
  not panic.

## Non goals

- Runtime lattice estimation.
- A shared ring dimension list for A, B, D, and F.
- A shared coefficient ladder for all roles.
- A dense base 32 row grid.
- Cell interpolation.
- Reusing a modulus profile for another modulus of the same size.
- Treating the scalar estimate as a proof against every structured attack.
- Compatibility with the replaced SIS policy identity.

## Evaluation

### Acceptance criteria

- [x] The only production security policy is `Quantum128BitADPS16`.
- [x] The estimator accepts only certified ADPS16 quantum scores at or above
      128.
- [x] Generic infinite and failed estimates stop generation.
- [x] The policy ID commits to all acceptance semantics.
- [x] Exact modulus profiles replace size only family selection.
- [x] Role coverage comes from the planner domain.
- [x] B and D cover every supported lower dimension, including dimension 32 for
      the 128 bit field.
- [x] A covers dimension 64 and every larger dimension the planner may choose.
- [x] F has explicit coverage.
- [x] The scalar table is the deduplicated union of required role cells.
- [x] The generator does not create unreachable dense grid cells.
- [x] Coefficient cells are selected per role.
- [x] Cap hits use `ScalarCutoff::AtLeast`.
- [x] Runtime lookup uses checked arithmetic and fails closed.
- [x] Generated tables, audit data, schedules, book text, and operational docs
      share the new identities and claim language.
- [x] Formatting, lint, tests, and documentation guardrails pass.

### Testing strategy

Pin the ADPS16 quantum estimator configuration:

```text
reduction model = ADPS16(mode = quantum)
shape model = LGSA
target = 128 bits
```

Test these cases:

- certified pass and fail results around 128;
- every unclassified infinite and numeric failure path;
- disagreement between discovery search and certificate search;
- security that does not increase as `m` grows;
- exact scalar equivalence across two role origins;
- exact modulus mismatch rejection;
- Q128 B and D requests at dimension 32;
- A requests at dimension 64 and above;
- role specific coefficient rounding;
- a missing required role cell;
- an omitted unreachable scalar cell;
- exact and cap hit cutoffs;
- multiplication overflow in runtime lookup;
- table and audit digest agreement.

### Performance

Runtime performs static table lookup only.

Offline generation estimates one copy of each required scalar cell. Discovery
may use the local search. Certification uses the exhaustive configured search
at the accepted and rejected boundary.

## Design notes

### Architecture

```text
Quantum128BitADPS16
        |
        +-- hard gate: ADPS16 quantum score >= 128
        |
role coverage from planner
        |
        +-- A dimensions and A coefficient cells
        +-- B dimensions and B coefficient cells
        +-- D dimensions and D coefficient cells
        +-- F dimensions and F coefficient cells
        |
union and scalar deduplication
        |
(modulus_profile, B, n) -> Exact(max_m) or AtLeast(cap)
        |
runtime role lookup: n = rank*d, m_need = width*d
```

### Transition

1. Keep this file as the only live production SIS policy and table design
   record.
2. Replace policy and modulus identity types in one cutover.
3. Add canonical role coverage shared by planner, generator, tests, and runtime.
4. Generate the union of required scalar cells.
5. Add boundary certificates and classified estimate results.
6. Regenerate production tables and affected schedules.
7. Update the book and operational docs.
8. Mark this spec implemented after all checks pass.

### Alternatives considered

| Option | Verdict |
|--------|---------|
| Keep a second quantum review line | Rejected because the production policy has one hard ADPS16 quantum gate |
| Accept generic infinite estimates | Rejected because one value covers both high cost and numeric failure |
| Use one dimension list for every matrix | Rejected because mixed dimension planning gives the roles different domains |
| Use one coefficient ladder for every matrix | Rejected because the role formulas and useful cells differ |
| Generate every multiple of 32 to one global maximum | Rejected because most cells are unreachable |
| Put role in the scalar estimator key | Rejected because role does not change an identical scalar instance |
| Use field bit length as modulus identity | Rejected because security depends on the exact modulus |
| Use only local search for final acceptance | Rejected because a missed attack can overstate security |

### Change control

Changing the hard target, ADPS16 mode, norm, shape model, estimator revision,
certificate domain, or estimate result semantics requires a new policy ID.

Changing role dimensions, role bounds, rank limits, exact modulus profiles,
search caps, or generated cells requires a new table digest and regenerated
dependents. A modulus value change also requires a new modulus profile ID.

## Documentation

This file is the single live design record for production SIS policy, role
coverage, and scalar table geometry. The estimator crate spec may describe
estimator APIs, but it must not redefine production acceptance.

Durable narrative belongs in `book/src/how/security.md`.

## References

- ADPS16 reduction and quantum cost implementation in the pinned
  `third_party/lattice-estimator` checkout used by the estimator goldens.
- Langlois, Stehle, *Worst Case to Average Case Reductions for Module Lattices*,
  [IACR ePrint 2012/090](https://eprint.iacr.org/2012/090).
- [`sis-infinity-estimator-crate.md`](sis-infinity-estimator-crate.md), Rust
  infinity estimator profiles and reduction model APIs.
