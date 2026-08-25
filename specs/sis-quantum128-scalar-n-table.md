# Spec: SIS ADPS16 Quantum 128-Bit Policy and Role Driven Scalar Table

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-13 |
| Status        | implemented |
| PR            | |
| Supersedes    | the SIS policy and table specs deleted in this cutover |
| Superseded-by | |
| Book-chapter  | book/src/how/security.md |

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
and ranks the generator must cover directly.

A, B, D, and compression do not share one forced geometry. Production B and D commitment
matrices use ring dimensions 64 or larger. A currently uses dimension 64 or
larger and may use dimensions above the other matrices. Smaller dimensions are
limited to the separate fixed compression cells. The generator filters requests
through the canonical role coverage, then deduplicates two requests only when
they produce the same scalar SIS key.

Policy identifier:

```text
Quantum128BitADPS16V2
```

The old policy identity and scalar `min_security_bits` identity are removed in
the same cutover. Unsupported policy and table identities fail closed.

### Delivered implementation parameters

The implementation pinned by this specification uses ADPS16 quantum exponent
`0.2650`, LGSA shape, coefficient `L-infinity` norm, target `128.0`, maximum
module rank `20`, and a per-cell search cap of `6_400_000_000_000`. The exact
modulus profiles are `Q32Offset99`, `Q64Offset59`, and `Q128OffsetA7F7`.
Production arithmetic otherwise uses the documented `f64` backend. Integer
small-box branch boundaries are compared exactly, with a fast log-space
precheck away from equality. The unimplemented high-precision backend fails
closed rather than silently executing in `f64`.

The canonical role coverage has Inner/A dimensions `64, 128, 256, 512, 1024,
2048` for q32, `64, 128, 256, 512, 1024` for q64, and `64, 128, 256, 512` for
q128. Outer/B and Open/D have dimensions `64, 128, 256`. The separate fixed
compression cells use their protocol-specific dimensions. Every
commitment-matrix cell has maximum module rank `20`. Inner/A uses the explicit
planner bucket union
`2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383,
32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607,
16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823,
2147483647, 4294967295, 8589934591, 17179869183, 34359738367, 68719476735,
137438953471, 274877906943, 549755813887, 1099511627775, 2199023255551,
4398046511103, 8796093022207, 17592186044415`. The q32 profile stops at
`268435455`, q64 stops at `2199023255551 = 2^41 - 1`, and q128 stops at
`17592186044415 = 2^44 - 1`.
Outer/B and Open/D use the exact gadget anchors
`3, 7, 15, 31, 63, 127, 255`.

The generator accepts the complete rectangular CLI domain for reproducibility,
then discards every dimension and bound pair that is not reachable from the
three commitment-matrix roles or the fixed compression cells. The checked-in
Rust files contain the resulting scalar union. The checked-in audit records
each ring origin, including its accepted and rejected boundary witnesses. The
runtime projection takes the minimum scalar cutoff when different ring origins
map to the same scalar key.

The checked-in production audit contains 10,980 ring-origin rows: 3,360 for
q32, 4,100 for q64, and 3,520 for q128. The q128 Inner/512 cell has 880 direct
estimator requests: 44 coefficient buckets times 20 module ranks.

### Quantum cost-model disposition

The `0.2650` exponent is a deliberate conventional Core-SVP policy, not an
oversight about newer asymptotic quantum sieves. BCSS23 reports
`2^(0.2563 * beta + o(beta))` time by reusing quantum walks across collision
searches. Akita does not promote that exponent to the production gate because
the transfer assumes heuristic asymptotics, exponential reusable-sieve
storage, writable QRAQM with coherent reads and writes, unit-cost or
polylogarithmic-cost coherent access, and then a transfer from the idealized
SVP oracle to BKZ and repeated infinity-norm short-vector generation. It is not
a concrete fault-tolerant resource estimate.

This disposition was evaluated rather than inferred by exponent rescaling. At
commit `00bc2210877c8a8f6bbc46bdbef300f9fa437457`, the generator independently
optimized the ADPS16 classical, ADPS16 quantum, and idealized BCSS23 models for
6,240 table rows. It used 124 bits as the BCSS review line because
`128 * 0.2563 / 0.2650 = 123.80...`; zero accepted ADPS16-quantum rows fell
below that review line. Commit `6384b57756b9116127c70ea397388096b2a420da`
then removed the non-gating BCSS implementation and audit columns as unused
production scaffolding. Regeneration under an additional BCSS gate is therefore
not required by this policy.

Promoting BCSS23 or another idealized quantum sieve to a hard constraint
requires a new policy identifier. The review must address finite-dimensional
costs, quantum memory size and access, fault-tolerant implementation, the BKZ
oracle transfer, and measured rank or proof-size impact; a smaller asymptotic
exponent alone is insufficient. The ADPS16 paranoid `0.2075 * beta` list-size
line is likewise not an end-to-end attack-time estimate.

### Shape-model disposition

LGSA models an attacker rerandomizing the q-ary basis so BKZ forgets the
canonical q-vectors. It is the production shape because the attacker may choose
that basis and, in the small-box branch used by the widened q64 and q128 rows,
LGSA's clipped profile has a first Gram-Schmidt vector no longer than ordinary
GSA at the same `(beta, zeta)`. A shorter first vector only increases the
modeled coefficient-wise success probability. When the LGSA unit tail
disappears, LGSA and GSA coincide exactly.

The pinned Sage estimator at
`c667a48546f140c3a5454c7503c3ca44a264cce2` was also used for an offline
profile comparison on the proposed widened representative rows. Independent
local beta and zeta optimization produced:

| Scalar SIS row | LGSA | GSA | CN11 | CN11 after forgetting q-structure |
|---|---:|---:|---:|---:|
| q64, `n=1024`, `m=1810`, `B=2^41-1` | 130.910 (`beta=494`, `zeta=0`) | 130.910 | 132.235 (`beta=499`, `zeta=0`) | 132.235 |
| q128, `n=1024`, `m=4096`, `B=2^44-1` | 172.515 (`beta=651`, `zeta=1`) | 172.515 | 173.045 (`beta=653`, `zeta=0`) | 173.045 |

Thus LGSA is the cheapest modeled attack among those determinant-preserving
profiles on both representative rows. CN11 remains an offline audit oracle: a
single full local q128 optimization took about 85 seconds while its
forget-q-structure variant took about 107 seconds on the audit machine, making
either inappropriate for the production table sweep.

The pinned ZGSA implementation is not a valid counterexample on arbitrary
unbalanced shapes. Its symmetric transition preserves determinant only when
there are at least as many identity vectors as q-vectors. On the q64 row above,
the profile loses 5,485.85 bits of log2 lattice volume at `beta=494` and creates
a spurious 90.895-bit result. The Rust compatibility path must reject that
domain rather than treat it as an attack. On supported ZGSA inputs, generated
profiles must preserve `log2(det Lambda) = n * log2(q)`.

The probability regime is also defined on the reduced instance. After the
attacker projects away `zeta` coordinates, the active dimension is
`d_eff = d - zeta`; therefore the small-box test is
`sqrt(d_eff) * B <= q`, not `sqrt(d) * B <= q`. The pinned Sage implementation
uses the original dimension at this branch. That is not a conservative choice
in general: an audit of 2,562 representative q32 cells found 35 cells where
the original-dimension branch reported a lower trial probability and hence a
higher attack cost. For example, `n=1024`, `d=65537`, `B=2^24-1`,
`beta=343`, and `zeta=57345` has `d_eff=8192` and a corrected quantum attack
cost of 118.916 bits. A separate audit of the 40 current exact q32 table
boundaries exposed to this branch change found no accepted/rejected boundary
reversal. The tables must nevertheless be regenerated after the search and
reach changes because that boundary sample is not a replacement for full
generation and certification. For integer production bounds, the corrected
branch comparison falls back to exact integer arithmetic at the boundary.

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
norm LGSA optimizer under the dedicated ADPS16 quantum cost model with exponent
`0.2650`.

Base-table generation uses `local-minimum` discovery, then certifies its
accepted boundary and immediate rejected successor with proven-pruned beta and
full-valid-domain zeta search. The valid tall q-ary domain is
`0 <= zeta < d - n`; an effective dimension `d - zeta <= n` is not an SIS
lattice instance priced by this attack model. The decision threshold is an
explicit estimator configuration value supplied by the policy profile. The
beta search visits values from 40 through the capped Euclidean baseline and
stops once the monotone ADPS16 reduction-cost lower bound exceeds the best
complete candidate. When the best visited attack and the lower bound for every
unvisited beta both exceed 128 bits, the estimator returns a classified
above-target result instead of representing the much larger exact cost. For
`B > 1`, the global infinity estimate explicitly takes the minimum with that
ordinary Euclidean SIS attack: any vector with `L2 <= B` also has
`L-infinity <= B`. Thus the Euclidean beta is both included as a real attack
and provides the monotone upper endpoint for the beta sweep; it is not merely
used as a heuristic search cutoff. For `B <= 1`, Euclidean dimension optimization is
undefined. The separate diagnostic compression table contains the production
`B = 1` instances of this edge case; those cells omit the Euclidean candidate
without substituting `B = 2` and sweep the full supported beta range instead.
For fixed beta under ADPS16/LGSA,
the search scans every effective dimension before the profile stabilizes; the
modeled stable tail adds only unit vectors and is minimized at one of its two
endpoints within either probability regime. If the active-dimension small-box
condition changes inside that tail, the search also checks the two dimensions
straddling the transition. This covers the wide D512 domain without changing
its width policy.

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

The scalar cutoff search starts at the first tall Module-SIS geometry
`width = rank + 1`. If that instance fails, the row records cutoff zero rather
than assigning an attack cost to a square or wide matrix. If it passes, smaller
widths inherit security from the certified tall instance by column restriction.

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

Any change to the hard model that can change whether the same scalar SIS cell
passes requires a new policy ID and regenerated artifacts. A change to the
search profile for a table extension requires a new table digest.

The active-dimension correction, complete pre-stable search, and explicit
Euclidean candidate in this hardening patch can change that decision. This
regeneration therefore introduces the revisioned `Quantum128BitADPS16V2`
runtime policy ID and a new table digest. Every dependent schedule is
regenerated in the same atomic cutover.

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
    Inner,
    Outer,
    Open,
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
  matrices. Their minimum production commitment dimension is 64.
- A includes every ring dimension that the planner may assign to A. Its current
  minimum is 64. Its cells may use larger dimensions than B and D.
- Q128 additionally exposes Inner/A dimension 512 for future mixed-ring use.
  This cell has ranks `1..=20` and is gated by its extension digest.
- Compression has its own fixed cells outside `SisMatrixRole`. It does not
  inherit commitment-matrix coverage.
- A new mixed dimension planner choice must update the matching role coverage
  and generated cells in the same change.

The spec does not force the three role cell sets or compression cells to be
equal. The implementation must use the actual planner domain as the source of
truth. It must not form an extra product of all dimensions and bounds within
one role.

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

A uses the explicit planner bucket set listed in the delivered implementation
parameters. The set is a deliberate collision-bucket contract, not an implicit
geometric ladder and not a runtime interpolation rule. If planner workloads
ever require a bound outside the set, the coverage and generated table must be
updated together.

F uses the bounds required by its own formula and planner domain.

Each role helper rounds a raw bound up within that role's allowed cells. The
generator stores the union of the resulting `(B, n)` requests. It does not
generate the full product of every role's bounds and every role's row
dimensions unless those cells are actually reachable.

Changing role bounds requires regenerated scalar cells and affected schedules
in the same change. The table digest and catalog identity must change.

### Search caps

The current generator uses the configured policy table cap for every scalar
cell:

```text
DEFAULT_M_SEARCH_CAP = 6_400_000_000_000
```

The cap is a generation limit, not an exact security boundary. A cap hit emits
`ScalarCutoff::AtLeast` and a review record. The table digest includes every
cell cap. The role-cell `required_max_width` field is coverage metadata and
does not silently raise or lower this cap.

Generation must fail if a required runtime demand exceeds its cell cap.

### Runtime lookup

Given an audited `(policy, table_digest, modulus_profile, role, d, B, width)`:

```text
validate exact modulus profile
validate role permits d and B
validate digest permits d
widths = require generated slice at (profile, d, B)
for (rank, max_width) in widths:
    if width <= max_width:
        return rank
reject
```

A missing required cell, unsupported role geometry, unsupported policy, or
table digest mismatch returns `AkitaError`.

`min_secure_rank` is the single canonical rank chooser. `AjtaiKeyParams`
constructors call it directly.

### Provenance

Generation provenance includes:

- policy ID and table digest;
- estimator revision and certificate domain;
- exact modulus values;
- norm, shape model, exponent, and target;
- the canonical role coverage from which every scalar origin is reconstructed;
- accepted and rejected boundary status for each scalar cell;
- monotonicity checks;
- exact or cap hit status;
- coefficient cell rules and role coverage;
- search caps and review margins.

Long generation may be split into deterministic, content-addressed work items.
Each completed result is immutable and commits to the evaluator identity and
canonical evaluator input. Partial runs may checkpoint or exchange any subset
of these results, but they are not tables. Runtime artifacts may be assembled
only after every work item required by the requested coverage is present and
the complete row set passes certificate, monotonicity, and coverage validation.
Changing security-relevant evaluator behavior changes the evaluator identity,
so results from the prior computation cannot silently satisfy the new plan.

The checked in table and audit artifact must have a shared digest. The audit
records accepted and rejected beta and zeta witnesses for every generated ring
origin. Role admission remains canonical runtime data and is committed
indirectly through the set of generated origins and the policy review.

The digest is SHA3-256 over the fixed UTF-8 domain tag
`akita-sis-table-digest-adps16-quantum-128bit\0`, followed in this order by the generated files
`q32.rs`, `q64.rs`, `q128.rs`, `policy_audit.csv`, and `policy_review.txt`.
Each file is encoded as an unsigned little endian 64-bit byte length, its
UTF-8 filename, a NUL byte, and its exact bytes. This encoding is independent
of host word size, map iteration order, and parallel generation order.

The q128 Inner/512 cells are part of the unified artifact and its shared table
digest. Dependent schedule catalogs embed that same digest.

## Invariants

- The hard gate is the ADPS16 quantum score at 128 bits.
- A generic infinite estimate never passes.
- Every non-cap accepted boundary has a complete certificate; cap rows are
  explicitly marked `AtLeast` and require review.
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

- [x] The only production security policy is `Quantum128BitADPS16V2`.
- [x] The estimator accepts only certified ADPS16 quantum scores at or above
      128.
- [x] Generic infinite and failed estimates stop generation.
- [x] The policy ID commits to all acceptance semantics.
- [x] Exact modulus profiles replace size only family selection.
- [x] Role coverage comes from the planner domain.
- [x] B and D cover every supported commitment dimension, starting at 64.
- [x] A covers dimension 64 and every larger dimension the planner may choose.
- [x] Compression has explicit fixed-cell coverage.
- [x] The scalar table is the deduplicated union of canonical reachable role
      cells.
- [x] The generator filters unreachable rectangular CLI requests before
      estimation and does not emit them.
- [x] Coefficient cells are selected per role from the explicit A and gadget
      anchor sets.
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
- rejection of B and D commitment requests below dimension 64;
- A requests at dimension 64 and above;
- role specific coefficient rounding;
- a missing required role cell;
- an omitted unreachable scalar cell;
- exact and cap hit cutoffs;
- multiplication overflow in runtime lookup;
- table and audit digest agreement.

### Performance

Runtime performs static table lookup only.

Offline generation certifies every reachable ring origin. The runtime
projection deduplicates equal scalar keys. Every cell retains exhaustive-beta,
proven-pruned-zeta boundary certification. The checked-in artifact is generated
offline; runtime never invokes the estimator.

## Design notes

### Architecture

```text
Quantum128BitADPS16V2
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
| Use only local search for every existing cell | Rejected because it would unnecessarily change base-table certification |
| Derive a dimension's rows from another dimension | Rejected because each supported dimension is generated directly |

### Change control

Changing the hard target, ADPS16 mode, norm, shape model, estimator revision,
or estimate result semantics requires a new policy ID.

Changing role dimensions, role bounds, rank limits, exact modulus profiles,
search or certificate profiles, search caps, or generated cells requires a
new table digest. Generated
dependents change only when the base generated table or a schedule-selected
digest changes. A modulus value change also requires a new modulus profile ID.

## Documentation

This file is the single live design record for production SIS policy, role
coverage, and scalar table geometry. The estimator crate spec may describe
estimator APIs, but it must not redefine production acceptance.

Durable narrative belongs in `book/src/how/security.md`.

## References

- ADPS16 reduction and quantum cost implementation in the pinned
  `third_party/lattice-estimator` checkout used by the estimator goldens.
- Bonnetain, Chailloux, Schrottenloher, Shen, *Finding Many Collisions via
  Reusable Quantum Walks*, [IACR ePrint 2022/676](https://eprint.iacr.org/2022/676).
- Cho, Hhan, Kim, Lee, Shen, *Does Quantum Lattice Sieving Require Quantum
  RAM?*, [IACR ePrint 2024/1700](https://eprint.iacr.org/2024/1700).
- Ducas et al., *CRYSTALS-Dilithium*,
  [round-3 specification](https://pq-crystals.org/dilithium/data/dilithium-specification-round3-20210208.pdf),
  Appendix C.3.
- Chen, Nguyen, *BKZ 2.0: Better Lattice Security Estimates*, ASIACRYPT 2011.
- Ducas, van Woerden, *NTRU Fatigue*,
  [IACR ePrint 2021/999](https://eprint.iacr.org/2021/999).
- Langlois, Stehle, *Worst Case to Average Case Reductions for Module Lattices*,
  [IACR ePrint 2012/090](https://eprint.iacr.org/2012/090).
- `crates/akita-sis-estimator/` — Rust infinity estimator profiles and
  reduction model APIs.
