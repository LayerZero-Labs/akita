# Spec: SIS Classical-138 / Quantum-128 Policy with Idealized BCSS Diagnostic

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-12 |
| Status        | approved |
| PR            | |
| Supersedes    | |
| Superseded-by | |
| Book-chapter  | |

## Summary

Akita's checked-in SIS width table currently enforces one 138-bit classical
ADPS16/LGSA estimate. That is a useful floor, but the scalar
`min_security_bits` name does not say which attack model it covers and does not
support an honest post-quantum claim. This spec adopts a descriptive, versioned
policy with two hard constraints on every production row:

1. at least 138 bits under the ADPS16 classical Core-SVP cost
   `log2(rop) = 0.2920 * beta`; and
2. at least 128 bits under the conventional ADPS16 quantum Core-SVP cost
   `log2(rop) = 0.2650 * beta`.

The policy also records one non-gating diagnostic: the BCSS quantum sieving cost
`log2(rop) = 0.2563 * beta`. BCSS is deliberately not called simply "the
quantum estimate": it relies on idealized asymptotic and writable-QRAQM
assumptions described below. A BCSS score below 124 bits requires explicit
security review, but does not automatically change a generated table.

The descriptive policy identifier proposed for code and generated artifacts is
`Classical138Quantum128WithIdealizedBcssV1`. Do not shorten this to
`Balanced`, `Default`, `Standard`, or another name that hides the actual
constraints.

## Intent

### Goal

Make the SIS security claim, generated-table acceptance rule, estimator
provenance, and runtime identity agree on one versioned policy: hard classical
138-bit and conventional-quantum 128-bit estimates, plus exactly one explicitly
idealized BCSS diagnostic.

### Decision

For a candidate SIS table row, run the complete infinity-norm LGSA optimizer
separately under each cost model. Let `C_model(row)` be the minimum finite
`log2(rop)` found over the model's supported `(beta, zeta)` search space. The row
is production-eligible exactly when both hard predicates hold:

```text
C_ADPS16_classical(row) >= 138
C_ADPS16_quantum(row)   >= 128
```

The production cutoff for each module rank is the largest width satisfying the
intersection. A generator must not infer the quantum score by multiplying the
classical result by an exponent ratio: changing the cost model can change the
optimizing `beta`, `zeta`, short-vector count, or feasibility boundary.

For every accepted boundary and its first rejected successor, also compute:

```text
C_BCSS_idealized(row), using log2(rop) = 0.2563 * beta
```

This value is provenance and review evidence, not a third production
predicate. If an accepted production boundary has `C_BCSS_idealized < 124`,
generation must flag the row for manual review and the policy/table update must
not be merged without a written disposition in the PR or a follow-up design
record. The value 124 is a review line, not a claim of 124-bit security. It is
the rounded comparison point suggested by applying the BCSS-to-ADPS16 exponent
ratio to the 128-bit conventional quantum target:

```text
128 * 0.2563 / 0.2650 = 123.80...
```

The diagnostic must nevertheless be computed by the full optimizer. The ratio
only explains the review line.

### Claim language

Until this spec is implemented and the tables are regenerated, accurate public
language is:

> Akita's current generated SIS table enforces a 138-bit classical ADPS16/LGSA
> estimate. A joint classical/quantum policy has been approved but is not yet
> enforced by the checked-in table.

After implementation and regeneration, accurate language is:

> Akita's generated SIS table targets at least 138 classical bits and 128 bits
> under the conventional ADPS16 quantum Core-SVP model. Akita separately
> reports an idealized BCSS writable-QRAQM diagnostic.

Do not compress the second statement into an unqualified "128-bit
post-quantum security" claim. It is a concrete estimator policy, not a proof
that all quantum attacks cost at least `2^128`, and the BCSS result is not a
physical qubit, gate, depth, or wall-clock estimate.

### Why BCSS is diagnostic rather than gating

BCSS improves the asymptotic quantum sieve exponent by reusing a prepared
quantum data structure across collision searches. In the high-level sieve
picture, a list of about `2^(0.2075 beta)` vectors is prepared, and quantum
walks search for enough reducing pairs without repaying the entire setup cost
for every collision. Optimizing the list filters, update/check costs, and walk
parameters yields the asymptotic `2^(0.2563 beta)` time exponent.

That is meaningful cryptanalytic evidence and must not be hidden. It is not,
however, a neutral drop-in replacement for the conventional quantum hard gate.
The estimate assumes all of the following:

- heuristic random-sphere and lattice-sieving behavior;
- the asymptotic regime, omitting lower-order and finite-dimensional constants;
- exponential classical list storage of roughly `2^(0.2075 beta)` entries;
- writable quantum random-access memory (QRAQM), including coherent reads and
  writes to the sieve data structure in superposition;
- unit-cost or polylogarithmic-cost QRAQM access in the augmented gate model;
- coherent quantum memory and data structures large and reliable enough to
  support the reusable quantum walks; and
- transfer of the exact-SVP/sieving oracle cost to the BKZ block oracle and
  then to Akita's repeated-short-vector infinity-norm SIS estimator.

These are idealized algorithmic assumptions, not engineering estimates for a
fault-tolerant quantum computer. Work on quantum speedups without QRAM also
supports keeping the memory model visible: in broad query models, the speedup
depends on QRAM or on a sufficiently capable bounded replacement. Because
there is unresolved uncertainty in both directions—ignoring BCSS can
overstate security, while making it a hard gate can charge present-day Akita
for an idealized machine—the model is a single visible diagnostic with a
review line.

### Invariants

- **Both hard constraints price the same protocol bound.** Classical and
  conventional quantum estimation consume the same modulus, ring dimension,
  coefficient-`L∞` bound, width, shape model, and short-vector semantics that
  production verification enforces.
- **Hard constraints form an intersection.** A row is accepted only when both
  the 138-bit classical and 128-bit conventional quantum predicates pass. No
  averaging or weighted score may admit a row that fails either predicate.
- **BCSS never silently changes ranks.** The idealized BCSS score and its review
  status are recorded, but table acceptance is unchanged unless a later,
  explicit policy revision promotes BCSS to a hard constraint.
- **Exactly one quantum diagnostic.** This policy does not expose a ladder of
  "paranoid", Chailloux-Loyer, practical-resource, or other optional scores in
  normal table output. Further models belong in research tooling or a policy
  revision, not the production policy surface.
- **No exponent rescaling shortcut.** Each hard model and BCSS run the complete
  optimizer. Boundary rows retain their own `(beta, zeta)` witnesses and
  margins.
- **Policy is part of identity.** Runtime SIS keys, Ajtai descriptors, generated
  table metadata, and any catalog or schedule identity affected by SIS sizing
  encode the versioned policy, not only a scalar bit count.
- **Unsupported policies fail closed.** Verifier-reachable lookup does not
  substitute the current table, round to another policy, or run the estimator.
  It rejects an unavailable policy using `AkitaError`.
- **Offline estimation only.** The model intersection and BCSS diagnostic run
  during generation. Proving and verification consume checked-in tables.
- **Reproducible provenance.** Generated artifacts identify the estimator
  revision, optimizer profile, all exponents, targets, the BCSS diagnostic
  status, and a digest of the canonical policy and table inputs.
- **Changing assumptions changes the version.** A new exponent, target,
  optimizer, shape model, norm interpretation, or promotion of BCSS from
  diagnostic to hard gate creates a new policy identifier and regenerated
  artifacts.

### Non-Goals

- Claiming concrete fault-tolerant quantum resources from a Core-SVP exponent.
- Treating BCSS as either irrelevant or fully realistic.
- Adding multiple diagnostic columns to routine planner or verifier output.
- Retaining the ADPS16 `Paranoid` exponent as a production policy component.
- Settling whether writable QRAQM of exponential scale can be built.
- Running lattice estimation in verifier-reachable code.
- Preserving compatibility with scalar `min_security_bits` descriptors.
- Replacing the estimator's LGSA or coefficient-`L∞` SIS model in this policy.

## Evaluation

### Acceptance Criteria

- [ ] Add a public, versioned SIS policy identifier named descriptively, with
      `Classical138Quantum128WithIdealizedBcssV1` as the initial policy.
- [ ] Replace production identity that depends only on
      `min_security_bits = 138` with identity that commits to the policy.
- [ ] Generate every production cutoff from the intersection of full
      classical and conventional quantum optimizer runs.
- [ ] Add a distinct BCSS reduction-cost model with exponent `0.2563`; do not
      overload `Adps16Mode::Quantum` or `Adps16Mode::Paranoid`.
- [ ] Record BCSS scores, optimizer witnesses, margins, and whether the 124-bit
      review line was crossed for each generated boundary pair.
- [ ] Keep the checked-in runtime table compact: the BCSS diagnostic may live
      in generation CSV/metadata rather than verifier-facing row data.
- [ ] Make generation visibly fail or produce a review-blocking result when an
      accepted hard-policy boundary is below the BCSS review line.
- [ ] Add fixed-cost tests for all three policy exponents and optimizer tests
      showing that all models are independently optimized.
- [ ] Add a regression where the row passing classical 138 fails conventional
      quantum 128, proving the table generator takes the intersection.
- [ ] Add a regression where BCSS falls below 124 but hard constraints pass,
      proving the row is flagged rather than silently rejected or accepted
      without provenance.
- [ ] Regenerate all production SIS tables and any schedules whose selected
      ranks change; retain the existing pure-DP/table expansion drift guard.
- [ ] Update the book's security language from the pre-implementation wording
      to the post-implementation wording only after regenerated tables land.
- [ ] Run `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D
      warnings`, `cargo test`, and `./scripts/check-doc-guardrails.sh`.

### Testing Strategy

Unit tests should pin the exact log-cost conventions:

```text
classical:             beta=500 -> 146.00 bits
conventional quantum:  beta=500 -> 132.50 bits
idealized BCSS:        beta=500 -> 128.15 bits
```

Golden estimator tests should use representative production-shaped rows across
the Q32, Q64, and Q128 modulus families. They must compare independently
optimized scores and witnesses, not only the fixed-beta formulas. Width-table
tests should cover both hard predicates passing; classical passing while
conventional quantum fails; BCSS above and below its review line; and an
unsupported policy identifier.

The generated-table boundary audit remains two-sided: record the last accepted
width and the first rejected width under each hard model. Cap-hit rows remain
explicit lower bounds as required by
[`sis-linf-table-cutover.md`](sis-linf-table-cutover.md).

### Performance

The runtime target is no additional estimator work and negligible lookup cost:
policy identity replaces scalar-floor identity, while generated table lookup
remains static. Generated artifacts need not carry per-row diagnostic fields
used only during review.

Offline generation will cost approximately three optimizer executions per
candidate instead of one. The generator should share only model-independent
precomputation; it must not cache a model-dependent optimum across exponents.
Parallel row generation remains appropriate.

The efficiency objective is to pay in production ranks only for the two stated
hard constraints. BCSS may motivate human review, research, or a future policy
revision, but it does not impose an automatic runtime or proof-size tax under
this version.

## Design

### Architecture

The implementation should preserve one source of truth for policy semantics:

```text
Classical138Quantum128WithIdealizedBcssV1
                 |
                 +-- hard: ADPS16 classical >= 138
                 +-- hard: ADPS16 quantum   >= 128
                 +-- diagnostic: BCSS idealized; review below 124
                                   |
offline generator -> provenance + generated intersection table
                                   |
runtime policy key -> table lookup -> Ajtai rank/descriptor/catalog identity
```

The canonical policy definition belongs in the estimator/security boundary,
not as repeated constants in the CLI, generator, and runtime crates. Runtime
code may use a compact stable identifier, while offline code resolves that
identifier to complete model configuration.

The current `SisTableKey { min_security_bits, ... }` and generated table guard
support only a scalar 138-bit floor. This spec intentionally permits a breaking
cutover. A target shape is:

```rust
pub enum SisSecurityPolicyId {
    Classical138Quantum128WithIdealizedBcssV1,
}

pub struct SisTableKey {
    pub policy: SisSecurityPolicyId,
    pub family: SisModulusFamily,
    pub ring_dimension: u32,
    pub coeff_linf_bound: u128,
}
```

The exact type placement is an implementation choice, but no thin wrapper or
parallel scalar-floor API should remain. Type methods may assemble canonical
arguments; policy logic must not be copied between crates.

Generation metadata should include at least the policy ID and serialization
version; estimator revision and command; norm, shape, and optimizer modes;
model exponents and targets; accepted/rejected scores and `(beta, zeta)`
witnesses; BCSS review status; cap-hit status; and a digest covering policy,
inputs, and output table.

The policy identifier must flow through the same descriptor and catalog paths
that currently commit to `min_security_bits`. If a policy changes selected
ranks, generated schedules must be regenerated as a coherent snapshot.

### Transition Plan

1. Add the BCSS cost model and policy type to `akita-sis-estimator`, with unit
   and representative-row tests.
2. Extend width generation to evaluate the two hard models independently,
   intersect their accepted widths, and emit the one BCSS diagnostic.
3. Review representative Q32/Q64/Q128 rows before starting a full table sweep.
   Record rank/proof-size deltas relative to the current classical-only table.
4. Replace scalar-floor runtime identity with the descriptive policy ID and
   remove obsolete pass-through APIs in the same cutover.
5. Regenerate production tables and affected schedules, then run the complete
   validation suite.
6. Update this spec to `implemented`, attach the PR, check the criteria, and
   update the book to the post-implementation claim.

### Alternatives Considered

#### A policy named `Balanced`

Rejected. It is subjective and gives no clue which targets or models are
enforced. A descriptive identifier is longer but makes logs, descriptors,
generated metadata, and review discussions self-explanatory.

#### Make BCSS a third hard constraint at 128 bits

Rejected for this version. That treats the ideal writable-QRAQM augmented-gate
model as the expected attacker and can impose avoidable rank and proof-cost
increases. BCSS remains visible and reviewable without silently converting an
uncertain asymptotic model into production cost.

#### Ignore BCSS

Rejected. It is the strongest directly relevant published asymptotic quantum
sieve exponent considered here. Omitting it would make the uncertainty
one-sided and encourage an overbroad post-quantum claim.

#### Report several quantum diagnostics

Rejected. A ladder of close exponents obscures the decision and invites users
to select whichever number supports a preferred conclusion. This policy keeps
one conventional hard gate and one clearly labeled idealized diagnostic.

#### Use the ADPS16 paranoid exponent as the diagnostic

Rejected. The `0.2075 * beta` list-size exponent is not a supported end-to-end
attack-time estimate for this policy. Presenting it alongside time estimates
without its different meaning is more confusing than conservative.

#### Derive all scores by exponent rescaling

Rejected. The optimizer can select different attack parameters under different
cost functions. Full optimization is required both for correctness and for
useful provenance.

### Promotion and Change Control

Promoting BCSS to a hard gate requires a new policy identifier and a focused
design review. Evidence should address at least finite-dimensional performance,
memory and coherent-access accounting, the exact QRAM/QRAQM model, integration
with BKZ and repeated short-vector generation, and production rank/proof-size
impact. A better exponent alone is insufficient.

Likewise, weakening or removing either hard target requires a new policy and
an explicit claim update. Existing generated artifacts are immutable snapshots
of their named policy; they must not acquire changed semantics under the same
identifier.

## Documentation

While implementation is pending, the Akita Book should link this approved spec
and continue to state that the checked-in table enforces only the current
138-bit classical estimate. The implementation PR owns the durable explanation
in `book/src/how/security.md`; after that fold, this spec should be marked
`implemented` and archived under the normal pruning policy.

`AGENTS.md` does not need a new operational command for the design-only PR. The
existing offline generation command remains authoritative until implementation
changes its policy selection or output format.

## References

- Albrecht, Ducas, Pöppelmann, Schwabe, *Post-quantum Key Exchange—A New Hope*,
  USENIX Security 2016 / [IACR ePrint 2015/1092](https://eprint.iacr.org/2015/1092).
  Source of the rounded ADPS16 classical and quantum Core-SVP conventions.
- Laarhoven, *Search Problems in Cryptography: From Fingerprinting to Lattice
  Sieving*, PhD thesis, 2016
  ([TU Eindhoven record](https://research.tue.nl/en/publications/search-problems-in-cryptography-from-fingerprinting-to-lattice-si)).
  Background for the conventional quantum lattice-sieving exponent.
- Bonnetain, Chailloux, Schrottenloher, Shen, *Finding Many Collisions via
  Reusable Quantum Walks: Application to Lattice Sieving*, EUROCRYPT 2023 /
  [IACR ePrint 2022/676](https://eprint.iacr.org/2022/676). Source of the
  idealized `0.2563 * beta` BCSS time exponent and reusable-walk construction.
- Cho et al., *Does Quantum Lattice Sieving Require QRAM?*,
  [IACR ePrint 2024/1700](https://eprint.iacr.org/2024/1700). Context for making
  the quantum memory assumption explicit rather than treating it as hidden.
- [`sis-infinity-estimator-crate.md`](sis-infinity-estimator-crate.md) — Rust
  infinity-norm estimator design and reduction-model surface.
- [`sis-linf-table-cutover.md`](sis-linf-table-cutover.md) — current 138-bit
  classical production table, boundary provenance, and runtime lookup contract.
- [`book/src/how/security.md`](../book/src/how/security.md) — owning narrative
  security chapter.
