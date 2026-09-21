# Spec: Compile-Time Field-Tier Pruning

| Field         | Value      |
|---------------|------------|
| Author(s)     | Quang Dao  |
| Created       | 2026-09-21 |
| Status        | active     |
| PR            | pending    |
| Supersedes    |            |
| Superseded-by |            |
| Book-chapter  |            |

## Summary

Akita's executable crates currently monomorphize dispatch arms for every
supported prime-field tier even when an application uses only one tier. This
feature adds positive, additive Cargo capabilities for Fp32, Fp64, and Fp128 so
an integrator can omit unused executable arms without narrowing protocol policy,
schedule catalogs, or the ring-degree surface of any selected tier.

## Intent

### Goal

Make executable field support an explicit compile-time capability owned by
`akita-types` and forwarded consistently through `akita-prover`,
`akita-verifier`, `akita-setup`, and `akita-pcs`.

### Invariants

- Default builds retain all three field tiers and therefore preserve current
  behavior.
- `field-fp32`, `field-fp64`, and `field-fp128` are positive and additive. A
  selected tier retains every role, NTT, and compression ring degree in the
  canonical protocol policy.
- The defining `akita-types` crate chooses which macro bodies exist. A
  downstream crate must not need matching local feature names for pruning to
  work.
- A missing executable field tier is rejected at scheme, setup, commitment,
  prover, and verifier admission with an error that names the required feature.
- Protocol policy and offline planner inspection remain complete in zero-field
  builds. Field-tier features do not prune schedule rows or regenerate catalog
  artifacts.
- Selecting Fp64 preserves its complete Q64 CRT parameters, prepared-cache
  representation, and exactness-tail policy.
- Cargo feature unification remains visible: any dependency that enables an
  additional tier widens the final executable graph.

### Non-Goals

- Pruning individual ring degrees within a selected tier.
- Adding an application-specific profile or feature.
- Pruning arithmetic representations, CRT primes, or tracked schedules.
- Changing proof bytes, transcript behavior, security estimates, or planner
  policy.

## Evaluation

### Acceptance Criteria

- [x] Default execution-crate features select all three tiers.
- [x] Every execution crate forwards each field capability without widening it.
- [x] Single-tier, additive multi-tier, and zero-tier builds compile and expose
  the expected admission behavior.
- [x] Downstream fixture crates prove that defining-crate macro selection drops
  unavailable bodies.
- [x] Fp64-only tests cover all dispatch degrees and the Q64 CRT/cache path.
- [x] CI profile selectors resolve transitively to exactly one intended tier.
- [x] The Book and verifier-only recipe document selection, defaults, zero-tier
  behavior, and Cargo feature unification.
- [ ] A downstream Fp64 integration records clean-build and code-size evidence
  against an otherwise identical all-tier build.

### Testing Strategy

Run the feature-selection matrix in `.github/workflows/ci.yml`, including all
three singleton builds, all pairwise additive builds, a zero-field build, and
the downstream fixtures. Run the repository's normal formatting, dependency,
Clippy, test, and documentation gates with all field tiers selected wherever
the gate exercises runtime behavior.

### Performance

Measure an Fp64 application using fresh, separate target directories for the
all-tier and Fp64-only configurations. Keep the Akita revision, application
revision, compiler, job count, profile, and non-field features fixed. Report
wall time and final library size without projecting savings from earlier
experiments that also pruned ring degrees.

## Design

### Architecture

The canonical dispatch tables remain unconditional. They route each field tier
through a hidden exported helper macro whose definition is selected inside
`akita-types`; an enabled helper expands every canonical degree arm, while a
disabled helper emits only the capability error. Public capability queries and
admission checks share one tier-to-feature mapping. Execution crates call that
check before performing work, while catalog decoding remains available to
offline tooling.

### Alternatives Considered

Downstream `cfg(feature = ...)` branches were rejected because exported macros
would inspect the consuming crate's features rather than `akita-types`'s
features. Negative disable-features were rejected because Cargo features are
additive. Application-specific umbrella profiles were rejected in favor of the
three generic protocol capabilities.

## Documentation

The durable integrator guidance lives in
`book/src/usage/feature-flags.md`; the minimal verifier recipe lives in
`book/src/usage/verifier-only.md`. After the implementation lands, mark this
record implemented, set its PR and Book chapter, and archive it under the
current quarter.

## References

- `crates/akita-types/src/dispatch/policy.rs`
- `crates/akita-types/tests/compiled_fields.rs`
- `scripts/check_profile_ci_features.sh`
