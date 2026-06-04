# Spec: Akita Crate Follow-Up and Jolt Trait Integration

| Field       | Value        |
|-------------|--------------|
| Author(s)   | @quangvdao   |
| Created     | 2026-05-05   |
| Status      | implemented  |
| PR          | #65          |

## Summary

This PR is a retroactive spec for the crate-decomposition follow-up after the
main Akita crate split. It tightens the internal module organization of the
larger extracted crates, starts a small Jolt integration by adopting Jolt-shaped
field/algebra traits and a Jolt-backed transcript engine, removes
velocity-restricting protocol regression vectors, and fixes the CI coverage
needed to keep these new boundaries honest.

## Intent

### Goal

Make the post-decomposition Akita workspace easier to review, easier to
integrate with Jolt, and less coupled across prover/verifier/config roles
without taking on a full Jolt sumcheck or polynomial API migration in this PR.

The key surfaces modified are:

- `akita-prover`: internal module ownership under `api`, `backend`, `kernels`,
  and `protocol`.
- `akita-types`: public protocol data ownership under `layout`, `proof`, and
  `transcript`.
- `akita-verifier`: internal module ownership under `proof`, `protocol`, and
  `stages`.
- `akita-field` and `akita-algebra`: adoption of Jolt-style trait names and
  behavior where it is already relevant to Hachi/Akita.
- `akita-transcript`: implementation backed by `jolt-transcript`, while
  preserving Akita's label-aware convenience API.
- `akita-config`: optional planner dependency and generated schedule-family
  selectors.

### Invariants

1. Prover and verifier must continue to share one set of proof, commitment,
   setup, layout, schedule, and transcript-append data types from
   `akita-types`.
2. `akita-verifier` must not depend on `akita-prover`, `akita-planner`, profile
   examples, benches, or prover-only polynomial backends.
3. Module moves must preserve crate-root public exports where those exports are
   intended public API.
4. The first Jolt integration slice must not force call sites into a verbose
   Jolt transcript API; Akita keeps a label-aware transcript convenience layer.
5. Jolt field integration must avoid the BN254-specific field trait path and
   instead track the more general field-trait refactor direction.
6. `BalancedDigitLookup` and halving/two-inverse support remain Akita-specific
   performance-facing capabilities rather than being folded into broad umbrella
   wrappers.
7. Optional planner support must fail explicitly on table misses when planner
   fallback is unavailable, and planner-dependent tests must be feature-gated.
8. Profiling and benchmark targets must compile under CI's warning-deny
   settings.
9. Protocol regression fixtures may be deleted in this PR because protocol
   behavior is expected to keep changing during active development.

### Non-Goals

1. Full cutover to Jolt's field implementation or any BN254-specific Jolt field
   trait path.
2. Full cutover to Jolt's transcript call-site API.
3. Full cutover to Jolt sumcheck, polynomial, or transcript-engine abstractions
   across every Akita call site.
4. Preserving removed protocol regression vectors as new canonical fixtures.
5. Changing the PCS protocol deliberately beyond the small schedule/profile and
   trait-bound changes required by this integration slice.
6. Adding compatibility shims for old internal module paths.
7. Introducing a new evaluation framework or porting Jolt's eval framework.

## Evaluation

### Acceptance Criteria

- [x] `akita-prover` internals are grouped into `api`, `backend`, `kernels`,
      and `protocol` modules.
- [x] Akita-specific prover sumcheck stages live under
      `akita-prover::protocol::sumcheck`.
- [x] `akita-types` helpers are grouped into `layout`, `proof`, and
      `transcript` modules.
- [x] `akita-verifier` internals are grouped into `proof`, `protocol`, and
      `stages` modules.
- [x] The large `akita-scheme` inline test module is moved to `src/tests.rs`.
- [x] `docs/crate-graph.md` documents the intended crate ownership and
      dependency graph.
- [x] Dependency hygiene CI checks include `akita-config`, `akita-setup`, and
      `akita-scheme`.
- [x] `akita-field` exposes a Jolt-trait compatibility layer and adopts
      Jolt-style random sampling, primitive integer conversion, byte encoding,
      challenge, and inversion traits where appropriate.
- [x] Akita-specific balanced digit lookup and halving support remain small
      focused traits/helpers.
- [x] Ring and field values implement standard traits needed by downstream code
      and tests.
- [x] `akita-transcript` depends on `jolt-transcript` internally while keeping
      Akita's `append_*` and `challenge_*` label-aware API.
- [x] `akita-config` makes planner fallback optional behind `planner`.
- [x] Planner-table misses report explicit errors when planner fallback is not
      compiled in.
- [x] Planner-dependent tests are gated with `#[cfg(feature = "planner")]`.
- [x] fp128 full/onehot schedule-family selectors choose among generated
      presets.
- [x] `examples/profile.rs` handles the generated `D64Full` schedule choice.
- [x] Protocol regression-vector fixtures are removed.
- [x] All-target/all-feature and no-default-feature CI paths compile cleanly.

### Testing Strategy

Required local and CI checks:

- `cargo fmt -q`
- `cargo fmt --all --check`
- `cargo check --workspace --all-targets`
- `cargo check -p akita-pcs --example profile --all-features`
- `cargo check -p akita-verifier --no-default-features`
- `cargo check -p akita-pcs --all-targets`
- `cargo check -p akita-pcs --tests --message-format=short`
- `cargo test --no-run --message-format=short --all-features`
- `cargo test -p akita-config fp128_family_selector --all-features`
- `cargo test -p akita-prover --lib`
- `cargo test -p akita-pcs --lib`
- `cargo test -p akita-pcs --test transcript`
- `cargo test -p akita-pcs --test sparse_challenge`
- `cargo test -p akita-pcs --test single_poly_e2e single_onehot_nv10 -- --exact`
- `cargo test -p akita-sumcheck --test drivers`
- `cargo clippy --all --all-targets --all-features -- -D warnings`
- `cargo clippy --all --all-targets --no-default-features -- -D warnings`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo build --release --quiet --example profile`
- `scripts/check-crate-deps.sh akita-verifier`
- `scripts/check-crate-deps.sh akita-prover`
- `scripts/check-crate-deps.sh akita-config`
- `scripts/check-crate-deps.sh akita-setup`

### Performance

This PR is mostly structural, but it touches field/ring trait implementations,
profiling schedule selection, and transcript internals. Performance expectations
are:

- no intentional regression in commitment/proving/verification code paths;
- no extra runtime cost from keeping Akita's label-aware transcript convenience
  API over the Jolt transcript engine;
- no NTT halving regression from replacing compile-time two-inverse access with
  slower runtime inversion.

The relevant performance smoke checks are the profile example and the existing
criterion benches. The onehot 32-variable benchmark must continue to build and
run in CI.

## Design

### Architecture

The PR refines the crate graph produced by the first crate-decomposition PR
rather than introducing new crates. The main architectural move is to make the
largest crates internally navigable:

```text
akita-prover/
  api/        public prover/setup/commitment trait surfaces
  backend/    dense, one-hot, multilinear, recursive witness storage
  kernels/    NTT, CRT, matrix, and low-level linear kernels
  protocol/   Akita proving flow, ring switch, PRG, sumcheck stages

akita-types/
  layout/     flat matrices, digit math, params, opening-point helpers
  proof/      public commitment/proof/setup/relation shapes
  transcript/ transcript-append traits for protocol objects

akita-verifier/
  proof/      verifier-facing claims and direct proof helpers
  protocol/   batched/level/ring-switch verification flow
  stages/     Akita stage-1 and stage-2 verifier instances
```

The Jolt integration is intentionally narrow. `akita-field` learns how to speak
the general Jolt trait vocabulary that Akita needs now, and
`akita-transcript` uses Jolt's transcript engine internally. Akita keeps its own
small convenience traits for protocol-specific needs such as balanced digit
lookup and fast halving/two-inverse access.

### Alternatives Considered

1. Full Jolt transcript API cutover.
   This was rejected because it made call sites verbose without improving the
   Akita protocol surface. The chosen design preserves the label-aware Akita API
   while still removing duplicate transcript-engine code.

2. Jolt `Field` trait cutover.
   This was rejected because the currently relevant Jolt field trait path is too
   BN254-specific for Akita's needs. The chosen design tracks the broader Jolt
   field-trait refactor direction and keeps Akita-specific helpers explicit.

3. Keeping protocol regression vectors.
   This was rejected because these fixtures would slow iteration while the
   protocol is still intentionally fluid. End-to-end and focused protocol tests
   remain the right guardrails for this stage.

4. Leaving large crates flat after decomposition.
   This was rejected because the extracted crates were easier to depend on but
   still hard to review internally. The chosen module hierarchy gives clearer
   ownership without another crate split.

## Documentation

- Add `docs/crate-graph.md` to document crate ownership and dependency
  direction.
- Add this retroactive spec under `specs/` so large-PR review has an explicit
  architectural record.
- The PR description should summarize the full diff against `main`, not just
  the final CI cleanup commits.

## Execution

Implemented order:

1. Group large crate internals under clearer module families.
2. Tighten dependency hygiene checks for the newly extracted role crates.
3. Remove protocol regression vectors to avoid pinning rapidly changing
   behavior.
4. Adopt the small Jolt-relevant field/algebra trait slice.
5. Restore focused Akita-specific traits for balanced digit lookup and halving.
6. Switch transcript internals to Jolt while keeping Akita's convenience API.
7. Fix Bugbot findings and all-target CI import fallout.
8. Add this retroactive spec.

## References

- `specs/akita-pcs-crate-decomposition.md`
- `docs/crate-graph.md`
- PR #65: <https://github.com/LayerZero-Labs/hachi/pull/65>
- Jolt repository field-trait refactor context from the sibling Jolt project.
