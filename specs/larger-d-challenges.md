# Spec: D=256 fp128 A-role dispatch for future mixed-ring schedules

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     |                                |
| Created       | 2026-07-24                     |
| Status        | implemented                    |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  |                                |

## Summary

Future mixed-ring-dimension schedules may assign `D = 256` to the inner
(A) commitment role at selected levels while assigning smaller dimensions to
other roles or later levels. The fp128 protocol dispatch policy currently
prevents that experiment because its A-role arms stop at `D = 128`, even though
the existing production challenge ladder and audited A-role SIS tables already
cover `D = 256`.

This PR adds only the missing fp128 A-role `D = 256` dispatch arm. It does not
make `D = 256` a uniform fp128 configuration, add a preset, generate or resolve
a `D = 256` schedule, or import mixed-ring implementation code. Those are
separate features that can decide independently which levels and roles should
use the larger dimension.

## Intent

### Goal

Allow fp128 protocol code to dispatch the inner (A) role at `D = 256`, so a
later production-ready mixed-ring feature can use that role/dimension pair
without first widening the uniform configuration surface.

The implementation surface is intentionally limited to:

- `crates/akita-types/src/dispatch/policy.rs` — add `256` to both sources of
  fp128 inner-role dispatch policy:
  `__dispatch_for_field_inner!` and the `protocol_dispatch_policy!` `inner`
  list.
- `crates/akita-types/src/dispatch/mod.rs` — update the role-specific dispatch
  tests to accept fp128 inner `D = 256` while preserving all existing
  acceptance and rejection boundaries.

The existing `D = 256` production challenge configuration
`SparseChallengeConfig { count_pm1: 23, count_pm2: 0 }` and the audited
`A_ROLE_RING_DIMS` entry are unchanged.

### Invariants

- **Role-specific capability only.** The change widens only fp128 inner/A-role
  dispatch. It must not widen another role, field tier, envelope, or NTT
  policy.
- **No configuration becomes selectable.** No preset, planner candidate,
  schedule catalog entry, generated table, or runtime fallback is added or
  changed. Existing user-facing configurations therefore remain unchanged.
- **Existing dispatch is stable.** fp128 inner `D = 64` and `D = 128` continue
  to dispatch identically. Unsupported dimensions, including `D = 32` and
  `D = 512`, remain rejected for that role.
- **The two dispatch authorities agree.** The macro arms used for
  monomorphization and the declarative protocol policy list contain the same
  fp128 inner dimensions: `{64, 128, 256}`.
- **Security remains fail-closed.** The change relies on the existing audited
  A-role SIS coverage for `D = 256`; it does not change SIS bounds, estimator
  inputs, challenge parameters, or certification logic.
- **No proof or transcript change.** Because no schedule can newly select this
  arm in this PR, existing proofs, transcript bytes, schedule identities, and
  generated artifacts remain unchanged.

### Non-Goals

- A uniform fp128 `D = 256` preset or schedule.
- Any generated `D = 256` schedule table or runtime-DP-backed `D = 256`
  schedule.
- Mixed ring dimensions per level or per role, schedule splicing, role
  compression, or a three-band schedule.
- Deciding which roles or levels a future mixed-ring configuration assigns
  `D = 256`.
- Prover, verifier, planner, config, profile-harness, or end-to-end PCS changes.
- Changes to the challenge ladder, sampler, entropy floor, SIS tables, NTT
  limits, or backend kernels.
- Support for `D = 512` or larger dimensions.
- Copying or adapting implementation from the mixed-dimension demo branch.

## Evaluation

### Acceptance Criteria

- [x] fp128 inner-role dispatch accepts exactly `D ∈ {64, 128, 256}`.
- [x] fp128 inner-role dispatch continues to reject `D = 32` and `D = 512`.
- [x] The fp128 outer, opening, envelope, and NTT policies are byte-for-byte
      unchanged.
- [x] The fp64 and fp32 dispatch policies are byte-for-byte unchanged.
- [x] No file under `crates/akita-config/`, `crates/akita-planner/`,
      `crates/akita-schedules/`, `crates/akita-prover/`, `crates/akita-verifier/`,
      or `crates/akita-pcs/` changes.
- [x] No generated artifact or schedule catalog identity changes.
- [x] Repository formatting, dependency, unit-test, and Clippy gates pass.

### Testing Strategy

Add or update the focused `akita-types` dispatch unit test so it exercises the
fp128 inner slot at `32`, `64`, `128`, `256`, and `512`. The test must prove
acceptance of `64`, `128`, and `256` and rejection of `32` and `512`.

Run the cheap repository preflight first, then the focused `akita-types` tests
and the three CI Clippy feature graphs. Existing schedule identity tests may be
run as a regression check, but this PR must not require regeneration or new
schedule fixtures.

An end-to-end `D = 256` proof test belongs to the later PR that introduces a
mixed schedule capable of selecting the new arm. Adding such a schedule here
would violate this spec's scope.

### Performance

There is no runtime or proof-size claim. The new monomorphization is unreachable
from existing configurations, so existing benchmark modes and proof artifacts
must be unaffected. Any performance claim for selectively using `D = 256`
belongs to the mixed-ring feature that selects it.

## Design

### Architecture

`dispatch_for_field!` maps a runtime field tier, protocol role, and ring
dimension to a const-generic protocol implementation. For the inner role, the
fp128 tier currently exposes only `D = 64` and `D = 128`. Adding the `256` arm
makes the already-certified implementation available to future callers without
creating such a caller in this PR.

The policy is duplicated intentionally in two macro sites:

1. `__dispatch_for_field_inner!` defines the concrete const-generic match arms.
2. `protocol_dispatch_policy!` defines the role-aware validation policy.

Both sites must be changed together. No wrapper API or new abstraction is
needed.

### Alternatives Considered

- **Add an fp128 `D256` preset.** Rejected because a preset represents a
  selectable uniform configuration and pulls schedule resolution into scope.
  The intended consumer is a later mixed-ring configuration, not a uniform
  `D = 256` mode.
- **Generate a `D = 256` schedule now.** Rejected because no schedule is needed
  to establish the role-specific dispatch capability, and a generated table
  would prematurely encode policy for all levels and roles.
- **Import the mixed-dimension demo.** Rejected because the demo combines
  several independently reviewable features. Production mixed schedules,
  per-role compression, and their prover/verifier changes require separate
  specs and PRs.
- **Widen every fp128 role together.** Rejected. Dispatch support is
  role-specific, and future mixed schedules may need `D = 256` for only a
  subset of roles.

## Documentation

No Book, `AGENTS.md`, or architecture documentation change is required because
this PR does not expose a new configuration or change proof behavior. Update
this spec's status and PR field when the implementation lands.

## Execution

1. Add `256` to the fp128 branch of `__dispatch_for_field_inner!`.
2. Add `256` to the fp128 `inner` list in `protocol_dispatch_policy!`.
3. Update only the focused role-dispatch unit test.
4. Verify that the diff contains no preset, schedule, planner, prover,
   verifier, PCS, challenge-policy, SIS-table, or generated-artifact changes.
5. Run the required repository checks.

## References

- Dispatch policy: `crates/akita-types/src/dispatch/policy.rs`
- Dispatch tests: `crates/akita-types/src/dispatch/mod.rs`
- Existing D=256 challenge entry:
  `crates/akita-challenges/src/config.rs`
  (`PRODUCTION_FOLD_CHALLENGE_LADDER`)
- Existing audited A-role coverage:
  `crates/akita-types/src/sis/ajtai_key.rs` (`A_ROLE_RING_DIMS`)
- Mixed-ring architectural context only:
  `specs/runtime-ring-cutover.md`
