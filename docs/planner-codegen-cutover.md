# Planner Codegen Cutover

## Status

Accepted design direction for the next schedule/preset cleanup tranche.

This note locks in the plan to reduce schedule authority to one real source of
truth and to stop treating checked-in tables as semi-manual artifacts.

## Problem

Schedule truth is currently split across multiple layers:

- the offline exhaustive planner in `planner/src/search.rs`
- the runtime planner in `src/protocol/commitment/schedule.rs`
- checked-in preset artifacts in `src/protocol/commitment/schedule_tables.rs`
- profile fallback muxing in `src/protocol/commitment/profile.rs`
- config/policy parameter synthesis in `src/protocol/commitment/config.rs`
- heuristic non-planned stopping in `src/protocol/commitment_scheme.rs`

This is too many authorities. It allows drift, makes review harder, and makes
it unclear which layer is actually shipping the preset behavior.

The current checked-in Rust table file is also not backed by an in-repo
generator; it was produced out of tree. That is not an acceptable long-term
workflow.

## Decision

For blessed preset families, the only schedule chooser should be:

1. the exhaustive planner
2. an in-repo generator
3. checked-in generated Rust artifacts

Runtime should validate and execute pinned artifacts. It should not choose
shipped schedules for blessed families.

For blessed families, the cutover should fully delete the current
non-authoritative schedule-choice paths rather than merely de-emphasizing them.
Any future experimental or unblessed planner path must be explicitly separate
from the shipping preset flow.

## Target Source Of Truth

### Authoritative

- planner search + codegen
- generated Rust artifact modules

### Non-authoritative

- runtime schedule reconstruction
- config/policy hooks
- dynamic root heuristics

These non-authoritative layers may recompute values as consistency checks, but
they must not be allowed to silently choose different shipped schedules.

For blessed families, these schedule-choice paths should be removed entirely
once the generated path is in place.

## Target Artifact Model

The generated artifact must match the current runtime proof model.

The planner can no longer emit only `levels + tail`. It must emit the same
step-based shape the runtime now uses:

- `Fold`
- `Direct`

Each generated step should pin the security-sensitive schedule data, not just
`(current_w_len, log_basis)`.

Required pinned data per step:

- step kind
- `D`
- `log_basis`
- `n_a`
- `n_b`
- `n_d`
- stage-1 challenge family
- `current_w_len`
- `next_w_len` for fold steps
- direct witness shape for direct steps

Runtime may still recompute:

- layout sizes
- `next_w_len`
- proof bytes

But any mismatch against the pinned artifact must fail closed.

## Generated Layout

Generated files should live under:

- `src/protocol/commitment/generated/`

Use one generated module per blessed prime profile, for example:

- `generated/fp128_prime275.rs`
- later `generated/fp64_<prime>.rs`
- later `generated/fp32_<prime>.rs`

`schedule_tables.rs` should stop being the hand-maintained central artifact
file. It can be replaced by generated modules plus a small registry layer, or
deleted entirely once the generated path is complete.

## Prime Profile Direction

Prime profiles should become the data-backed registry for schedule generation.

Each blessed profile should declare:

- field type / modulus identity
- supported root `D` values
- shipped proof families
- stage-1 challenge family map
- security tables / thresholds
- any profile-specific dynamic-root policy inputs

Adding a new blessed prime should become:

1. define the profile manifest
2. add security data
3. run generator
4. check in generated Rust

Not:

1. touch runtime planning logic by hand
2. add ad hoc schedule tables manually

## Scope Boundaries

### Phase 1

Do this first:

- singleton planner-backed families
- step-based planner output
- in-repo Rust generator
- generated-only consumption for blessed singleton families

Keep static presets out of codegen. They already encode their schedule directly
and do not need generated artifacts.

### Phase 2

Do this after singleton cutover is stable:

- batch-keyed generated root artifacts
- broader fp64/fp32 blessed profile rollout

## Migration Plan

1. Upgrade `planner/src/search.rs` to emit the same step-based schedule model as
   runtime.
2. Introduce a shared generated artifact schema for pinned step specs.
3. Add an in-repo generator binary or `xtask` that emits Rust files under
   `src/protocol/commitment/generated/`.
4. Generate the current blessed fp128 prime275 artifacts from that pipeline.
5. Make `profile.rs` consume only generated artifacts for blessed families.
6. Remove silent `generated_or_planned` fallback for those families.
7. Demote runtime planning in `schedule.rs` to debug/validation tooling for
   non-blessed or experimental families.
8. Strengthen tests so they compare:
   - generated artifact vs exhaustive planner
   - runtime validation vs pinned artifact
   - end-to-end proofs at boundary `nv` values

## Rules

- Do not manually edit checked-in schedule artifacts as the normal workflow.
- If a generated preset row is wrong, fix planner/codegen, then regenerate.
- Blessed preset families must not silently fall back to live planning.
- Runtime must fail closed on artifact/derived-parameter mismatch.
- For blessed families, delete superseded schedule-choice paths instead of
  leaving them as dormant or parallel alternatives.
- If an experimental live-planner path remains, it must be clearly separate
  from blessed preset execution and impossible to reach by silent fallback.

## Success Criteria

The cutover is complete when all of the following are true:

- planner output is step-based
- generated Rust artifacts are reproducible in-repo
- blessed families are generated-only
- non-authoritative schedule-choice paths for blessed families are deleted
- runtime no longer chooses shipped schedules for blessed families
- artifact validation covers full pinned step specs
- adding a new blessed fp32/fp64/fp128 prime is a profile + codegen operation
