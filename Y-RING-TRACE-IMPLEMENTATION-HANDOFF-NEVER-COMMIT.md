**NEVER COMMIT THIS FILE.**

# Y-Ring Trace Implementation Handoff

- **Handoff reason:** phase complete
- **Summary:** The branch now has a pushed commit that removes `y_ring` / `y_rings` wire payloads and implements the degree-one, non-EOR fused stage-2 trace term. The remaining implementation blocker is the K > 1 / extension-opening-reduction final binding, which still relied on removed `y_ring` data.
- **Goal and scope:** Fully implement `specs/y-ring-trace-internalization.md`: remove on-wire `y_ring` / `y_rings`, internalize the trace check as a fused stage-2 term with `gamma_tr`, remove public-output M rows, update proof sizing/planner tables, finish K > 1 and ZK handling, and get CI green.
- **Current state:** tests failing. Commit `68301c67` (`feat(trace): internalize degree-one stage2 checks`) was pushed to `layerzero/quang/y-ring-trace-internalization`. Current working tree has only local never-commit notes untracked.

## Work Completed

- Removed proof-wire `y_ring` / `y_rings` from level/root/terminal proof payloads, shapes, wire serialization, and tests in `crates/akita-types/src/proof/levels.rs`, `crates/akita-types/src/proof/shapes.rs`, and `crates/akita-types/src/proof/wire.rs`.
- Reconstructed and committed the shared trace stage-2 helper module at `crates/akita-types/src/trace_weight/stage2.rs`.
- Added the stage-2 trace term plumbing through prover and verifier paths, including `CHALLENGE_TRACE_BATCH` and trace-wire construction.
- Fixed the trace-bearing stage-2 prover by disabling only the fused prefix-x cache path when trace is active at `crates/akita-prover/src/protocol/sumcheck/akita_stage2/lifecycle.rs:180`.
- Fixed padded-column trace handling by treating out-of-range padded trace entries as zero at `crates/akita-prover/src/protocol/sumcheck/akita_stage2/mod.rs:252`.
- Reworked closed-form trace-weight evaluation to support actual, non-power-of-two digit segments and basis-correct packed inner openings at `crates/akita-types/src/trace_weight/eval.rs:33`, `crates/akita-types/src/trace_weight/eval.rs:118`, and `crates/akita-types/src/trace_weight/eval.rs:168`.
- Fixed root verifier trace weighting for batched degree-one claims by matching the prover's claim-scaled block layout at `crates/akita-verifier/src/protocol/levels.rs:876`.
- Moved shared `generate_y` into `akita-types` and updated prover imports.
- Updated the local implementation worklog at `Y-RING-TRACE-WORKLOG-NEVER-COMMIT.md`.

## Files Modified

- `crates/akita-types/src/trace_weight/stage2.rs`: new shared trace stage-2 helper layer.
- `crates/akita-types/src/trace_weight/eval.rs`: direct final-point evaluator for trace weights, using `inner_opening_ring` for basis correctness.
- `crates/akita-types/src/trace_weight/layout.rs`: removed the old power-of-two closed-form validation helper.
- `crates/akita-types/src/trace_weight/mod.rs`, `crates/akita-types/src/lib.rs`: exported trace stage-2 helpers.
- `crates/akita-types/src/proof/levels.rs`, `crates/akita-types/src/proof/shapes.rs`, `crates/akita-types/src/proof/wire.rs`, `crates/akita-types/src/proof/tests.rs`: removed `y_ring` / `y_rings` from proof data and fixtures.
- `crates/akita-types/src/proof/relation.rs`, `crates/akita-types/src/proof/mod.rs`, `crates/akita-types/src/proof/ring_relation.rs`: shared relation helper updates, including `generate_y`.
- `crates/akita-types/src/proof_size.rs`, `crates/akita-types/src/schedule.rs`: proof-size and synthetic proof fixture updates after removing wire fields.
- `crates/akita-prover/src/protocol/flow.rs`, `crates/akita-prover/src/protocol/flow/inputs.rs`, `crates/akita-prover/src/protocol/flow/recursive.rs`, `crates/akita-prover/src/protocol/flow/root_fold.rs`: prover trace compact table construction and constructor/call-site updates after wire removal.
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2/*.rs`: fused trace term, folding, prefix behavior, and tests.
- `crates/akita-prover/src/protocol/ring_relation.rs`, `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`: removed local `generate_y` and adapted relation wiring.
- `crates/akita-verifier/src/protocol/levels.rs`, `crates/akita-verifier/src/protocol/levels/recursive.rs`, `crates/akita-verifier/src/stages/stage2.rs`: verifier trace wire, input claim, and final oracle updates.
- `crates/akita-transcript/src/labels.rs`: added trace batching transcript label.
- `crates/akita-pcs/src/scheme/tests/*.rs`: proof shape/test fixture updates for removed wire fields.
- `Y-RING-TRACE-WORKLOG-NEVER-COMMIT.md`: local never-commit implementation worklog. Keep local, do not stage.
- `Y-RING-TRACE-IMPLEMENTATION-HANDOFF-NEVER-COMMIT.md`: this local handoff note. Keep local, do not stage.

## Context Files

- `specs/y-ring-trace-internalization.md`: source spec. Key requirements: fused oracle at lines 72-107, final-point evaluation at lines 127-184, invariants at lines 193-201, acceptance criteria at lines 214-223, recommended execution order at lines 270-280.
- `crates/akita-verifier/src/protocol/levels/extension_opening_reduction.rs`: EOR verifier still expects `y_ring` rows. See `EorRow` and `expected_output_claim` around lines 32-38 and 92-100.
- `crates/akita-verifier/src/protocol/levels.rs`: non-ZK root EOR verifier currently builds `zero_y_rings` in the removed-wire world around lines 565-589.
- `crates/akita-prover/src/protocol/flow/root_extension.rs`: root EOR prover has the actual public partials, row coefficients, sumcheck `rho`, final claim, and factors by point. See `RootExtensionOpeningReduction` around lines 14-19 and final oracle logic around lines 362-400.
- `crates/akita-prover/src/protocol/flow/root_fold.rs`: root EOR caller still checks prover final claim against reconstructed internal y-rings before finishing the root fold around lines 337-382 and 747-769.

## Key Decisions and Rationale

- Treat `stash@{0}` as a design draft, not a patch to apply wholesale. It conflicted with the public M-row commits and referenced a missing `trace_weight/stage2.rs`.
- Implement the K = 1 / non-EOR trace path first because the spec explicitly names it as the first target and it exercises the main proof-wire removal without solving the harder EOR final binding.
- Do not use compatibility shims for removed proof fields. This repository has no backward-compatibility guarantee, and the requested cutover is a full removal.
- Disable only the trace-bearing fused prefix-x cache, not all prefix folds. The cached path omitted trace coefficients for the next round; ordinary prefix folds remain correct.
- Evaluate trace weights by direct finite sum over the real digit segment. The earlier tensor-only shortcut imposed a false power-of-two digit-depth constraint and broke monomial basis.

## Blockers / Errors

- Current failing command:

```text
cargo test -p akita-pcs --no-default-features --lib
```

- Current result:

```text
19 passed; 2 failed

failures:
  scheme::tests::fp32_ring_subfield::fp32_ring_subfield_outer_extension_uses_root_tensor_projection
  scheme::tests::fp32_ring_subfield::fp32_ring_subfield_multipoint_extension_uses_root_tensor_projection

Both fail with Err(InvalidProof) from verifier replay.
```

- Investigation result before cleanup diagnostics: the fp32 failures came from `ExtensionOpeningReductionVerifier`, where the expected final oracle was zero because the non-ZK verifier path used placeholder zero y-rings after the wire removal.
- The spec calls this out directly: K in `{2,4,8}` must also move the extension-opening-reduction final binding away from on-wire `y_ring` (`specs/y-ring-trace-internalization.md:199`, `specs/y-ring-trace-internalization.md:206`).

## Open Questions / Risks

- How exactly should non-ZK root EOR verifier compute the final oracle without y-rings?
  Likely direction: derive it from `ExtensionOpeningReductionProof::partials`, row coefficients, the sampled `eta`, and final `rho`, rather than trying to reconstruct y-rings.
- Recursive EOR has the same conceptual dependency and must be audited after root EOR is fixed.
- ZK path is not implemented. The spec requires removing y-ring masks and adding a deferred trace relation analogous to stage-2 final relation.
- Proof sizing and shipped planner tables still need the final cutover values, especially after confirming whether public-output M rows and quotient digits are fully removed for every path.
- Transcript hardening has not been re-run after the new `gamma_tr` sample and y-ring absorb removal.

## Cleanup Needed

- No debug prints are intentionally left in committed source.
- Keep these untracked local files out of commits:
  - `Y-RING-TRACE-WORKLOG-NEVER-COMMIT.md`
  - `Y-RING-TRACE-IMPLEMENTATION-HANDOFF-NEVER-COMMIT.md`
- Stashes still exist:
  - `stash@{0}: trace-wip`
  - `stash@{1}: wip2`
  - plus older unrelated stashes
  Do not blindly apply them; recover by intent only if needed.
- Before any future commit, run `git status --short` and stage explicit source paths only. Do not use `git add -A` while never-commit notes exist.

## Tests and Commands Run

- `cargo fmt -q`: pass.
- `ReadLints` on touched trace/stage-2/verifier files: no IDE linter errors.
- `cargo test -p akita-types --no-default-features`: pass earlier in the slice, `139 passed`.
- `cargo test -p akita-pcs --no-default-features --lib scheme::tests::single::verify_passes_for_consistent_opening -- --nocapture`: pass.
- `cargo test -p akita-pcs --no-default-features --lib scheme::tests::batched::batched_verify_passes_for_consistent_openings -- --nocapture`: pass.
- `cargo test -p akita-pcs --no-default-features --lib scheme::tests::single::monomial_basis_prove_verify_round_trip -- --nocapture`: pass.
- `cargo test -p akita-pcs --no-default-features --lib`: fail, `19 passed; 2 failed`, both fp32 EOR verifier tests.
- `git status -sb`: clean tracked tree after push, with only never-commit notes untracked.
- `git diff --stat`: empty after push, before this handoff note was created.
- `git stash list`: `trace-wip`, `wip2`, and older stashes still present.
- Committed and pushed:

```text
68301c67 feat(trace): internalize degree-one stage2 checks
```

## Next Steps

1. Fix non-ZK root EOR verifier final oracle without `y_ring`.
   Start at `crates/akita-verifier/src/protocol/levels.rs` around the `zero_y_rings` construction and replace the placeholder with a value derived from public EOR partials plus the final sumcheck challenges.
2. Mirror the same no-y-ring EOR final-binding design in recursive EOR verifier paths.
3. Enable K > 1 trace internalization instead of `trace_stage2_enabled` refusing EOR paths.
   The spec's K > 1 formula is at `specs/y-ring-trace-internalization.md:175-184`.
4. Add algebraic unit anchors for K in `{1,2,4}` that compare trace-weight contraction against `TraceOpen(sum_j b_j * e_folded_j)`.
5. Finish ZK cutover: remove y-ring masks, add deferred trace relation, and close hiding cursor accounting.
6. Update proof sizing, planner scoring, and regenerate shipped tables.
7. Add negative tests for tampered `e_hat` trace projection at root, recursive, and terminal levels.
8. Run transcript hardening, nextest non-ZK/all-features, and profile proof-size shrink checks.

## How to Resume

1. Run:

```bash
cd /Users/quang.dao/Documents/SNARKs/akita-y-ring-trace-internalization
git status -sb
cargo test -p akita-pcs --no-default-features --lib scheme::tests::fp32_ring_subfield::fp32_ring_subfield_outer_extension_uses_root_tensor_projection -- --nocapture
```

2. Read:

```text
specs/y-ring-trace-internalization.md:175-206
crates/akita-verifier/src/protocol/levels.rs:565-589
crates/akita-verifier/src/protocol/levels/extension_opening_reduction.rs:32-100
crates/akita-prover/src/protocol/flow/root_extension.rs:362-400
```
