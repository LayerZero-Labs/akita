**NEVER COMMIT THIS FILE.**

# Y-Ring Trace Internalization Worklog

## Goal & Scope

Track implementation work for `specs/y-ring-trace-internalization.md` and PR #154.
The goal is the full cutover: remove on-wire `y_ring` / `y_rings`, internalize the trace check as a fused stage-2 term, update sizing/planner tables, finish ZK handling, and verify the PR to approval quality.

## Starting State

- Branch: `quang/y-ring-trace-internalization`
- Base commit: `0f388293` (`fix(ring-switch): complete public M-row cutover after y-ring removal`)
- Related spec: `specs/y-ring-trace-internalization.md`
- Related PR: `https://github.com/LayerZero-Labs/akita/pull/154`
- Relevant stash: `stash@{0}` (`trace-wip`)

## Plan

This continuation must proceed slice-by-slice. After each slice: run the
slice's acceptance commands, perform an adversarial review, append a worklog
retrospective, commit only the relevant source files, and push
`quang/y-ring-trace-internalization`.

Guardrails:
- No backward-compatibility proof fields, serde aliases, or migration shims.
- Do not apply `stash@{0}` / `stash@{1}` wholesale; use them only as design
  references if a specific detail is missing.
- Keep `Y-RING-TRACE-*NEVER-COMMIT.md` untracked and never stage them.
- Verifier-reachable code must return `AkitaError`, not panic/unwrap/assert.
- Every protocol slice updates prover, verifier, transcript order, tests, and
  docs together.
- No hidden public-row padding unless explicitly justified here first.
- ZK must keep true claims, masked transcript handles, and LC masks separate.

Slice 0: Stabilize the branch before protocol work.
Fix stale trace tests, clippy `needless_question_mark`, all-features cfg imports,
profile report y-ring references, root-fold line cap, Bugbot's stale ZK
public-row test issue, and the multipoint root trace-weight bug.
Acceptance:
- `cargo test -p akita-types --no-default-features trace_weight -q`
- `cargo check -p akita-pcs --examples --no-default-features -q`
- `scripts/check-rust-file-lines.sh`
- `cargo doc -q --no-deps --all-features`

Slice 1: Soundness derivation gate.
Add/update the spec with the extraction argument for dropped public M rows plus
the fused trace term, including EOR and ZK masking. No protocol code beyond
doc/spec.
Acceptance:
- Spec text explains the extractor, `gamma_tr` binding, public-row removal,
  EOR final-claim handoff, ZK masking, and the added soundness error.
- Adversarial review finds no missing assumption about EOR final claim,
  multipoint batching, or ZK public input.

Slice 2: Generalize trace weights.
Replace the single-opening trace wire with multi-term trace data: per-term
block range, scale, prepared inner point, and K=1/K>1 opening representation.
Acceptance:
- K=1,2,4,8 dense-table versus closed-form tests pass.
- Witness-dot tests equal `TraceOpen(sum_j b_j * e_folded_j)`.
- Multipoint batched root trace weights use each claim's own prepared point.

Slice 3: Non-ZK EOR cutover.
Use the EOR sumcheck's returned `(rho, final_claim)` instead of zero y-ring
placeholders. Enable trace for EOR by scaling trace terms with EOR final factors
and setting trace input to the EOR `final_claim`.
Acceptance:
- Both fp32 EOR tests pass.
- `cargo test -p akita-pcs --no-default-features --lib` passes.

Slice 4: ZK cutover.
Remove y-ring hiding masks/absorbs. Split true trace claim from masked
transcript claim. Add trace mask into stage-2 initial-claim mask and add the
trace term to the ZK stage-2 final R1CS relation.
Acceptance:
- ZK hiding cursor closes exactly.
- `cargo nextest run --profile ci-all-features` passes.

Slice 5: Sizing, planner, tables, profile.
Finalize `level_proof_bytes`, profile reporting, generated schedules, and regen
drift.
Acceptance:
- Generated table drift tests pass after regeneration.
- Profile reports `y_ring_bytes = 0` and the expected proof shrink.

Slice 6: Negative tests and transcript hardening.
Add tampered `e_hat` rejection for recursive, root, terminal, and EOR cases.
Re-run logging transcript tests after `gamma_tr` / absorb changes.
Acceptance:
- `cargo test -p akita-pcs --features logging-transcript --test transcript_hardening`
  passes.
- Required tamper tests fail with `InvalidProof`.

Final gate:
- `cargo fmt -q`
- `cargo clippy --all --all-targets --all-features -- -D warnings`
- `cargo nextest run --profile ci-non-zk`
- `cargo nextest run --profile ci-all-features`
- `AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile`
- PR #154 CI green and Bugbot comments addressed.

Current prover-performance intervention:

Slice P0: Adversarial no-code performance audit.
Profile the trace-bearing prover modes and collect independent subagent
reviews before editing tracked files. The audit must separate construction
cost (`build_trace_stage2_compact`, row materialization, table slicing) from
sumcheck-round cost (`AkitaStage2Prover` folding/accumulation paths).
Acceptance:
- fp32/fp64 trace-bearing release profiles identify the hot spans and total
  prove/verify wall times.
- At least two adversarial reviews agree on the likely waste source or explain
  a concrete disagreement.
- Pre-implementation review records any existing correctness hazard that must
  be fixed together with the performance work.

Slice P1: Remove avoidable trace-weight construction work.
Replace dense trace-table materialization for prover compact trace weights with
direct live-column compact construction, and remove repeated full ring
multiplication from ring-row materialization when a single product plus shifts
is equivalent.
Guardrails:
- Dense table builders stay as the reference oracle for tests.
- No verifier-visible proof fields, compatibility shims, or trace disabling.
- Verifier-reachable helpers keep returning `AkitaError` on bad shape.
Acceptance:
- Direct compact trace weights equal dense-table slicing for field and ring
  openings, including partial `live_x_cols`.
- Existing trace-weight tests pass for K = 1, 2, 4, 8.
- fp32/fp64 release profile construction spans improve without verify
  regression.
- Commit and push this slice before moving to another tracked-code slice.

Slice P2: Remove avoidable per-round trace sumcheck work if profiling still
shows the trace term in the hot path.
Pre-scale trace compact weights by `gamma_tr` once and simplify per-round trace
accumulation so the prover does not multiply every trace sample by the same
challenge repeatedly. Only extend prefix/two-round fusion if an adversarial
review proves the algebra still matches the fused trace relation.
Guardrails:
- The initial claim remains `gamma_tr * trace_opening_claim`.
- ZK and non-ZK transcript-visible claims remain unchanged.
- No prefix optimization is re-enabled by assumption; it needs a local proof
  and tests first.
Acceptance:
- Stage-2 prover tests and PCS e2e tests pass.
- Release profiles show no remaining trace-specific prover regression above
  noise, or the residual is recorded with a concrete reason.
- Commit and push this slice before CI triage.

Slice P3: CI and Bugbot closure.
Run local final gates, push all commits, inspect PR #154 checks and comments,
address actionable Bugbot/CI feedback, and only declare the goal complete after
CI is green.
Acceptance:
- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `scripts/check-rust-file-lines.sh`
- `cargo test`
- Representative release profiles for fp32/fp64 and the canonical fp128 mode.
- PR #154 checks green, including Bugbot.

## Decisions

- **[2026-06-07] Small-field verifier regression is in trace-weight final-point evaluation, not in the ring-switch row evaluator.**
  Local fp32/fp64 profiles show root `stage2_expected_output_claim` dominating verification while `stage2_ring_switch_row_eval` remains small.
  Chosen fix: keep the fused trace term enabled, but evaluate the K=2/K=4 trace MLE by linearly combining the ring-coordinate shifts once and applying one trace decode per block, instead of recomputing a trace reduction for every ring coordinate.
  Guardrail: dense trace-weight table equivalence tests remain the acceptance oracle; disabling trace or reintroducing y-ring proof data is out of scope.
- **[2026-06-07] Prover regression is representation fallout, not a harder sumcheck relation.**
  Local fp32/fp64 profiles and two adversarial reviews agree that the trace term fell off fast paths: `build_trace_stage2_compact` materializes a mostly-zero full table and the trace-bearing stage-2 prover recomputes dense prefix terms.
  Chosen first fix: direct-to-live compact trace construction plus shared-product ring-row materialization, with dense-table slicing kept as the reference oracle.
  Guardrail: do not re-enable two-round/prefix fusion until dense round-polynomial equivalence is tested; construction-only equivalence is not enough for that riskier slice.
- **[2026-06-07] P2 preweighting belongs in compact trace construction, not in `AkitaStage2Prover::new`.**
  A constructor-local scan removes per-round `gamma_tr` multiplies, but fp32 single-run profiling showed the scan can eat the small-field gain.
  Chosen fix: thread the trace batch scalar into the root/recursive compact trace builders and write preweighted live entries directly, then remove `gamma_tr` from the stage-2 prover constructor.
  Guardrail: keep the public unscaled `build_trace_stage2_compact` helper as the dense-equivalence oracle and add a scaled compact equivalence test rather than changing verifier wire semantics.
- **[2026-06-06] Treat `stash@{0}` as a design draft, not a patch to land wholesale.**
  Reason: it applies with conflicts against the two local public M-row commits and re-exports a missing `trace_weight/stage2.rs` module.
  Working approach: recover intent file-by-file and recreate missing helper code from the existing trace-weight primitives plus the spec.
- **[2026-06-06] EOR trace target is the EOR final claim, not the unweighted protocol-point opening.**
  Reason: the EOR sumcheck final oracle includes the public tail equality factor(s). The no-y-ring stage-2 trace term must therefore scale its public trace weights by the recursive `final_factor` or root `factors_by_point[point_idx]`, with input contribution `gamma_tr * final_claim`.
  Working approach: write this into the spec before EOR implementation, then use it as the Slice 3 verifier/prover acceptance gate.
- **[2026-06-06] Non-ZK EOR verification is split at the final-oracle boundary.**
  The verifier will use `SumcheckProof::verify` to absorb/check all EOR rounds and recover `(final_claim, rho)`, then bind that `final_claim` through the fused stage-2 trace wire.
  Reason: after removing on-wire y-rings, the verifier cannot independently evaluate the EOR final oracle before the trace proof; forcing the trace term to be the final-oracle check preserves soundness without reintroducing witness data.
- **[2026-06-06] ZK trace binding carries both true and masked trace claims.**
  Prover stage-2 arithmetic uses the true trace target, while transcript-visible ZK sumcheck input uses the masked public handle and adds its mask to `AkitaStage2Verifier::initial_claim_mask`.
  For recursive non-EOR this masked handle is the carried `next_w_eval`; for ZK EOR it is the masked final EOR sumcheck claim replayed from the masked EOR proof.
  Reason: this preserves the existing masked-sumcheck convention while removing y-ring hiding masks entirely.

## Deviations

## Tradeoffs

- **[2026-06-07] Deferred trace-aware two-round prefix until after construction cleanup.**
  Considered immediately re-enabling the old two-round prefix with a trace relation grid, but that changes round 0/1 polynomial generation and cached round-2 handoff.
  Starting with direct compact construction is lower risk, has a simple dense-table oracle, and gives cleaner profiles for deciding whether deeper sumcheck surgery is still needed.
- **[2026-06-07] Avoid field-size heuristic gating for P2.**
  Considered preweighting only for large extension fields because fp128 benefits more visibly than fp32.
  Rejected that as a brittle type-size heuristic; construction-time preweighting removes the extra scan for every field and preserves one semantic representation for the sumcheck.

## Open Questions

## Slice Retrospectives

### 2026-06-07 retrospective: P1 direct compact trace construction

Commit: `c1da312c fix(trace): build compact trace weights directly`

**Bottom line:** no blockers.
The prover trace construction path now builds the live compact table directly
instead of materializing a mostly-zero full Boolean trace table, and K > 1 ring
row materialization shares one ring product across all coordinate shifts.

- `Bug fixed:` `build_trace_stage2_compact` used to build a full
  `layout.table_len()` table and then copy `live_x_cols`.
  The direct compact builders preserve witness order and skip ring-row work for
  blocks whose opening-digit columns are outside the live witness.
- `Performance:` same info-profile mode improved
  `onehot_fp32_d128 nv28` prove from `2.950168834s` to `2.170337125s`, and
  `onehot_fp64_d128 nv28` prove from `2.368386084s` to `1.544877417s`.
  Verify stayed stable (`0.121516459s` to `0.114498791s` fp32;
  `0.080819709s` to `0.079669291s` fp64).
- `Adversarial review:` Epicurus and Beauvoir independently found the same
  representation waste: dense full-table construction plus trace disabling
  prefix optimizations. Pasteur found the wider PR matrix also has
  ring-switch quotient and schedule-shape regressions, so this construction
  slice is necessary but not the whole prover-performance fix.
- `Risk:` the row optimization relies on commutativity:
  `(ring · X^c) * sigma_-1(packed) == (ring * sigma_-1(packed)) · X^c`.
  Dense ring compact tests compare against the previous table semantics.
- `Deferred:` trace-aware prefix/two-round optimization and the fp128 schedule
  shape regression remain for subsequent slices.
- `Non-issue checked:` small-field verifier speedup remains intact in the same
  representative profiles.
- `Verification:`
  - `cargo fmt -q`
    → passed
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    → `15 passed; 0 failed`
  - `cargo check -p akita-pcs --all-features -q`
    → passed
  - `scripts/check-rust-file-lines.sh`
    → `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    → passed
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp32_d128 AKITA_NUM_VARS=28 cargo run --release --example profile > /tmp/akita-profile-fp32-prover-after-p1.txt 2>&1`
    → `akita batched prove complete levels=6 elapsed_s=2.170337125`; `verify OK label="onehot_fp32_d128" elapsed_s=0.114498791`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp64_d128 AKITA_NUM_VARS=28 cargo run --release --example profile > /tmp/akita-profile-fp64-prover-after-p1.txt 2>&1`
    → `akita batched prove complete levels=6 elapsed_s=1.544877417`; `verify OK label="onehot_fp64_d128" elapsed_s=0.079669291`

### 2026-06-07 retrospective: P2 construction-time trace preweighting

Commit: `b5ef7caa fix(trace): preweight compact stage2 trace`

**Bottom line:** no blockers.
The trace compact table is now built already multiplied by `gamma_tr`, so the
stage-2 prover avoids repeated per-round trace multiplications without adding a
constructor-local scan over the compact table.

- `Bug fixed:` the first P2 attempt preweighted inside
  `AkitaStage2Prover::new`; fp32 showed that the extra read/branch/write pass
  could eat the small-field win. Moving the scale into compact construction
  writes only live trace entries once and keeps the sumcheck table invariant
  simple: `trace_compact = gamma_tr * trace_weight`.
- `Performance:` same one-off profile mode improved/held the trace-bearing
  shapes:
  `onehot_fp32_d128 nv28` `2.182631292s` prove, within noise of P1
  `2.170337125s` and better than the rejected constructor-scan `2.396017041s`;
  `onehot_fp64_d128 nv28` `1.3406125420000001s` prove;
  `onehot_fp128_d64 nv32` `1.479183917s` prove;
  `onehot_fp128_d128 nv32` `1.386525417s` prove.
- `Adversarial review:` Kant recommended exactly this construction-time
  preweighting and deferring planner/ring-switch changes. Volta agreed the
  algebra is sound but suggested a field-size policy branch; rejected for this
  slice because construction-time preweighting removes the fp32 scan without
  carrying two trace table representations through the stage-2 prover.
- `Risk:` `AkitaStage2Prover` now assumes any non-`None` trace table is already
  batched by `gamma_tr`. This is private prover plumbing; the public unscaled
  `build_trace_stage2_compact` remains as the dense-equivalence oracle and the
  scaled helper has an explicit `scaled == gamma_tr * dense_slice` test.
- `Deferred:` trace-aware two-round prefix remains deferred. Profiles now point
  at it only as a possible future optimization, not as a correctness or P2
  acceptance blocker.
- `Non-issue checked:` the fixed-thread benchmark matrix
  (`RAYON_NUM_THREADS=8`, 3 runs + 1 warmup) completed all cases, but its
  absolute timings are not comparable to the P1 one-off baselines because P1
  was measured with default Rayon settings.
- `Verification:`
  - `cargo fmt -q`
    → passed
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    → `16 passed; 0 failed`
  - `cargo test -p akita-prover --no-default-features stage2_ -q`
    → `23 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features fp32_ring_subfield_extension_rejects_tampered_reduction_partial -- --nocapture`
    → `test scheme::tests::fp32_ring_subfield::fp32_ring_subfield_extension_rejects_tampered_reduction_partial ... ok`
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    → passed
  - `scripts/check-rust-file-lines.sh`
    → `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp32_d128 AKITA_NUM_VARS=28 ./target/release/examples/profile > /tmp/akita-profile-fp32-pr-after-p2-builder-scaled.txt 2>&1`
    → `akita batched prove complete levels=6 elapsed_s=2.182631292`; `verify OK label="onehot_fp32_d128" elapsed_s=0.122929834`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp64_d128 AKITA_NUM_VARS=28 ./target/release/examples/profile > /tmp/akita-profile-fp64-pr-after-p2-builder-scaled-seq.txt 2>&1`
    → `akita batched prove complete levels=6 elapsed_s=1.3406125420000001`; `verify OK label="onehot_fp64_d128" elapsed_s=0.074682125`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 ./target/release/examples/profile > /tmp/akita-profile-fp128-d64-pr-after-p2-builder-scaled-seq.txt 2>&1`
    → `akita batched prove complete levels=8 elapsed_s=1.479183917`; `verify OK label="onehot_fp128_d64" elapsed_s=0.031577583`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 ./target/release/examples/profile > /tmp/akita-profile-fp128-d128-pr-after-p2-builder-scaled.txt 2>&1`
    → `akita batched prove complete levels=6 elapsed_s=1.386525417`; `verify OK label="onehot_fp128_d128" elapsed_s=0.036632583`

### 2026-06-06 retrospective: trace-stage2 helper reconstruction

**Bottom line:** no blockers.
The missing `trace_weight/stage2.rs` helper layer has been reconstructed as shared glue around existing trace-weight table builders and closed-form eval.

- `Risk:` `trace_stage2_enabled` currently refuses the extension-opening-reduction bridge by construction.
  This is intentional for the helper slice, but the final PR still needs a real K > 1 / EOR path rather than a silent skip.
- `Non-issue checked:` The dense trace-weight algebra already covered K = 1, K = 4, and K = 8.
  The helper tests only add compaction and verifier-wire dispatch anchors.
- `Verification:`
  - `cargo fmt -q && cargo test -p akita-types --no-default-features trace_weight`
    → `test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 130 filtered out`

### 2026-06-06 retrospective: proof-wire removal in `akita-types`

**Bottom line:** no blockers.
`AkitaLevelProof`, `TerminalLevelProof`, and `AkitaBatchedFoldRoot` no longer carry `y_ring` / `y_rings` in `akita-types`, and the shape/serialization byte tests pass locally.

- `Risk:` This is only the data-shape layer.
  Prover/verifier call sites still need the fused trace term before `akita-pcs` can pass.
- `Non-issue checked:` Headerless shape serialization remains internally consistent after removing the y-ring coefficient counts.
- `Verification:`
  - `cargo fmt -q && cargo test -p akita-types --no-default-features`
    → `test result: ok. 139 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`

### 2026-06-06 retrospective: non-EOR stage-2 trace debugging

**Bottom line:** K = 1, non-EOR trace internalization is now coherent for the PCS no-default lib paths exercised by single, batched, onehot, and monomial tests.

- `Bug fixed:` the fused prefix-x cache path computed the next round while folding the current x challenge, but omitted the trace term.
  Trace-bearing stage-2 now avoids that fusion path while keeping the ordinary prefix folds.
- `Bug fixed:` odd live-width x folds can pair a live column with a padded column.
  The trace accumulator now treats missing padded trace entries as zero, matching the padded witness convention.
- `Bug fixed:` the verifier trace evaluator assumed power-of-two opening digit depth and raw Lagrange inner coordinates.
  It now directly sums the actual opening-digit segment and evaluates the basis-correct `inner_opening_ring`.
- `Bug fixed:` root verifier trace wiring for batched claims used one unweighted block group.
  It now mirrors the prover's claim-scaled block layout.
- `Remaining blocker:` extension-opening-reduction verifier paths still depend on the removed `y_ring` final oracle.
  The current non-ZK EOR verifier uses a zero placeholder where it previously needed y-ring data, so fp32 ring-subfield EOR tests still fail until K > 1 / EOR trace internalization is completed.
- `Verification:`
  - `cargo test -p akita-pcs --no-default-features --lib`
    → `19 passed; 2 failed`, with the remaining failures isolated to fp32 extension-opening-reduction verifier tests.

### 2026-06-06 retrospective: Slice 0 branch stabilization and Bugbot triage

**Bottom line:** Slice 0 is complete and ready to commit.
The branch now compiles through the planned stabilization gates, the stale ZK tests no longer reference removed public y-ring proof fields, and the root trace wire no longer assumes all batched root claims use one opening point.

- `Bug fixed:` Bugbot's multipoint trace issue was a real representation bug, not just a wrong loop.
  `TraceStage2OpeningOwned::Field` now carries explicit block-offset terms, each with its own packed inner opening; prover and verifier both build one term per claim using `claim_to_point`.
- `Bug fixed:` stale `TraceOpeningAtPoint::Field` tests now use the basis-correct packed inner opening field.
- `Bug fixed:` profile reporting no longer reads removed `y_ring` / `y_rings` proof fields and reports internalized y-ring bytes as zero.
- `Bug fixed:` ZK materialized-M regression now sizes public rows as zero for the updated relation, and stale ZK y-ring tamper/shape checks were replaced with live proof-field checks.
- `Guardrail checked:` I searched for old `public_rows().first()` / `prepared_points.first()` trace construction patterns after the fix; remaining first/zero point uses are recursive-singleton/EOR-specific paths, not root multipoint K = 1 trace batching.
- `Risk:` the K > 1 / EOR verifier still has planned zero-y-ring placeholders and remains for later slices.
  This slice intentionally fixed K = 1 multipoint trace wiring without pretending EOR is done.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    → `10 passed; 0 failed`
  - `cargo check -p akita-pcs --examples --no-default-features -q`
    → passed
  - `scripts/check-rust-file-lines.sh`
    → passed, cap 1500
  - `cargo doc -q --no-deps --all-features`
    → passed
  - `cargo test -p akita-pcs --test multipoint_batched_e2e multipoint_dense_round_trip_with_bundles_per_point --no-default-features -- --nocapture`
    → passed
  - `cargo test -p akita-pcs --features zk --test zk zk_multipoint_ring_switch_relation_matches_materialized_m -- --nocapture`
    → passed

### 2026-06-06 retrospective: Slice 1 soundness derivation gate

**Bottom line:** Slice 1 is complete and ready to commit.
The spec now contains the extraction/soundness argument for dropping public-output `M` rows and adding the fused trace term, including the EOR-specific target rule.

- `Decision locked:` EOR trace binding uses `gamma_tr * final_claim` and scales trace weights by `final_factor` / `factors_by_point`.
  This avoids verifier reconstruction from removed y-rings and avoids inverting a possibly awkward public factor.
- `Test anchor added:` trace-weight ring tests now include the K = 2 `Ext2` path, so the algebraic anchors cover K = 1, 2, 4, and 8 locally.
- `Adversarial review:` older acceptance/execution wording that only said `opening` was updated to mention the EOR final-claim variant explicitly, preventing Slice 3 from binding the wrong target.
- `Risk:` this slice is a gate, not the EOR implementation.
  The verifier still has zero-y-ring placeholders until Slice 3 replaces them.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    → `12 passed; 0 failed`

### 2026-06-06 retrospective: Slice 2 term-general trace weights

**Bottom line:** Slice 2 is complete and ready to commit.
The trace-weight opening representation is now term-based for both K = 1 scalar block weights and K > 1 ring block weights.

- `Change:` added `TraceRingBlockOpening`, `build_trace_weight_table_ring_terms`, and `trace_stage2_opening_owned_ring_terms`.
  The old single-ring helper now constructs a one-term opening through the same path.
- `Bug-prevention:` ring final-point evaluation and table materialization now sum terms by block offset, so later EOR wiring can scale root terms by `factors_by_point` and recursive terms by `final_factor` without needing a second representation.
- `Test anchor added:` a K = 4 multi-term regression checks both dense final-point evaluation and witness-dot contraction with two different packed inner points.
- `Adversarial review:` searched for the old one-ring `TraceOpeningAtPoint::Ring { block_rings, packed_inner_point }` shape; only the new term APIs and compatibility constructor remain.
- `Risk:` no EOR caller is switched on yet; `trace_stage2_enabled` still gates EOR until Slice 3.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    → `13 passed; 0 failed`
  - `cargo check -p akita-pcs --examples --no-default-features -q`
    → passed
  - `cargo doc -q --no-deps --all-features`
    → passed
  - `scripts/check-rust-file-lines.sh`
    → passed

### 2026-06-06 retrospective: Slice 3 non-ZK EOR trace cutover

**Bottom line:** Slice 3 is complete and ready to commit.
Non-ZK root and recursive EOR now verify sumcheck rounds to recover `(final_claim, rho)` and bind that final claim through the fused stage-2 trace term, with root per-claim trace scales from `factors_by_point` and recursive trace scale from `final_factor`.

- `Bug fixed:` root and recursive non-ZK verifier paths no longer construct zero y-ring placeholders for EOR final-oracle checks.
- `Bug fixed:` prover trace construction no longer uses the unweighted recovered protocol-point opening as the EOR trace target.
  EOR trace input is `gamma_tr * final_claim`, and trace weights are scaled by the public tensor factors.
- `Guardrail kept:` ZK trace remains disabled in `trace_stage2_enabled` under the `zk` feature.
  Existing masked y-ring final-relation code is intentionally left fenced for Slice 4.
- `Adversarial review:` searched for `zero_y_ring`, `zero_y_rings`, `ExtensionOpeningReductionVerifier`, K=1-only trace helpers, `internal_claims`, and `check_extension_opening_reduction_output` across prover/verifier protocol paths.
  The deleted verifier name had one stale comment, now fixed; remaining internal-claim checks are ZK-fenced or prover-local construction checks.
- `Risk:` the non-ZK split relies on the stage-2 trace term being present whenever EOR is present.
  This is now enforced by enabling supported K = 2/4/8 non-ZK trace dispatch, but future unsupported extension degrees must keep returning `AkitaError` from the trace evaluator.
- `Verification:`
  - `cargo fmt -q`
  - `cargo check -p akita-pcs --examples --no-default-features -q`
    → passed
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    → `13 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features fp32_ring_subfield_outer_extension_uses_root_tensor_projection -- --nocapture`
    → `1 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features fp32_ring_subfield_multipoint_extension_uses_root_tensor_projection -- --nocapture`
    → `1 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features --lib`
    → `21 passed; 0 failed`
  - `cargo check -p akita-pcs --all-features -q`
    → passed
  - `scripts/check-rust-file-lines.sh`
    → `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`

### 2026-06-06 retrospective: Slice 4 ZK trace cutover

**Bottom line:** Slice 4 protocol work is complete.
ZK no longer masks or absorbs y-rings; root/recursive EOR and non-EOR trace claims now carry a true prover target plus a transcript-visible masked handle, and Stage 2 records the trace mask and final trace oracle in the ZK R1CS relation.

- `Bug fixed:` root ZK EOR branches still skipped `CHALLENGE_TRACE_BATCH` and passed `C::zero()` into Stage 2.
  This shifted verifier transcript replay once the verifier sampled trace challenges unconditionally. Root ZK now samples/passes `gamma_tr` in every root fold branch.
- `Bug fixed:` recursive ZK EOR initially used the unreduced internal opening with scale `1` as the true trace target while the public handle was the EOR final claim.
  It now matches the non-ZK rule: true target is `final_claim`, and trace weights are scaled by `final_factor`.
- `Bug fixed:` the obsolete y-ring hiding witness slots and verifier y-ring recovery relations are removed.
  ZK hiding cursor alignment is now EOR partial masks/round pads, Stage 1 pads, Stage 2 pads, and next-w eval masks only.
- `Adversarial review:` searched for stale `y_rings_masked`, `take_ring`, `zk_recovered_y_ring_lc`, cfg-gated `gamma_tr`, and `C::zero()`/`L::zero()` trace placeholders.
  Remaining matches were either removed or changed to describe the actual EOR scaling state.
- `Deferred:` `cargo nextest run --profile ci-all-features` fails only at `akita-config::generated_tables::generated_schedule_tables_match_find_schedule`.
  The failure reports 207 generated-table drift issues and asks to run `cargo run --release -p akita-config --bin gen_schedule_tables -- crates/akita-planner/src/generated`.
  This is the planned Slice 5 table-regeneration gate, not a remaining ZK trace verifier failure.
- `Verification:`
  - `cargo fmt -q`
  - `cargo check -p akita-pcs --all-features -q`
    → passed
  - `cargo test -p akita-pcs --features zk --test zk zk_fp32_extension_opening_reduction_folded_root_verifies -- --nocapture`
    → `1 passed; 0 failed`
  - `cargo test -p akita-pcs --features zk --test zk zk_multipoint_ring_switch_relation_matches_materialized_m -- --nocapture`
    → `1 passed; 0 failed`
  - `cargo test -p akita-pcs --features zk --test zk -q`
    → `10 passed; 0 failed`
  - `cargo check -p akita-pcs --examples --no-default-features -q`
    → passed
  - `cargo test -p akita-pcs --no-default-features --lib -q`
    → `21 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features fp32_ring_subfield_outer_extension_uses_root_tensor_projection -- --nocapture`
    → `1 passed; 0 failed`
  - `scripts/check-rust-file-lines.sh`
    → `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    → passed
  - `cargo nextest run --profile ci-all-features`
    → `604/865 tests run: 603 passed, 1 failed, 10 skipped`; failed test `akita-config::generated_tables generated_schedule_tables_match_find_schedule` due generated schedule-table drift.

### 2026-06-06 retrospective: Slice 5 schedule tables and profile sizing

**Bottom line:** Slice 5 is complete and ready to commit.
The generated planner tables have been regenerated for both default and ZK feature surfaces, and the `onehot_fp128_d128` profile reports no serialized y-ring payloads at any fold level.

- `Bug fixed:` the all-features generated-table drift required running the generator with `--features zk`, not only the default generator command.
  Both feature surfaces now agree with `find_schedule`.
- `Adversarial review:` inspected `git status --short` and `git diff --stat` after regeneration.
  The only source changes are generated schedule tables; the worklog/handoff remain untracked and no profile trace artifact is staged.
- `Non-issue checked:` the optimizer sometimes chooses an extra compact fold after y-ring removal.
  This is acceptable because the table equality test compares shipped tables to the runtime DP, and the profile reports the actual proof-size outcome.
- `Size check:` `onehot_fp128_d128` at `nv=32` now reports `total=146384` bytes with seven fold levels and `y_rings=0 bytes (internalized)` at root, every intermediate level, and terminal.
  The mandatory same-schedule removed payload is `7 * 128 * 16 = 14336` bytes; the profile's structured report also emits `y_ring_bytes = 0`.
- `Open Questions audit:` none; the Open Questions section is empty.
- `Verification:`
  - `cargo run --release -p akita-config --bin gen_schedule_tables -- crates/akita-planner/src/generated`
    → wrote the 8 default generated schedule table files.
  - `cargo run --release -p akita-config --features zk --bin gen_schedule_tables -- crates/akita-planner/src/generated`
    → wrote the 8 ZK generated schedule table files.
  - `cargo test -p akita-config generated_schedule_tables_match_find_schedule -- --nocapture`
    → `test generated_schedule_tables_match_find_schedule ... ok`
  - `cargo test -p akita-config --all-features generated_schedule_tables_match_find_schedule -- --nocapture`
    → `test generated_schedule_tables_match_find_schedule ... ok`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=warn AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile`
    → `proof: total=146384 bytes, akita_fold=45184 bytes, tail=101200 bytes, framing=0 bytes, levels=7`; every fold-level breakdown printed `y_rings=0 bytes (internalized)`.
  - `cargo fmt -q`
    → passed
  - `cargo test -p akita-config regen_diff -- --nocapture`
    → `regen_diff_vs_shipped_tables ... ignored, diagnostic`; command completed successfully.

### 2026-06-06 retrospective: Slice 6 negative coverage and PR feedback audit

**Bottom line:** Slice 6 is complete and ready to commit.
The internalized trace relation now has targeted negative coverage for root fold handles, recursive fold handles, terminal packed witness digits, and fp32 extension-opening reduction partials; transcript hardening still passes with the trace-y-ring payload removed.

- `Tests added:` non-ZK end-to-end tamper tests now mutate the root `v` trace handle, the first recursive `next_w_commitment` trace handle, and the terminal `e_hat` packed digit.
  Each verifies the malformed proof is rejected with `AkitaError::InvalidProof`.
- `Tests added:` the fp32 subfield outer-extension fixture now mutates the first extension-opening reduction partial and verifies rejection with `InvalidProof`.
  This directly guards the EOR trace cutover against accepting stale or forged reduction handles.
- `Bug fixed while testing:` the terminal tamper test initially derived its witness layout from the root fixture layout.
  It now derives the terminal layout from the last runtime fold step, matching the actual packed terminal witness.
- `Bugbot audit:` PR #154 currently has two Bugbot review threads.
  The high-severity multipoint trace-weight thread is resolved and outdated; the medium-severity ZK stale-public-M thread is resolved.
  No unresolved actionable Bugbot threads remain at the current audit point.
- `Adversarial review:` inspected the Slice 6 diff for accidental production changes, verifier-reachable panics, stale ZK compilation issues, line-cap drift, and tests that only prove panic behavior.
  The new panic/expect sites are limited to test-fixture assertions, all source edits are in test files, and the targeted tamper paths exercise verifier rejection rather than fixture construction failure.
- `Caveat:` `cargo test -p akita-pcs --all-features fp32_ring_subfield_extension_rejects_tampered_reduction_partial -- --nocapture` compiled the all-features surface successfully but ran zero matching tests because this module is transparent-feature scoped under all-features.
- `Open Questions audit:` none; the Open Questions section is empty.
- `Verification:`
  - `cargo fmt -q`
    -> passed
  - `cargo test -p akita-pcs --test akita_e2e --no-default-features trace_internalization_rejects_tampered_root_fold_handle -- --nocapture`
    -> `1 passed; 0 failed`
  - `cargo test -p akita-pcs --test akita_e2e --no-default-features trace_internalization_rejects_tampered_recursive_fold_handle -- --nocapture`
    -> `1 passed; 0 failed`
  - `cargo test -p akita-pcs --test akita_e2e --no-default-features trace_internalization_rejects_tampered_terminal_e_hat_digit -- --nocapture`
    -> `1 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features fp32_ring_subfield_extension_rejects_tampered_reduction_partial -- --nocapture`
    -> targeted library test passed
  - `cargo test -p akita-pcs --features logging-transcript --test transcript_hardening -- --nocapture`
    -> `8 passed; 0 failed`
  - `cargo check -p akita-pcs --all-features -q`
    -> passed
  - `scripts/check-rust-file-lines.sh`
    -> `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    -> passed
  - `cargo test -p akita-pcs --all-features fp32_ring_subfield_extension_rejects_tampered_reduction_partial -- --nocapture`
    -> command completed successfully; matching test filtered out under all-features transparent-module gating.

### 2026-06-06 retrospective: final gate and CI closeout

**Bottom line:** The planned implementation is complete.
All slices have been committed and pushed, local final verification passed, PR #154 CI is green on pushed head `4bd8e76992055394d21f54074773a9fc2c190210`, and Bugbot has no unresolved actionable threads.

- `Remote CI:` `gh pr checks 154 --repo LayerZero-Labs/akita --watch --interval 30` reached all-pass.
  The final long `Test` check passed in 31m32s after remote non-ZK and all-features test passes both succeeded; `Profile benchmarks` passed in 13m44s.
- `Bugbot audit:` Cursor Bugbot is green on the pushed head, and its latest summary comment for commit `4bd8e76992055394d21f54074773a9fc2c190210` reports the expected trace-internalization summary without new issue threads.
  The two older Bugbot threads remain resolved: wrong multipoint trace block weights is resolved/outdated; ZK stale public M rows is resolved.
- `Local final gate:` both local nextest CI profiles passed, the stricter all-target/all-feature clippy gate passed, and the release profile still reports `y_rings=0 bytes (internalized)` for root, every intermediate fold, and terminal.
- `Profile evidence:` `onehot_fp128_d128` at `nv=32` reports `proof: total=146384 bytes, akita_fold=45184 bytes, tail=101200 bytes, framing=0 bytes, levels=7`.
  A noisy first profile run created an ignored Perfetto trace JSON, which was removed after the quiet evidence run.
- `Adversarial review:` checked PR review threads, PR comments, PR check rollup, local status, and final profile output.
  The branch is pushed at the final commit; no source changes remain unstaged or unpushed, and the only local untracked files are the required never-commit handoff/worklog notes.
- `Open Questions audit:` none; the Open Questions section is empty.
- `Verification:`
  - `cargo fmt -q`
    -> passed
  - `cargo clippy --all --all-targets --all-features -- -D warnings`
    -> passed
  - `cargo nextest run --profile ci-non-zk`
    -> `869 tests run: 869 passed (5 slow), 10 skipped`
  - `cargo nextest run --profile ci-all-features`
    -> `869 tests run: 869 passed (2 slow), 10 skipped`
  - `AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile`
    -> passed
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=warn AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile`
    -> `proof: total=146384 bytes, akita_fold=45184 bytes, tail=101200 bytes, framing=0 bytes, levels=7`; every fold-level breakdown printed `y_rings=0 bytes (internalized)`.
  - `gh pr checks 154 --repo LayerZero-Labs/akita --watch --interval 30`
    -> all checks passed, including `Test`, `Profile benchmarks`, and `Cursor Bugbot`.
  - GitHub review-thread audit via `_list_pull_request_review_threads`
    -> two Bugbot threads found, both resolved; no unresolved Bugbot comments.

### 2026-06-07 retrospective: small-field verifier trace regression

Commit: `06bb07bc` (`fix(trace): speed small-field verifier trace eval`)

**Bottom line:** no blockers. The fp32/fp64 verifier regression came from the extension trace-weight final-point evaluator recomputing one trace reduction per ring coordinate; it now forms the MLE-weighted shifted ring once and decodes a single linear trace per block.

- `Bug fixed:` local `onehot_fp32_d128` at `nv=28` verify time dropped from `0.679022s` before the fix to `0.112886s` after the fix, with proof size unchanged at `116288 bytes`.
- `Bug fixed:` local `onehot_fp64_d128` at `nv=28` verify time dropped from `1.287189s` before the fix to `0.079579s` after the fix, with proof size unchanged at `118016 bytes`.
- `Adversarial review:` rejected disabling trace internalization for small fields because that would hide the regression by weakening the soundness cutover.
  The source diff is confined to `crates/akita-types/src/field_reduction.rs`; it does not alter proof wire shape, transcript order, or prover trace-table materialization.
- `Non-issue checked:` the old ring-switch row evaluator was not the bottleneck.
  Before the fix, root `stage2_ring_switch_row_eval` was about `11.4ms` while `stage2_expected_output_claim` was about `882ms`; after the fix the root expected-output span is about `43.8ms` with the row eval still about `11.4ms`.
- `Risk:` the optimized path relies on the same F-linear trace map extended over the verifier claim field.
  The dense trace-weight table equivalence tests are the acceptance oracle for K=1,2,4,8 and passed after the change.
- `Open Questions audit:` none; the Open Questions section is empty.
- `Verification:`
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    -> `13 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features fp32_ring_subfield_extension_rejects_tampered_reduction_partial -- --nocapture`
    -> targeted library test passed
  - `cargo check -p akita-pcs --all-features -q`
    -> passed
  - `cargo fmt -q`
    -> passed
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    -> passed
  - `scripts/check-rust-file-lines.sh`
    -> `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`
  - `cargo test`
    -> passed, including trace-internalization tamper tests and `akita-types` trace-weight tests
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=warn AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp32_d128 AKITA_NUM_VARS=28 cargo run --release --example profile`
    -> `[onehot_fp32_d128] verify OK: 0.112886s`; `proof: total=116288 bytes`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=warn AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp64_d128 AKITA_NUM_VARS=28 cargo run --release --example profile`
    -> `[onehot_fp64_d128] verify OK: 0.079579s`; `proof: total=118016 bytes`
  - `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot_fp64_d128 AKITA_NUM_VARS=28 cargo run --release --example profile`
    -> root `stage2_expected_output_claim` `43.8ms`; root `stage2_ring_switch_row_eval` `11.4ms`; `[onehot_fp64_d128] verify OK: 0.079776s`

### 2026-06-07 retrospective: prover trace/sumcheck performance closeout

Commits:
- `c1da312c` (`fix(trace): build compact trace weights directly`)
- `b5ef7caa` (`fix(trace): preweight compact stage2 trace`)

**Bottom line:** the prover regressions were from avoidable trace-table construction and an extra compact-table scan in the trace-bearing stage-2 sumcheck path, not from verifier wire semantics. The final pushed head is `b5ef7caaf0f42a296826d14add09681d6253b34a`; PR #154 CI is all green on that head, and Cursor Bugbot has no unresolved actionable threads.

- `Bug fixed:` `build_trace_stage2_compact` no longer materializes a mostly-zero dense trace table and slices it back down to live columns. Field and ring compact builders now construct the live compact trace weights directly, using dense-table slicing only as the test oracle.
- `Bug fixed:` the first P2 shape would have multiplied `gamma_tr` in `AkitaStage2Prover::new`, adding a second scan over the compact table. The final implementation threads `gamma_tr` into root/recursive compact trace construction and writes `trace_compact = gamma_tr * trace_weight` once.
- `Adversarial review:` Kant and Volta both flagged the constructor pre-scan as the likely small-field prover regression. I rejected field-size heuristic gating because it would create two semantic representations and would not address the root duplicated scan.
- `Risk checked:` no transcript, proof-shape, or verifier acceptance semantics changed. The trace-bearing stage-2 prover now assumes preweighted compact entries, while public unscaled compact builders remain available as dense-equivalence oracles.
- `Performance evidence:` local release runs after P2:
  - `onehot_fp32_d128 nv28` prove `2.182631292s`, verify `0.122929834s`.
  - `onehot_fp64_d128 nv28` sequential prove `1.3406125420000001s`, verify `0.074682125s`.
  - `onehot_fp128_d64 nv32` sequential prove `1.479183917s`, verify `0.031577583s`.
  - `onehot_fp128_d128 nv32` prove `1.386525417s`, verify `0.036632583s`.
- `Remote CI:` `gh run watch 27080912683 --repo LayerZero-Labs/akita --interval 30 --exit-status` passed. The final `Test` job passed in `37m36s`; `Profile benchmarks` passed in `12m51s`; Cursor Bugbot passed in `1m49s`.
- `Bugbot audit:` GraphQL review-thread audit found the two older Cursor Bugbot threads. `Wrong trace block weights multipoint` is resolved/outdated; `ZK test stale public M rows` is resolved. The latest `Cursor Bugbot` check is green on `b5ef7caa`.
- `Git state:` branch `quang/y-ring-trace-internalization` is pushed to `layerzero/quang/y-ring-trace-internalization` at `b5ef7caa`; the only local untracked files are this never-commit worklog and the never-commit handoff.
- `Open Questions audit:` none; the Open Questions section is empty.
- `Verification:`
  - `cargo fmt -q`
    -> passed
  - `cargo test -p akita-types --no-default-features trace_weight -q`
    -> `16 passed; 0 failed`
  - `cargo test -p akita-prover --no-default-features stage2_ -q`
    -> `23 passed; 0 failed`
  - `cargo test -p akita-pcs --no-default-features fp32_ring_subfield_extension_rejects_tampered_reduction_partial -- --nocapture`
    -> targeted library test passed
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    -> passed
  - `scripts/check-rust-file-lines.sh`
    -> `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`
  - `gh pr checks 154 --repo LayerZero-Labs/akita`
    -> all checks passed, including `Test`, `Profile benchmarks`, and `Cursor Bugbot`

### 2026-06-07 retrospective: remove resurrected 4x30 one-hot E2E

Commit: `c8d0d27a` (`test(pcs): remove resurrected 4x30 onehot e2e`)

**Bottom line:** the slow `akita-pcs::akita_e2e::batched_onehot_4x30_keeps_folding_past_oversized_tail` test was merge fallout from `6959457e` and contradicted `specs/profile-bench-coverage-matrix.md`, which says that full E2E proof was removed and replaced by planner-level schedule coverage.

- `Bug fixed:` deleted the resurrected full `nv30 x np4` one-hot E2E proof from `crates/akita-pcs/tests/akita_e2e.rs` instead of merely adding `#[cfg(not(feature = "zk"))]`.
- `Coverage preserved:` `akita-config::proof_optimized::tests::batched_onehot_4x30_plan_keeps_terminal_witness_bounded` still checks the 4x30 final-witness bound cheaply, and `akita-pcs::akita_e2e::batched_onehot_same_point_round_trip` still covers recursive-suffix truncation rejection on a smaller E2E fixture.
- `Adversarial review:` guarding the old test as non-zk-only would still leave a >1000s outlier in one CI pass and would conflict with the spec's explicit cleanup note. Removal is the intended full cutover.
- `Remote CI:` PR #154 is green on pushed head `c8d0d27a`. `Test` passed in `23m54s`, `Profile benchmarks` passed in `12m56s`, and Cursor Bugbot passed in `1m45s`.
- `Timing artifact:` downloaded `ci-test-timing-data` from run `27083506371`; `batched_onehot_4x30_keeps_folding_past_oversized_tail` is absent from `summary.json`, `report.md`, and `comment.md`. The non-zk slowest table now tops out at `179.9s` (`aggregated_dense_nv17_batch5`) instead of the old `1045.2s` 4x30 E2E outlier.
- `Verification:`
  - `cargo fmt -q`
    -> passed
  - `cargo nextest list --profile ci-non-zk --no-default-features --features parallel,disk-persistence -E 'test(batched_onehot_4x30_keeps_folding_past_oversized_tail)'`
    -> no matching tests
  - `cargo nextest list --profile ci-all-features --all-features -E 'test(batched_onehot_4x30_keeps_folding_past_oversized_tail)'`
    -> no matching tests
  - `cargo nextest run --profile ci-non-zk --no-default-features --features parallel,disk-persistence -E 'test(batched_onehot_4x30_plan_keeps_terminal_witness_bounded) or test(batched_onehot_same_point_round_trip)'`
    -> `2 tests run: 2 passed, 885 skipped`
  - `cargo nextest list --profile ci-all-features --all-features -E 'test(batched_onehot_4x30_plan_keeps_terminal_witness_bounded) or test(batched_onehot_4x30_keeps_folding_past_oversized_tail)'`
    -> no matching tests under all-features/zk
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    -> passed
  - `scripts/check-rust-file-lines.sh`
    -> `Rust file line-cap check passed: scanned 407 tracked Rust files; cap=1500; baseline_entries=0.`
  - `gh pr checks 154 --repo LayerZero-Labs/akita --watch --interval 30`
    -> all checks passed
  - `gh run download 27083506371 --repo LayerZero-Labs/akita --name ci-test-timing-data --dir /tmp/akita-ci-test-timing-c8d0d27a`
    -> downloaded timing artifact; old 4x30 E2E test name absent

### 2026-06-07 retrospective: CI slow-test trim

Commit: pending (`test(pcs): trim redundant slow e2e coverage`)

**Bottom line:** the PR timing comment showed CI was no longer dominated by the resurrected 4x30 E2E, but several full prove/setup tests were still duplicated across equivalent shapes. This slice trims representative-only coverage while keeping each unique protocol path exercised.

- `Trimmed:` removed `recursive_onehot_nv25`; `recursive_onehot_nv20` still proves/verifies Recursive setup mode, and `recursive_onehot_cross_mode_rejects_nv20` still checks Direct/Recursive structural rejection.
- `Trimmed:` removed `aggregated_onehot_nv23_batch4`; aggregated coverage still has one-hot root-direct, one-hot folded irregular batch, dense root-direct, dense folded irregular batch, and mixed dense/onehot folded E2E.
- `Trimmed:` simplified `fp128_degree_one_batched_proof_roundtrip_is_stable` to build one full proof, serialize it twice, deserialize, compare, and verify. This keeps the serialization roundtrip and deterministic-byte check without paying for a second identical proof.
- `Trimmed:` collapsed three batched verifier fixture tests into one proof generation that checks accept, wrong opening rejection, and oversized folded payload rejection.
- `Trimmed:` removed D64 ZK commitment-rerandomization and folded-v hiding duplicates; D32 and D128 still cover the same ZK paths at the low/high decomposition boundaries.
- `Protected:` did not trim `generated_schedule_tables_match_find_schedule`; `specs/ci-test-timing.md` explicitly names it the sole schedule drift guard in both CI passes.
- `Adversarial review:` I avoided trimming the setup-capacity matrix in this commit because it is broader than a duplicate-row deletion: converting those full E2Es into boundary checks needs a separate, explicit acceptance argument for every preset/capacity axis.
- `Verification so far:`
  - `cargo test -p akita-pcs --test recursive_setup_e2e recursive_onehot -- --nocapture`
    -> passed, `2 passed`
  - `cargo test -p akita-pcs --test batched_aggregated_e2e -- --nocapture`
    -> passed, `5 passed`
  - `cargo test -p akita-pcs batched_verify_accepts_consistent_openings_and_rejects_bad_inputs -- --nocapture`
    -> passed
  - `cargo test -p akita-pcs fp128_degree_one_batched_proof_roundtrip_is_stable -- --nocapture`
    -> passed

## Follow-ups

### 2026-06-07 council fixes (autonomous pass)

Context: user reviewed the council report and directed (1) apply the revised PR
title/description, (2) drop the dedicated `gamma_tr` challenge and instead batch
the trace term via the existing stage-2 batching challenge (FS-ordering fix),
(3) D32 tables are non-load-bearing (keep or remove), (4) confirm the `r_hat`
shrink is an intentional cutover. They left and asked for autonomous execution.

CI bench regression baseline (`60e4312` vs main `f4e0022`):

| Case | Mode | Prove Δ | Verify Δ |
| --- | --- | --- | --- |
| fp32 onehot D128 nv28 | onehot_fp32_d128 | +20.68% | +128.21% |
| fp64 onehot D128 nv28 | onehot_fp64_d128 | -32.48% | +220.30% |
| fp128 dense D64 nv24 | dense_fp128_d64 | +0.19% | +5.56% |
| fp128 onehot D64 nv32 np1 | onehot_fp128_d64 | +46.65% | -15.51% |
| fp128 onehot D64 nv30 np4 | onehot_fp128_d64 | +46.43% | -15.08% |

Two distinct hotspots:
- D128 fp32/fp64 = K>1 claim field. Verifier blew up (+128/+220%).
- D64 fp128 onehot = K=1 claim field. Prover blew up (+46%).

**Done this pass:**

1. `gamma_tr` removed; trace batched as `γ²` of `CHALLENGE_SUMCHECK_BATCH`
   (sampled after the next-level witness is bound). Field/struct renamed
   `gamma_tr -> trace_coeff`. Terminal levels sample a dedicated `trace_gamma`
   from the same label while the virtual `batching_coeff` stays zero. Removed
   `CHALLENGE_TRACE_BATCH`. Fixes the FS-ordering soundness bug.
2. Silent-skip footgun replaced: `trace_stage2_enabled` (bool) ->
   `ensure_trace_stage2_supported` (`Result`), hard-erroring on unsupported
   claim-field degrees so a dropped `y_ring` can never leave the opening unbound.
3. **Verifier K>1 fold-blocks-first** (`field_reduction.rs`, `trace_weight/eval.rs`):
   the trace-open pipeline (`weighted_negacyclic_shift_sum` -> ring product ->
   `Tr_H` -> decode) is `E`-linear in the fold-block ring, so per term we now
   fold every block into one block-weighted lifted `E`-ring and take a single
   `Tr_H`, instead of one `Tr_H` (and one `h_exponents` recompute) per block.
   `trace_open_ring_mle_dot(F-ring, per-block)` -> `trace_open_folded_ring_mle_dot`
   (pre-folded `E`-ring). Turns the verifier K>1 path from `O(num_blocks·D²)`
   into `O(num_blocks·D)` fold + one `O(D²)` trace per term. All K2/K4/K8
   `trace_weight` unit tests pass. This targets the +128/+220% verifier
   regressions.
4. `r_hat` shrink confirmed as intentional cutover (see `schedule.rs:276`,
   `m_row_count_for(num_points, 0, layout)` zeroes public M-rows).

**Deferred: prover K=1 separable trace (`prover-separable`).**

The +46% prover regression on D64 fp128 onehot has two new costs vs main:
(a) the dense witness-sized `trace_compact` table folded every round, and
(b) `has_trace` disabling the stage-2 `two_round_prefix` bivariate skip
(`lifecycle.rs:104,187`). The prefix-y/x skips already fold the dense trace
(`accumulate_witness_relation_at_trace_indices*`), so only the two-round skip
is lost.

The clean fix is to represent the K=1 trace by its separable factors
(`tc · Σ_t colfactor_t(x) ⊗ inner_t(y)`, rank = #terms), fold `colfactor`
alongside `m_compact` and `inner` alongside `alpha_compact`, fuse the
contribution into the per-corner `p0/p1`, delete the dense table+fold, and
thread the rank-r term through the `two_round_prefix` `{0,1,∞}²` skip grid.

I deferred this rather than land it unsupervised because it is soundness-critical
hot-path surgery: it rewrites all six stage-2 round-poly variants plus the
intricate bivariate-skip grid, and full recovery *requires* the skip-grid
change (the dense-table deletion alone only recovers cost (a), a partial win).
I can't validate it against the CI bench locally, only against correctness
tests. Authoring is not the constraint; review + bench validation is. Design and
regression data captured above so it can be greenlit and landed as a focused
follow-up.

**Dead-code sweep (done + scoped).**

Done: removed the `combine_root_y_rings` -> `y_rings` -> `RingRelationProver`
data flow. After the public M-row removal that combined `y_rings` was computed
(`evaluate_root_claims_at_prepared_points` -> `combine_root_y_rings`) and threaded
through `finish_*root_fold_with_prepared_openings` only to feed a `y_rings.len()`
length check in `RingRelationProver::new` / `new_recursive_multipoint`. Deleted
the function, dropped the param from both constructors and both `finish_*`
helpers, removed the four call sites + the two test callers (`zk.rs`,
`ring_switch.rs`). The recursive per-point `y_rings` local stays (it still
computes `trace_eval_target` via `recover_ring_subfield_inner_product`); only the
dead constructor arg was removed. This kills the misleading data flow that
implied on-wire `y_ring` is still load-bearing. `akita-prover` + tests compile.

Scoped out (deliberately):
- `num_public` M-row dimension: still threaded through verifier-reachable M-row
  layout (`ring_switch/evals.rs`, verifier `levels.rs:635`) and schedule sizing.
  The r_hat witness-shape side already cut over to `m_row_count_for(.., 0, ..)`,
  but proving the M dimension itself is fully inert needs protocol analysis and
  is a soundness-sensitive change, not a mechanical sweep. Left intact.
- `trace_weight` public surface (`build_trace_weight_table_*`,
  `trace_stage2_supported`): genuinely used as in-crate test oracles / internal
  helper. Tightening visibility is churn, not dead-code removal. Left as is.

**D32 tables: kept (recommend split, not delete).**

User flagged the fp128 D32 schedule tables as experimental / not-load-bearing /
"keep or remove". Investigated commit `03c590f7`: it does NOT introduce the D32
config (`D32Full` etc. are pre-existing and used across verifier, setup, pcs
tests, benches, examples); it only ships the planner schedule tables + registers
`fp128_d32_full`/`fp128_d32_onehot` in `generated_families`. Removing those
tables would make D32 fall back to the runtime DP, which *slows down* every D32
setup/e2e test (more tests than the single `generated_schedule_tables_match_find_schedule`
check it would speed up) and is ~3000 lines of generated-table churn for marginal
benefit. Kept in this branch. The commit is cleanly isolated (no later commit
touched its files), so if PR review focus is the concern, split `03c590f7` into
its own PR rather than delete the work. Flagged for the user.
